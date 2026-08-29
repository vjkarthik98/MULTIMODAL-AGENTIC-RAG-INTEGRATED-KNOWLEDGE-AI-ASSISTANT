# GPU memory management — model lifecycle on a single GPU box

How MAGIK decides what stays in VRAM, what gets evicted, and when — plus how
this problem is solved in production systems, and the point at which this
design would need to be replaced rather than tuned.

## The two classes of model

The defining split is **who waits**. A model on the query path is one a user is
actively blocked on; a model on the ingestion path is one a background job is
blocked on. They get opposite treatment.

| | Models | Policy | Why |
|---|---|---|---|
| **Hot set (pinned)** | `llm` (via llama-server), `text_embedder` (BGE), `reranker` (BGE-reranker), `siglip` + `image_embedder` + `siglip_text_embedder` | **Never evicted.** Warmed at startup via `WARMUP_MODELS`. | Every query needs them. A reload mid-query is a multi-second stall a user directly feels. |
| **Cold set (evictable)** | `whisper`, `blip`, `qwen2_vl`, `qwen2_vl_video`, `trocr`, `diarizer`, `ner`, `finbert` | Loaded on first use of that modality; **evicted when idle or under VRAM pressure**. | Only touched during ingestion. Users do not upload continuously, so keeping ~15GB of VLM/ASR weights resident between uploads buys nothing. |

**SigLIP is pinned even though it sounds ingestion-only.** Its *text* encoder
runs on every query for cross-modal search — visible in any Tier-2 log as
`siglip_text_embed_success` on each retrieval. Evicting it would put a
multi-second reload on a live user's query. This is the kind of thing that is
only obvious from tracing callers, so it is asserted in
`tests/unit/core/test_model_reaper.py` rather than left to a comment.

## The policy

`app/core/model_reaper.py` runs a sweep every `MODEL_REAPER_INTERVAL_SEC`:

```
if gpu_busy():                        skip entirely
elif free_vram < WATERMARK_GB:        pressure -> evict least-recently-used
else:                                 routine  -> evict anything idle > TTL
```

### Why two triggers and not just a TTL

A TTL alone is wrong in both directions:

- **Too eager.** On a card with 30GB free, evicting a model nothing is
  competing for pays a ~25s reload to reclaim memory no one wanted.
- **Too slow.** A model idle for four minutes is still resident while a large
  ingest is about to OOM.

The watermark handles the case that actually matters — memory under
*contention*. The TTL is the baseline that keeps a quiet box from drifting
full. This is the same least-recently-used-under-pressure policy KServe
ModelMesh applies across a model pool.

### Why the sweep skips while a GPU slot is held

Eviction only drops the *loader's* reference. A job that already called the
getter holds its own reference, so Python refcounting keeps those weights alive
— dropping mid-job is **memory-safe and never crashes**. But it would force a
reload on the next stage and briefly hold two copies, which is the opposite of
the point. `gpu_admission.gpu_busy()` already knows when heavy GPU work is in
flight, so gating on it makes that impossible rather than merely unlikely.

### Why free VRAM is read from the driver

`device_manager.free_vram_gb()` uses `torch.cuda.mem_get_info()`, **not**
`total_memory - torch.cuda.memory_reserved()`. The latter counts only the
calling process's PyTorch caching allocator, which on this deployment is a
minority of what is resident: the llama-server process holds the 14B GGUF, and
during Tier-2 the eval judge holds another ~5GB in its own worker process.
This is not hypothetical — `qwen_judge.py` shipped with the allocator-only
version, which reported tens of GB "free" on a nearly-full card and left its
CPU-fallback guard permanently inert.

### Why full unload, not CPU offload

The usual next step is sleep/offload rather than unload — vLLM's
`sleep(level=1)` moves weights to host RAM and wakes in ~1s instead of ~25s
from disk; `accelerate` CPU offload and DeepSpeed ZeRO-Inference do similar.

**Deliberately not used here.** g6e.xlarge has 32GB host RAM, and Tier-2 has
already been killed by the host OOM killer (run 30745527098, exit 137 =
SIGKILL). Pushing ~15GB of weights into host RAM would worsen a failure mode
we have actually observed on this box. Full unload trades reload latency for
host-RAM safety, which is the right trade at this memory ratio. On a box with
128GB+ host RAM the answer would flip.

## Configuration

All in `app/core/config.py`; defaults are production-tuned.

| Setting | Default | Notes |
|---|---|---|
| `MODEL_IDLE_EVICTION_ENABLED` | `true` | Master switch; `false` restores the old always-resident behaviour. |
| `MODEL_IDLE_TIMEOUT_SEC` | `300` | Long enough that a user uploading several files in one sitting never pays a reload between them. |
| `MODEL_REAPER_INTERVAL_SEC` | `60` | Floored at 5s (`_MIN_INTERVAL_SEC`) so a `0` cannot busy-wait the driver. |
| `MODEL_EVICT_VRAM_WATERMARK_GB` | `6.0` | Urgent trigger. `0` disables pressure eviction, leaving TTL only. |

## Observability

Scraped by the existing Prometheus/Grafana stack (`monitoring/`):

- `model_loaded{model}` — 1/0, flips to 0 on eviction.
- `model_evictions_total{model,reason}` — `reason` is `idle` or `pressure`.
  A rising `pressure` count means the box is genuinely undersized for its
  workload, not merely idle-trimming; that is the signal to act on.
- `gpu_vram_free_gb` — published every sweep, whether or not anything is
  evicted (the gauge is most useful when flat).

Log events: `model_evicted_idle`, `model_evicted_pressure`,
`model_reaper_swept`, and `vram_below_watermark_nothing_evictable` — the last
means the memory is held by the pinned hot set or another process and no amount
of sweeping will help.

---

## How this is done in industry

Worth stating plainly: **an in-process reaper is not how a large production
system solves this.** It is the right answer for a single-box deployment, and
the wrong answer past a certain scale. What real systems do:

### 1. Tier separation — the dominant answer

Online serving and batch/ingestion are **different services with different
scaling policies**:

- **Online tier** (embedder, reranker, LLM): always on, autoscaled on latency
  or QPS, models permanently resident. Never evicts anything.
- **Ingestion tier** (Whisper, VLM captioners, OCR, diarization): workers that
  pull from a queue and **scale to zero when it drains**. VRAM is reclaimed by
  the worker *process exiting* — the most reliable eviction there is, because
  it cannot leak, fragment, or forget.

Typical stack: Ray Serve / KServe / Triton for serving; Celery, SQS, or Kafka
for the queue; KEDA or Ray autoscaling to drive replicas to zero.

This is precisely the intuition behind "once the ingestion work is over, why
should it stay in GPU" — the industry answer is that it shouldn't even be in
the same *process*, let alone the same GPU allocation.

### 2. Model servers with an explicit load/unload control plane

- **Triton** `--model-control-mode=explicit` — `POST /v2/repository/models/{name}/load`
  and `/unload` give an external orchestrator direct control.
- **TorchServe** — register/unregister a model at runtime.
- **KServe ModelMesh** — the closest formal analog to what this repo
  hand-rolls: an LRU cache of models across a pool of serving pods, with
  capacity-aware placement and automatic eviction under memory pressure.

### 3. Sleep/offload tiers

Rather than binary loaded/unloaded, production inference stacks increasingly
keep a middle state: weights in host RAM (vLLM sleep mode), or sharded across
GPU/CPU/NVMe (DeepSpeed ZeRO-Inference, `accelerate` `device_map="auto"` with
`max_memory`). Turns a ~25s cold load into a ~1s wake, at the cost of host RAM.

### 4. Node-level scale-to-zero

MAGIK already does this: the wake-gateway Lambda (`deploy/aws/`) stops the
whole EC2 box when idle and starts it on the next request. For a demo, this is
strictly more cost-effective than any in-process eviction, because it also
reclaims the *instance* cost, not just VRAM. In-process eviction complements it
by keeping a single active session from OOMing itself.

### When this design should be replaced

Move to the separate-worker architecture (1) when any of these becomes true:

- Ingestion volume is high enough that models rarely go idle, so the reaper
  stops firing and VRAM stays full anyway.
- Ingestion latency starts mattering to users — repeated cold loads (~25s per
  VLM) become the dominant cost, and you need warm dedicated workers instead.
- You need to scale ingestion and serving **independently**, which a single
  process fundamentally cannot do.
- More than one GPU is involved — placement across devices is exactly the
  problem ModelMesh/Ray exist to solve, and hand-rolling it is a mistake.

Until then, the reaper plus node-level scale-to-zero covers the same ground at
a fraction of the operational complexity.
