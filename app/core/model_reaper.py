"""Background VRAM reaper — the unload half of the model lifecycle.

WHY THIS EXISTS
---------------
`app/core/model_loader.py` has had `unload_idle_models(idle_seconds=...)`
since 2026-08-01, but its only caller was `_oom_guard(loading=...)`, which
runs exclusively when ANOTHER heavy model is about to load. Eviction was
therefore reactive to demand pressure and never to idleness — the
`idle_seconds` argument was only ever evaluated at the moment of the next
load. In practice: a user uploads one image, BLIP2 and Qwen2-VL load, and if
nothing else is ever uploaded those models sit in VRAM for the life of the
process.

`WARMUP_MODELS` already gives load-on-demand (vision/audio/video models load
on first use of that modality via `model_registry.ensure_for_modality()`).
This module is the missing symmetric half: unload-on-idle.

POLICY
------
Two triggers, one guard, evaluated every MODEL_REAPER_INTERVAL_SEC:

    if gpu_busy():                    skip entirely
    elif free_vram < WATERMARK_GB:    pressure -> evict least-recently-used
    else:                             routine  -> evict anything idle > TTL

Only the 8 ingestion-only models in `_EVICTABLE_MODELS` are ever touched.
The query hot set — llm, text_embedder, reranker, and the SigLIP family
(its text path runs on every query for cross-modal search) — is pinned in
the loader and unreachable from here, so a live query never waits on a
reload.

WHY BOTH TRIGGERS
-----------------
TTL alone is simultaneously too eager and too slow. Too eager: on a card
with plenty of headroom it evicts models nothing was competing for, paying a
~25s reload to free memory nobody wanted. Too slow: a model idle for four
minutes is still held while a large ingest is about to OOM. The watermark
covers the case that actually matters — memory under contention — which is
the same least-recently-used-under-pressure policy KServe ModelMesh applies
across a model pool. See docs/runbooks/gpu-memory-management.md.

WHY NOT CPU-OFFLOAD INSTEAD OF UNLOAD
-------------------------------------
vLLM-style sleep/offload (weights to host RAM, ~1s wake instead of ~25s from
disk) is the usual next step, and is deliberately NOT used here: g6e.xlarge
has 32GB host RAM and Tier-2 has already been SIGKILLed by the host OOM
killer (run 30745527098, exit 137). Moving ~15GB of weights into host RAM
would worsen a failure we have actually observed. Full unload is correct on
this box.
"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.gpu_admission import gpu_busy
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Floor on the sweep interval, so a misconfigured MODEL_REAPER_INTERVAL_SEC=0
# cannot spin this loop into a busy-wait against the CUDA driver. Exposed as a
# module constant rather than inlined so tests can lower it and exercise the
# loop at speed — without it, any test of the loop's behaviour either takes
# 5+ real seconds or passes vacuously by never completing an iteration.
_MIN_INTERVAL_SEC = 5.0


def _sweep_once() -> list[str]:
    """One synchronous eviction pass. Returns the models evicted, if any.

    Blocking by design — takes the loader's RLock and, on eviction, runs
    gc.collect() + torch.cuda.empty_cache() + torch.cuda.synchronize(), which
    can cost hundreds of ms. The caller runs it via asyncio.to_thread() so it
    never stalls the event loop serving live queries.
    """
    # Publishes gpu_vram_free_gb for Grafana on every tick, whether or not
    # anything is evicted — the gauge is useful precisely when it is flat.
    free_gb = model_loader.refresh_vram_gauge()

    watermark = settings.MODEL_EVICT_VRAM_WATERMARK_GB
    if watermark > 0 and free_gb is not None and free_gb < watermark:
        evicted = model_loader.unload_until_free(watermark)
        if evicted:
            return evicted
        # Under the watermark with nothing evictable left to drop: the memory
        # is held by the pinned hot set or another process, and no amount of
        # sweeping will help. Say so once per tick rather than silently
        # looking like the reaper is doing something about it.
        logger.warning(
            "vram_below_watermark_nothing_evictable",
            free_vram_gb=round(free_gb, 2),
            watermark_gb=watermark,
        )
        return []

    return model_loader.unload_idle_models(idle_seconds=settings.MODEL_IDLE_TIMEOUT_SEC)


async def run_model_reaper_loop() -> None:
    """Periodic idle/pressure eviction. Started from app/main.py's lifespan.

    Every iteration is wrapped: an unhandled exception here would kill the
    task silently and turn VRAM reclamation off for the life of the process
    — precisely the failure this module exists to prevent. (The neighbouring
    run_online_eval_loop() has no such guard; a dead eval loop only costs a
    stale dashboard, whereas a dead reaper costs the GPU.)
    """
    if not settings.MODEL_IDLE_EVICTION_ENABLED:
        logger.info("model_reaper_disabled")
        return

    interval = max(settings.MODEL_REAPER_INTERVAL_SEC, _MIN_INTERVAL_SEC)
    logger.info(
        "model_reaper_started",
        interval_sec=interval,
        idle_timeout_sec=settings.MODEL_IDLE_TIMEOUT_SEC,
        watermark_gb=settings.MODEL_EVICT_VRAM_WATERMARK_GB,
    )

    # Let startup warmup finish before the first sweep, so the reaper never
    # races _warmup_models_async() and evicts something it just loaded.
    await asyncio.sleep(interval)

    while True:
        try:
            # Checked HERE, on the event loop, not inside the worker thread:
            # the counter it reads is mutated only by coroutines on this loop
            # (app/core/gpu_admission.py's gpu_slot), so this is the one place
            # it can be read without a race.
            #
            # Skipping while a slot is held keeps the reaper from dropping a
            # model an ingestion job is mid-way through. That is memory-safe
            # either way — the job holds its own reference, so refcounting
            # keeps the weights alive — but it would force a reload and
            # briefly hold two copies, which is the opposite of the point.
            if gpu_busy():
                logger.debug("model_reaper_skipped_gpu_busy")
            else:
                evicted = await asyncio.to_thread(_sweep_once)
                if evicted:
                    logger.info("model_reaper_swept", evicted=evicted, count=len(evicted))
        except asyncio.CancelledError:
            # Normal shutdown path — lifespan cancels background tasks.
            logger.info("model_reaper_stopped")
            raise
        except Exception as exc:  # noqa: BLE001 — must never kill the loop
            logger.warning("model_reaper_iteration_failed", error=str(exc))
        await asyncio.sleep(interval)
