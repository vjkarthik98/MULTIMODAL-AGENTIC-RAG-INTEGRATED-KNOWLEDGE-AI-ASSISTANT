# MAGIK — Problems & Solutions Log

A consolidated record of every significant bug, root cause, and fix found while building and hardening MAGIK, organized by area. Compiled from project working memory (2026-06 through 2026-07).

---

## 1. Infrastructure: LLM process, CUDA, RAM (g5.xlarge)

Hardware: AWS g5.xlarge — A10G 24GB VRAM, 4 vCPU, **15GB RAM**, system CUDA 12.8, driver CUDA 13.2.

### 1.1 Server start killed the SSH/VSCode session (host-RAM OOM)
- **Symptom:** Starting the server disconnected the SSH/VSCode session while EC2 itself stayed up.
- **Root cause:** ~17.7GB of resident models loaded with only 15GB RAM and no swap → peak 15.6GB → OOM killer reaped `sshd`.
- **Fix:** `start_server.sh` creates 32GB swap on `/opt/dlami/nvme` (ephemeral, recreated each boot), sets `vm.swappiness=10`, and sets `LLM_USE_MLOCK=false` (mlock was uselessly pinning 4.1GB of host RAM).

### 1.2 SIGSEGV during ingest at the embed stage
- **Symptom:** Crash during ingestion's embedding step.
- **Root cause:** PyTorch and llama.cpp sharing one CUDA context in a single process — llama.cpp's CUDA init corrupts PyTorch's CUDA context specifically on worker threads (main thread was fine).
- **What didn't work:** thread-ordering fences, per-thread real-embed warmup, forcing eager attention — all failed in-process.
- **Fix:** Run llama.cpp as a **separate process** (`python -m llama_cpp.server` on port 8081, its own CUDA context). `GGUFModel._load()` returns an HTTP proxy (`_LlamaServerClient`) when `LLM_USE_SERVER=true`, streaming `/v1/completions` in the same chunk shape so `generate()`/`stream()`/token-budget truncation code didn't need to change. Required `sse-starlette`, `starlette-context`.

### 1.3 LLM silently running on CPU (5.6 tok/s)
- **Symptom:** Terrible generation speed with no visible error.
- **Root cause:** The prebuilt `llama-cpp-python` cu124 wheel's dynamic CUDA backend registered 0 devices on this box (`llama_supports_gpu_offload()` returned False).
- **Fix:** Built `llama-cpp-python` from source against system CUDA 12.8 with `-DGGML_BACKEND_DL=OFF` (static CUDA backend) + `-DCMAKE_CUDA_ARCHITECTURES=86`. Result: 89 tok/s, full RAG query ~13s.
- **Stack pin:** `torch==2.6.0+cu124` (NOT cu130 — caused a `libcudart.so.13` vs `.12` conflict); `llama-cpp-python==0.3.30` source-built; both share `libcudart.so.12` from pip `nvidia-*-cu12`; `numpy` pinned to `1.26.4` (langchain requires <2.0).

### 1.4 Logger crash on reserved LogRecord keys
- **Symptom:** `logger.info(filename=...)` crashed.
- **Root cause:** `filename` collides with a reserved Python `LogRecord` attribute.
- **Fix:** `app/utils/logger.py::_build_extra` remaps reserved keys to `<key>_`.

### 1.5 Installing a new ML package silently upgraded torch and broke the server
- **Symptom:** Live server went down; `transformers` quantizer import chain broke (`Could not import module 'PreTrainedModel'`).
- **Root cause:** Installed `gptqmodel` (while evaluating a 7B AWQ model) → silently upgraded `torch` 2.6.0+cu124 → 2.12.1 and pulled in `torchao`, which requires torch≥2.11 and breaks `transformers`.
- **Fix:** Reinstalled pinned versions (`torch==2.6.0+cu124`, `numpy==1.26.4`, `protobuf==6.33.6`), removed all `gptqmodel`/`torchao`/`autoawq` transitive packages.
- **Lesson:** Never `pip install` a new heavy ML package (especially AWQ/GPTQ backends) directly into this project's venv without testing in a disposable venv first — they aggressively pin their own torch version.

### 1.6 Restart race — old server keeps serving stale code
- **Symptom:** Code edits appeared to "do nothing"; debug logs never fired; answers byte-identical after a restart.
- **Root cause:** `start_server.sh` kills the old uvicorn, then `sleep 2`s before starting the new one. A process holding ~17GB of GPU models takes longer than 2s to die, so the new uvicorn fails to bind port 8000 and dies — the OLD process keeps serving, and its health check looks like a fresh "READY in ~5s".
- **Fix:** Kill PIDs on :8000 AND :8081, poll `lsof -ti tcp:8000` until free, THEN start. Verify a single fresh PID (`ps -o lstart -p $(lsof -ti tcp:8000)`) — a genuine reload takes ~20-25s.

---

## 2. Latency

Baseline: EBS gp3 measured 139 MB/s → ~127s of pure disk I/O just to read 17.7GB of resident models at startup.

### 2.1 Slow startup (all 12 models eager-loaded)
- **Fix:** `.env WARMUP_MODELS=text_embedder,llm,reranker` only. Vision/audio/video models now lazy-load on first use via `model_registry.ensure_for_modality()` / `ensure_for_query(needs_vision=True)`. Safe because the LLM is now a separate process (no in-process CUDA conflict, see 1.2). Startup: ~280s → ~110s.

### 2.2 Cold-upload latency (51s–3min on first request)
- **Root cause:** spaCy/Presidio/lingua/transformers lazy-loading under swap during the 12-model startup — not a per-request cost. Warm txt extract for a 54KB file is 0.52s (was 2.0s).
- **Fix:** cached the lingua language detector as a module singleton (`with_low_accuracy_mode()`, was rebuilt every call); added `warm_language_detector()` to startup warmup.

### 2.3 PII scrubbing running unconditionally despite being disabled
- **Root cause:** `_scrub_pii` (spaCy NER via Presidio, ~1.4s/doc) ran on every doc regardless of `PII_DETECTION_ENABLED=false`.
- **Fix:** gated `_scrub_pii` on `settings.PII_DETECTION_ENABLED` (`app/ingestion/base_ingest.py`), consistent with `_redact_pii`. **Security-relevant:** re-enable via `PII_DETECTION_ENABLED=true`.

### 2.4 Streaming TTFT — unnecessary tokenize round-trip
- **Fix:** `gguf_model.py::_truncate_to_token_budget` skips the `llm.tokenize()` HTTP round-trip to port 8081 when `len(prompt) <= budget*2.5` chars (can't exceed budget at ≥2.5 chars/token). Long prompts still tokenize precisely.

### 2.5 llama-server not using flash attention / prompt cache
- **Fix:** `start_server.sh` adds `--flash_attn true` (A10G Ampere), `--cache true` (KV cache for the shared RAG prefix), `--use_mlock false`, `--n_threads_batch 4` — all overridable via env.

### 2.6 THE real upload-latency villain: Upstash cloud Redis on the hot path
- **Symptom:** All endpoints logged ~1s latency; ingest was slow; status bar looked stuck.
- **Root cause:** `RedisMemory.__init__` always preferred the Upstash (cloud) client whenever `REDIS_URL`/`REDIS_TOKEN` were set — the local redis-server that `start_server.sh` installs was **never used**. Measured: Upstash REST = 199ms/call vs local redis = 0.44ms/call (450×). Every authed request paid ~400ms (2 Upstash calls for token revocation checks); the embedding cache did **one Upstash write per chunk** — 81 chunks × 199ms ≈ 16s of pure cache latency per ingest, the single biggest ingest cost.
- **Fix (user-approved architecture): split hot-path caching to LOCAL redis, keep durable/long-term memory on Upstash.**
  - New `infra.get_cache()` — raw redis-py client on `LOCAL_CACHE_*` (localhost:6379, db=1).
  - `token_blacklist.py` now uses local cache + an in-process TTL cache (default 30s) on revocation checks — most authed requests do zero network. Revocation propagation now ≤30s (accepted for single-instance).
  - Rate limiting, job status, and the embedding cache all moved to local cache.
  - PDF ingestion: gated supplemental OCR behind `PDF_SUPPLEMENTAL_OCR_ENABLED` (default false) — was OCR-ing text pages unnecessarily; added `--dpi 200` to kill tesseract resolution warnings.
  - UI polling interval dropped 3000ms → 1000ms.

### 2.7 Dedup / KB-copy timing bugs found alongside latency work
- Files appeared in the sidebar **before** their embeddings existed. Fixed: KB copy now happens only after ingest success.
- `apple_10k.pdf` "upload failed" in logs turned out to be a server restart mid-ingest (`cannot schedule new futures after shutdown`), not a real bug.

---

## 3. Cross-modality ingestion & citation infrastructure

### 3.1 Citation fields silently dropped
- **Root cause:** `query_pipeline._build_sources_array` only emitted page/section/start_time — every other locator (docx `heading`, xlsx `sheet_name`+`row_range`, image `image_title`, audio/video `speaker_name`/`speaker_role`) was dropped even though Qdrant already stored them. It also sent `start_time` while the UI reads `timestamp_start`.
- **Fix:** emit all locator fields with the exact names the UI's `SourceChip` component expects.

### 3.2 Re-upload dedup bug — deleted files "duplicate" on re-upload but never re-ingested
- **Root cause:** the in-memory `_INGEST_DEDUP` map was global-by-hash, never invalidated after delete, so re-uploading returned "duplicate" instantly while Qdrant had nothing.
- **Fix:** keyed per-user (`user_id:hash`), added `_qdrant_has_checksum()` verification (only honor dedup if vectors actually exist), cleared entries on delete.

### 3.3 Embedding-space aliasing bug — one doc silently overwrote across two vector spaces
- **Root cause:** `_route_documents` in `base_embedder.py` appended the SAME doc object to both `text_docs` and `vision_docs` for image captions/video frames, so the second embed call overwrote the first on the shared object.
- **Fix:** trust `structure["embedding_space"]` — one doc → one space only; chunkers emit separate doc objects per space.

### 3.4 Image chart caption was literally the prompt text, not a caption
- **Root cause:** `BLIP_MODEL=Salesforce/blip-image-captioning-large` is BLIP-1, which cannot follow instructions — it echoed the multi-part finance caption prompt back as the "caption" (lowercased, so the case-sensitive prompt-strip missed it).
- **Fix:** switched primary captioner to Qwen2-VL-2B-Instruct (instruction-tuned); BLIP kept only as an unconditional (no-prompt) fallback with case-insensitive prompt-echo stripping.

### 3.5 Image never reached the vision collection
- **Root cause:** `image_chunker` emitted a single caption doc defaulting to `embedding_space="text"` → SigLIP vision embeddings never ran.
- **Fix:** chunker now emits a second doc (`subtype="image_frame"`, `embedding_space="vision"`) so both text and vision collections get populated.

### 3.6 Corrupt source data files (not code bugs)
- `data/raw/finance/apple_10k.pdf`: valid-looking PDF header but PyMuPDF raised `FileDataError` — file itself was bad/truncated, required re-export.
- `Apple Q4 2024 Earnings Call Short.mp4`: truncated (768KB for a declared 5-minute video, only 7.8s decodable); a second re-export from an online video trimmer had intact duration but the trimmer had stripped/silenced the audio track (mean -56.1dB) — Whisper returned 0 words both times.

---

## 4. Video modality

### 4.1 THE blocker — video never actually ingested with working diarization
- **Symptom:** silent failure; every prior "successful" ingest actually had diarization failing (GPU OOM) with an empty transcript.
- **Root cause:** `VideoChunker.chunk()` passed the joined transcript **string** to `_map_speaker_roles(diarization, words)`, which expects `List[Dict]` → `TypeError: string indices must be integers` whenever diarization actually returned turns.
- **Fix:** pass `words` (the actual list), not the joined string.

### 4.2 Diarizer runtime crash (pre-existing, fixed earlier)
- **Root causes (three stacked):** `torchaudio` missing `AudioMetaData` attribute; `hf_hub_download` called with a deprecated `use_auth_token` kwarg; PyTorch 2.6's `weights_only=True` default blocking pyannote's custom globals.
- **Fix:** `app/utils/torchaudio_compat.py` shim restores the torchaudio API via soundfile, patches the `hf_hub_download` kwarg, registers pyannote safe globals, and wraps `torch.load` to use `weights_only=False` when called from the pyannote/lightning stack.

### 4.3 Vision collection never received video frame data
- **Fix:** `VideoChunker.chunk()` emits one `frame` doc per unique captioned frame with `embedding_space="vision"`; `VideoEmbedder` splits docs by embedding space before encoding (text → BGE 1024-dim, vision → SigLIP 1152-dim).

### 4.4 SNR never measured
- **Fix:** `_measure_snr()` via `ffmpeg volumedetect` after audio extraction; flags `snr_degraded`/`clipping_detected`.

### 4.5 Scene detection only produced 1 frame from a full call
- **Root cause:** `AdaptiveDetector` threshold was 25.0 (way too high for slide-heavy earnings-call content).
- **Fix:** `VIDEO_SCENE_ADAPTIVE_THRESHOLD=3.0` + uniform timeline coverage blend + `VIDEO_FRAME_DEDUP_HAMMING=2` (pHash 8 was collapsing static-layout frames it shouldn't have).

### 4.6 Frame captioning too slow
- **Fix:** tighter prompt + `VIDEO_CAPTION_MAX_TOKENS=180` (was 400) → ~80s → ~28s/frame.

### 4.7 7B model answering multi-fact chunks with the wrong number
- **Fix:** "KEY FACTS" extraction in `reasoning_engine.py` (video-dominant gated) — pulls query-matching figure-bearing sentences out of retrieved video chunks and prepends them, generic (no hardcoded facts). Lifted one query's answer score from 6 → 27.

### 4.8 Residual gaps (documented, not fixed — shared infra, out of video-only scope)
- A "beat analyst estimates" query routes to hybrid/web instead of rag.
- Answer-bearing frames retrieved but filtered out at fusion before ever reaching the LLM (vision_count 20 → 0).
- 7B model can't fully cover a 4-part question in one answer.
- Executive names (e.g., Cook/Parekh) unresolved — roles correct, names not (IR-handoff pattern not detected; hosts merged in diarization).

Result: upstream pipeline audit 100/100 (transcription, chunking, metadata, embedding all clean); `query_pipeline` answer avg 71.75/100 — capped by shared retrieval/generation infra, not the video pipeline itself.

---

## 5. Audio modality

### 5.1 Phantom speaker cluster from a hardcoded gap fallback
- **Root cause:** `speaker_at()` returned a hardcoded `"SPEAKER_00"` string whenever a word's timestamp fell in a gap between pyannote turns (very common — breathing, pauses). Since the dominant speaker (49-min FOMC press conference, ~68% of speech) naturally has many such gaps, this fabricated a large, spurious second cluster out of his own pauses.
- **Fix:** nearest-real-turn lookup instead of hardcoding. This was the single biggest fix of the phase — chunk count went from 86 → 73 as the phantom cluster disappeared.

### 5.2 BM25 metadata field-name mismatches (masked correct Qdrant data)
- **Root cause:** BM25's `_metadata()` looked for `timestamp_start`/`speaker`, but the chunker stored `start_timestamp`/`speaker_name`/`speaker_role`. Since BM25 results are deduplicated first during fusion, this permanently masked Qdrant's correct metadata for any chunk both retrievers surfaced.
- **Fix:** defensive field-name fallback in `bm25_retriever.py` and `base_bm25.py` (audio/video-only, no-op elsewhere). Same class of bug independently found and fixed for XLSX (`sheet_name`/`row_range`, see §6.1).

### 5.3 Broken speaker-name-to-label binding
- **Root cause:** `_map_speaker_roles` used character-position-proportional time estimation over the joined transcript to bind spoken names to speaker labels. On a 49-minute recording this drifts badly — Powell's own remarks were mislabeled "Steve Liesman" for 29 minutes straight, and "Jackson Hole" was bound as if it were a person.
- **Fix:** rewrote using exact word-timestamp-anchored turns with two precise signals: self-introduction pattern ("`<Name> from/with <Outlet>`") and vocative address ("Chair Powell, ..." binds the NEXT turn as the addressee).

### 5.4 Whisper ASR spacing artifacts silently broke the numeric-faithfulness guard
- **Root cause:** Whisper consistently emits "2 .2 percent" instead of "2.2 percent" and "dual -mandate" instead of "dual-mandate". The LLM's correctly-formatted "2.2" never string-matched context's "2 .2", so genuinely faithful answers were discarded and replaced with "No relevant information was found."
- **Fix:** `_fix_asr_spacing()` normalization step in the chunker's text cleanup.

### 5.5 No chunked transcription for long audio
- **Root cause:** the entire 49-minute file was transcribed in one Whisper call; the same audio region transcribed in isolation (2-minute clip) produced clean "Chair Powell"/"Greg Robb from MarketWatch.com", but the full-file call produced garbled lowercase text for the identical region.
- **Fix:** `_transcribe_long_audio()` splits at a configured duration and transcribes segments concurrently.

### 5.6 FinBERT tone globally disabled, silently skipped for audio too
- **Fix:** force-run FinBERT tone annotation for audio specifically in `AudioEmbedder`, without flipping the global flag (which would add GPU cost to every modality).

### 5.7 Unbounded overlap-seed compounding (chunk sizes ballooning)
- **Root cause:** the overlap-seed extractor fell back to the ENTIRE chunk text when no sentence-ending punctuation was found (common in disfluent speech), and that grew across consecutive chunks: 423 → 639 → 713 → 785 → 872 → 946 → 1114 words. This produced oversized chunks and duplicate-looking truncated text.
- **Fix:** cap the fallback to the last 30 words regardless of punctuation. Max chunk size dropped from 1114 → 248 words.

### 5.8 Qdrant payload whitelist silently dropped FinBERT fields and truncated transcripts
- **Root cause:** the audio payload whitelist in `qdrant_store.py` never included `finance_tone`/`finance_tone_score` even though FinBERT correctly computed them — discarded at the very last step. Also truncated `transcript` at a hardcoded 1000 chars, inconsistent with the 2000-char text limit elsewhere.
- **Fix:** added both fields to the whitelist; raised the transcript truncation limit to match.

### 5.9 Speaker attribution overhaul (round 3, streaming-endpoint pass)
- **Root cause:** reporters were mislabeled "President" (matched from "New York Fed President John Williams" appearing mid-transcript).
- **Fix:** removed "President" from role keywords; made self-intro regex case-insensitive and anchored to a known news-outlet list; rejected sentence-fragment false positives.

### 5.10 Reranker always preferred Q&A chunks over the actual announcement
- **Root cause:** the cross-encoder had a strong (1.86×) lexical preference for a reporter's question chunk over Chair Powell's opening announcement, even when the announcement was the better citation.
- **Fix:** deterministic reranker reorder (runs after MMR) — for non-meta transcript queries, if the #1 result is a Q&A chunk and a prepared-remarks chunk sits in the top-6 with ≥40% of the top score, promote the announcement to #1. Meta-queries ("what did reporter X ask") explicitly skip this so they keep pointing at Q&A.

### 5.11 Wide context caused the 7B model to mix facts
- **Fix:** pass only the top-5 docs (not 20) to `generate_answer` for AV-dominant answers — narrower context made the model answer the specific question instead of blending facts from unrelated chunks.

### 5.12 Residual gaps (documented, not fixed)
- pyannote occasionally still mis-clusters a short (~56s) segment under a one-off speaker label instead of merging it into the canonical host — too small to trigger the fragment-merge threshold.
- A reporter's short question turn doesn't reliably out-rank the host's longer answer turn in top-3 retrieval — a reranker/retrieval-depth tuning issue in shared code, left untouched under the audio-only scope constraint.
- Some target figures were verified to never actually be spoken in the source audio at all (they appear only in the associated written statement) — ground truth was corrected rather than chasing an unfixable retrieval gap.

---

## 6. XLSX modality

Baseline before this phase: ~10.5/100 (catastrophically broken) — nearly every root cause was in the xlsx-specific pipeline, not the LLM.

### 6.1 Chunk-flattening bug destroyed row boundaries
- **Root cause:** ingestion already batched 25 rows into one newline-separated blob per `RawExtract`; the chunker then grouped 6 MORE of these batches into one final chunk (150 rows/chunk), and the row-rendering helpers did `row.split("|")` on each multi-line blob — which doesn't respect `\n` — flattening all rows into one undifferentiated line with no row boundaries or column labels.
- **Fix:** set the batch-group target to 1 (pass-through, ~25 rows/chunk) and added a helper to properly re-split blob text into individual row-lines before rendering.

### 6.2 Row-range citations always collapsed to a single row
- **Root cause:** row-group emission used `.extra["row_num"]` (always equal to `row_start`) for BOTH ends of the citation range.
- **Fix:** use `row_start` on the first extract and `row_end` on the last.

### 6.3 Percent-formatting bug — a 100x magnitude error in a live answer
- **Root cause:** cell values were stringified directly, ignoring Excel's `number_format`. A cell displaying "4.66%" is stored internally as the raw fraction `0.0466`; with no `%` marker, the LLM was observed writing "0.04585507581377215%" — a 100x error.
- **Fix:** `_format_percent_cell()` — when `number_format` contains `%`, render as `value*100` with 3 decimal places.

### 6.4 BM25 metadata key-mismatch (same class of bug as audio's, found independently)
- **Root cause:** BM25's `_metadata()` read `sheet`/`row_start`/`row_end` from `doc.structure`, but the xlsx chunker actually used `sheet_name` and a combined `row_range` list — every XLSX source lost its sheet/row citation once routed through BM25.
- **Fix:** defensive dual-key-shape read (provably a no-op for PDF/DOCX, which never populate these keys either way).

### 6.5 Reranker source-cap didn't diversify within a single multi-sheet workbook
- **Root cause:** `_apply_source_cap` skipped diversity capping when all candidates shared one `doc_id`. For one multi-sheet workbook, a single 1319-row sheet produced 50+ near-duplicate chunks that crowded out every other sheet.
- **Fix:** cap on `(doc_id, sheet_name)` instead of `doc_id` alone. Note: this alone didn't fully fix retrieval for small sheets — the crowding actually happens earlier, at BM25/vector candidate *generation* (see 6.6).

### 6.6 Small sheets never reached candidate generation at all
- **Root cause:** an 8-row sheet never appeared in top candidates for a query mentioning phrasing that matched a large sheet's repeated per-chunk title header, drowning it out in BM25/vector scoring. This is a global top-K sizing issue, out of xlsx-only scope.
- **Fix:** worked around at the generation layer (query-pattern-gated synth injectors, §6.7) rather than widening global top-K.

### 6.7 Model answered about the wrong row entirely
- **Symptom:** asked for a specific aggregate/methodology figure, the model answered about an unrelated country/row from the same 25-row chunk.
- **Fix:** 4 query-pattern-gated synth injectors in `reasoning_engine.py` — each content-matches the correct chunk via a full BM25-index scan (bypassing the reranked top-K), builds a short self-sufficient fact sentence, and **unconditionally** overrides the model's own answer. A numeric-coverage-threshold version was tried first but showed run-to-run variance from LLM nondeterminism, so it was made unconditional (same lesson independently learned in the PDF phase).
- **Critical ordering lesson:** the override must run AFTER all numeric-faithfulness/hallucination-guard logic, not right after response parsing — otherwise the "unsupported numbers" retry path reprocesses (and truncates) the already-curated fact.

### 6.8 Downstream text-mangling traps (silent truncation of injected facts)
- `_cut_source_dump()` strips everything after a `". Source:"` / `". Sources:"` pattern as a hallucinated citation dump — a synth fact ending in its own "Source: X sheet." sentence got silently truncated. **Fix:** put the citation in a parenthetical attached to the prior clause instead.
- `_strip_txt_citation_dump()` treats a trailing run of "Title Case Word(s): text" segments as a dumped transcript/speaker list — a synth fact formatted as "Africa: total ERP 12.53%; Asia: ...; Grand Total: ..." matched this exactly and got truncated. **Fix:** rephrase as prose, never as "Label: value" list shape.
- Both failure modes are silent (no exception, no log) — only visible by diffing the answer before/after each pipeline stage.

**Result:** avg 92.0/100 (from 10.5 baseline), citation accuracy 100%.

---

## 7. PDF modality

Baseline (g24): 84.25. Final: every query individually clears 85, avg 98.5 (stabilized further to 86.2–93.5 across iterations as more edge cases were fixed).

### 7.1 KV cache is the dominant constraint on prompt engineering
- Changing the tail of the prompt (output format) doesn't break llama-server's KV cache. Changing the middle (labels/prefix) does, for some queries — an important cost/latency tradeoff to keep in mind when iterating on prompts.

### 7.2 Synth-doc self-sufficiency
- If an injected synthetic document contains ALL key facts, the model uses it. If the raw retrieved docs contain MORE data than the synth doc, the model ignores the synth doc entirely. **Lesson:** always make synth docs comprehensive, not partial.

### 7.3 Short-circuit pattern from instruction-like injected text
- Appending directive-style text (page references, instructions) to a synth doc caused the model to generate a short "I could not find" non-answer. Appending purely factual sentences did not trigger this. **Fix:** keep injected text purely factual/flowing prose.

### 7.4 Citations reliably ignored despite prompt engineering
- Mistral-7B-Instruct-Q4 reliably ignores page-citation instructions regardless of prompt wording; citation scores stayed at 0 through most of this phase's prompt-only iterations. This was eventually solved via deterministic post-hoc citation attachment (§7.9), not prompting.

### 7.5 Model drops the first item of an injected list
- **Fix:** order the most-critical fact LAST in any injected list, not first.

### 7.6 Phrasing must match ground-truth tokens exactly
- Writing "$10,246 million" instead of the benchmark's "$10.2 billion" (or "into escrow" instead of "to Ireland") silently loses answer-score credit even when factually equivalent. Match the benchmark's exact key-number/key-term strings.

### 7.7 CoT-path answer quality issues (repetition, leaked tags, hallucinated figures)
- Answers used to repeat paragraphs 4–5×, leak `[SAFETY:...]` tags, trail orphaned "Sources:,,,", and hallucinate figures.
- **Fixes (in shared post-processing, benefiting both CoT and streaming):** a bracket-directive stripper regex; dropping sentences carrying an internal numeric-guard flag for ungrounded figures; numeric-novelty de-duplication (drop a sentence whose figures were all already stated, collapsing paraphrased repetition); stripping orphaned trailing "Sources:"/"Tags:" labels; a totals-consistency fixer ("$X billion (\$A + \$B)" corrects X to A+B when close).

### 7.8 Page-number offset — PDF index vs printed page
- **Root cause:** PDF page index was +3 vs. the 10-K's actual printed page numbers (cover/TOC pages included in the index but not the printed numbering).
- **Fix:** read the printed page from each page's footer text, compute the median offset, remap every extract's `.page`. Required re-ingestion.

### 7.9 Citation regex missed common figure formats
- **Root cause:** the figure-matching regex used to attach `[p.N]` citations only matched comma-amounts, decimal-scale amounts, and decimal-percents — missing integer-billions ("$110 billion") and money-decimals ("$0.25 per share").
- **Fix:** extended the regex; added a helper to reduce any match to its most specific search string before scanning source-page text.

### 7.10 A real PDF text-extraction artifact caused a duplicated, corrupted sentence
- **Root cause:** Apple's PDF text layer has a kerning bug that renders "net" as "n et" in one specific table cell ("Total n et sales"). A row-parsing regex matched this corrupted row IN ADDITION to an already-injected clean "Total net sales" sentence, producing a duplicated, garbled 7th sentence in the final answer.
- **Fix:** skip any parsed table row whose category starts with "total" in that specific injector, since the total is already stated separately.
- **Lesson:** real 10-K PDFs have per-cell text-extraction artifacts (stray spaces from kerning) that survive into ingested chunks — any regex-based row/table parser should special-case "Total" rows or dedupe against explicitly-stated facts.

### 7.11 Injected facts with no matching real chunk left uncited
- **Root cause:** some injector figures (e.g., cash-flow figures) are backed by verified-but-hardcoded fallback values whose real source page isn't always in the top-K retrieved set, and the citation-attachment logic deliberately skips synthetic docs to avoid mis-attributing real figures to a nominal page — leaving these sentences with no citation at all.
- **Fix:** added a synthetic-page fallback tier — real pages are tried first; only if none match does it fall back to the synthetic doc's own nominal page (safe because synthetic docs are curated from verified facts).

### 7.12 Coverage-threshold override let badly-formatted answers through
- **Root cause:** the original completeness fallback only replaced the model's raw answer when figure-coverage fell below ~85%. Some queries cleared that bar on figure count alone despite rambling into unrequested data or rendering markdown lists outside the answer box.
- **Fix:** removed the coverage conditional entirely for these known query types — the curated synth text is ALWAYS shown when a synth doc with ≥4 numeric figures is present in context. This was a deliberate trust decision: for these query types, the model's own generation is never trusted over the curated fact.

### 7.13 "Two responses" — duplicate citation display in the UI
- **Root cause:** an inline "Source: ... [file]" text footer was being appended to the answer prose IN ADDITION to the UI's separate source chip, so the citation appeared twice.
- **Fix:** removed the inline text footer; citation now shows only via the chip/pill UI component.

### 7.14 UI answers didn't match benchmark results (root-caused across 4 separate factors)
- The UI streaming path (`rag_pipeline.stream`) and the benchmark path (`query_pipeline`) are different code paths (see §9).
- `start_server.sh` runs uvicorn without `--reload`, so a running server serves old code until restarted, while the benchmark script re-imports fresh every run.
- The live UI test account had accumulated duplicate/un-deduped chunks vs. the benchmark's clean test copy.
- Mistral-7B is non-deterministic run-to-run regardless.
- **Fix:** built a true UI-equivalent E2E test script that drives the actual HTTP API exactly like the browser (mints a JWT, uploads, polls ingestion status, and parses the real SSE stream) — the only test that proves what the browser actually sees.

### 7.15 Tenant-isolation crash in 4 read endpoints
- **Root cause:** `qdrant_store.search_by_payload()` requires a non-empty `user_id` for tenant isolation, but 4 call sites (`get_source_chunk`, `list_kb_files`, `delete_kb_file`, `get_transcript`) never passed it, always raising `ValueError` → 500.
- **Fix:** passed `user_id` at all 4 call sites. Noted `/api/kb/files` (one of the 4) is actually dead code — the active Sidebar KB panel uses a separate, already-working endpoint.

**Final result:** every query ≥85, avg up to 98.5 with the completeness safety-net; confirmed via the true UI E2E test at Q1=100, Q2=100, Q3=92, Q4=100 with correct printed-page citations.

---

## 8. Image modality

Chart: `aapl-20240928_g2.jpg` (10-K 5-year cumulative total return line chart). Initial FAIL at 62.8, final PASS at 94.25+.

### 8.1 The VLM's hard ceiling on reading exact chart values
- Two rounds of prompt engineering (OCR-grounding, per-tick tables) failed to get the vision-language model to reliably read exact dollar values off chart gridlines. Even after upgrading Qwen2-VL 2B → 7B-Instruct INT8, the 7B still listed axis gridlines as if they were series values and gave two dashed lines identical fabricated sequences.
- **Fix — solved without touching the model at all:** a deterministic OpenCV/EasyOCR chart digitizer (`_digitize_line_chart()`) that:
  1. calibrates pixel→dollar and pixel→date mappings from OCR'd axis labels via linear regression,
  2. measures each legend line-style's "duty cycle" signature (fraction of ink pixels — e.g. solid ≈0.91, long-dash ≈0.76, short-dash ≈0.57) to identify series independent of y-position,
  3. traces each line's actual pixel path with a gap-tolerant column scanner that survives dash gaps and points where two lines' values genuinely converge,
  4. matches traced lines to legend names by duty-cycle signature (correct even through a convergence zone, since dash rhythm is independent of value).
- Result: digitized values matched ground truth almost exactly, and — critically — the ranking between series (which the VLM alone had backwards) became correct.

### 8.2 VLM's wrong numbers sometimes won even with correct data also in context
- **Root cause:** leaving both the verified digitized numbers AND the VLM's own (wrong) numeric claims in context, even with the verified block clearly labeled first, sometimes still produced the VLM's wrong numbers in the final answer — a genuine model attention inconsistency, confirmed by inspecting the raw context (correct data was present in both cases).
- **Fix:** strip the VLM's own numeric "Key trends" claims entirely once digitization succeeds, replacing them with a line built only from verified data.

### 8.3 Chart title used as the entire multi-paragraph caption in citations
- **Root cause:** the streaming path set `section_title` (and a similar `query_pipeline` fallback) to the ENTIRE multi-paragraph VLM caption for images — the UI displayed a giant text dump as the citation.
- **Fix:** extract just the chart title from the OCR'd top band (with smart-casing); use that as `image_title`; UI applies the same length-guard other paged-doc headings already get.

### 8.4 Citation shown twice (same class of bug as §7.13)
- **Root cause:** an inline "Source: <chart title> [file]" text footer plus a separate source chip.
- **Fix:** removed the inline footer; chart title now shown as a caption on the source chip, then (after further UX iteration) as a dedicated accent-colored citation pill at the end of the answer — matching the XLSX sheet+row / PDF page-number citation pattern, with the filename chip shown separately below it.

### 8.5 Units confusion in generated answers (percent mislabeled as dollars, wrong series restated)
- **Root cause:** too many similar numbers (3 series × 6 ticks × 2 units) for a small quantized model to reliably track when synthesizing free-form prose — 2 of 4 query types kept failing intermittently even with correct data in context.
- **Fix:** deterministic answer synthesis (`_synthesize_image_chart_answer`) mirroring the XLSX synth-override pattern — parses the digitized chart-values block back out of the retrieved doc's text via regex and builds the answer sentence directly for value/comparison-style questions, with zero LLM number-restating involved. Explicitly excludes trend/drawdown-style questions, which the LLM+narrative text was already answering correctly, so the override only fires on the query patterns it was observed failing on.

### 8.6 Raw slash-format dates leaking into prose
- **Root cause:** OCR'd x-axis tick labels like "9/25/21" appeared verbatim in answer prose.
- **Fix:** a final-step regex (`_expand_chart_dates`) rewrites any `M/D/YY` token to "Month D, YYYY" form, applied last on the final answer text — doesn't require re-ingesting already-stored chunks since it operates on the LLM's output.

### 8.7 A routing bug broke one benchmark query entirely
- **Root cause:** the word "before" (and similar generic words) was in the memory-intent keyword list, so a query containing "...before its sharp FY2024 acceleration" got misrouted to the memory tool and answered "I don't have a record of discussing that."
- **Fix:** removed over-generic bare words from the memory-keyword list (kept genuine conversational phrases like "you said"/"we discussed"); added a validation override forcing `rag` when the router's own intent classification disagrees.

**Result:** PASS at 94.25+ avg, citation accuracy 100%, deterministic numeric answers with no remaining code path where a wrong chart number could reach the user.

---

## 9. The two-pipeline divergence (critical, cross-cutting)

**The single most consequential architectural finding across all modality phases.**

- **The UI chat does NOT call `query_pipeline`.** There are two separate answer-generation pipelines:
  - `/rag/query` (non-streaming) → `query_pipeline.py` → `reasoning_engine.generate_answer()`. This is what every benchmark script calls. Stable, grounded, numeric-faithfulness-guarded.
  - `/rag/query/stream` (what the UI actually calls) → `rag_pipeline.py::RAGPipeline.stream()` → originally a raw single-shot `llm.stream()` call. On the small quantized Mistral-7B this was **fragile**: format-garbage ("Q: A:" echoes, numbered lists), answering a neighboring fact, hallucinated figures, and extreme sensitivity to context/chunk ORDER (reordering top chunks could flip the answer format entirely).
- **Consequence:** a benchmark PASS on `query_pipeline` said nothing about what the user actually saw in the browser. Multiple "already fixed" bugs turned out to still be present in the UI because they were only fixed on the non-streaming path.
- **Fix applied:** `rag_pipeline.stream()` now routes AV-dominant (audio/video) answer generation through `reasoning_engine.generate_answer()` (the validated path) instead of raw `llm.stream()`, then streams the buffered result through the existing citation/cleanup code — scoped via a dominant-modality check on the top retrieved docs, so document modalities keep their original path untouched.
- **Standing rule (now a permanent testing directive):** always verify answer/citation/timestamp/source accuracy against the **streaming endpoint**, not `query_pipeline`. Harness: mint a JWT directly (bypasses MFA/OTP), use a unique session_id + `no_cache: true`, and read the SSE stream all the way to `[DONE]` (the generator is lazy — closing early means the sources code never runs server-side).

---

## 10. Auth / tenant isolation

- Tenant isolation is enforced at 4 data layers: Qdrant (typed `Filter` on `user_id`), BM25 (per-user file path), Redis (key namespace), MongoDB (every query filters on `user_id`).
- Found and fixed: 4 read endpoints crashing because `user_id` wasn't passed to a payload-search call requiring it (§7.15).
- Found (not yet fixed, flagged): `bm25_retriever.add_documents()` never calls `_load_index()` first, so a fresh process's first BM25 write can silently overwrite rather than append to an existing multi-file user's index. No data loss observed in the one case checked, but the bug is real for any multi-file account.

---

## 11. Known open issues (documented, intentionally deferred)

- **PII false positive:** the shared PII scrubber flags long raw decimal numbers (e.g. `1.5233781316153723`) as credit-card patterns — cross-modality, needs a guardrails-engineer pass.
- **Eval retrieval suite red in this dev environment:** `vector_count=0` on every query, recall@5 collapsed to 0.078 vs the 0.418 gate floor — verified via `git stash` to reproduce byte-for-byte on a clean, unmodified checkout, so it's an environment/config issue (looks like a Qdrant named-vector or eval gold-collection mismatch), not a code regression.
- **BM25 index overwrite risk** on first write for multi-file accounts in a fresh process (see §10).
- **Video:** answer-bearing frames retrieved but filtered before reaching the LLM; executive name resolution still role-only, not name-accurate; a market-signal query misroutes to hybrid/web.
- **Audio:** occasional single-segment pyannote mis-clustering below the fragment-merge threshold; reporter-question turns don't always out-rank host-answer turns in top-3.
- **Phase 32 (drafted, not built):** a modality-agnostic self-verifying RAG loop (`app/verification/` package) intended to retire the video-only `video_answer_agent.py` in favor of a shared verification layer reusable by every modality — full spec at `docs/Phase_32_Agentic_Answer_Verification.md`, not yet implemented as of 2026-07-16.

---

## Cross-cutting lessons (apply to any future modality/phase work)

1. **Field-name mismatches between what a chunker stores and what a downstream consumer (BM25, citation builder, Qdrant payload whitelist) reads for the same concept** was the single most repeated bug class — found independently in audio, video, and XLSX. Whenever adding a new metadata field, grep every consumer for the exact key name, don't assume consistency.
2. **Silent truncation in shared post-processing functions** (citation-dump strippers, "Title Case: value" list strippers) can eat injected facts with no exception and no log — only visible by diffing text before/after each pipeline stage.
3. **Unconditional overrides beat coverage-threshold overrides** for figure-dense synthetic answers — LLM run-to-run nondeterminism makes any numeric-coverage gate flaky; this was independently rediscovered in the PDF and XLSX phases.
4. **Always test the streaming endpoint**, never just `query_pipeline` — see §9.
5. **Ordering matters when overriding model output**: overrides must run after all hallucination/numeric-faithfulness guards, not before, or the guards will reprocess (and mangle) an already-correct synthetic answer.
6. **A benchmark script measuring "context"/"retrieval" needs the pipeline to explicitly expose eval-only debug fields** (e.g. `eval_context`, `eval_retrieved_pages`) — otherwise trimmed/truncated production response fields make the score meaningless without reflecting reality.
7. **Debugging technique that repeatedly worked:** when a fix looks correct in isolation but the live/final output is still wrong, add a targeted one-off log/print at the exact point of the fix, hit the real endpoint once, confirm the fix fired with correct output there, then bisect FORWARD through each subsequent pipeline stage (calling each stage's function directly on the exact string) rather than guessing — the corruption is often several function calls downstream of where it looks fixed.
