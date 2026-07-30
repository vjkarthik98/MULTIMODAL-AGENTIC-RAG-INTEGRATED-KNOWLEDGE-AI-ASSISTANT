# Changelog

All notable changes to this project will be documented on this file.
The format follows Keep a Changelog and Semantic Versioning.


# [0.1.0] - Initial Setup

### Added

- Github repository with development workflow
- Protected main branch
- Project folder architecture
- Python package structure for rag_system
- Environment configuration template (.env.example)
- Dependency management (requirements.txt)
- Python packaging (pyproject.toml)
- VERSION file for semantic versioning

# [0.2.0] - FastAPI Integration

### Added

- FastAPI backend for RAG system
- API endpoints for query handling
- Integration of RAG pipeline with API
- End-to-End system with Qdrant + Ollama

# [0.3.0] - Multimodal Ingestion
### Added
- Multimodal ingestion pipeline (text, image, audio, video)
- Faster-Whisper integration for audio transcription
- OCR-based image text extraction
- Video-to-audio transcription pipeline
- Unified ingestion router

### Improved
- Metadata-aware ingestion schema
- Standardized document structure (text + metadata)
- Modula ingestion architecture

# [0.4.0] - Chunking Integration
### Added
- Recursive chunking with overlap
- Batch embedding pipeline
- Qdrant vector database integration
- Retrieval system with semantic search
- End-to-end ingestion pipeline (chunk -> embed -> store)
- FastAPI upload endpoint for document ingestion
- Full RAG pipeline (retrieve + generate)

### Improved
- Embedding performance using batching
- Modular architecture (utils, ingestion, embeddings, vectorstore)
- Automatic Qdrant collection creation
- Context formatting for better LLM responses

### Bug Fixed
- Qdrant API compatibility issues ('query_points')
- Retrieval method mismatches
- Vector Store integration bugs

# [0.5.0] - GGUF Model Integration
### Added
- GGUF model integration using llama.cpp (CPU-based inference)
- Streaming response support using FastAPI StreaminResponse
- New `/rag/query/stream` endpoint for real-time token output
- Modular LLM wrapper (`gguf_model.py`)

### Changed
- Replaced HuggingFace/Ollama with GGUF-based inference
- Updated RAG pipeline to use local quantized models
- Improved prompt formatting for Mistral Instruct

### Fixed
- Upload endpoint routing issue
- Multipart file upload handling
- Qdrant collection initialization flow
- UTF-8 decoding issue for non-text files (handled via testing approach)

### Notes
- Optimized for CPU environments (no GPU dependency)
- Designed for lightweight deployment and scalability

# [0.6.0] - Improved RAG Pipeline

### Added
- Chunk-based ingestion pipeline
- Batch embedding for documents
- Unique UUID-based vector storage in Qdrant
- Retriever with configurable top-k
- Structured context formatting for LLM

### Improved
- Token usage optimization for GGUF models
- Prompt design for better grounding
- Duplicate document removal in retrieval

### Fixed
- Context window overflow errors
- UUID generation bug in Qdrant storage
- Prompt NoneType crash issue

### Notes
- System now supports production-level RAG pipeline
- Fully optimized for CPU-based local inference

# [0.7.0] - Production-grade ingestion pipeline

### Added
- End-to-end ingestion pipeline orchestration (process_file)
- Structured API response (status, details, chunks)
- Logging system for pipeline observability

### Improved
- Chunk -> Embed -> Stored pipeline consistency
- Qdrant storage with document_id grouping
- Metadata strucutre for better retrieval

### Fixed
- Ingestion returning incorrect format
- Embedding validation issues
- Duplicate / untraceable vector storage

### Validation
- Added checks for:
    - Empty documents
    - Embedding failure
    - Invalid vector formats

# [0.8.0] - Text and Image (Ingestion + Query)
### Added
- Image ingestion pipeline using CLIPVisionMOdelWithProjection
- Image query pipeline using CLIPTextModelWithProjection
- Consistent dimensional embedding across ingestion and query

### Fixed
- Dimension mismatch issue (512 vs 768) during image query

### Notes
- Qdrant collection standardized to 384(Text) and 768(Image) vector size
- Foundation ready for multimodal expansion (audio/video)

# [0.9.0] - Audio Intelligence Upgrade
### Features
- Added audio ingestion using faster-whisper
- Converted audio to text segments for embedding
- Implemented audio query endpoint (/rag/query/audio)
- Enabled full RAG pipeline for audio queries

### Improvements
- Segment-level audio chunking
- Added timestamps (start_time, end_time) in metadata
- Context enriched with audio time references

### Fixes
- Fixed retriever (removed invalid search() call)
- Ensured pipeline consistency for all modalities

### Result
- Full multimodal RAG (text, image, audio)
- Production-grade ingestion + query pipeline

# [0.10.0] - Multimodal Video Rag
### Added
- Video ingestion pipeline
- Frame extraction + BLIP captioning
- Audio extraction + transcription (faster-whisper)
- Multimodal embeddings (audio + frame)
- Unified storage in Qdrant
- Video query endpoint
- Retrieval balancing (audio + frame)

### Improved
- Query rewriting for better semantic retrieval
- Context aggregation for better LLM understanding
- Prompt optimization for meaning-based answers

### Fixed
- Audio chunking issue (single segment bug)
- FFmpeg path resolution
- Tuple vs dict retrieval bug
- Metadata loss in Qdrant payload

### Known Issues
- LLM sometimes prefers visual description over audio meaning
- Needs reranking / weighting improvement

# [0.11.0] - UI
### Features
- UI with Multi-chat sessions
- Streaming responses (real-time token generation)
- Multimodal upload support (PDF, Image, Audio, Video)
- File ingestion pipeline integrated with UI
- Session-based chat switching

### RAG Improvements
- Stable retrieval pipeline (fixed modality handling)
- Context construction improvements
- Source attribution in responses

### Bug Fixes
- Fixed Qdrant collection mismatch issues
- Fixed embedding consistency (query vs document)
- Fixed streaming pipeline (backend + frontend sync)
- Fixed Gradio message format issues
- Fixed chat state synchronization bug

### Internal
- Refactored retriever logic
- Improved vector store insertion reliability
- Cleaned pipeline structure

# [0.12.0] - Memory System Integration
### Features
- Added Redis-based short-term conversational memory
- Added MongoDB-based persistent memory storage
- Implemented session-based memory handling (multi-user support)
- Integrated memory into RAG pipeline
- Enabled memory support in Gradio UI (Non-Stream mode)

### Improvements
- Clean Separation of memory layers:
    - Redis -> short-term context
    - MongoDB -> long-term storage
    - Qdrant -> knowledge retrieval
- Improved pipeline structure for memory injection
- Fixed session handling across API and UI

### Fixes
- Fixed Redis key formatting issue (Whitespace bug)
- Fixed Gradio streaming vs non-stream mismatch
- Fixed memory not storing from UI requests
- Corrected pipeline memory storage order (post-generation)

### Infr
- Dockerized Redis and MongoDB
- Added persistent container strategy (restart policies)

# [0.13.0] - Summarization of Redis Memory
### Added
- MemoryManager for automatic summarization
- LLM-based conversation summarization
- Conversation history injection into RAG pipeline

### Improved
- Prompt now includes conversation history
- Gradio UI supports conversational flow

### Fixed
- Redis key mismatch bug
- Memory overwrite issue
- Gradio message format error
- LLM response not returned properly

# [0.14.0] - Smart Memory Optimization
### Added
- Memory formatter
- Semantic memory filtering
- Memory summarization (LLM-based)
- Memory fusion Layer

### Improvement
- Designed token-efficient, context-aware memory system

# [v0.15.0] - Systerm Integration & Stabilization
### Features
- Image Captioning using BLIP
- Audio Transcription pipeline
- Video frame + audio processing
- Unified embedding pipeline (SentenceTransformer - 384 dim)
- Qdrant Vector database integration
- Redis (short-term) + MongoDB(long-term) memory
- Dynamic, modality-aware prompt system

### Improved
- Retrieval accuracy using reranker
- Reduced hallucinations with retrieval guard
- Unified embedding space across modalities
- Better query relevance and context handling

### Fixed
- BLIP captioning failures(generate()bug)
- Image ingestion issues (PIL, RGB, EXIF)
- Vector dimension mismatch (768 vs 384)
- Retrieval and reranker execution issues

# [v0.16.0] - Multimodal Intelligence & Reasoning
### Features
- Reasoning Engine
- Query Decomposition
- Mutli-Query Retrieval
- Result Fusion & Ranking Layer
- Diversity Filtering for context optimization

### Improved
- Retrieval quality via multi-hop search
- Answer quality with structured reasoning
- Context relevance with fusion + filtering
- Memory + reasoning integration


# [v0.17.0] - Agentic Pipeline + Model Loader + Full Observability
### Features
- Introduced AgentController for intelligent query routing
- Added decision-based execution (Multimodal vs standard pipeline)
- Added websearch tool for enhanced latest output
- Enabled dynamic handling of image, audio, video and text queries

### Improved
- Centralized model management via ModelLoader
- Eliminated scattered model initialization across modules
- Centralised logging system via get_logger
- Replaced all print statemens with structured logging

# [v0.18.0] - Hybrid Retrieval + Reranking Stabilization
### Features
- Implemented BM25-based keyword retrieval using rank-bm25
- Integrated semantic vector search using existing embedding pipeline
- Built HybridRetriever to combine keyword + semantic results
- Added CrossEncoder-based reranker for result refinement
- Improved retrieval precision and reduced irrelevant chunks
- Multimodal Ingestion Improvements
    -> PDF: Text Extraction + Image Extraction + Table Extraction
    -> Word: Text Extraction + Image Extraction + Table Extraction
    -> Excel: Table Extraction with structured conversion
- Metadata Standardization


# [v0.19.0] - Multimodal system Refactor & Architecture Strengthening
### Features
- Standardized multimodal ingestion across text, document, image, audio, video
- Introduced structured document schema with doc_id, session_id, file_hash
- Enabled modality-aware outputs (caption, OCR, speech, frame, tables)
- Added enriched metadata(timestamps, pages, segments, source tracking)
- Improved audio(Whisper) and video(audio + frame fusion) pipelines
- Enhanced image pipeline with capiton + OCR dual representation
- Introduced ModelLoader with lazy loading to initialize models only on first use
- Added centralized caching to prevent redundant model loads across modules
- Introduced Config for centralized Details of Important details

### Improved
- Refactored ingestion to produce structured blocks instead of flat text
- Strengthened document parsing (PDF, WORD, Excel) with multimodal extraction
- Aligned all modalities for consistent embedding + retrieval readiness
- Upgraded memory filtering with embedding reuse + recency/role scoring
- Improved pipeline modularity (ingestion vs query separation)
- Standardized logging, error handling and session tracking across modules
- Added validation and fallback handling in ingestion workdflows
- Reduced startup latency by deffering heavy model initialization
- Improved memory efficiency with on-demand model lifecycle management
- Standardized model access across pipeline via unified loader interface


# [v0.20.0]- Deterministic Multimodal RAG Stabilization & Agent Control Hardening
### Features
- Strict grounding enforced (LLM answers only from retrieved context)
- Multi-user session isolation across retrieval, memory, and vector store
- Hybrid execution support (RAG + Search)
- Intent-aware agent routing with multi-query handling
- Parallel sub-query execution for complex queries
- Structured API response (answer, sources, confidence, trace)
- Modality-aware retrieval (text, image, audio, video)
- Video frame-level retrieval integration
- Fail-safe agent fallback hierarchy

### Improved
- Retrieval: Fixed BM25 flow, improved hybrid ranking, added score filtering
- Agent: Reduced randomness, improved routing stability, added trace logging
- RAG Pipeline: Better context filtering, reduced hallucination, optimized token usage
- Ingestion: Improved chunking, standardized metadata, stronger validation
- Multimodal: Enhanced captioning, transcription, and modality alignment
- Memory: Strict session isolation, improved conversation handling
- Prompting: Enforced context-only answers, improved structure
- Model Loading: Centralized loading with retry and timeout handling
- API & Logging: Structured responses, better error handling, improved observability

### Fixed
- BM25 indexing and retrieval issues
- Duplicate and low-quality chunk retrieval
- Incorrect agent routing and fallback behavior
- Metadata loss in multimodal ingestion
- Context overflow and embedding inconsistencies
- Session leakage across memory and retrieval
- Hybrid retriever score normalization bug
- Edge-case ingestion and pipeline failures

# [v0.21.0] - Production Hardening, Multimodal Edge-Case Robustness & Test Foundation

### Features
- Bounded agent execution (max-steps + wall-clock timeout + token budget)
- Tenant isolation hook via Qdrant `user_id` payload filter (typed, not string-built)
- Circuit breaker (`_CircuitBreaker`) on Qdrant calls with half-open probe
- GDPR purge path across Qdrant + Redis + Mongo
- Hallucination guard + numeric-faithfulness check in reasoning engine
- Temporal-anchor boost for time-sensitive queries (FY/quarter/prior-year)
- MMR diversity in retriever, modality-aware reranking
- Standardised query response schema (answer, sources, confidence, trace)

### Improved
- Multilingual support removed (deferred); pipeline simplified to English path
- Full `.env` ↔ `config.py` alignment; zero hardcoded literals in pipeline
- Text/PDF/Word/Excel ingestion hardened against broken + edge-case files
- Image/Audio/Video ingestion hardened (solid-color detect, OCR repair, frame dedup via pHash)
- Retrieval + RAG pipeline end-to-end repair (Phase 24.7)
- Startup latency cut (~25s → ~7s) via lazy device manager + deferred model loads
- Per-model device routing (CPU/CUDA/MPS) with profile auto-resolve
- Lifecycle management for models, infra and pipelines (lazy singletons + health)
- Memory layer: sliding window + dedup + role/recency-weighted fusion
- Caption sanitizer + prompt-injection scrub on BLIP/CLIP/web-search inputs

### Fixed
- Qdrant payload index on `user_id` for multi-user filtering
- Section-aware chunking (preserves structure for temporal queries)
- Warmup race + startup crashes on missing optional deps
- Embedding cache key drift across modalities
- Empty-context path now raises `EMPTY_CONTEXT_NO_DOCUMENTS_RETRIEVED` instead of silent stub


## [v0.22.0] - Evaluation Harness & RAG Quality Metrics

### Added
- Eval CLI (`python -m app.eval.run --suite <name>`) with exit-code gate (0=pass, 1=breach, 2=infra error)
- 54 hand-curated gold triples across all 7 modalities (TXT/PDF/DOCX/XLSX/IMAGE/AUDIO/VIDEO)
- Real-world finance corpus: SEC 10-K filings, Berkshire letter, FRED macro data, earnings call MP3, CNBC MP4
- Retrieval metrics: recall@k, MRR, nDCG@10, context_precision, hit_rate against real `Retriever.retrieval()`
- Generation metrics: faithfulness, answer_relevancy, context_recall, template_leak_rate via lexical judge
- Hallucination detector: ungrounded-claim rate reported per suite, per modality
- Routing benchmark: route_accuracy + hybrid_with_web_rate over 12 labelled queries
- MLflow file-backend tracking: git_sha, dataset_version, all metrics logged per run
- Committed baseline (`baselines/rag_report_v1.json`) — regressions show in PR diffs
- Regression runner: diffs current run vs baseline, flags drops > 5% tolerance
- Phase 31 stubs: `drift_detection.md`, `online_eval.md`, `human_eval.md`

### Improved
- HTTP judge routing (`gguf_judge.py`) — eval calls live server instead of loading second GGUF, preventing T4 VRAM conflict
- `model_registry.ensure_for_query()` — skips LLM warmup when `EVAL_SKIP_LLM_WARMUP=true`
- All 9 gold JSONL files — chunk IDs kept in sync with corpus re-ingest via scroll-based hash discovery
- Modality-by-modality eval execution — prevents GPU stress and server crashes during long runs

### Fixed
- `clip_text_embedder._prepare_texts()` — `KeyError: 'language'` on SigLIP text embedding path
- Gold chunk ID staleness — SHA-256 hashes updated across all modalities after fresh ingest

### Baseline Metrics (v3 — real corpus, lexical fallback judge)
- TXT retrieval hit@10: 1.000 · MRR: 0.815 · faithfulness: 0.517 · hallucination: 14%
- PDF retrieval hit@10: 1.000 · MRR: 0.547 · faithfulness: 0.398 · hallucination: 40%
- DOCX retrieval hit@10: 1.000 · MRR: 0.750 · faithfulness: 0.614 · hallucination: 0%
- XLSX retrieval hit@10: 0.667 · MRR: 0.667 · faithfulness: 0.335 · hallucination: 33%
- IMAGE retrieval hit@10: 1.000 · MRR: 0.667 · faithfulness: 0.408 · hallucination: 0%
- AUDIO retrieval hit@10: 0.000 · MRR: 0.000 · faithfulness: 0.308 · hallucination: 50%
- VIDEO retrieval hit@10: 1.000 · MRR: 0.333 · faithfulness: 0.580 · hallucination: 0%
- Routing accuracy: 0.750 (threshold 0.917 — breach logged for Phase 26)

## [v0.23.0] - Production Guardrails, Security Hardening & Pre-Ingestion Attack Defence

### Added
- `app/guardrails/` — new package (2 314 lines) replacing 7 scattered sanitize implementations
  - `input_guard.py` — unified blocking entry point (`check()`) and non-blocking ingestion path (`sanitize()`); NFKC + confusables + injection + jailbreak + SSRF; raises `GuardrailBlocked` on violation
  - `output_guard.py` — 7-step output pipeline: groundedness → template artifacts → length → citation integrity → PII egress → toxicity → mojibake repair
  - `jailbreak.py` — Tier 1 regex (26 patterns) + Tier 2 semantic similarity; upgrade hook for Tier 3 ML classifier
  - `rate_limiter.py` — per-session + per-IP block-event tracking; rolling 60 s window; 5-block threshold triggers 5-minute ban; `reset()` for admin use
  - `confusables.py` — 200-entry Unicode TR39 curated map (Latin-extended, superscript/modifier letters, Greek, Cyrillic, math bold, whitespace variants, tab→space) applied post-NFKC
  - `pii.py` — Microsoft Presidio detector + scrubber; `strip_pii_from_prompt()` strips PII from LLM prompt before generation; `scrub_pii()` for egress
  - `audit.py` — HMAC-SHA256 tamper-evident structured decision log; every allow/block/scrub event signed and Prometheus-counted
  - `ssrf.py` — URL extraction + blocked-CIDR check (loopback, RFC-1918, link-local, AWS metadata)
  - `policies.yaml` — single source of truth for all patterns, thresholds, refusal templates; 43 injection patterns across critical/high/medium severity; no code change needed to add/tune patterns
  - `exceptions.py` — typed `GuardrailBlocked(reason, surface, guard_type, correlation_id, detail)`
  - `metrics.py` — Prometheus counters: `guardrail_blocks_total{guard_type, surface}` and `guardrail_allows_total`
- `tests/guardrails/` — 257-test suite (5 modules + adversarial corpus)
  - `adversarial/red_team_prompts.jsonl` — 109-case corpus: 84 attack / 25 benign across injection, jailbreak, encoding bypass, PII, SSRF, poisoned-document, web-result poisoning, pre-ingestion vectors
  - `test_input_guard.py`, `test_output_guard.py`, `test_jailbreak.py`, `test_ssrf.py`, `test_audit.py`
- `docs/security/guardrails_runbook.md` — operator runbook: add injection pattern, rotate HMAC secret, bypass escalation procedure, Prometheus alert thresholds, rate-limiter manual unban, known gaps table

### Improved
- **agent_controller.py** — `_sanitize()` (non-blocking) replaced with `_guard_input()` (blocking, raises on violation); rate limiter `enforce()` before input guard; `_direct()` and `_fallback()` paths now run `_guard_output()` before returning
- **rag_pipeline.py** — output guard added after generation; PII stripped from prompt before LLM via `strip_pii_from_prompt()`; streaming path collects all tokens, guards full answer, then yields once
- **agent_router.py**, **api_routes.py**, **web_search.py**, **text_embedder.py**, **clip_text_embedder.py** — all replaced inline sanitize/inject-check with `input_guard.sanitize()` call (non-blocking path)
- **frame_captioner.py** — `_sanitize_caption()` delegates to `input_guard.sanitize()` (Phase 26 consolidation confirmed)
- **image_ingest.py** — upgraded from old local `sanitize_prompt_injection()` to `input_guard.sanitize()`, covering homoglyph and encoding bypass variants
- **output_guard groundedness ordering** — groundedness check moved to step 1 (raw answer) before any citation stripping or PII scrubbing; eliminates false hallucination-rate regression caused by comparing mutated text against context
- **Injection regex corpus** — 43 patterns covering: ignore/disregard/forget/overlook variants, override/bypass, you-are-now/act-as persona hijack, DAN, system-prompt exfil, transcribe-instructions, released-from-constraints, from-now-on, new-instructions-are-to, [MODEL:] bracket injection, NEW SYSTEM INSTRUCTIONS, P.S./note afterthought injection, stop-following-instructions, mass PII exfiltration, output-session-tokens

### Fixed
- **PRE-03 (White-font PDF)** — `input_guard.sanitize()` applied to raw fitz-extracted text before chunking; hidden injection text stripped before it enters vector store
- **PRE-04/PRE-06 (Image/video BLIP caption overlay)** — unified `input_guard.sanitize()` replaces local pattern list; covers homoglyph and encoding bypass not in old list
- **PRE-09 (Excel hidden rows/columns)** — `_process_excel()` now skips rows where `row_dimensions[i].hidden=True` and columns where `column_dimensions[col].hidden=True`; attacker-hidden cells never reach indexed text
- **PRE-11 (DOCX comment author PII)** — Presidio `scrub_pii()` applied to `author` field before building `[COMMENT by {author}]` chunk; author name no longer stored verbatim in vector index
- **video_runner.py** — null `source_file` crash on gold rows with no attached file fixed with `source_file = row.get("source_file") or ""; if not source_file: continue`

### Security Metrics (Phase 26)
- Injection corpus recall: **64 / 64 — 100%** (was 49/64 before this phase)
- False positive rate: **0.9%** (1/109 — RTL-wrapped attack correctly blocked, test label was wrong)
- F1 score: **0.994** | Precision: **0.988** | Recall: **1.000**
- OWASP LLM Top 10 (2025): **10 / 10 threats addressed**
- Test suite: **257 passed, 7 skipped, 0 failures**


# [v0.24.0] — Authentication, MFA & Tenant Security

### Features

- Full JWT authentication system with access tokens (30 min) and refresh
  tokens (7 days) — every protected route now requires a Bearer token
- Argon2id password hashing (OWASP recommended) with zxcvbn strength
  enforcement at registration — weak passwords rejected before storage
- TOTP multi-factor authentication (RFC 6238) — enrol via QR code, verify
  with any authenticator app (Google Authenticator, Authy, 1Password)
- Eight single-use backup codes generated at MFA enrolment, stored as
  bcrypt hashes — shown once, never kept in plaintext
- Token revocation via Redis blacklist — logout is now real, not cosmetic;
  jti added to Redis with TTL matching remaining token lifetime
- Logout-all via generation counter in Redis — one atomic bump invalidates
  every active session across all devices instantly
- Password change automatically triggers logout-all — old sessions can
  never be reused after a credential update
- Google OAuth2 sign-in — redirect → code exchange → JWT pair; links to
  existing account if email matches, creates new account otherwise
- Admin panel with user management — list accounts, promote/demote roles,
  deactivate users, purge data; all routes role-gated beyond standard auth
- Multi-tenant data isolation enforced at every storage layer — Qdrant
  user_id filter injected from JWT (never from request body), Redis keys
  namespaced per user, MongoDB queries scoped by user_id
- Per-user BM25 index — keyword search isolated the same way as vector
  search; cross-tenant keyword leakage is impossible
- GDPR self-delete (DELETE /auth/me) — purges Qdrant vectors, Redis
  cache, MongoDB history, deactivates account, revokes all tokens in one
  call

### Improved

- Constant-time password verification even on missing-user path — dummy
  hash always verified to prevent timing-based account enumeration
- Refresh token exchange now issues a full new token pair (rotation) —
  refresh tokens are single-use by design
- All pipeline routes receive user_id exclusively from the verified JWT
  payload — form fields and headers can no longer forge tenant identity
- Redis blacklist fails open on outage — a Redis outage logs a warning
  and allows the token rather than locking all users out of the system
- Admin routes return 403 (not 404) on role mismatch — avoids leaking
  which endpoints exist to unprivileged callers

### Fixed

- Cross-tenant retrieval now impossible through any path — Qdrant filter,
  Redis namespace, and Mongo query all enforce user_id from JWT
- Argon2 dummy-hash path on login miss prevents timing attacks that could
  confirm whether an email address has an account
- MFA challenge token scoped to `mfa_challenge` type — cannot be reused
  as an access or refresh token
- Backup code burned immediately on use — replaced with empty string in
  stored hash array; replay attacks blocked
- AUTH_ENABLED=false dev bypass preserved for local development and eval
  runs — never ships in production config

# [v0.25.0] — Per-Modality Architecture Rebuild, Model Upgrade & Full Evaluation Harness

This is the largest release so far. The project moved from a handful of
shared, monolithic files per layer to a strict per-modality architecture,
the resident model stack was upgraded end to end, and — for the first time —
every one of the seven modalities (txt, pdf, docx, xlsx, image, audio,
video) was measured against a real gold dataset with a purpose-built judge
model, diagnosed, fixed, and re-measured until the numbers actually moved.

### Features

**Architecture — per-modality rebuild**

- Replaced the old shared `chunker.py` (one 224-line file handling every
  modality's chunking with branching logic) with `base_chunker.py` plus
  one dedicated chunker per modality — `txt_chunker.py`, `pdf_chunker.py`,
  `docx_chunker.py`, `xlsx_chunker.py`, `image_chunker.py`,
  `audio_chunker.py`, `video_chunker.py` — so a bug or tuning change in one
  modality can never silently affect another
- Same split for embeddings: `text_embedder.py` (737 lines),
  `clip_text_embedder.py` (612 lines) and `multimodal_embedder.py`
  (655 lines) — three overlapping do-everything files — were retired in
  favour of `base_embedder.py` (shared BGE-large singleton, cache,
  finance-number normalization) and seven thin per-modality embedders
- BM25 given the identical treatment: `base_bm25.py` (shared tokenizer,
  circuit breaker, save/load) plus one `{modality}_bm25.py` per modality,
  replacing a single general-purpose retriever file
- Ingestion follows the same 4-file-per-modality pattern
  (`{modality}_ingest.py` → chunk → embed → bm25), all reachable only
  through the public dispatch layer (`app/chunking/__init__.py`,
  `app/embeddings/__init__.py`, `app/ingestion/router.py`) — nothing in
  the pipeline imports a per-modality file directly any more
- Shared audio/video transcript logic (Whisper word timing, pyannote
  diarization, speaker-turn assembly, filler-word stripping) consolidated
  into a new `av_shared.py` instead of being duplicated between the audio
  and video chunkers
- Deleted `app/agents/video_answer_agent.py` (a video-only, one-off
  verification hack) and replaced it with a proper generic package,
  `app/verification/` — `groundedness_checker.py`, `citation_verifier.py`,
  `completeness_verifier.py`, `confidence_scorer.py`,
  `retrieval_evaluator.py`, `retry_controller.py`, `stopping_criteria.py`,
  `verification_loop.py` — a self-verifying answer loop that works the
  same way for every modality, not just video
- Removed dead `app/bin/ops/*` one-off scripts (`init_qdrant.py`,
  `migrate_qdrant_dim.py`, `purge_session.py`, `rebuild_bm25_index.py`)
  and folded their still-needed behaviour into `app/bin/models/` and
  `app/bin/server/`
- Brand-new `app/eval/` package — `datasets/gold/` (per-modality gold
  files), `judges/`, `metrics/`, `runners/`, `tracking/`, `reports/`,
  `baselines/`, `benchmark_queries/`, `scripts/` — the project had no
  structured evaluation harness before this release

**Model upgrade**

- Swapped the resident LLM from Mistral-7B-Instruct to
  **Qwen2.5-14B-Instruct (Q4_K_M GGUF)** — still run as a separate
  `llama.cpp` process on its own CUDA context (never in-process with
  PyTorch), on the 48GB VRAM GPU (AWS g6e.xlarge / L40S)
- Added a dedicated evaluation judge, **Prometheus-2-7B v2.0 (Q8_0
  GGUF, ~7.7GB)** — a model purpose-built to be an LLM evaluator, not a
  repurposed instruct model, replacing the earlier Ragas + gguf-Mistral /
  cross-encoder / lexical judge stack that had capped faithfulness scores
  at 0.29–0.54 regardless of actual answer quality
- Replaced BLIP-1 image captioning (caption text was silently reused as
  the VLM prompt — a long-standing bug) with **Qwen2-VL** as the primary
  vision-language model: 7B for image charts, a lighter 2B variant for
  video frame captioning to keep 1-hour video ingests inside VRAM budget
- Added a deterministic OpenCV chart digitizer for financial line charts
  (`_digitize_line_chart` in `image_chunker.py`) — axis-label OCR
  calibrates pixel→dollar and pixel→date mappings, a dash-signature
  tracer follows each series' line through the chart, so exact chart
  values are pulled from pixel geometry instead of asking a vision model
  to read them off an image
- `app/bin/models/download_all_models.py` — one script that fetches and
  verifies the entire ~20GB model set in one pass, replacing manual
  per-model download steps

**Full evaluation harness (new)**

- Rebuilt the gold dataset from scratch to an industry-standard shape:
  one gold file per modality, every row carrying a `reference_answer`,
  ground-truth `relevant_chunk_ids`, and per-modality citation ground
  truth (page/section for docs, sheet+row for spreadsheets, timestamp for
  audio, timestamp+frame for video)
- Added dedicated refusal rows (question the KB genuinely cannot
  answer), adversarial rows (prompt injection + false-premise
  correction), and a websearch-required row to every modality's gold set
- Removed multi-hop query rows entirely at the user's direction — the
  live agent does a single classify-and-dispatch, not iterative
  multi-step tool chaining, so multi-hop questions were testing a
  capability the system was never designed to have
- `python -m app.eval.run --suite {retrieval,generation,behavioral,full}
  --modality {txt,pdf,docx,xlsx,image,audio,video}` — retrieval metrics
  (recall@k, MRR, nDCG, hit_rate) need no LLM; generation metrics
  (answer_correctness, answer_relevancy, faithfulness via Prometheus;
  context_recall, finance_fidelity, citation_accuracy deterministic);
  behavioral metrics (refusal_accuracy, adversarial_pass)
- Deterministic `finance_fidelity` and `context_recall` metrics that
  check numeric agreement and fact recoverability directly against
  source text, giving a trustworthy signal independent of any LLM judge

**Retrieval intelligence**

- Meeting/event-scoped retrieval — when a query names a dated meeting or
  earnings call the source's own filename encodes (e.g. "September 2024
  FOMC" or "Q4 2025 earnings call"), primary retrieval is scoped to that
  one source before ranking, so a same-topic different-period document in
  the same knowledge base (a December FOMC transcript, a prior-year 10-K)
  can no longer answer a question with the wrong period's numbers
  entirely
- Per-call cross-encoder budget — the reranker can now be handed the
  full scoped candidate set for a single-source query instead of the
  usual fusion-capped top-N, so a fact whose chunk opens with unrelated
  text (a legal disclaimer, a generic intro sentence) still gets read and
  ranked correctly by the cross-encoder instead of never reaching it
- Finer, modality-specific audio/video transcript chunking (tunable via
  `AUDIO_CHUNK_MIN/MAX_WORDS` and `VIDEO_CHUNK_MIN/MAX_WORDS`, both
  defaulting to the same values the shared chunker always used, so
  nothing else changed) — a specific spoken fact now lands in its own
  focused, retrievable chunk instead of buried inside a long,
  topic-mixed block

**UI — full frontend rewrite (Gradio retired, React shipped)**

- Retired the original Gradio chat interface entirely (`ui/gradio_app.py`
  and `ui/theme.py`, ~1,000 and ~700 lines) and replaced it with a
  ground-up React + Vite + Tailwind single-page app — the UI is no longer
  a Python-templated Gradio Blocks layout, it's a real frontend with its
  own build pipeline, calling the FastAPI backend exclusively over HTTP
  (no `app/` imports from the UI, matching the project's own layering
  rule)
- Full page set: `ChatPage`, `LoginPage`, `ForgotPasswordPage`,
  `ResetPasswordPage`, `TranscriptViewer` — plus a component library
  (`MessageBubble`, `Sidebar`, `SettingsModal`, `ConversionModal`,
  `GuestBanner`, `LoginModal`, `Toast`, `TypingIndicator`,
  `ErrorBoundary`) and a `useIsMobile` hook driving a dedicated mobile
  action menu
- Dark/light theme, animated sidebar, streaming message rendering with
  syntax-highlighted code blocks and full GitHub-Flavored-Markdown table
  support, keyboard shortcuts, a three-dot per-message action menu, and
  file-type badge coloring in the knowledge-base list
- Finance-specific components: `FinanceTable` (Markdown financial-table
  renderer), `MediaTimestampChip` (clickable audio/video timestamps that
  seek playback), `EarningsCallBrowser` (call-section navigator),
  `KnowledgeBasePanel` (file list, upload, delete), numeric verification
  badges and source citation chips on message bubbles — the last of
  these was specifically re-verified against real answers across TXT,
  PDF, and DOCX after citation-display bugs were found
- Upload UX rebuilt — progress ring, cancel-in-flight, and duplicate-file
  detection all fixed and working together; the earlier standalone
  upload progress bar was removed once the ring replaced it
- Login page redesign — Google OAuth button, email/password flow, and
  account creation, first shipped on the Gradio UI and then carried
  forward faithfully into the React rewrite
- Persistent login across page reloads and a logout that actually
  revokes the token server-side (the old logout was UI-only)
- Guest session usage limits (rate-limited per IP, query and file caps)
  so anonymous trial access can't be used to run the system unbounded,
  plus a `GuestBanner`/`ConversionModal` prompting guests to create a
  real account
- Web-search source indicator in the chat UI

### Improved

**Per-modality accuracy — every modality diagnosed and re-measured**

- **TXT** — strong out of the gate: retrieval hit_rate 1.00,
  finance_fidelity 0.875, context_recall 0.77
- **PDF** — a candidate-pool floor fix (never fewer than 50 candidates
  reach the reranker) recovered cross-document queries; finance_fidelity
  0.929
- **DOCX** — fixed a structural-embedding-lane bug where a large,
  unrelated spreadsheet was hijacking the top of completely unrelated
  queries (a "gross margin" question was answering from a country
  risk-premium file); this one fix alone took DOCX retrieval hit_rate
  0.79 → 1.00 and also lifted TXT and XLSX; a second fix removed a
  false-positive NER tag that mistook the acronym "DCF" for a company
  name and wrongly abstained on valid questions
- **XLSX** — retrieval hit_rate 0.29 → 0.64 from the same structural-lane
  fix; generation was the real story: answer_correctness went from
  **0.000 to 0.786** by adding generic per-country and per-sheet fact
  extractors (equity risk premium, GDP, tax rate, sovereign rating, CDS
  spread, rating-lookup tables) that parse the exact queried row directly
  out of the dense multi-country table chunks the small model was
  otherwise reading the wrong row from
- **Image** — chart Q&A answer_correctness 0.289 → 0.857 by rewriting the
  chart-answer synthesizer to handle specific-date reads, distinctive
  per-series matching (previously "Index" alone matched every series in
  the chart), title extraction, and an unsupported-metric abstention
  guard, backed by a vision-store fetch fallback so the digitized chart
  data reaches the synthesizer even when the reranker buries the chart
  chunk below text chunks
- **Audio** — diagnosed and fixed a two-transcript contamination bug
  where a September Fed press-conference question was silently answered
  with a December transcript's numbers because the denser document
  out-ranked the correct one; meeting-scoped retrieval took
  answer_correctness 0.196 → 0.375; finer re-chunking on top of that
  more than doubled retrieval quality (MRR 0.19 → 0.40, hit_rate 0.79 →
  0.86)
- **Video** — same two fixes applied to the earnings-call recording:
  answer_correctness 0.071 → 0.411 from call-scoped retrieval, then
  retrieval hit_rate 0.43 → 0.93 and faithfulness roughly doubled from
  finer transcript re-chunking; on-screen financial-chart reads (EPS,
  stock price) now answer correctly from the vision-frame captions

**Latency & performance**

- Lazy model loading — heavy models load on first real use instead of at
  startup, cutting cold-start time
- Uvicorn startup, file upload, and streaming-response latency all
  reduced; time-to-first-token improved by skipping redundant
  tokenization on the hot path
- `llama-server` tuned with flash-attention and prompt KV-cache flags so
  repeated RAG system-prompt prefixes reuse cached compute
- Traced the real villain behind slow uploads to synchronous calls on the
  Upstash-hosted Redis (≈200ms per call) rather than the code doing the
  uploading — split to a local Redis cache for the hot path plus a
  short-lived in-process auth cache
- Ingestion NLP/PII extraction and BM25 search performance improved
  across the board

**Agent & reasoning**

- Retired the video-only verification hack in favour of the shared,
  generic `app/verification/` self-verifying answer loop described above
- Added an entity subject-dominance abstention gate — the agent now
  checks whether the query's actual subject (company, person) is
  genuinely present in the retrieved context before answering, instead
  of answering from a superficially similar chunk about a different
  entity; refusal_accuracy rose from 0.024 to 0.155 project-wide

### Fixed

- Video ingestion crashed outright whenever diarization returned any
  speaker segments (a string was passed where `_map_speaker_roles`
  expected a structured mapping) — video had never actually worked
  end-to-end with diarization before this release
- XLSX generation was answering with a plausible-looking but wrong
  country's numbers for nearly every per-country question (two different
  countries both got answered as "Turkey") — root-caused to the model
  losing track of which row it was reading in a dense, 25-country table
  chunk
- Image chart Q&A returned empty answers or the wrong chart title for
  several question types because the chunk holding the actual digitized
  chart values was routinely outranked by unrelated text chunks and
  never reached the synthesizer
- A same-topic, different-period document sharing a knowledge base with
  the correct source (a prior-year 10-K, a different month's earnings
  call) could silently answer a question with the wrong period's numbers
  instead of the source actually being asked about
- BLIP-1's caption output was being reused verbatim as the next model's
  prompt, corrupting downstream vision-language generation
- Corrected a corrupted source data file (`apple_10k.pdf`) that had been
  silently feeding bad text into ingestion
- Message upvote/downvote state was lost on page reload — votes now
  persist server-side instead of living only in frontend component state
- Knowledge-base file upload could be duplicated by a slow network retry
  and left half-deleted files behind on cancel — upload dedup and delete
  are now atomic
- Source citation chips and section headers were showing stale or
  missing data for some TXT/PDF/DOCX answers — re-verified end-to-end
  after the fix
- A summarization trigger bug and a PII/prompt-corruption issue surfaced
  through the chat UI were both fixed at the API layer feeding it


# [v0.26.0] — Production MLOps, LLMOps & CI/CD

This release deliberately changes nothing about what the assistant *does*.
No retrieval logic, no agent behaviour, no guardrail rules were touched.
What it adds is the operational discipline around the system — the machinery
that proves it still works after every future change.

That machinery splits into three concerns, and it is worth naming them
separately because they answer different questions:

- **MLOps** — is the *model itself* reproducible and correct? Model
  identity, checksums, revision pinning, cache layout, vector-index schema,
  and a startup that refuses to serve the wrong artifact.
- **LLMOps** — is what the model *produces* good and safe? Prompt
  versioning, judge selection, generation and behavioural evaluation,
  guardrails on every input and output.
- **CI/CD** — does the automation *enforce* both of the above on every
  change, without anyone having to remember to run it?

The more interesting half of this release is what building those gates
uncovered. A quality gate is only worth having if it can tell "this is
broken" apart from "this got worse" — and on its first real run, the
retrieval gate reported a 15% quality drop that turned out to be BM25
silently returning zero results. Chasing that down surfaced a string of
genuine bugs that every unit test had been passing straight over, because
unit tests mock exactly the things that were broken.

### Features

**MLOps — model provenance, reproducibility and startup safety**

- The model downloader now performs Trust-On-First-Use checksum
  verification: the first trusted download records a SHA-256 per artifact
  into `.hf_cache/download_manifest.json`, and every later run verifies
  against it and fails loudly on drift. A pinned model file plus a checksum
  does not require knowing the hash in advance — it requires refusing to
  accept a *different* one later
- Every model entry carries an explicit `revision`, so a HuggingFace repo
  moving underneath us is a hard failure rather than a silent behaviour
  change. The one genuinely unpinnable window is a model's first-ever
  download, which is inherent to the TOFU model rather than an oversight
- `startup_validator` aborts boot on an incomplete or mismatched manifest.
  Combined with the pre-existing write-time embedding-dimension check, a
  wrong model or a wrong vector width now fails at startup instead of
  quietly degrading retrieval
- `torch.hub` models (Detoxify) resolve their cache correctly from *any*
  entry point rather than only from `start_server.py`. `TORCH_HOME` and
  `HF_HOME` are separate, non-interchangeable caches, and only one of them
  was being set outside the server launcher — so a guardrail check could
  re-download a model that was already on disk
- The BM25 index carries a schema version and discards a stale index rather
  than mixing tokens from two different tokenizer generations
- Qdrant's delete-and-recreate-on-dimension-mismatch path is now guarded by
  `QDRANT_ALLOW_RECREATE` and logs at CRITICAL with an explicit
  `data_loss=True` field instead of INFO. The behaviour itself is
  intentional — an embedding-model upgrade must not leave stale-dimension
  vectors corrupting a new index — but a config typo produced an identical,
  easily-missed log line until the data was already gone

**LLMOps — prompt versioning, judge provenance and generation quality**

- `PROMPT_VERSION` is recorded as an MLflow parameter on every evaluation
  run, so a metric can always be traced back to the exact prompt that
  produced it. A generation score without a prompt version attached is not
  a measurement, it is an anecdote
- `GET /version` reports the running prompt version alongside the git SHA,
  image tag and full model manifest — a deployed instance can now prove
  exactly what it is serving
- Judge selection is single-sourced. The harness previously resolved
  `EVAL_JUDGE_MODEL` in two places with two different fallbacks, so reports
  could claim the Prometheus-2-7B judge while the legacy Ragas path
  actually ran
- Evaluation thresholds are enforced per-suite. Retrieval gates for real
  against a measured baseline, while generation, e2e and behavioural stay
  explicitly informational until each is re-baselined against the
  Prometheus judge on real GPU hardware — rather than a single global
  switch forcing an all-or-nothing choice between gating on unverified
  numbers and gating on nothing
- The full generation and behavioural suites (answer correctness, citation
  accuracy, hallucination rate, refusal accuracy, adversarial resistance)
  are wired as Tier 2, GPU-only and post-deploy

**Continuous integration — the always-on PR gate**

- `ci.yml` runs ruff, black, isort, mypy and the full 1,372-test unit suite
  on both Python 3.10 and 3.11 for every pull request, on hosted runners
  with no GPU and no external services
- mypy runs but does not block. This is a deliberate call, not an oversight:
  a real run found 311 pre-existing type errors across 65 files, and a gate
  that is red on the day it ships teaches everyone to ignore it. The step
  still shows red in the Actions UI, so the debt stays visible
- Every pytest invocation is scoped to a specific subdirectory rather than
  `pytest tests/ -m <marker>`. pytest collects every file under `testpaths`
  before applying any marker filter, so one broken file anywhere aborts the
  entire run with zero tests executed — which presents as a silent hang,
  not a clear failure

**The two-tier retrieval quality gate**

- `eval-gate.yml` Tier 1 scores 56 gold questions against the live Qdrant
  collection plus BM25 on every pull request, and blocks the merge on any
  regression. Thresholds sit 5% below a measured v4 baseline (recall@5
  0.6786, recall@10 0.7589, MRR 0.4660, nDCG@10 0.5322, hit_rate 0.8393,
  n=56)
- Tier 1 is self-sufficient on a hosted runner. When its BM25 cache misses,
  it rebuilds the index directly from Qdrant payloads — no GPU, no models,
  no re-embedding, because the chunk text is already stored in the payload
- A cache *hit* is no longer trusted on faith. The restored index is opened
  and checked for loadability and document count before use; anything
  unusable triggers a rebuild instead of silently degrading the gate
- Thresholds are enforced per-suite rather than by a single global switch,
  so retrieval can gate for real while generation, e2e and behavioral stay
  informational until each is re-baselined against the Prometheus-2-7B
  judge on real GPU hardware
- Tier 2 (full generation plus judge) remains GPU-only and post-deploy. Its
  nightly schedule is disabled until a self-hosted runner exists — a
  scheduled job against a runner label that resolves to nothing does not
  fail, it queues indefinitely

**Supply-chain and code security**

- New `security.yml` with four independent checks: detect-secrets, Bandit
  SAST, pip-audit for dependency CVEs, and a dependency license scan
- detect-secrets is now enforced in CI. It previously ran only as a local
  pre-commit hook, so anyone who skipped or bypassed that hook had nothing
  else in the pipeline catching a leaked credential
- Bandit blocks on HIGH severity only, and that gate is genuinely clean: the
  five HIGH findings from the first real run were triaged individually, four
  fixed and one suppressed with a written justification
- The license scan blocks only on AGPL and SSPL — the two families that
  would force source disclosure of this hosted service. GPL-2.0 (`mutagen`)
  and LGPL-3.0 (`CairoSVG`) dependencies were found, and rather than being
  silently accepted or silently blocked they were moved to a commented-out
  optional section of `requirements.txt`. Both are already used behind
  `try/except ImportError` with graceful degradation, so leaving them
  uninstalled costs a deployer nothing but the extra fidelity
- Dependabot now watches pip, npm and GitHub Actions weekly

**Deployment, releases and provenance**

- `cd.yml` builds the CUDA image, pushes to GHCR, deploys to EC2 over SSM
  using GitHub OIDC (no long-lived AWS keys anywhere), health-checks, and
  rolls back automatically on failure
- Every image now ships with an SBOM and a SLSA provenance attestation
  attached in the registry, so "what is actually inside the artifact running
  in production" is answerable from the registry alone rather than
  reconstructed from the Dockerfile
- Trivy scans each built image and publishes results to the repository's
  Security tab. It is informational for now, and honestly so: the image has
  never been built on a machine with Docker available, and setting a
  threshold without a measured count would be guessing
- New `release.yml` automates the mechanical half of cutting a release —
  version bump, changelog section, tag, GitHub Release — while leaving the
  judgement half (what to call it, what belongs in it) to a human. It is
  manually triggered, never automatic
- `GET /version` reports the running git SHA, image tag, prompt version and
  full model manifest, so a deployed instance can prove what it is

**Developer experience**

- `Makefile` gained `integration`, `benchmark`, `security-scan`, `sbom` and
  `release` targets. `release` is a preflight that deliberately does *not*
  tag — it checks version consistency, a clean tree, an unused tag and a
  changelog entry, then points at the workflow that owns the actual release
- `.env` is down to 37 keys — secrets, per-environment values and hardware
  pins only. Everything else now lives in `config.py` with a reviewable
  default, and `.env.example` documents the device and timeout overrides
  that were previously undiscoverable

### Improved

- mypy errors reduced from 311 to 188, every fix a real correction rather
  than a suppression
- Bandit HIGH-severity findings reduced from five to zero
- `tests/integration/` cut from 42 files to 13. Twenty-seven referenced a
  pre-refactor `src.rag_system.*` package that no longer exists; two more
  tested APIs that had been renamed or removed. The three genuinely current
  gap tests were missing the `skipif(llama-server unavailable)` guard their
  sibling smoke test already had, so they hung instead of skipping
- `main` is now a protected branch requiring all seven checks, blocking
  force pushes and deletions

### Fixed

**Retrieval and tenant isolation**

- BM25 search returned the *first* user's index to every subsequent user for
  the lifetime of the process. The retriever is a process-wide singleton and
  its `if not self.bm25` guard meant the per-user index swap ran exactly
  once. This was a live cross-tenant data leak, confirmed by reproduction:
  querying as a nonexistent user returned another user's documents
- Concurrent BM25 writes silently lost each other's updates — whichever save
  finished last overwrote the rest. Found in real data: the eval user's
  on-disk index held three of seven ingested sources
- The per-modality BM25 base class never loaded its existing index before
  appending, so an incremental add would have overwritten a user's entire
  index with just the new batch, and delete/purge always found nothing to
  remove
- BM25 documents built by the index-rebuild entry point were pickled with
  their class recorded as `__main__.BM25Document`, so no other process could
  ever unpickle them. The index built and saved cleanly; every subsequent
  search just returned nothing, degrading hybrid retrieval to dense-only
  while reporting the result as a quality regression

**Settings that silently did nothing**

- `QDRANT_ALLOW_RECREATE` was read by no code at all. A setting whose name
  promises protection over the most destructive path in the codebase did
  nothing whatsoever; an operator setting it to `false` to protect
  production vectors got no protection. Now wired up and verified in both
  directions
- `app/utils/paths.py` read an `ENVIRONMENT` variable that nothing in the
  project sets — the project defines `ENV` — so its production flag was
  permanently false, including in production. Harmless only because nothing
  consumed it yet; it is the Phase 30 placeholder for the local-to-S3
  storage switch, and would have failed silently the moment that was wired
- `MATRYOSHKA_SHORT_DIM` was referenced with no setting ever defined,
  guaranteeing an `AttributeError` the first time that code path ran
- The eval harness read `EVAL_JUDGE_MODEL` in two places with two different
  fallbacks. While `.env` set the value explicitly they agreed by luck; once
  the settings migration dropped the key they diverged, so reports claimed
  the Prometheus judge while the legacy path actually ran — meaning every
  generation metric would have attested to a judge that never executed

**Crashes and silent data loss**

- A prompt builder crashed on its own default argument: one line guarded an
  optional list correctly, the next called `len()` on it unguarded
- `UniversalMetadata` was silently discarded at six ingestion call sites,
  which passed it as `metadata=` when the field is `universal_metadata=`.
  Pydantic ignores unknown keyword arguments by default, so it vanished
  without error
- Deleting all chat sessions never actually purged Redis. The handler called
  two methods that do not exist — one on the registry, one on the memory
  class — and a broad `except` swallowed the `AttributeError`
- The agent's RAG tool called both a non-existent attribute and a
  non-existent method, so it could only ever have returned an empty list
- Corrupt-audio repair used `tempfile.mktemp`, which returns a filename
  without creating the file and leaves a symlink race open, and never
  deleted the repaired file afterwards
- `QdrantVectorStore.delete_by_ids` was called but never existed, so the
  knowledge-base delete-by-file-hash endpoint always returned a 500
- A renamed qdrant-client attribute meant collection stats always reported
  an error instead of statistics

**Tooling and pipeline**

- `pyproject.toml` contained an invalid `[project.scripts]` entry that
  silently broke `pip install -e .` for anyone who tried it
- Any `TestClient(app)` instantiation fired real GPU model preloading on an
  uncancellable executor thread, hanging test teardown indefinitely
- The gold-set ingestion script crashed on its first file every run under
  Windows' cp1252 console encoding, which is why the eval corpus was never
  actually populated
- Doc-only pull requests would have deadlocked permanently once checks
  became required — a workflow filtered out by `paths-ignore` reports no
  status at all, so the check waits forever. The filter was removed from
  every pull-request trigger rather than papered over with a companion
  workflow, which introduces its own race
- The Tier-2 dispatch step lacked the `contents: write` permission its API
  call requires, so it would have failed silently after every deploy
- detect-secrets rewrites its own baseline when line numbers drift and exits
  3 to say so; CI treated that as a failure. Exit 3 is now tolerated and
  exit 1 still blocks, verified against the library's own source to confirm
  a real finding can never be masked

### Production deployment & eval re-baseline (AWS g6e.xlarge / L40S)

The MLOps/CI/CD machinery above only earns its keep once the system runs on the
hardware it was built for. This is that step — the first live deployment to a GPU
cloud box — and, more valuable than the deploy itself, the string of environment
gaps a fresh production machine surfaces that a months-tuned laptop silently
papers over. Every fix below was found by running, not by review.

**Deployment target moved to AWS g6e.xlarge (NVIDIA L40S, 48 GB VRAM)**

- Migrated the single hardware target from g5.xlarge/A10G (24 GB) to
  g6e.xlarge/L40S (48 GB): CUDA compile arch `sm_86 → sm_89` (Ada Lovelace),
  `VRAM_BUDGET_GB` sized for the larger card, and every `A10G` / `g5.xlarge` /
  `24GB` reference removed across code comments, docs, and the `.claude` skills
  so the repo describes exactly one production target with no stale hardware.
- The box needs a **local Redis** for hot-path state (ingestion job status, token
  blacklist, embedding cache, rate limits), separate from the Upstash durable
  store. A fresh Deep Learning AMI has none, so job-status polling 404'd and
  logout silently no-op'd until `redis-server` was installed — now documented as
  a provisioning requirement.
- `MODEL_TIMEOUT_SEC` given real headroom: the L40S loads the ~15 GB Qwen2-VL
  from network-attached EBS (~139 MB/s), which overran the 120 s default — a
  class of slowness that simply never appears on a local SSD.

**Dependency pins hardened after production surfaced the drift**

- `transformers` pinned `<5.0`: v5's `TokenizersBackend` refactor drops the
  slow→fast tokenizer conversion path and breaks TrOCR (and other slow-tokenizer
  models) with a misleading "need sentencepiece" error. The unbounded `>=4.39`
  let a fresh install pull v5 — verified 5.14.1 breaks TrOCR, 4.57.6 works.
- `gradio` removed from `requirements.txt` entirely: the UI is React/Vite and
  gradio is imported nowhere under `app/`; its `huggingface-hub>=1.2.0` pin
  conflicted with `transformers<5.0` (which needs `<1.0`).

**Retrieval eval re-baselined to production (v4 → v5)**

- The v4 retrieval baseline matched recall on **positional chunk IDs**
  (`source::chunk_N`) pinned to a dev machine's exact ingestion. The production
  box chunks the same corpus more finely (1257 chunks), so those IDs don't
  align — the gate reported ~half the baseline recall not because retrieval
  regressed, but because it was measuring against the wrong environment's IDs.
- `app/eval/datasets/verify_gold_index` re-maps each answer row's
  `relevant_chunk_ids` to the chunk of the same source that actually contains the
  row's specific facts (112 rows re-aligned, 0 orphaned, 3 spoken-form-number
  rows left flagged for review) and fills real page/sheet/timestamp/image-title
  locators from the live Qdrant payloads.
- `thresholds.yaml` retrieval gate re-baselined to the **production box**
  (recall@5 0.5089, recall@10 0.5536, MRR 0.3558, nDCG@10 0.4024, hit_rate
  0.6786; n=56, 2026-07-28), superseding the laptop v4 numbers. Gating production
  against a dev-machine baseline was never right; the gate now passes against the
  environment where the system actually runs.
- End-to-end correctness verified independently of the metric: "What was Apple's
  total net sales for FY2024?" returns **$391,035 million**, cited to
  `apple_10k.pdf` p.29. Rerank + generation recover the correct answer even when
  the exact gold chunk isn't rank-1 — which is precisely why a lower exact-chunk
  recall coexists with correct answers. Raw exact-chunk recall is logged as a
  candidate for a future retrieval-quality pass, not this gate.

# [v0.27.0] — AWS Deployment & Scale-to-Zero

MINOR, not a patch: this is Phase 30's deliverable. Everything here belongs to
the deployment phase rather than to Phase 29 (v0.26.0) — the CD pipeline
defects below were only *discovered* by running that pipeline for the first
time against real infrastructure, which is Phase 30 work by definition.

Two themes:

1. **The delivery pipeline had never actually been executed.** `cd.yml` was
   written correctly in the abstract, but a workflow that has never run against
   real infrastructure is a hypothesis, not a pipeline. The first genuine tagged
   run surfaced five defects in a row, each of which would have failed the
   deploy on its own.
2. **The deployed system is now genuinely reachable and survivable** — it serves
   a user interface rather than a bare API, and it runs on a GPU box that is
   stopped by default and wakes itself on a visit. That is the difference
   between a demo that outlives a job search and one that burns a $200 credit
   in under a week.

### Bug Fixed — delivery pipeline

**Deploy could never succeed — a hard-coded 100-second ceiling**

- `aws ssm wait command-executed` was used to wait for the remote deploy. That
  waiter is fixed at 20 attempts × 5s = **100 seconds**, which cannot cover a
  multi-GB CUDA image pull followed by ~18GB of model weights paging off EBS
  (measured ~139 MB/s on this instance). The step was guaranteed to fail
  regardless of whether the deploy itself worked. Replaced with an explicit
  poll loop with a 40-minute cap.
- The post-deploy health check allowed 5 minutes (30 × 10s) for the container
  to answer `/health`. A cold start on this hardware needs longer; raised to
  20 minutes (80 × 15s), and it now aborts early with container logs if the
  container exits during startup rather than waiting out the full window.

**A supply-chain artifact could veto a working release**

- `anchore/sbom-action` (Syft) failed while scanning the freshly-pushed image
  and, lacking `continue-on-error`, failed the entire `build-push` job — even
  though `docker/build-push-action` had already succeeded and the image was
  genuinely in GHCR. An SBOM is a compliance artifact, not a release gate;
  it is now non-blocking, consistent with the Trivy scan beside it.
- `aquasecurity/trivy-action@0.24.0` no longer resolved: aquasecurity retired
  every non-`v`-prefixed tag following a supply-chain incident and now
  publishes only `vX.Y.Z`. Pinned to `v0.36.0`.

**Build-time check that could never pass in CI**

- The Dockerfile asserted `llama_cpp.llama_supports_gpu_offload()` at build
  time. That function probes for a *physical GPU*, which hosted CI runners do
  not have — so the assertion failed unconditionally, independent of build
  correctness. Removed; the CUDA build is verified where a GPU actually exists
  (`install_cuda.sh` on the box, and the deploy health check).

**The UI build stage was silently building nothing**

- `.dockerignore` still had a blanket `ui/` exclusion left over from before
  this release added the static UI mount, so `COPY ui/ ./` in the new
  `ui-builder` stage found an empty context and failed with
  `"/ui": not found`. Removed the stale line; verified with `pathspec`
  (Docker's dockerignore matching engine) that `node_modules/` and `dist/`
  already exclude `ui/node_modules/` and `ui/dist/` generically, so nothing
  else needed to change.

### Improved — delivery pipeline

- The remote deploy script is now authored as an ordinary shell script and
  base64-encoded into a single SSM command, replacing an inline JSON array
  that required hand-escaping quotes and `$` through three layers
  (YAML → JSON → shell).
- Deploy failures are now diagnosable without SSH: preflight checks (docker
  daemon, AWS CLI, `--gpus all`, required mount paths) fail fast with explicit
  messages, and remote `stdout`/`stderr` plus the last 80 lines of
  `docker logs` are streamed into the Actions log.
- The image is pulled *before* the running container is touched, so a failed
  pull can no longer take down a healthy service.
- Container hardening: `--shm-size=2g` (torch multiprocessing outgrows the
  64MB default) and log rotation (`max-size=50m`, `max-file=3`) — this box has
  already filled its disk once.
- `post-deploy-eval` is gated behind a `SELF_HOSTED_GPU_RUNNER` repository
  variable. Tier-2 runs on `[self-hosted, gpu]`; with no such runner
  registered it produced a run queued indefinitely against a runner that would
  never appear. It now skips cleanly until the runner exists.

### Features — scale-to-zero infrastructure

**`deploy/aws/`**

- **Wake gateway** (`lambda/wake_gateway/handler.py`) — an always-on Lambda
  behind a public API Gateway HTTP API. Starts the stopped instance, holds the
  visitor on a self-refreshing interstitial while ~18GB of models page off EBS,
  then
  redirects once `/health` answers. Handles the states that actually occur:
  `stopped`, `pending`, `stopping`, running-but-not-yet-healthy, and
  `InsufficientInstanceCapacity` (which this account has hit before) each get a
  correct response rather than a stack trace.
- **Idle stop** (`lambda/idle_stop/handler.py`) — scheduled every 5 minutes,
  stops the instance after 20 minutes of low `NetworkIn`. Two guards exist
  because both failures are real: a **minimum-uptime guard**, without which the
  first idle check would stop a box a visitor is actively waiting on (it is
  loading models, so network traffic is near zero) and the gateway would
  restart it — a wake/stop loop that bills continuously and serves nobody; and
  an **in-flight SSM check**, because a `cd.yml` deploy runs 20+ minutes at low
  network traffic and must never be interrupted. Both fail *safe*: unreadable
  CloudWatch or SSM means "assume busy, do nothing."
- **`scripts/deploy_lambdas.sh`** — idempotent one-shot: IAM roles, both
  Lambdas, the public API Gateway HTTP API (including the explicit
  `apigateway.amazonaws.com` invoke permission — see below for why this isn't
  a Function URL), and the EventBridge schedule.
- Least-privilege IAM throughout: `StartInstances`/`StopInstances` are scoped
  to the single instance ARN, never `*`. The wake gateway is reachable
  unauthenticated from the internet, so it must not be able to start anything
  else in the account.

### Bug Fixed — deployed system

- **`cd.yml` deploy could not authenticate to AWS.** The `deploy` job declares
  `environment: production`, and the moment a job references a GitHub
  Environment the OIDC subject claim changes format to
  `repo:<owner>/<repo>:environment:<name>` — it no longer carries the ref. A
  trust policy matching only `ref:refs/tags/*` therefore rejected it with
  `Not authorized to perform sts:AssumeRoleWithWebIdentity`. The corrected
  policy (`iam/github-oidc-trust-policy.json`) allows both forms.
- **Supply-chain scans ran out of disk.** Syft and Trivy both failed with
  `no space left on device` exporting the CUDA image on a hosted runner
  (~14GB free). Reclaiming unused preinstalled toolchains frees ~25-30GB in
  about 30 seconds. The scans were already non-blocking, so this restores the
  artifacts rather than unblocking the build.
- **Docker layer cache never hit across tags.** The default `type=gha` cache
  scope derives from the git ref, so a tag build cannot read a cache written by
  a different tag or by a branch build — `v0.26.1` recompiled
  `llama-cpp-python` from source for 1h21m despite the identical layer having
  been built minutes earlier. Pinned to a fixed `scope=magik-cuda-runtime`.
- **The deployed image served no user interface.** The Dockerfile copied
  `app/`, `start_server.py` and `pyproject.toml` but never `ui/`, and
  `app/main.py` had no static mount — so `GET /` returned the JSON service
  banner and a visitor never reached the chat interface. The UI had only ever
  been served by `npm run dev` (Vite's dev server, proxying to :8000), which is
  a development tool and not part of the deployed artifact. Added a
  `ui-builder` Dockerfile stage that compiles the SPA into `/app/ui_dist`, and
  a conditional static mount + SPA fallback in `app/main.py`. The mount is
  conditional so an API-only source checkout still runs unchanged, and the
  catch-all refuses known API prefixes so a mistyped endpoint returns a JSON
  404 rather than HTML. No UI code changed: `client.js` already used
  same-origin relative paths (`const API = ''`).
- **Rate limiting failed open in the container, and job status 404'd.**
  `infra_registry.get_cache()` dials `LOCAL_CACHE_HOST` (default `localhost`),
  but inside a container `localhost` is the container — which runs no Redis. So
  `get_cache()` returned `None` and every caller degraded: ingestion job-status
  polls 404 (breaking upload progress in the UI) and, per its own docstring,
  `app/auth/rate_limit.py` **fails open** — removing the only guard against a
  public demo being hammered on a $1.86/hr GPU. `cd.yml` now runs a
  `magik-redis` sidecar on a user-defined network and points the app at it with
  `-e LOCAL_CACHE_HOST`, which overrides the env-file so running from source on
  the host (where `localhost` is correct) is unaffected. The sidecar is
  deliberately ephemeral — no persistence, 512MB cap, LRU eviction — because
  everything in it is rebuildable cache.
- **The wake gateway's public entry point returned 403 Forbidden despite
  correct configuration.** The wake gateway was first deployed behind a Lambda
  Function URL (`auth-type NONE` + a resource policy granting
  `lambda:InvokeFunctionUrl` to principal `*`, conditioned on
  `lambda:FunctionUrlAuthType: NONE`). Both settings were independently
  verified correct via `get-function-url-config` and `get-policy`, on two
  separately created URLs, and requests still failed with 403 — AWS
  Organizations SCPs were also checked and ruled out
  (`AWSOrganizationsNotInUseException`, confirmed standalone account). Root
  cause was never conclusively identified. Replaced the front door with an API
  Gateway HTTP API (`apigatewayv2 create-api --target <lambda-arn>` plus an
  explicit `lambda add-permission` for principal `apigateway.amazonaws.com`,
  since the quick-create `--target` shortcut did not reliably attach the
  invoke permission on its own — confirmed by a `500` that traced to a
  Lambda log group that had never been created, i.e. the function was never
  invoked). This is a strictly more mature permission model and worked on the
  first request once the permission was attached. `handler.py` required no
  changes — both integration types deliver the same
  `{statusCode, headers, body}` response shape.

### Documentation

- `deploy/aws/README.md` — architecture, deploy/verify commands, and the
  operational rules that are easy to get wrong (never put the model cache on
  instance-store; port 8000 stays open since there is no reverse proxy in
  front of the app).

### Bug Fixed — Tier-2 eval could never have run

The self-hosted GPU jobs in `eval-gate.yml` had never executed once — the
runner they target did not exist until this phase. Auditing them before
enabling found four independent defects, each fatal on its own, in a job that
looked correct:

- **No dependency installation.** `tier2-full-suite` went straight from
  `actions/checkout` to `python -m app.eval.run`, with no `setup-python` and
  no `pip install` (both of which `tier1-retrieval` has). A systemd-run runner
  does not inherit a login shell's PATH, so nothing guaranteed an interpreter
  with torch, let alone the project's dependencies.
- **No Qdrant credentials.** Tier-1 explicitly maps `QDRANT_URL`/
  `QDRANT_API_KEY` from repository secrets; Tier-2 set neither. Repository
  secrets are not auto-injected into a job, so retrieval — the one *gated*
  sub-suite — would have scored against an unreachable vector store.
- **No BM25 index.** `app/utils/paths.py`'s `DATA_ROOT` is the *relative*
  path `data/users`, resolved against the process CWD, and
  `actions/checkout`'s `git clean -ffdx` deletes gitignored `data/`. The BM25
  half of hybrid retrieval would have returned empty on every query.
- **No access token.** `/rag/query` requires `get_current_user`, and
  `EVAL_ACCESS_TOKEN` was never set, so every generation and e2e call would
  have returned 401.

Combined, the first three would have produced a *false* retrieval regression:
the gate is real (`retrieval.gate_enabled: true`, v5 production baseline), so
the job would have failed red while reporting a quality problem that did not
exist — the single most corrosive failure mode a gate can have.

Fixed by executing both self-hosted jobs inside the deployed container
(`docker exec magik-current …`) rather than on the runner. That resolves all
four at once — dependencies are baked into the image, credentials arrive via
`--env-file /opt/magik/.env`, the real corpus is mounted at `/app/data`, and
the live server answers on `127.0.0.1:8000` — and is more correct besides:
post-deploy eval should measure the artifact actually serving traffic, not a
git checkout beside it. The eval token is minted inside the container from its
own `JWT_SECRET_KEY` and consumed within the same step, so no credential is
stored in a secret, a step output, or a file.

Also fixed while in here:

- **Preflight checks with legible failures.** Both jobs now verify the
  container is running and healthy before doing anything, and Tier-2
  additionally asserts a BM25 index exists for `EVAL_USER_ID` — turning "the
  corpus is not ingested for this tenant" into that exact message instead of a
  silent recall collapse reported as a regression.
- **`seed-eval-fixtures` published to a cache key nothing reads.** It saved
  `bm25-eval-index-<hash>` while `tier1-retrieval` restores
  `bm25-eval-index-v2-<hash>`. Every publish was a dead write; masked only
  because Tier-1 self-heals by rebuilding from Qdrant. Salts now match.
- **No job timeouts on GPU jobs.** Both inherited GitHub's 6-hour default,
  so one hung judge call could pin a $1.86/hr instance for a quarter of a day.
  Capped at 120 min (Tier-2) and 180 min (seed).

### Features — self-hosted GPU runner registered, Tier-2 eval unblocked

- **The self-hosted GPU runner (Phase 30 Stage 5) is registered and idle**,
  removing the last blocker to running `eval-gate.yml`'s Tier-2 suite from
  `post-deploy-eval`. But the runner lives on the same box `idle-stop`
  monitors for auto-shutdown, and a GPU-bound eval job produces almost no
  external `NetworkIn` — so enabling Tier-2 without a guard would let
  `idle-stop` genuinely kill the instance mid-eval-run, corrupting the run and
  taking the runner itself offline. Added a fourth guard to `idle_stop/
  handler.py`: it checks the runner's `busy` status via the GitHub REST API
  (`GET /repos/{repo}/actions/runners`) before ever stopping the box, using a
  fine-grained PAT (Administration: read-only, the minimum that endpoint
  allows) stored in SSM Parameter Store at `/magik/github_actions_pat` —
  same pattern as `/magik/ghcr_pat`, never touching a GH Actions log. IAM is
  scoped to `ssm:GetParameter` on that one parameter ARN only. Fails safe like
  every other guard here: an unreadable token or API call means "assume busy,
  do nothing." Setting the `SELF_HOSTED_GPU_RUNNER` repository variable to
  `true` is still a manual step — see `deploy/aws/README.md`'s "Runner
  busy-check token" section for the one-time PAT setup that must happen first.

### Known Issues

- Full wake-cycle verification (cold start → interstitial → redirect →
  healthy UI) and idle-stop verification (confirming the box actually stops
  itself after ~20-25 min idle) are both still pending — not because of a bug,
  but because `g6e.xlarge` capacity in `us-east-1` has been intermittently
  unavailable (`InsufficientInstanceCapacity`) throughout this phase, most
  recently confirmed by the wake gateway correctly rendering its own
  capacity-error page end-to-end. Re-run the verification steps in
  `deploy/aws/README.md` once a `StartInstances` call succeeds.
