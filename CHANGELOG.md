# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0-rc1] - 2026-08-23

Release candidate — deploying today, monitoring production for one week
before promoting to the final v1.0.0 tag. v0.33.0's own tagged build also
failed to reach production; that failure surfaced a second, deeper bug in
the same area as v0.33.0's original fix, documented below. RAG system
retrieval, agent, guardrail, and memory logic is unchanged from
[0.33.0]/[0.32.0] below — every fix in this release closes gaps found
during two consecutive real production promotion attempts, not new
product behavior.

### Fixed
- `app/bin/models/download_all_models.py`'s "already cached" fast path
  (`_handle_cached()`) verified a model's on-disk checksum but never wrote
  a `download_manifest.json` entry for it. A model whose files were
  already present — including `Qwen/Qwen2-VL-7B-Instruct` itself, left on
  disk by v0.33.0's own crash-and-restart cycle — stayed permanently
  invisible to `startup_validator.py`'s strict manifest check: every
  deploy attempt found the files, skipped re-downloading, and never fixed
  the manifest, so the exact same `Required models not cached` crash
  repeated on every restart with no way to self-heal. `_handle_cached()`
  now writes the manifest entry the first time it finds a model cached
  with no existing record, reusing the checksum it already computes —
  self-healing on the next deploy, no manual box intervention needed.

### Known limitations carried forward

Unchanged from [0.33.0]/[0.32.0] below — see README.md's Known
Limitations & Roadmap section.

## [0.33.0] - 2026-08-23

Fixes a real deploy-blocking gap discovered when v0.32.0's tagged build was
actually promoted to production for the first time, plus release-readiness
governance docs and a CI fix found in the same pass. RAG system retrieval,
agent, guardrail, and memory logic is unchanged from [0.32.0] below — this
release is about closing gaps found during v0.32.0's real production
promotion attempt, not new product behavior.

### Added
- `SECURITY.md` — private vulnerability disclosure process, supported
  versions, scope.
- `CONTRIBUTING.md` — branch workflow, commit conventions, pre-PR checklist.
- `.github/PULL_REQUEST_TEMPLATE.md`.

### Fixed
- `app/bin/models/download_all_models.py` was missing `Qwen/Qwen2-VL-7B-Instruct`
  from its manifest entirely, even though `startup_validator.py`'s
  `REQUIRED_MODELS` has required it all along — silently absent until
  `MODEL_CACHE_REQUIRE_MANIFEST` was turned on for the first time in
  v0.32.0. v0.32.0's real production promotion failed at container startup
  as a direct, correct consequence (`RuntimeError: Required models not
  cached (1 missing): Qwen/Qwen2-VL-7B-Instruct`) — the strict manifest
  check did exactly what it was built for, refusing to serve rather than
  degrading silently, and surfaced a gap that had been invisible until this
  release turned the check on. Added the missing manifest entry
  (`qwen2vl_7b`, 16.59GB, revision-pinned) so the model downloads and
  checksum-verifies like every other required model.
- `quality-live.yml`'s `ragas-report`/`deepeval-report` jobs crashed at
  import time (`JWT_SECRET_KEY must be at least 32 characters long`) on
  every run — the jobs never set the placeholder `SECRET_KEY`/
  `JWT_SECRET_KEY` env vars that `ci.yml` and `eval-gate.yml` already carry
  for the identical `Settings.validate()` import-time requirement. Neither
  job had ever completed successfully before this fix, which is why
  `quality-reports/ragas/` and `quality-reports/deepeval/` had only ever
  held a `.gitkeep`.
- `detect-secrets` flagged the two placeholder `SECRET_KEY`/`JWT_SECRET_KEY`
  lines added above as unaudited "Secret Keyword" findings — the same
  literal strings already exist safely in `ci.yml`/`eval-gate.yml`, but this
  was a new file location the baseline hadn't seen. Allowlisted inline with
  `pragma: allowlist secret` rather than regenerating the baseline for two
  known-fake values.

### Known limitations carried forward

Unchanged from [0.32.0] below — see README.md's Known Limitations &
Roadmap section.

## [0.32.0] - 2026-08-22

Final hardening pass before v1.0.0. Found and fixed live against the running
v0.31.0 production system.

### Added
- Online-eval sampling enabled in production (`ONLINE_EVAL_ENABLED=true`,
  `ONLINE_EVAL_SAMPLE_RATE=1.0`) — the "RAG Quality" Grafana dashboard had
  been empty because live-traffic sampling was never turned on.
- Session-scoped caching for web-search answers, closing an inconsistent-
  answer report and eliminating redundant Tavily API calls.
- Qdrant collection snapshot/restore capability (`create` / `list` / `restore`
  CLI), with a confirmation prompt before any destructive restore.
- Python dependency lock file (`requirements.lock.txt`) via pip-compile.
- Exact commit-SHA pinning for all 14 HuggingFace-hosted models, replacing
  default-branch tracking.
- Production config hardening: PII detection, image EXIF/GPS stripping,
  startup model-manifest verification, and INT8 vision-model loading all
  enabled after an audit found each defaulting off and never explicitly set.
- `app/bin/seed_eval_reporter.py` — creates the dedicated, OTP-skip account
  (pinned to the existing `EVAL_USER_ID` tenant, which already owns the
  ingested gold-set corpus) that the Ragas/DeepEval report workflows log in
  as. Run once against production and verified end-to-end (a real
  `POST /auth/login` against the live server returned a genuine token, no
  OTP challenge) — not just assumed working from the Mongo write succeeding.
- Ragas/DeepEval now also run automatically once per real release, appended
  to `cd.yml`'s `promote-production` job — production is already awake and
  healthy at that point, so this costs no extra wake. Non-blocking
  (`continue-on-error`) and never gates or rolls back the deploy. Report +
  regenerated badges are committed back to `main` only if a run actually
  produced new content.

### Fixed
- A knowledge-base file deletion by any user silently flushed the *entire*
  shared query-response cache instead of only the entries referencing that
  file.
- Long-term conversation summaries were generated correctly but never
  persisted to MongoDB, due to a missing `user_id` on the storage call —
  long-term memory recall had silently never worked in production.
- `GET /knowledge-base` could permanently delete a user's uploaded file from
  disk on a transient Qdrant read (no undo). The endpoint is now a
  side-effect-free listing; cleanup of a genuinely failed upload happens
  once, at the point of failure.
- `install_cuda.sh` installed PyTorch with no version pin and a
  `llama-cpp-python` bound inconsistent with the Docker image. Both
  provisioning paths now pin identical, verified-compatible versions.
- Ragas/DeepEval "live" report generation (`quality-live.yml`) could never
  actually work: it ran on a bare GitHub Actions runner with no access to
  production's real `JWT_SECRET_KEY`, so every in-process-minted token was
  signed with the wrong secret and silently, permanently rejected — the
  reason `quality-reports/ragas/` and `quality-reports/deepeval/` had never
  held anything but a `.gitkeep`. `EvalAuth` (`app/eval/http_client.py`) now
  supports a real login-based token path (`EVAL_REPORTER_EMAIL`/
  `EVAL_REPORTER_PASSWORD`), correct by construction since the server issues
  the token itself.

### Removed
- Repository cleanup ahead of the final version: a stray personal-data folder
  that had landed inside `data/`, a scratch notebook, MLflow experiment-run
  artifacts, an orphaned top-level BM25 index file with zero real references,
  eleven empty throwaway test-user directories, and every local tool cache
  (`.coverage`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.hypothesis`,
  `.lighthouseci`, `__pycache__`, build artifacts). No source code, tests, or
  real project/user data touched — verified before and after.

### Documentation
- `README.md` fully overhauled: the live demo link now points at
  `launch.vk-ai.online`'s status page instead of the raw AWS URL, a real
  per-modality generation-quality scorecard (sourced from
  `docs/modality_scorecard_2026-08-20.pdf`) replaced the old qualitative
  accuracy table, and the "Agentic query routing" feature description now
  states plainly that RAG requires an explicit file selection and web search
  is heuristic-by-default with an explicit toggle for a deterministic
  contract — both deliberate design choices, not gaps. Caught and fixed two
  stale claims in the same pass: the test-file count (137, not the
  previously-stated 99) and a missing `tests/api_contract/` entry.
- `CHANGELOG.md` itself reformatted to standard Keep a Changelog style
  (newest-first, consistent Added/Changed/Fixed/Removed sections, real dates
  from git tag history) — same content, condensed from ~4,800 lines to
  ~600 without dropping any version's substance.
- `deploy/aws/scripts/restore_ssm_secrets.py` — one authoritative manifest of
  every `/magik/*` SSM parameter the project depends on, with a `--check-only`
  verification mode and a real restore path (from a live container's own
  environment, or from an offline backup file), replacing the hand-typed
  `aws ssm put-parameter` commands previously documented as the only way to
  (re)populate them. Uses boto3 directly rather than the AWS CLI, so there is
  no shell in the path to mangle an argument.

## [0.31.0] - 2026-08-21

Observability made live in production, a blocking staging quality gate,
Tier-2 eval reliability, and a full video-modality accuracy pass.

### Added
- Staging deploy gate: every tag now deploys to a private staging box first,
  runs the full Tier-2 quality suite against it, and only promotes to
  production if it passes; a failure rolls staging back and never touches
  production.
- Second EC2 GPU box for staging (zero inbound rules, SSM-only access),
  cloned from a production AMI, with its own self-hosted CI runner.
- `tier2-eval.yml`, a reusable workflow extracting the Tier-2 eval/rollback
  logic so staging and production's nightly run share one implementation.
- Conversational tone rewrap on the final answer, applied only after every
  accuracy-critical stage and discarded if it alters any cited figure.
- "Summarize this document" as a whole-document map-reduce operation,
  distinct from top-k retrieval.
- `app/core/model_reaper.py` — background sweep that evicts idle
  ingestion-only models from VRAM under a watermark.
- Full AWS account rebuild after the prior account's EC2 instances/EBS
  volumes were deleted: all infrastructure (VPC, EC2, EBS, IAM, security
  groups) codified in Terraform; Wake Gateway and idle-stop Lambdas
  redeployed; monitoring stack (Prometheus/Grafana/Loki/Tempo) redeployed to
  production.
- Wake gateway rebuilt as a live, multi-step status page (AJAX-polled state
  machine: waking / loading / stuck / capacity / error / ready) replacing a
  blind auto-refresh, served from a dedicated `launch.vk-ai.online` domain.

### Fixed
- Prometheus/OTel were never actually enabled in the production environment
  file, so the fully-built monitoring stack had nothing to scrape.
- Trace/log correlation was dead — the log formatter never injected the
  active span's trace ID, so Grafana's log-to-trace jump never worked.
- Tier-2 eval could never complete a run: the judge model loaded in-process
  and corrupted the shared CUDA context, crashing the suite; moved to a
  subprocess judge, matching the resident LLM's own architecture.
- Idle GPU models never had their memory reclaimed between eval sub-suites,
  causing repeated CUDA OOM and multi-hour timeouts; VRAM-watermark eviction
  now runs on every model load, not only on an idle timer.
- Video-modality retrieval, sentence-scoring, chart lookup, and query
  routing bugs that capped answer correctness at 0.46; fixed to 0.88 with no
  regression on any other modality.
- Production's model cache moved off a 93%-full root volume onto a
  dedicated EBS volume, mounted by UUID instead of a fragile symlink chain.

### Known limitations
- Uptime Kuma and a hardened Grafana `basic_auth` credential are configured
  but not provisioned/rotated.

## [0.30.0] - 2026-08-07

Demo-account reliability, answer regeneration, and a bundle of citation and
web-search correctness fixes found during real end-to-end use.

### Added
- `regenerate` as a first-class, explicitly-gated request flag — sampling
  temperature floor, fresh seed, and a rewrite directive — so "Regenerate"
  produces a genuinely different answer instead of replaying a deterministic
  one. The default answer path remains fully reproducible.
- httpOnly cookie-based session storage, replacing tokens in `localStorage`
  and in OAuth redirect URLs.
- CSRF middleware for cookie-authenticated mutating requests.

### Fixed
- The demo account (`testuser@ragdev.local`) could fall through to a normal
  OTP email challenge on any environment where its database flag hadn't
  been (re-)seeded; the login bypass now also matches on a configured email
  address and self-heals the flag.
- Password change and "sign out everywhere" revoked active sessions but left
  a 30-day trusted-device OTP exemption alive on the browser.
- Image citations were silently dropped depending on which retriever (BM25
  vs. dense) matched a chunk first, due to a metadata-loss bug in both BM25
  implementations and a fusion step that discarded the richer of two
  duplicate hits instead of merging them.
- A refusal could reach the user dressed as a real answer with source chips
  attached, when a streaming refusal check drifted out of sync with an
  earlier guard in the same function.
- Web-search mode could flicker from a correct web answer to a
  knowledge-base answer, because the non-streaming fallback endpoint
  accepted `force_web` and never read it.

### Security
- Access, refresh, and device tokens were readable from `localStorage` by
  any XSS payload or browser extension, and were briefly exposed in the
  OAuth redirect URL. Both are closed by the httpOnly-cookie change above.

## [0.29.0] - 2026-08-03

Testing and quality-reporting initiative: API contract testing, a second
independent LLM-eval framework, load/multi-tenant simulation, browser
performance, DAST, and passive uptime monitoring.

### Added
- Schemathesis property-based API fuzzing against the live OpenAPI schema.
- Ragas and DeepEval as two independent evaluation frameworks scoring the
  same gold dataset, both backed by MAGIK's own resident judge model (never
  a third-party API).
- k6 load, stress, soak, and multi-tenant concurrency tests, asserting zero
  cross-tenant data leakage under real concurrent load.
- Lighthouse CI (browser performance) and OWASP ZAP (passive + opt-in active
  DAST).
- Passive Uptime Kuma push-monitor hooks in the existing wake/idle-stop
  Lambdas — monitoring reports status as a side effect of work already
  being done, and can never itself trigger a wake.
- `quality-reports/` and shields.io badges, tracked in git and linkable from
  the README.
- Idle-eviction for ingestion-only models in `ModelLoader`, freeing VRAM
  after 5 minutes of inactivity.

### Changed
- Consolidated onto a single evaluation judge model (Qwen2.5-7B-Instruct),
  removing two retired judges and the self-evaluation bias of judging the
  resident RAG model with itself.
- API rate limiting moved from per-IP to per-authenticated-user, closing a
  gap where a fully-built per-user limiter existed but was never called.

### Fixed
- A duplicate Prometheus metric registration caused a permanent, repeating
  crash on ingestion and stalled Tier-2 eval runs.
- Video ingestion silently dropped its audio transcript on every run due to
  a thread pool that didn't propagate request context.
- The model-download script re-verified and re-downloaded already-cached
  models on every boot, due to a checksum comparison that included files
  written after download completed.
- `cd.yml`'s deploy job was missing its repository checkout step, failing
  the first real production deploy at the first file read.

### Known limitations
- The Uptime Kuma host is not yet provisioned.
- CI's local-mode Schemathesis/k6/ZAP/Lighthouse jobs cannot complete on a
  hosted runner (no GPU, and the full model set cannot be cached within
  GitHub Actions' cache size limit); tracked as a follow-up requiring a
  second self-hosted runner.

## [0.28.0] - 2026-07-31

Monitoring and observability stack, Tier-2 auto-rollback, and secrets
migration to AWS SSM Parameter Store.

### Added
- Prometheus, Grafana, Tempo, and an OpenTelemetry Collector, additive to
  the production container, with three dashboards (system health, RAG
  quality, logs) and unified alerting to a Slack-compatible webhook.
- Live-traffic online evaluation: deterministic sampling of real queries,
  scored with reference-free metrics and pushed as Prometheus gauges.
- Log aggregation via Loki + Promtail, with bidirectional log-to-trace
  correlation through existing trace IDs.
- Automatic rollback when a post-deploy Tier-2 run regresses the gated
  retrieval section specifically (every other section remains
  informational-only against an unvalidated baseline).
- Migrated five app secrets (`GOOGLE_CLIENT_SECRET`, `SMTP_PASSWORD`,
  `SECRET_KEY`, `JWT_SECRET_KEY`, `MONGO_URI`) from plaintext on the
  instance to AWS SSM Parameter Store, fetched fresh on every deploy.
- Rate-limited OTP resend endpoint, replacing a workaround that silently
  failed during registration.

### Changed
- Prometheus judge (`prometheus-7b-v2.0`) downloaded by default at
  provisioning time instead of being excluded as "eval-only."

### Fixed
- The app's own metrics port collided with Prometheus's own default port,
  which would have silently broken scraping.
- Conversation memory was silently empty on the live streaming path — the
  code path the UI actually calls never fetched history at all, a gap the
  evaluation harness's separate code path never exercised.
- Image and video captioning were fully broken in production because the
  runtime image was missing a C compiler required by the vision model at
  inference time.
- The evaluation harness re-ran full audio transcription/diarization up to
  11 times on the same file across gold rows that shared a source,
  dominating total suite runtime.

### Removed
- Guest/anonymous trial mode — replaced by a single, permanent, pre-verified
  demo account for recruiter/hiring-manager evaluation, after the
  guest-to-account data migration path proved to be a recurring source of
  silently orphaned data.
- Unused optional integrations (Cohere, SerpAPI, Langfuse) — declared in
  config but never read anywhere in the application.

### Known limitations
- Finance numeric fidelity is not yet scored on live sampled traffic, only
  at the offline CI gate.

## [0.27.0] - 2026-07-31

AWS production deployment and scale-to-zero infrastructure.

### Added
- Scale-to-zero via two Lambdas: a wake gateway that starts the stopped
  instance and holds the visitor on an interstitial page until the app is
  healthy, and an idle-stop scheduler that stops the instance after a
  sustained low-traffic window (guarded against stopping mid-deploy or
  mid-wake).
- Custom domain (`magik.vk-ai.online`) with Caddy + Let's Encrypt,
  terminating the public entry point in front of the app.
- React SPA served directly by the production image via a new Docker build
  stage and static mount, replacing a dev-only proxy setup that had no
  deployed equivalent.
- GPU admission control shared across ingestion and query paths, replacing
  a semaphore that existed but was never actually wired into the live
  request path.

### Fixed
- The deploy pipeline used a fixed 100-second wait for a multi-gigabyte
  image pull and model download, guaranteeing failure regardless of deploy
  health; replaced with an explicit 40-minute poll.
- A supply-chain SBOM scan failure could block a successful, already-pushed
  image; made non-blocking.
- The container port-rename step during deploy didn't stop the previous
  container first, so every deploy after the first failed to bind its port.
- Rate limiting failed open inside the container because it dialed
  `localhost` for a Redis instance that only exists on the host; a
  dedicated Redis sidecar container closes the gap.
- The Tier-2 self-hosted eval jobs had never actually run: missing
  dependency install, missing credentials, missing BM25 index, and missing
  auth token, any one of which would have produced a false "regression."
  Fixed by running eval inside the already-provisioned production
  container instead of a bare CI runner.
- A JWT-shape extraction bug meant every authenticated eval request failed
  instantly with a raw decode error, ending a 1-3 hour suite in ~15 minutes.

### Known limitations
- Five app secrets remained in plaintext on the instance at the time of this
  release (resolved in v0.28.0).

## [0.26.0] - 2026-07-28

Production MLOps, LLMOps, and CI/CD — no retrieval, agent, or guardrail
behavior changed in this release; this is the operational discipline layer
around the system.

### Added
- Trust-on-first-use checksum verification and explicit revision pinning
  for every downloaded model, with startup validation that aborts on a
  mismatched or incomplete model cache.
- `PROMPT_VERSION` and full model manifest recorded on every evaluation run
  and exposed via `GET /version`.
- CI (`ci.yml`): ruff, black, isort, mypy, and the full unit test suite on
  every pull request across two Python versions.
- Two-tier retrieval quality gate: a blocking PR-time gate against a
  measured baseline (Tier 1), and a full generation/behavioral suite
  against a real judge model, GPU-only, post-deploy (Tier 2).
- Supply-chain and code security scanning: secret detection, SAST, CVE
  auditing, and a dependency license scan, enforced in CI.
- Automated container build/push/deploy pipeline with SBOM, provenance
  attestation, and automatic rollback on a failed health check.

### Fixed
- BM25 search returned the first user's index to every subsequent user for
  the life of the process — a live cross-tenant data leak in the retrieval
  singleton's initialization guard.
- Concurrent BM25 writes silently overwrote each other's updates, and
  documents pickled by the index-rebuild path were unreadable by any other
  process, degrading hybrid search to dense-only with no visible error.
- Several settings that appeared to gate destructive or security-relevant
  behavior were read by no code at all, including a Qdrant recreate-on-
  mismatch guard.
- Multiple silent data-loss and crash bugs across memory deletion, the
  agent's RAG tool, corrupt-file repair, and vector store deletion, each
  masked by an overly broad exception handler.

### Production deployment
- First live deployment to AWS (g6e.xlarge, NVIDIA L40S, 48GB VRAM), with
  the retrieval evaluation baseline re-measured against the real production
  environment rather than a developer machine.

## [0.25.0] - 2026-07-22

The largest release to date: a full per-modality architecture rebuild, a
model-stack upgrade, and the project's first structured evaluation harness
measuring all seven modalities against a real gold dataset.

### Added
- Per-modality architecture: one dedicated chunker, embedder, and BM25
  implementation per modality, replacing large shared files with
  branching logic, reachable only through a public dispatch layer.
- `app/eval/` — gold datasets, judges, metrics, and runners for retrieval,
  generation, and behavioral evaluation, none of which existed before this
  release.
- `app/verification/` — a generic self-verifying answer loop (groundedness,
  citation, completeness, confidence, retry) replacing a one-off,
  video-only verification hack.
- Full React/Vite/Tailwind frontend, replacing the original Gradio
  interface: streaming chat, finance-specific components (financial
  tables, clickable media timestamps, an earnings-call browser), knowledge
  base management, and persistent login.
- Deterministic OpenCV chart digitizer for financial line charts, reading
  exact values from pixel geometry instead of asking a vision model to read
  them off an image.

### Changed
- Resident LLM upgraded to Qwen2.5-14B-Instruct; added a dedicated
  evaluation judge model (Prometheus-2-7B) rather than reusing the RAG
  model to judge itself.
- Vision-language model upgraded to Qwen2-VL, replacing BLIP-1 and a
  long-standing bug where BLIP's caption text was reused verbatim as the
  next model's prompt.

### Fixed
- Video ingestion crashed outright whenever diarization returned any
  speaker segments — video had never worked end-to-end with diarization
  before this release.
- A same-topic, different-period document sharing a knowledge base with
  the correct source could silently answer with the wrong period's
  numbers; fixed with meeting/event-scoped retrieval.
- Per-modality accuracy improvements across the board following the new
  evaluation harness — most notably XLSX generation accuracy from 0.000 to
  0.786, and image chart Q&A from 0.289 to 0.857.

## [0.24.0] - 2026-05-29

Full authentication, MFA, and tenant security.

### Added
- JWT authentication with access/refresh token rotation, Argon2id password
  hashing, and TOTP multi-factor authentication with single-use backup
  codes.
- Token revocation via a Redis blacklist and a logout-all mechanism that
  invalidates every active session across all devices.
- Google OAuth2 sign-in.
- Admin panel with role management and account deactivation.
- Multi-tenant data isolation enforced at every storage layer — Qdrant,
  Redis, MongoDB, and a per-user BM25 index, all scoped by the verified
  JWT's `user_id`.
- GDPR self-delete, purging a user's data across every storage layer in one
  call.

### Fixed
- Constant-time password verification on a missing-user login, preventing
  timing-based account enumeration.
- Refresh tokens rotate on every use and are single-use by design.

## [0.23.0] - 2026-05-27

Production guardrails and pre-ingestion attack defense.

### Added
- `app/guardrails/` — a unified input/output guardrail package replacing
  seven scattered, inconsistent sanitization implementations: prompt
  injection and jailbreak detection, PII scrubbing, SSRF protection, an
  audit log, and a per-session/per-IP rate limiter.
- 257-test guardrail suite including a 109-case adversarial corpus spanning
  injection, jailbreak, encoding bypass, PII, SSRF, and pre-ingestion
  attack vectors.

### Fixed
- Hidden prompt-injection text in white-on-white PDF text, hidden Excel
  rows/columns, and image/video caption overlays could all reach the
  vector index unfiltered before this release.
- Author metadata in DOCX comments was stored verbatim in the vector index
  without PII scrubbing.

### Security
- Injection corpus recall improved from 49/64 to 64/64 (100%), with a
  0.9% false-positive rate and all 10 OWASP LLM Top 10 (2025) threat
  categories addressed.

## [0.22.0] - 2026-05-27

Evaluation harness and RAG quality metrics.

### Added
- Evaluation CLI with an exit-code gate, 54 hand-curated gold Q&A pairs
  across all seven modalities, and a committed baseline report for
  regression detection in pull requests.
- Retrieval metrics (recall@k, MRR, nDCG, hit rate), generation metrics
  (faithfulness, relevancy, context recall), and a routing accuracy
  benchmark.

## [0.21.0] - 2026-05-23

Production hardening, multimodal edge-case robustness, and a test
foundation.

### Added
- Bounded agent execution (step, wall-clock, and token budgets).
- Tenant isolation via a typed Qdrant payload filter.
- Circuit breaker on Qdrant calls, and a GDPR purge path across every
  storage layer.
- Hallucination guard and numeric-faithfulness check in the reasoning
  engine.

### Changed
- Startup latency reduced from ~25s to ~7s via lazy model loading and a
  deferred device manager.

### Fixed
- Section-aware chunking now preserves document structure for
  time-sensitive queries.
- An empty retrieval result now raises a clear error instead of a silent
  stub answer.

## [0.20.0] - 2026-04-27

Deterministic multimodal RAG stabilization and agent control hardening.

### Added
- Strict grounding — the model answers only from retrieved context.
- Multi-user session isolation across retrieval, memory, and the vector
  store.
- Intent-aware agent routing with parallel sub-query execution.

### Fixed
- BM25 indexing and retrieval issues, duplicate/low-quality chunk
  retrieval, and session leakage across memory and retrieval.

## [0.19.0] - 2026-04-19

Multimodal system refactor and architecture strengthening.

### Added
- Standardized ingestion schema across text, document, image, audio, and
  video, with enriched, modality-aware metadata.
- `ModelLoader` with lazy loading and centralized caching, replacing
  scattered per-module model initialization.

## [0.18.0] - 2026-04-09

Hybrid retrieval and reranking stabilization.

### Added
- BM25 keyword retrieval combined with existing semantic vector search via
  a new hybrid retriever, plus a cross-encoder reranker.
- PDF/Word/Excel ingestion: text, image, and table extraction.

## [0.17.0] - 2026-04-08

Agentic pipeline, model loader, and full observability.

### Added
- `AgentController` for intelligent, decision-based query routing across
  the multimodal and standard pipelines, plus a web-search tool.
- Centralized model management via `ModelLoader`, and structured logging
  replacing scattered print statements.

## [0.16.0] - 2026-04-04

Multimodal intelligence and reasoning.

### Added
- Reasoning engine with query decomposition, multi-query retrieval, and
  result fusion/ranking.

## [0.15.0] - 2026-04-04

System integration and stabilization.

### Added
- BLIP image captioning, audio transcription, and video frame/audio
  processing on a unified embedding pipeline.
- Redis (short-term) and MongoDB (long-term) memory.

### Fixed
- Vector dimension mismatch between image and text embeddings.

## [0.14.0] - 2026-04-04

Smart memory optimization.

### Added
- Memory formatter, semantic memory filtering, and a memory fusion layer
  for a token-efficient, context-aware memory system.

## [0.13.0] - 2026-04-03

Redis memory summarization.

### Added
- `MemoryManager` for automatic, LLM-based conversation summarization,
  injected into the RAG pipeline as conversation history.

## [0.12.0] - 2026-04-02

Memory system integration.

### Added
- Redis-based short-term conversational memory and MongoDB-based
  persistent memory, with session-based multi-user support.

## [0.11.0] - 2026-04-01

UI.

### Added
- Multi-session chat UI with streaming responses and multimodal upload
  support (PDF, image, audio, video).

### Fixed
- Qdrant collection mismatch and embedding consistency issues between
  query and document paths.

## [0.10.0] - 2026-03-30

Multimodal video RAG.

### Added
- Video ingestion: frame extraction with BLIP captioning, audio extraction
  and transcription, and unified multimodal embeddings.

## [0.9.0] - 2026-03-28

Audio intelligence upgrade.

### Added
- Audio ingestion via faster-whisper with segment-level chunking and
  timestamp metadata, completing the first full multimodal RAG pipeline
  (text, image, audio).

## [0.8.0] - 2026-03-28

Text and image ingestion and query.

### Added
- Image ingestion and query pipeline using CLIP vision/text models.

### Fixed
- Embedding dimension mismatch between ingestion and query paths.

## [0.7.0] - 2026-03-25

Production-grade ingestion pipeline.

### Added
- End-to-end ingestion orchestration with structured API responses and
  pipeline observability logging.

## [0.6.0] - 2026-03-24

Improved RAG pipeline.

### Added
- Chunk-based ingestion, batch embedding, and a configurable top-k
  retriever.

### Fixed
- Context window overflow and a UUID generation bug in vector storage.

## [0.5.0] - 2026-03-24

GGUF model integration.

### Added
- Local GGUF model inference via llama.cpp (CPU-based), with a streaming
  `/rag/query/stream` endpoint.

### Changed
- Replaced hosted HuggingFace/Ollama inference with local quantized
  models.

## [0.4.0] - 2026-03-21

Chunking integration.

### Added
- Recursive chunking with overlap, batch embedding, Qdrant integration, and
  the first end-to-end RAG pipeline (retrieve + generate).

## [0.3.0] - 2026-03-20

Multimodal ingestion.

### Added
- Ingestion pipeline for text, image, audio, and video, with faster-whisper
  transcription and OCR-based image text extraction.

## [0.2.0] - 2026-03-18

FastAPI integration.

### Added
- FastAPI backend with query-handling endpoints, and an end-to-end system
  using Qdrant and Ollama.

## [0.1.0] - 2026-03-18

Initial setup.

### Added
- Project structure, dependency management, and semantic versioning
  scaffolding.
