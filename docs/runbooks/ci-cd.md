# CI/CD Runbook — Phase 29

How the pipeline works, how to seed/re-baseline the eval gate, how rollback
works, and exactly what's left to provision before `cd.yml` / Tier-2 can run
end-to-end.

## `tests/integration/` — was largely dead code, now resolved

Discovered by actually running `pytest tests/ -m "guardrails or auth"` for
real: 27 of `tests/integration/`'s 42 files failed to even import. Some
referenced an entirely different, pre-refactor package (`src.rag_system.*` —
the project's original architecture, predating the per-modality rebuild),
others called APIs that were renamed or removed (`app.retrieval.retriever`,
`app.embeddings.text_embedder`, `AgentController.decide()`, `ingest()`'s
signature changed). One file (`test_agent_brain.py`) had live, unguarded
code at module level — `agent = AgentController(); print(agent.decide(...))`
— that fired the instant the file was *imported*, not run as a test.

pytest collects every file under `testpaths` before applying any `-m`
filter, so **any bare `pytest tests/ -m <marker>` invocation walked into this
directory regardless of which marker was asked for**, and aborted with
"Interrupted: N errors during collection" — **zero tests ran**. This
presented as a silent hang (climbing CPU, no output) rather than a fast,
clear failure, and cost real debugging time before the actual cause was
found (it was mistaken for memory pressure at first, which was a red
herring — a clean, isolated re-run was *slower*, not faster, which is what
actually pointed at collection rather than resource contention).

**Scoping fix (still in place, defense in depth)**: `ci.yml`, `eval-gate.yml`,
and every `Makefile` test target scope pytest to a specific subdirectory
(`tests/unit/`, `tests/auth/`, `tests/guardrails/`) or explicitly exclude
the one remaining problem file — never bare `pytest tests/ -m <marker>`.
`CLAUDE.md`'s documented commands match.

**Resolved — the 27 broken files themselves**: deleted, not just excluded.
Running the survivors for real surfaced two more with the same root problem
(`test_qdrant.py` — calls a `create_collection` method that no longer
exists, has zero assertions; `test_text_chunking.py` — constructs an
`IngestedDocument` as chunker *input*, when it's actually the chunker's
*output* type, a fundamentally backwards premise relative to the current
API) — both deleted as superseded by `tests/unit/`'s equivalent, correctly-
mocked coverage. Three genuinely current files (`test_gap1_multidoc.py`,
`test_gap2_largedoc.py`, `test_gap3_multiturn.py` — real, well-written
multi-doc/large-doc/multi-turn-memory integration tests using the current
`query_pipeline` API) were missing the same `skipif(not llama-server-up)`
guard `test_llm_server_smoke.py` already had, so they hung instead of
skipping when no server was running; centralized into a new
`tests/integration/conftest.py::requires_llama_server` and applied to all
three. **Final state**: 42 → 13 files. `pytest tests/integration/
--ignore=.../test_document_pipeline.py` → 0 failures, 30 correctly-skipped
(no server), 12 real files pass. `test_document_pipeline.py` stays excluded
on its own — `import magic` *hangs* (not a clean failure) on this Windows
dev machine because `libmagic` isn't installed; a real Windows-only
environment gap, not a code bug — Linux CI/EC2 ship `libmagic` system-wide
and are not expected to hit this.

**Separately, root-causing a related hang** (`TestClient(app)`'s teardown
blocking indefinitely in `tests/auth/test_admin.py`) found that
`WARMUP_AT_STARTUP` defaults to `True` with no test-mode override anywhere,
so *any* `TestClient(app)` instantiation fires real GPU model preloading on
an executor thread that — per its own docstring — "never times out" and
cannot be cancelled mid-flight. Fixed the same way, at the root: `tests/
conftest.py` now sets `WARMUP_AT_STARTUP=false` as a test-session default
(`os.environ.setdefault`, so a developer who wants to test real warmup
behavior can still override it). This is very likely also why earlier
guardrails/auth runs this session ran far longer than the numbers justified
— not memory pressure as first suspected, but real background model loading
firing on every `TestClient` instantiation across the suite.

## Pipeline overview

```
PR opened/updated
  └─ ci.yml            lint + mypy + pytest -m unit         (hosted, ~minutes)
  └─ eval-gate.yml      tier1-retrieval                       (hosted, no LLM)
       └─ merge blocked if either fails (set both as required status checks
          in GitHub branch protection — not yet configured, do this once
          the first green run exists)
       └─ SKIPPED ENTIRELY if the PR only touches **.md / LICENSE
          (paths-ignore) — see the gotcha below before turning on
          required status checks

Tag pushed (vX.Y.Z)
  └─ cd.yml
       ├─ wait-for-ci-green   polls the tagged commit's CI + Security runs
       │                      until both conclude; fails fast if either isn't
       │                      a success — the literal "check all the things
       │                      to get green" step before anything is built
       ├─ build-push           compiles CUDA image once, pushes to GHCR
       ├─ deploy-staging        wakes the STAGING box (idempotent), SSM-
       │                        deploys, health-checks, auto-rolls-back on a
       │                        health-check failure — private box, no public
       │                        port, no Caddy (SSM is the only way in)
       ├─ tier2-staging-gate    calls tier2-eval.yml against STAGING,
       │                        BLOCKING. Alerts (Slack) + rolls staging back
       │                        to its own previous image on a retrieval-
       │                        section failure. Production is not touched.
       └─ promote-production    ONLY runs if tier2-staging-gate passed
                                 (skipped otherwise, by `needs:` — no image
                                 is ever deployed to prod without first
                                 clearing the staging gate). Deploys the SAME
                                 image tag already validated on staging.

Nightly 03:00 UTC
  └─ eval-gate.yml      tier2-nightly → tier2-eval.yml against PRODUCTION,
                         informational only (self-hosted GPU box, until
                         baseline v4 — see below). This does NOT gate any
                         deploy; it exists to catch drift on the already-live
                         image between releases.

On demand (a human clicks Run workflow)
  └─ quality-report.yml  wakes STAGING, runs Ragas + DeepEval in-container on
                         the staging GPU box, regenerates badges on a hosted
                         runner, stops staging. Gates nothing, blocks nothing,
                         touches production never. See below.
```

## Staging gate (champion/successor promotion)

Full design: `cd.yml`'s header comment and `.github/workflows/tier2-eval.yml`.
Short version — a tag no longer reaches production directly. It first deploys
to a second, private EC2 box (`magik-staging`), runs the full Tier-2 suite
against it, and only promotes the exact same image to production if that
suite passes with no retrieval-section regression. A failure alerts and rolls
staging back to its own previous image; production is left completely
untouched (the `promote-production` job is skipped, not run, whenever
`tier2-staging-gate` fails — there is nothing to roll back in prod because
nothing was deployed there).

`tier2-eval.yml` is a `workflow_call` reusable workflow — the exact same
eval/rollback logic runs for both the blocking staging gate and the nightly
informational production check, parameterized by container name and runner
label, so there is one copy of the fragile JWT-mint/BM25-preflight/rollback
logic instead of two that could drift apart.

**PREREQUISITES NOT YET PROVISIONED** (this pipeline is correct but cannot
succeed end-to-end until these exist — see `cd.yml`'s own header comment for
the up-to-date list):

1. A second EC2 instance for staging, tagged `Name=magik-staging` (or set repo
   variable `EC2_STAGING_INSTANCE_TAG` to whatever it's tagged).
2. Its instance profile — reuse
   `deploy/aws/iam/ec2-instance-profile-permissions.json` as-is (nothing in it
   is prod-specific); attach it to the staging box's own instance profile.
3. Its security group with **no inbound rule for 80/443/8000** — the second,
   independent layer enforcing "no HTTP" alongside `deploy-staging`'s own
   `-p 127.0.0.1:8000:8000` binding.
4. `AWS_DEPLOY_ROLE_ARN`'s own IAM policy (not captured in this repo; managed
   directly in AWS) extended with `ec2:StartInstances`/`DescribeInstances` +
   `ssm:SendCommand` scoped to the staging instance too, alongside prod's.
5. `deploy/aws/iam/github-oidc-trust-policy.json`'s new
   `environment:staging` sub pattern applied to the real IAM role
   (`aws iam update-assume-role-policy` or equivalent) — this repo file only
   documents the intended policy.
6. A second self-hosted GitHub Actions runner registered **on the staging
   box**, labelled `staging-gpu` (distinct from prod's `gpu` label). Until
   `gh runner list` shows it online, set repo variable
   `SELF_HOSTED_STAGING_GPU_RUNNER=true` — before that, `tier2-staging-gate`
   skips cleanly rather than queueing forever, but that also means **nothing
   gates promotion**, so this variable should only ever be "false" during
   initial bring-up, never in steady state.
7. On the staging box itself: `/opt/magik-staging/{.env,.hf_cache,data,logs}`
   present (same shape as prod's `/opt/magik/...`), with the full ~25GB model
   cache paged into `.hf_cache` — staging is a second real GPU box, not a
   lightweight clone, since a genuine champion/successor test has to run on
   hardware that isn't already serving prod traffic.

**Manual end-to-end validation**, once the above exists: push a real tag and
confirm, in order — `wait-for-ci-green` passes, `deploy-staging` brings up
`magik-staging-current` on the staging box only (`curl` to its public IP on
port 8000 must fail/refuse — proving "no HTTP" actually holds), `tier2-
staging-gate` runs against staging and not production, and `promote-
production` only fires after that gate is green. Then force a failure (e.g.
temporarily lower `thresholds.yaml`'s retrieval floor) and confirm: Slack
alert fires, `magik-staging-current` rolls back to `magik-staging-previous`,
and `magik-current`/production is untouched throughout.

## Manual quality report (`quality-report.yml`)

The portfolio-facing Ragas + DeepEval numbers. **Manual only, staging only.**

```
gh workflow run quality-report.yml -f tools=both -f ragas_limit=30 -f deepeval_limit=10
```

- **When to run it:** after a release you want a number for, or after changing
  retrieval/generation in a way you expect to move faithfulness. There is no
  schedule and no automatic trigger, by design.
- **What it costs:** staging wakes for the duration and is stopped again by the
  workflow's own `stop-staging` job — roughly 25–40 minutes at ~$1.86/hr, so
  well under a dollar per run. `keep_staging_running: true` suppresses the stop
  for debugging; nothing else will turn staging off if you do that, because the
  idle-stop Lambda watches `magik-prod` only.
- **What a red run means:** nothing is broken in production and no deploy is
  affected. Open the failing step; if it ends in `exit code 137` the eval was
  OOM-killed and the row limit needs lowering, not retrying.
- **What the badge says:** `(staging)`, not `(live)`. It grades the promoted
  image on a box cloned from production, and the badge should not claim more
  than that.

### Why it is not part of cd.yml any more

Until v1.0.0 this ran automatically after every promotion, against the live
production container, on a runner registered on the production box. In
v1.0.0-rc5 (CD run 33149188726) the DeepEval step recorded no conclusion and no
completion time, and the five `if: always()` steps after it never ran — the
runner process itself died. DeepEval loads Qwen2.5-7B as a second model stack
beside the live server's resident models; the OOM-killer fired and took
`magik-prod-runner` offline with it.

The lesson generalises past this one job: `continue-on-error` is evaluated *by
the runner*, so it cannot protect a job whose failure mode is the runner dying.
A report that gates nothing has no business running unattended on the box that
serves traffic. Both problems are fixed by location and trigger, not by more
guards.

The eval still runs *inside* the deployed container for the three reasons
tier2-eval.yml documents — local judge, no Qdrant credentials in a bare shell,
no BM25 index after `git clean -ffdx`. Only the box and the trigger changed.

## Why two tiers (not one eval gate)

The full generation suite needs the resident model stack CLAUDE.md documents
at ~17.7GB+, plus a live running API (generation calls route through HTTP to
`/rag/query`, not in-process — see `app/eval/run.py`'s
`EVAL_SKIP_LLM_WARMUP` comment). That doesn't fit:
- a 16GB laptop with no CUDA GPU,
- a standard GitHub-hosted runner (~7GB RAM).

It only fits the real GPU box. So Tier 2 runs there, post-deploy, not in any
CI container. Tier 1 (retrieval-only: BM25 + Qdrant + BGE embedder/reranker,
~3GB, no LLM) is cheap enough to block every PR on a hosted runner.

## The eval-corpus gap (read this before Tier 1 fails mysteriously)

The gold QA set (`app/eval/datasets/gold/*.jsonl`) is versioned in git. The
**ingested corpus it references is not** — `data/` is gitignored, and BM25
indexes are per-user local pickle files (`app/bm25/base_bm25.py`), not cloud
state. Qdrant Cloud persists on its own (reachable from CI via
`QDRANT_URL`/`QDRANT_API_KEY` secrets), but the BM25 half only exists on
whatever machine last ran ingestion.

`eval-gate.yml`'s `tier1-retrieval` job restores a GitHub Actions cache of
the BM25 index (keyed on a hash of the gold-set files) instead of rebuilding
it every PR. **That cache has to be seeded once, and re-seeded whenever the
gold set or corpus changes**, by manually running:

```
Actions → Eval Gate → Run workflow → seed: true
```

This runs `python -m app.eval.datasets.build_gold_set --ingest` on the
self-hosted GPU runner (the only place with `data/raw/finance` present),
then publishes the resulting BM25 index to the cache. If Tier 1 fails with
`no_gold_data` or a retrieval error on a PR that clearly didn't touch
retrieval code, this is almost certainly why — re-run the seed job.

## Re-baselining v4 and enabling the gate

`app/eval/thresholds.yaml` currently has `gate_enabled: false`. The v3
baseline was retired because the judge model changed (GGUF Mistral →
Prometheus-2-7B) and the gold set grew 14 → 164 rows — the old numbers
aren't comparable. The file's own header documents the procedure:

1. `EVAL_USER_ID=<owner> python -m app.eval.run --suite full` on the GPU box
   (ungated).
2. Record the Prometheus-2-7B / v1.0.0 numbers as baseline v4.
3. Set each metric's `min`/`max` at `baseline_v4 * 0.9` (safety-critical
   metrics) or `* 0.95` (the rest).
4. Flip `gate_enabled: true`.

Do this only after a few stable nightly Tier-2 runs — a gate wired to a
single noisy measurement will produce false-positive blocks. This step is
**eval-engineer's domain** (`app/eval/`), not devops — coordinate there
before flipping the flag.

## Rollback

Two independent rollback paths exist, at two different layers:

1. **Infra-level, both boxes** — `cd.yml`'s `deploy-staging` and
   `promote-production` steps each keep the previous container renamed
   (`magik-staging-previous` / `magik-previous`), not removed. If the new
   container fails its health check (`/health`, 80×15s retries), the SSM
   script automatically removes it and restores/restarts the previous one —
   the job fails loudly (so you know a rollback happened) but the service
   stays up on the last-known-good image.
2. **Quality-level, RAG-specific** — `tier2-eval.yml`'s auto-rollback step,
   scoped to a retrieval-section breach only (see "Staging gate" above). On
   staging this is what decides whether production ever sees the image at
   all; on the nightly production run it's the same mechanism reacting to
   live drift.

Manual rollback (if a bad deploy passes health checks but is wrong in some
other way) — same commands on either box, just the container names differ:
```bash
# production
docker rm -f magik-current
docker rename magik-previous magik-current
docker start magik-current

# staging
docker rm -f magik-staging-current
docker rename magik-staging-previous magik-staging-current
docker start magik-staging-current
```

## Doc-only PRs and required status checks — resolved by removing `paths-ignore`

`ci.yml`, `eval-gate.yml`, and `security.yml` originally had a
`paths-ignore: ["**.md", "LICENSE"]` filter on their `pull_request` trigger,
on the reasoning that a doc-only change can't affect lint, types, tests, or
retrieval quality.

**Why that had to change.** GitHub's required-status-checks feature does not
auto-pass a check that never ran — and a workflow filtered out by
`paths-ignore` reports *no status at all*, rather than a skipped/neutral one.
So the moment any of these jobs is marked "required", a PR touching only
`**.md` sits permanently on *"Expected — waiting for status to be reported"*:
not merged, not blocked with a reason, just unmergeable forever.

**Why not the documented companion-workflow workaround.** GitHub's suggested
fix is a second trivial workflow declaring *identical job names*, triggered on
the inverse `paths:` condition, that immediately reports success. It has a
real flaw: `paths-ignore` skips only when *every* changed file matches, while
`paths` fires when *any* does — so a PR touching both a doc and code triggers
**both** workflows, both report the same check name, and whichever finishes
last wins. That is an always-passes job racing a genuine failure for control
of a required check. Unacceptable for a gate whose whole job is to block bad
merges.

**What was actually done.** `paths-ignore` was removed from the `pull_request`
trigger of all three workflows. This repo is **public**, so GitHub Actions
minutes are unlimited and free — the optimization was saving nothing
measurable while creating a real trap and pushing us toward a racy workaround.
Running the full suite on a doc-only PR costs a few free minutes and removes
the failure mode by construction: every PR always gets a real status from
every required check, with nothing to keep in sync and no race.

`push: branches: [main]` **keeps** its `paths-ignore` — post-merge
re-verification carries no branch-protection semantics, so skipping a doc-only
commit there is pure upside with no trap attached.

## Pipeline monitoring vs. Phase 31 (runtime) monitoring

"Monitoring" means two different things across this plan, and Phase 29 only
owns one of them:

- **Pipeline/deploy observability (Phase 29, built)** — is the build green,
  did the eval gate pass, did the deploy succeed or roll back. This is what
  `ci.yml`/`eval-gate.yml`/`cd.yml` now surface directly:
  - README status badges (top of `README.md`) for at-a-glance CI/Eval
    Gate/CD state.
  - `$GITHUB_STEP_SUMMARY` on every eval-gate.yml and cd.yml run — the
    `rag_report.md` content (Tier 1/Tier 2) and deploy result are visible on
    the Actions run page itself, no external tool needed.
  - MLflow experiment tracking (`app/eval/tracking/mlflow_logger.py`,
    already wired into `app/eval/run.py`) — every eval run is a logged,
    comparable experiment. This is the "Experiment tracking" row from the
    plan's own Phase 29 tools table.
  - Optional Slack (or Discord, swap `text` for `content` in the payload)
    webhook alert on **Tier-2 failure** and **deploy failure/rollback** —
    the two events where nobody is watching the Actions tab live (a 3am
    scheduled run, a tag push with no one at the keyboard). No-op today:
    set repo secret `SLACK_WEBHOOK_URL` to activate it, nothing else to
    change.
  - Docker `HEALTHCHECK` + `cd.yml`'s post-deploy polling loop — deploy-time
    health, not ongoing runtime health.

- **Runtime/application observability (Phase 31, not built, correctly out of
  scope here)** — Prometheus scraping `/metrics` continuously, Grafana
  dashboards (latency, error rate, circuit-breaker state, RAG-quality trend
  over time), an OTel collector routing spans to Tempo/Jaeger, alert rules
  with error budgets, a scheduled online-eval job sampling live traffic.
  None of this exists yet. It doesn't belong in Phase 29: it's about
  watching the *already-deployed, already-running* system's behavior over
  time, not about gating what gets deployed. Build it once there's a
  deployed system worth watching (i.e., after Phase 30).

## Local dev

```
make install-dev     # pip install -e ".[dev]" + pre-commit install
make lint             # ruff + black --check + isort --check
make typecheck        # mypy app/
make test-unit         # pytest -m unit, mocked models, no external services
make compose-up         # local CPU-only stack: API + Qdrant + Redis + Mongo
```

`docker compose up` uses the Dockerfile's `dev-runtime` target — `python:3.12-slim`,
no CUDA base image, no from-source compile, fast to build. It is **not** a
substitute for the eval gate: it doesn't seed the eval corpus (see above),
and `--suite full`/`eval-full` needs a GPU box regardless of which Docker
target you're on, per `make eval-full`'s own warning banner.

## Security scanning (`security.yml`) — what was actually found and why each landed where it did

Four checks, each triaged against a REAL run (2026-07-25), not written blind —
see `security.yml`'s own inline comments for the authoritative, up-to-date
version of each rationale; this is the narrative summary.

- **detect-secrets** — CI-enforced now (was local-only before, a real gap: a
  contributor who skipped or bypassed pre-commit had nothing else catching a
  leaked credential). Clean against the current baseline.
- **Bandit SAST** — blocking on HIGH severity only (0 remain after fixing
  4 real MD5-for-content-dedup findings with `usedforsecurity=False`, and
  suppressing 1 confirmed false positive — `app/guardrails/input_guard.py`'s
  literal RTL-override characters ARE the injection-defense signature the
  function exists to strip, not a vulnerability, `# nosec B613` + rationale).
  297 Low / ~25 Medium findings are visible (full report uploaded every run)
  but non-blocking — dominated by `B110`/`B101`/`B112` (try/except/pass,
  assert, try/except/continue — ~250 of the total), which is this codebase's
  existing, deliberate non-fatal-fallback style throughout, not a pattern to
  mass-suppress without reviewing each site's actual blast radius individually.
  That's real future code-review-pass scope, not something to rush.
  `B615` (huggingface_unsafe_download, 25 findings) is a Bandit blind spot,
  not a real gap — `revision` IS threaded through every `from_pretrained()`
  call via `_dispatch_download()`'s `**kw` dict, Bandit's static analysis just
  can't trace it; the only real unpinned window is a model's first-ever
  download, which is inherent to this project's TOFU checksum design, not an
  oversight (see `app/bin/models/download_all_models.py`'s own comment).
- **pip-audit** — informational (`continue-on-error: true`), not because the
  findings aren't real (several are genuinely serious — SSRF/XXE/RCE-class)
  but because every one traced back to a specific, bounded reason it isn't
  live exposure here: the whole langchain/langsmith CVE chain comes from
  `ragas==0.1.21` (eval-harness-only, runs against a curated gold set, never
  live/untrusted input — fixing it means a real ragas major-version bump and
  an eval-harness compatibility pass, not a quick pin bump); `ecdsa`'s Minerva
  timing-attack CVE (no upstream fix planned, ever) requires ECDSA signing,
  and this app's JWT defaults to HS256 (HMAC) — confirmed unreachable, not
  just assumed; `keras`/`diskcache` are both pinned by their parent package
  (tensorflow / llama-cpp-python respectively). Dependabot
  (`.github/dependabot.yml`) is the standing mechanism that drives fixes as
  compatible patched versions land upstream.
- **License scan** — blocks only on AGPL/SSPL (network-copyleft — would force
  source disclosure of this whole hosted service; zero ambiguity, zero found).
  Found `mutagen` (GPL-2.0-or-later) and `CairoSVG` (LGPL-3.0-or-later)
  installed but NOT actually declared in `requirements.txt` — both are used
  behind `try/except ImportError` with graceful degradation already
  (`audio_ingest.py`'s duration fallback, `image_chunker.py`/`image_ingest.py`'s
  SVG→PNG conversion), so moved to `requirements.txt`'s new commented-out
  "OPTIONAL — license-sensitive" section: a deployer with GPL/LGPL concerns
  gets full graceful degradation for free by leaving them uninstalled; one who
  wants the extra fidelity opts in explicitly by uncommenting.

**mypy** (not part of `security.yml`, but the other quality-debt gate in
`ci.yml`): started this pass at 311 errors / 65 files, real work brought it to
204 / 62 — not by mass-suppressing, but by fixing what a triage surfaced as
real bugs first: a config value (`MATRYOSHKA_SHORT_DIM`) referenced with no
setting ever defined (would crash on first real call), a crashing import in
`txt_bm25.py`'s sub-index rebuild (`user_data_dir` doesn't exist — real
function, reachable on every text ingestion with speaker-tagged content), a
renamed Qdrant client attribute (`vectors_count` → `indexed_vectors_count`),
a missing `QdrantVectorStore.delete_by_ids` method (the KB-delete-by-file-hash
endpoint always 500'd), two wrong method names in the GDPR-adjacent
all-sessions-delete path (`infra.get_redis_memory()` / `RedisMemory.
delete_all_user_sessions()` neither exist — silently swallowed by a broad
`except`, so Redis was never actually being purged on session deletion), and
two dead, fundamentally-broken factory functions in `app/ingestion/schema.py`
(deleted, not patched — zero callers, and `ProcessingResult` doesn't even
accept the kwargs they were passing). The remaining 204 are mechanical/
scattered (PIL type stubs in `image_chunker.py`, a handful of structural
issues in `video_ingest.py`/`reasoning_engine.py`) — real, but lower-value
per-error than what's already fixed; left for a dedicated follow-up pass
rather than exhaustively grinding through a 65-file surface in one sitting.

## First real CI runs — what actually broke (2026-07-26)

Everything above was verified locally before it ever ran on a GitHub runner. The
first real runs confirmed `ci.yml` and `security.yml` pass on `ubuntu-latest`,
and surfaced two genuine faults in `eval-gate.yml` that local testing
structurally could not have caught. Both are fixed.

**1. Tier-1 failed on every PR — the BM25 cache could never exist.**
`tier1-retrieval` restored the BM25 index from a GitHub Actions cache that
`seed-eval-fixtures` was supposed to publish. But that job is
`runs-on: [self-hosted, gpu]` *and* `workflow_dispatch`-only, so on a repo with
no self-hosted runner registered the cache was never populated — not once.
Every PR then scored recall against an empty index and reported a *quality
regression*, when the truth was "there is no corpus here." Run 30185188038 shows
the chain verbatim: `Cache not found` → `bm25_no_saved_index` →
`bm25_empty_index_returning_empty_list`.

Fixed structurally, not by seeding once: Tier-1 now **rebuilds the index from
Qdrant on a cache miss** (`python -m app.retrieval.bm25_retriever --user_id …`)
and saves it back. The chunk text already lives in Qdrant payloads, so the
rebuild needs no GPU, no models, and no re-embedding — which removes the
self-hosted dependency from the PR gate entirely.

**2. The gate had no credentials at all, and said the wrong thing about it.**
No repository secrets or variables were configured, so `QDRANT_URL` /
`QDRANT_API_KEY` were empty and `EVAL_USER_ID` silently fell back to
`eval_default`. Tier-1 therefore had no data source *and* was pointed at the
wrong tenant, but still presented as "retrieval regressed." A gate that can't
distinguish **broken** from **regressed** teaches people to ignore it. There is
now a preflight step that fails in ~5 seconds naming exactly what is missing,
instead of installing torch for 20 minutes first, plus a warning when
`EVAL_USER_ID` is still the fallback.

**Still required from the repo owner** (credentials): add `QDRANT_URL` and
`QDRANT_API_KEY` under *Settings → Secrets and variables → Actions → Secrets*.
`EVAL_USER_ID` is already set under *Variables*. Until the secrets exist,
Tier-1 fails fast and loudly, which is the intended behaviour.

**⚠️ Add the same two values a SECOND time, under *Dependabot* secrets.**
This is not redundancy — it is the reason every early failure of this gate
happened. GitHub deliberately denies Dependabot-triggered workflow runs access
to repository *Actions* secrets; they read from a separate *Dependabot* secret
store instead. Every failing Eval Gate run in the first batch was on a
`dependabot/*` branch, and a generic "add your secrets" message would have sent
you to the Actions page, where the values already were. The preflight step now
detects `github.actor == 'dependabot[bot]'` and prints the Dependabot-specific
instruction instead — verified by executing the step's own shell across all
three cases (dependabot+missing → dependabot message; owner+missing → generic
message; owner+present → exit 0).

Leaving Dependabot secrets unset is not a safe shortcut once Tier-1 is a
*required* check: every dependency PR would be permanently red and unmergeable.
And skipping the gate for Dependabot would be worse — a bump to `rank-bm25`,
`sentence-transformers`, or `qdrant-client` is precisely when a retrieval
regression is most likely.

**3. The nightly Tier-2 schedule was queuing forever.** `tier2-nightly` (via
`tier2-eval.yml`) also targets `[self-hosted, gpu]`. A scheduled run against a
nonexistent runner does not fail — it sits `queued` indefinitely (one had
accrued 3h+ before being cancelled). The `schedule:` trigger is commented out
until a GPU runner is registered; re-enabling it is one line, and belongs with
the Phase 30 bring-up.

## Known gaps (flagged, not silently hidden)

- **`start_server.py`'s `atexit`-registered llama-server cleanup is not
  guaranteed to fire on SIGTERM** (`docker stop`'s default signal) — Python
  doesn't install a default SIGTERM handler, so an abrupt container stop can
  leave the llama-server child process orphaned inside the container until
  the container itself is removed. Not patched here — `start_server.py` is
  owned by the app, not this deploy layer. Worth a follow-up fix (a signal
  handler that calls `sys.exit()` so `atexit` fires, or forwards SIGTERM to
  the child explicitly) — flag for code-review/architect.
- **`.secrets.baseline` — resolved.** Started as a hand-written empty
  baseline; regenerated for real via `detect-secrets scan`, reviewed (6
  findings, all confirmed false positives: a config validation check, gold-set
  SHA-256 hashes, test-fixture dummy passwords), and is now CI-enforced (see
  `security.yml`'s `secrets-scan` job below) — not just a local pre-commit
  hook.
- **Self-hosted runner + public repo**: the `[self-hosted, *]` jobs (
  `eval-gate.yml`'s `seed-eval-fixtures`, and `tier2-eval.yml`'s job — called
  from `eval-gate.yml`'s `tier2-nightly` via `schedule`/`workflow_dispatch`
  and from `cd.yml`'s `tier2-staging-gate` via `needs:` chaining off a
  `push: tags:` trigger) are only ever reachable through triggers a fork
  cannot forge — deliberately never `pull_request`. Do not add a
  `pull_request` trigger to any self-hosted job, or to anything that calls
  one; a fork PR would then be able to execute arbitrary code on your GPU
  box(es).

## Phase 30 handoff — provisioning status

| # | Item | Status |
|---|---|---|
| 1 | EC2 `g6e.xlarge` launched, Elastic IP attached | done |
| 2 | IAM role for GitHub OIDC (`vars.AWS_DEPLOY_ROLE_ARN`) — least-privilege, scoped to the one instance. **Trust policy must allow BOTH subject forms**: a job with `environment:` gets `repo:<o>/<r>:environment:<name>`, not `ref:refs/tags/*` | done |
| 3 | Instance tagged `Name=magik-prod` (`vars.EC2_INSTANCE_TAG`) | done |
| 4 | SSM Agent + instance profile on the box | done |
| 5 | GHCR read PAT at SSM Parameter `/magik/ghcr_pat` (SecureString) | done |
| 6 | Self-hosted runner registered on the box, labeled `self-hosted, gpu`, installed as a systemd service so it survives stop/start | done |
| 7 | Scale-to-zero wake gateway + idle-stop Lambdas (`deploy/aws/`) | done |
| 8 | **Runner busy-check PAT at SSM `/magik/github_actions_pat`** (fine-grained, Administration: read-only). Idle-stop reads it to avoid stopping the box mid-eval | **required before #10** |
| 9 | **`deploy_lambdas.sh` re-run** after #8, to ship the idle-stop build that actually performs the runner busy-check | **required before #10** |
| 10 | **Repository variable `SELF_HOSTED_GPU_RUNNER=true`** — un-gates `eval-gate.yml`'s nightly `tier2-nightly` job. Until set, that informational run against production never fires (job skips grey) | **required for nightly Tier-2** |

Order matters for 8→9→10. Setting #10 first means a Tier-2 run (up to 2h) is
exposed to an idle-stop build that cannot see a busy runner; skipping #8 before
#9 means idle-stop fails safe and **never stops the box at all** — no error,
just a GPU billing continuously.

**Staging box provisioning is tracked separately** — see "Staging gate"
above and `cd.yml`'s own header comment for the up-to-date checklist. It is
NOT covered by this table (this table is prod-only, Phase 30 scope); the
staging box, its runner (`staging-gpu`), and `SELF_HOSTED_STAGING_GPU_RUNNER`
are a distinct, separately-tracked prerequisite set — and unlike the
production nightly run above, `SELF_HOSTED_STAGING_GPU_RUNNER` gates the
BLOCKING pre-promotion gate, not an informational one, so treat it as
required before relying on this pipeline at all, not optional polish.

### Tier-2 executes inside the deployed container

`tier2-eval.yml` (called by both `cd.yml`'s `tier2-staging-gate` and
`eval-gate.yml`'s `tier2-nightly`) and `seed-eval-fixtures` run their Python
via `docker exec <container> …`, not directly on the runner. This is deliberate.
Running on the runner could never have worked: no dependency install (unlike
tier1's `setup-python` + `pip install`), no `QDRANT_URL`/`QDRANT_API_KEY`
(repo secrets are not auto-injected into a job), no BM25 index
(`app/utils/paths.py`'s `DATA_ROOT` is the *relative* `data/users`, and
`actions/checkout`'s `git clean -ffdx` wipes gitignored `data/`), and no
`EVAL_ACCESS_TOKEN` (`/rag/query` requires `get_current_user`).

The container has all four: deps baked in, credentials via
`--env-file /opt/magik/.env`, the real corpus at `/app/data` (mounted from
`/opt/magik/data`), and the live server on `127.0.0.1:8000`. It is also the
more correct target — post-deploy eval should measure the artifact serving
traffic, not a checkout beside it. The eval access token is minted inside the
container from its own `JWT_SECRET_KEY` and consumed in the same step, so no
credential is stored anywhere.

### The judge model has two independent safety nets

`app/bin/models/download_all_models.py`'s default run (what `start_server.py`
calls on every boot) originally excluded `prometheus_judge`
(`"startup": False`) on the theory that an eval-only model shouldn't cost
anything on a normal boot — but the actual download step never loads the
model (no VRAM, just a `hf_hub_download` + hardlink to disk), so the only
real cost was a one-time ~4.4GB fetch on first boot. Confirmed live
(2026-07-31) that defaulting it off meant a full Tier-2 run silently graded
with a weaker lexical judge (`[eval] Prometheus GGUF not found — falling
back to lexical judge`, easy to miss in a 2-hour log) because nobody had run
the manual fetch step first. Fixed at the root — `prometheus_judge` is now
in the default download run, same as every other resident model, so a fresh
box already has it before Tier-2 ever starts.

As a second, independent safety net (not a substitute — belt and suspenders),
`app/eval/judges/prometheus_judge.py::ensure_available()` auto-downloads the
GGUF inline the first time Tier-2 actually calls for it, if it's somehow
still missing. Manual fetch is still available if needed:

```bash
docker exec magik-current python -m app.bin.models.download_all_models --only prometheus_judge
```

### Tier-2 result gates production promotion (staging gate)

Superseded design note, kept for history: earlier revisions of this pipeline
had `cd.yml` deploy straight to production and *dispatch* Tier-2
asynchronously afterward — the deploy went green regardless of the eval
outcome, and only the box's own health check could trigger a rollback.

That is no longer how this works. `cd.yml`'s `tier2-staging-gate` job runs
Tier-2 **synchronously, against staging, before production is ever touched**
— see "Staging gate" above. `promote-production` is a real dependency of
that job succeeding (`needs: [build-push, tier2-staging-gate]`), so a Tier-2
failure (including the same retrieval-section auto-rollback described
earlier, now scoped to `magik-staging-*`) means production is never deployed
at all, not deployed-then-rolled-back. The "needs the non-retrieval sections
re-baselined first" caveat still applies to *which* Tier-2 sections are
gating (only `retrieval` is `gate_enabled: true` today) — that part of the
design is unchanged, just relocated earlier in the pipeline.
