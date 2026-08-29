# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0-rc7] - 2026-08-29

rc6 built and pushed cleanly but could never be deployed: `deploy-staging`
failed on `docker pull` four times across four hours, each attempt taking the
staging box down with it via `stop-staging`. The cause was not in this
repository. No application behaviour changes.

### Fixed
- **GHCR refused to serve the image's largest layer, so no deploy could
  start.** `COPY --from=cuda-builder /opt/venv /opt/venv` shipped the entire
  virtualenv — torch plus the pip-installed CUDA libraries — as a single
  7.63GB blob, and ghcr.io answers requests for it with
  `429 TOOMANYREQUESTS {"message":"retry-after: 476ms"}`. Measured against the
  registry directly, every one of the image's other 17 layers served normally,
  including a 2.07GB one; `HEAD` on the failing blob returns 200, so the blob
  exists and is intact — the registry simply will not send the body.
  It is size-related rather than specific to one build: v1.0.0-rc5's
  equivalent 7.63GB layer 429s identically today, so rc5 could not be
  redeployed either. It is also not a transient window — one attempt came
  after 2h13m of complete quiet and failed in 35 seconds, and eight
  consecutive direct requests to the blob returned 429 every time, which is
  why no retry was added to the deploy: retrying demonstrably does not get
  past it.
  The venv is now split across six layers in the `runtime` stage — `torch`,
  `triton`, `nvidia/cudnn`, `nvidia/cublas`, the rest of `nvidia`, and the
  remainder of the venv — each landing at its original path so nothing
  downstream can observe the difference. `bin/` is deliberately left in the
  remainder: it holds the `python3.12` symlink into `/usr/bin` that the
  runtime stage's own interpreter install resolves, which has broken once
  before. The split step prints `du -sh` of each bucket into the build log so
  the achieved sizes are visible without reproducing the investigation.
  `dev-runtime` copies from `base-deps` and is unchanged.

### Known limitations
- The split targets the four subtrees that dominate a torch-CUDA venv. If a
  future dependency bump pushes one bucket back over the limit the same
  failure returns; the `du -sh` output in the build log is the early warning.
  The durable fix is a registry in the same region as the deployment target
  (ECR), which removes both the throttle and the cross-internet pull.
- The quality-report fixes from rc6 still have not executed on a GPU box —
  rc6 never deployed. Verification remains a manual `quality-report.yml` run
  against staging once this release lands.

## [1.0.0-rc6] - 2026-08-29

rc5 shipped the `--limit 30` cap for the RAGAS report and it was not enough:
the DeepEval step that runs immediately after was OOM-killed again, and this
time the kernel took the self-hosted runner offline with it, so the job
recorded no conclusion, no completion time, and none of its five
`if: always()` steps ever ran. That failure mode is the subject of this
release. No application behaviour changes.

### Changed
- **The post-release quality report no longer runs on the production box, or
  automatically.** `cd.yml`'s `report-quality-metrics` job ran Ragas +
  DeepEval inside the live production container, on a runner registered on the
  production instance. It is deleted, and replaced by
  `.github/workflows/quality-report.yml`: a `workflow_dispatch`-only workflow
  that wakes the staging box, evaluates the deployed image in-container on the
  staging GPU, regenerates badges on a hosted runner, and stops staging again.
  Two facts drove this. `continue-on-error` is evaluated *by* the runner, so
  it cannot protect a job whose failure mode is the runner dying — every
  mitigation built for that failure died with it. And a report that gates
  nothing had no business OOM-pressuring the instance serving real traffic.
  The release pipeline now ends at `promote-production`.
- **`quality-live.yml` no longer offers `ragas-report` / `deepeval-report`.**
  Both ran on `ubuntu-latest`, and both Ragas and DeepEval load the Qwen2.5-7B
  judge locally — `qwen_judge._start_worker()` raises rather than downloading
  the GGUF, so Ragas silently degraded to its lexical fallback and DeepEval
  errored every row while the run reported success. A broken path that stays
  clickable gets clicked.
- `docs/` is now tracked in git. It sat under the same blanket
  "regenerable, keep it local" rule that had already hidden
  `scripts/generate_quality_badges.py` from CI, while `cd.yml`,
  `tier2-eval.yml` and `quality-report.yml` all point readers at
  `docs/runbooks/ci-cd.md` by path.

### Fixed
- **RAGAS rows failed to parse because the judge wrapper returned the wrong
  `LLMResult` shape.** `AnswerRelevancy._ascore` requests `n=strictness`
  generations and reads them out of `result.generations[0]`, so all n must
  share one inner list. `QwenRagasJudge` built one generation per *outer*
  entry, so Ragas only ever saw a single response and `strictness` was
  silently always 1 regardless of what it asked for.
- **`answer_relevancy` now runs at `strictness=1`, explicitly.** `_ascore`
  discards a row entirely if *any* of its generations fails to parse
  (`if any(answer is None ...): return np.nan`), so the default of 3 is a
  three-fold amplifier on per-row judge flakiness — and three times the judge
  calls on the process whose memory footprint caused the OOM. Correcting the
  shape bug above without this would have switched that cost and that risk on
  for the first time.
- **The judge was given two contradictory formatting orders in one turn.**
  Ragas' own prompt ends "return only a pure JSON string surrounded by triple
  backticks"; the injected system prompt said "No markdown." Fenced replies
  are now explicitly allowed — `_extract_json_from_text` unwraps a fence in
  its first branch, making that the cheapest reply to parse.
- **Ragas' retry was wasted on every failure.** When a reply fails to parse,
  `RagasoutputParser.aparse` re-prompts with `FIX_OUTPUT_FORMAT`, whose text
  embeds the entire original prompt. Every content branch of
  `_build_ragas_prompt` matched that embedded copy and told the judge to
  answer the original question again — reproducing the reply that had just
  failed and burning the single retry Ragas allows. The repair prompt is now
  detected first.
- **Long answers truncated mid-JSON.** The Ragas judge path capped generation
  at 1024 tokens, but `context_recall` returns one object *per sentence* of
  the answer, each carrying a statement echo and a free-text reason. Long
  finance answers exceeded the cap and were cut mid-object — unparseable, and
  deterministically so on exactly the longest rows. Raised to 2048.
- **The eval process held two model stacks at its peak.** It runs via
  `docker exec` as a second process with its own `model_loader` singleton, and
  the query phase loads BGE-large, SigLIP and the SigLIP text encoder — none
  of which are in `_EVICTABLE_MODELS`, being core query-path models, so
  nothing ever released them. They stayed resident while the ~4.7GB judge
  worker spawned on top. The query and judge phases never overlap, so both
  report modules now free that stack at the end of the query phase
  (`release_full_context_models()`).

### Added
- `tests/unit/eval/test_ragas_judge_contract.py` — pins the `LLMResult`
  generation shape (sync and async), the repair-prompt branch ordering, and
  the fence-compatible system prompt, and keeps Ragas' exact
  `FIX_OUTPUT_FORMAT` instruction string as a canary for library upgrades.

### Known limitations
- **None of the quality-report fixes above have executed on a GPU box.** They
  are derived from the `ragas-0.1.21` source and pinned by unit tests; the
  memory fix is reasoned from rc4's log, where the kill lands in the query
  phase before the judge starts, rather than from a memory profile. The
  verification is a manual `quality-report.yml` run against staging.

## [1.0.0-rc5] - 2026-08-28

v1.0.0-rc4 deployed to production cleanly and its monitoring fixes are
confirmed working live (the promote log shows the Grafana deployment
annotation posting for the first time, the ntfy webhook verified current, and
the throwaway secret files finally removed). What rc4 did *not* manage was to
prove its own quality-report fixes: the `report-quality-metrics` job was
OOM-killed before it reached any of the corrected code. This release fixes
that, and nothing else — no application behaviour changes.

### Fixed
- **The RAGAS report was OOM-killed on the box, taking the CI runner with it.**
  `cd.yml` runs `ragas_report` as a *second* Python process inside the live
  `magik-current` container, so it loads its own text embedder and SigLIP
  beside the copies the app already has resident — both of which
  `model_loader._EVICTABLE_MODELS` deliberately refuses to evict, being core
  query-path models — while every gold row drives a full RAG query through the
  app in that same container. Unbounded that is 98 rows and ~17 minutes
  (measured on rc3) of two model stacks coexisting on a 32GB host. The rc4 run
  did not survive it: the kernel OOM-killer took the eval at 8m17s
  (`exit code 137`, about half way through the query phase, CD run
  33134484078) and the cascade knocked the self-hosted runner offline,
  cancelling the job — so `continue-on-error` never got the chance to absorb
  it and no artifact was produced at all. The 137 also rendered as a green
  tick in the job list, visible only in the step log.

  `ragas_report.py` now takes `--limit` (deterministic, gold-set order,
  mirroring the `--limit` `deepeval_suite.py` has always had) and `cd.yml`
  passes `--limit 30`, cutting the window in which both stacks are resident
  from ~17 minutes to a few. A capped run stays honest: the report records
  `n_queries`, and `generate_quality_badges.py` refuses to publish a Ragas
  badge whose graded rows cover less than half of it.

  Three candidate causes were ruled out with evidence rather than assumed:
  the idle-stop Lambda (its own CloudWatch log shows the `_runner_busy()`
  guard firing correctly at 05:08 and the box still running at 05:23), a
  container memory limit (none is set — this is the host OOM killer), and
  rc4's own judge fixes (rc3 ran the identical query phase to completion in
  16m55s, and the kill landed before `compute_generation_metrics_ragas`
  executes). Model eviction was considered and rejected: the models actually
  loaded here are precisely the non-evictable ones, so it would free nothing.
- `cd.yml`'s post-report warning now names the `exit code 137` signature and
  says to lower the limit rather than retry unchanged, so the next person
  meets a diagnosis instead of a cancelled job.

### Known limitations
- **Mitigated, not eliminated.** `--limit 30` shortens the overlap; it does
  not remove the second model stack from the container. A real fix is a
  dedicated eval container with its own memory budget, or a larger host.
  Neither belongs in a release-eve change.
- Whether the rc4 Ragas/DeepEval judge fixes actually work is **still
  unproven** — no run has yet got far enough to produce a report from the real
  judge. The check is this release's own artifact: `quality-reports/ragas/*.json`
  must carry `judge=qwen2.5_7b` in its `notes`, not `lexical_fallback`.
- Everything else unchanged from [1.0.0-rc4] below.

## [1.0.0-rc4] - 2026-08-27

v1.0.0-rc3 promoted to production cleanly — the first successful promotion
since v0.31.0 — and in doing so produced the first real output the
`report-quality-metrics` job had ever generated. That output was wrong, in
two independent ways that had been invisible for as long as the job had been
failing earlier. Both public quality badges were reporting numbers that did
not measure what they claimed. This release fixes those, the reason nothing
caught them, and the two release/deploy bugs found in the same pass. No
retrieval, agent, guardrail, or memory behaviour changes; nothing here alters
what the app does at request time.

### Fixed
- **Ragas never ran. At all.** `compute_generation_metrics_ragas()` builds
  `ragas.embeddings.HuggingfaceEmbeddings`, whose `__post_init__` decides
  bi-encoder vs cross-encoder with
  `bool(np.intersect1d(SEQ_CLASSIFICATION_ARCHITECTURES, config.architectures))`.
  `BAAI/bge-large-en-v1.5` is a `BertModel`, so that intersection is always
  empty, and `bool()` of an empty array is a hard `ValueError` on NumPy >= 2
  — surfaced through Ragas's pydantic dataclass as "1 validation error for
  HuggingfaceEmbeddings". Deterministic: this project's embedding model could
  never construct that class. The `except` around it silently re-scored every
  metric with the crude lexical judge, so
  `quality-reports/ragas/20260827-064744-live.json` reported 98 rows, 0
  errors, `faithfulness=0.87` — with `judge=lexical_fallback (ragas_error:
  ...)` buried in each metric's `notes` and a bright-green
  "ragas faithfulness 0.87" badge on the README. Replaced with a small
  `BaseRagasEmbeddings` adapter over MAGIK's own `TextEmbedder` singleton:
  no NumPy version sniff, no second copy of the same weights loaded
  (`_make_full_context_retriever()` has already loaded it in that process),
  and relevancy measured in the exact embedding space production retrieval
  ranks in. Length is preserved 1:1 because `TextEmbedder.embed_texts` drops
  entries it cannot encode while Ragas reshapes on the input length.
- **DeepEval's judge was fine; the JSON extractor was eating its answers.**
  `_extract_json_from_text()` searched for an array *before* an object,
  unconditionally, so every object-shaped reply was reduced to its inner
  list and the wrapper discarded — and every DeepEval schema is an object
  wrapping a list (`Truths` `{"truths": [...]}`, `Claims`, `Verdicts`).
  DeepEval's own parser locates the payload with `find("{")`/`rfind("}")`,
  finds neither in a bare array, parses the empty string, and reports
  "Evaluation LLM outputted an invalid JSON. Please use a better evaluation
  model." That is the error on every row of
  `quality-reports/deepeval/20260827-065236-live.json`: 5 of 6 metrics
  `mean=None, n=0`. The 6th scored exactly one row — the one reply that
  happened to arrive in a markdown code fence, the single branch that
  returned the object intact — and that lone score became the
  "deepeval avg 1.00" badge. Extraction is now decided by position: whichever
  of `{` or `[` opens first wins, with the other still tried as a fallback.
  Ragas's array-shaped replies are unaffected.
- **Both badges could publish a number that measured something else.**
  `generate_quality_badges.py` read the score and never looked at how it was
  produced. It now refuses to render a Ragas badge whose `notes` say
  `lexical_fallback`, and requires both frameworks to clear a coverage floor —
  DeepEval a majority of its metrics and half of all metric-row slots, Ragas
  half of `n_queries` — before showing an average. On the rc3 report those
  render "judge unavailable" and "insufficient data (1/6 metrics)". Excluding
  empty metrics from the average was correct on its own but, without a
  coverage floor, is what turned one lucky row into a perfect score.
- **`faithfulness` scored every ungradeable row as 1.0 — a perfect score.**
  "No statements to check", "no context", "the judge returned unparseable
  output" and "an exception was raised" all appended 1.0 to the mean. Same
  class of defect as the lexical fallback above, aimed at the same public
  badge, and newly dangerous: this block sits *after* the
  `HuggingfaceEmbeddings` construction that raised, so it had never once
  executed — fixing Ragas turns it on. Ungradeable rows are now excluded from
  the mean and counted in `notes`; if nothing grades, the metric is empty
  rather than perfect.
- **`faithfulness` judged answers against 800 characters of context** — about
  an eighth of what the model under test was given
  (`MAX_CONTEXT_CHARS=16000`), so any claim supported only by later context
  looked unfaithful because the judge could not see its own evidence. That is
  the exact failure `qwen_judge.grade_metric` documents for its own budget,
  which live measurement settled at 6000 against the same n_ctx=8192 judge.
  Matched to 6000. This cannot regress a published baseline, because the code
  path had never run.
- **Every production promote aborted its monitoring sync.**
  `deploy_monitoring.sh` fetches `GRAFANA_ADMIN_PASSWORD` from SSM into a
  file for Docker Compose, but the Grafana deployment-annotation step reads it
  as a *shell* variable, and nothing ever set one — so `set -u` killed the
  script at that line on every run ("line 129: GRAFANA_ADMIN_PASSWORD:
  unbound variable", CD run 33037783625). Two consequences beyond the missing
  annotation: the run reported failure on an otherwise healthy promote, and
  the cleanup at the bottom of the script became unreachable, leaving the
  Grafana admin password and the ntfy webhook URL in plaintext on disk and
  leaking a `/tmp/monitoring-compose.*.env` per run — the exact opposite of
  the fetch-use-delete lifecycle the script exists to implement. The fetch
  loop now also binds each value into the shell (`printf -v`, no eval),
  cleanup moved to an `EXIT` trap so it covers every exit path, and the
  annotation block is guarded so it can never again abort the deploy.
- **Grafana could keep serving a stale alerting webhook after a sync.**
  `NTFY_WEBHOOK_URL` reaches Grafana through `env_file:`, and Compose does not
  reliably re-read env_file content for a container it considers unchanged —
  a plain `restart` definitively does not, and rc3's promote logged
  "Container magik-grafana Running" (not recreated) on the very run meant to
  roll out a freshly-rotated ntfy topic. Because Grafana expands
  `${NTFY_WEBHOOK_URL}` itself at runtime inside the mounted
  `contact-points.yml`, a container holding the old value leaves all 11 alert
  rules provisioned but undeliverable — looking healthy while nothing pages.
  `deploy_monitoring.sh` now compares the value inside the running container
  against what it just fetched from SSM and force-recreates *only* Grafana
  when they differ, then re-checks and warns if it still does not match.
  Idempotent: a correct container is left alone.
- **`release.yml` could not have cut v1.0.0.** Its version gate compared with
  `sort -V`, which has no notion of a pre-release and ranks `1.0.0` *below*
  `1.0.0-rc3` — so the real release would have been rejected as "not greater
  than" its own release candidate — and its `^[0-9]+\.[0-9]+\.[0-9]+$` regex
  rejected every `-rcN` version outright. Replaced with real SemVer
  precedence. Its changelog step was also a plain `>>` append, writing each
  new release to the *bottom* of this file, below `[0.1.0]`, under a heading
  format (`# [vX.Y.Z] — Title`) matching nothing else in it; it now inserts a
  correctly-formatted `## [X.Y.Z] - YYYY-MM-DD` section above the newest
  entry.
- **`release.yml` also pushed to whatever branch it was dispatched on**, which
  no branch in this repo can accept: on `main` the push is rejected outright
  (PR + status checks required), and on `development` the bump *and* the tag
  both landed on `development`, so the release tag never pointed at `main`.
  Split into two dispatchable modes that match the documented flow —
  `prepare` (from `development`: bump, insert changelog, push a
  `release/vX.Y.Z` branch, open a PR into `main`) and `tag` (from `main`,
  after that PR merges: verify `main` really is at this version, tag it,
  publish the release using the notes already in `CHANGELOG.md` rather than
  asking for the same prose twice). Neither mode writes to a protected branch.
- **`release.yml` interpolated free-text inputs directly into shell commands**
  (`--title "… ${{ inputs.title }}"`, `--notes "${{ inputs.changelog_notes }}"`).
  A title containing a quote or `$(...)` would have been executed by the
  runner. All user-supplied values now reach the CLI through environment
  variables or a file.
- **`make release` preflight could never pass.** It asserted the substring
  `v<version>` appeared in `CHANGELOG.md`, but v0.32.0 normalised every
  heading to `## [X.Y.Z] - date` with no `v` inside the brackets, so the check
  had been a false negative on every release since. Now matches the real
  line-anchored heading.
- DeepEval reports were stamped `"judge": "magik-mistral-7b-gguf"`. Mistral
  was retired as the judge on 2026-08-01; every report written since has
  mislabelled which model graded it.
- DeepEval's judge calls used the 768-token default, which truncates the
  longer truths/claims lists mid-array — indistinguishable downstream from a
  model that cannot follow the format. Raised to 1536, and the retrieval
  context handed to each test case is now bounded by the same 6000-char
  budget `qwen_judge.grade_metric` already applies for the same n_ctx reason
  (it was previously unbounded).
- **`quality.yml`'s Schemathesis and ZAP jobs could be cancelled mid-build.**
  Both build the dev-runtime image with no layer cache (~13–14 min measured)
  inside a 20-minute budget, which the job's own comment admitted left "as
  little as ~1min" of headroom. Hosted-runner I/O varies by more than that: on
  two runs of the same job hours apart, the pure-filesystem layer
  `COPY --from=base-deps /opt/venv` went 83.2s → 173.6s (2.09×) while
  `pip install -r requirements.txt` went 416.5s → 594.9s (1.43×). The slower
  run was killed mid-`docker build` and reported as a failing API-contract
  check that never ran Schemathesis at all. Both budgets raised to 30 minutes;
  a timeout is a ceiling, not a target.

### Added
- `tests/unit/eval/test_judge_json_extraction.py` — pins both shapes the one
  extractor has to serve, in both directions, so DeepEval's objects and
  Ragas's arrays can't be traded off against each other again.
- `tests/unit/core/test_version_consistency.py` — pins `VERSION` and
  `pyproject.toml` to each other, checks the resolver returns that version and
  never silently serves its unknown-version sentinel, and fails if
  `app/main.py`'s header ever grows a hardcoded version again.

### Changed
- **The version is single-sourced.** It was four independent literals —
  `VERSION`, `pyproject.toml`, `app/core/config.py`'s `APP_VERSION` default
  plus that file's own `TestSettings` assertion, and `app/main.py`'s header
  comment — with nothing deriving one from another, and `release.yml` only
  ever rewrote two of them. A bump that missed one shipped an image whose
  `GET /version` and `GET /status` disagreed with the git tag that built it.
  `APP_VERSION` now resolves at import via `config._read_project_version()`,
  which reads `pyproject.toml` (the one version-bearing file the Dockerfile
  actually copies into the runtime image — `VERSION` is not) and falls back to
  `VERSION`, then to an explicit `0.0.0+unknown` sentinel rather than a
  plausible-looking wrong number. Parsed by regex, not `tomllib`, which is
  3.11+ while this project supports 3.10. `app/main.py`'s header no longer
  carries a version at all.
- `deepeval` upper-bounded to `>=4.1,<5` in both `requirements.txt` and
  `pyproject.toml`'s `[quality]` extra. `>=1.4` was unbounded, so 4.1.10
  installed itself — three majors past what the suite was written against —
  exactly the failure mode already documented for Schemathesis in the same
  file.

### Known limitations
- The image's installed NumPy is >= 2 despite `requirements.txt` pinning
  `numpy>=1.26,<2.0.0`: `Dockerfile`'s CUDA `llama-cpp-python` build runs
  `pip install --force-reinstall`, which reinstalls that package's
  dependencies without re-applying requirements.txt's ceiling. Identified,
  deliberately not changed here — the image works on NumPy 2 today, and
  forcing it back down on a release-eve build is a larger risk than the drift
  itself. The Ragas fix above is version-agnostic either way. Reconciling the
  declared and installed pins is a follow-up that needs its own build test.
- The post-release `report-quality-metrics` job runs the eval as a second
  process inside the live container, and has never yet completed a run whose
  Ragas numbers came from the real judge — v1.0.0-rc3 fell back to the lexical
  scorer, and this release's own run was OOM-killed before reaching the fixed
  code. Addressed in [1.0.0-rc5] above.
- Everything else unchanged from [1.0.0-rc3] below — see README.md's Known
  Limitations & Roadmap section.

## [1.0.0-rc3] - 2026-08-27

v1.0.0-rc2 deployed successfully. `cd.yml`'s `report-quality-metrics` job —
non-blocking, runs after every production promotion — had never once
produced a report; this release root-causes and fixes it, and moves the job
onto the box where it can actually succeed.

### Fixed
- **Missing script.** `scripts/generate_quality_badges.py` was caught by the
  blanket `scripts/*` entry in `.gitignore`, so it never existed on any
  runner. Added an explicit `!scripts/generate_quality_badges.py` negation.
- **Import-time crash.** `python -m app.eval.ragas_report` /
  `deepeval_suite` import `app.core.config`, which builds and validates a
  `Settings()` object at import time; validation requires `JWT_SECRET_KEY`
  to be ≥32 characters whenever `AUTH_ENABLED` is true (the default), so the
  bare runner crashed before either report could run a single query.
- **Wrong keyword argument.** Once the import crash was fixed, both report
  modules turned out to call `_query_via_server(..., access_token=...)` — a
  parameter that does not exist on that function's real signature
  (`auth=`, an `EvalAuth` instance). Every gold row raised `TypeError` into
  `errors`, so the report was written with zero rows regardless of the two
  bugs above. `behavioral_runner.py` and `generation_runner.py`'s own calls
  were already correct; only these two had drifted.
- **Missing file scope.** Neither module passed `sources` on the query or
  the grading-context lookup, so every row would additionally 400 against
  `/rag/query`'s FILE SCOPE REQUIRED gate. Now built from
  `relevant_doc_ids`/`source_file`, matching the two working runners, and
  threaded into `_full_contexts()` too — grading context must mirror the
  scope the answer was generated under, or faithfulness gets scored against
  a different file's chunks (see that function's docstring for the live
  audio-suite case where exactly this happened).
- **Badge report selection used file mtime.** A fresh `actions/checkout`
  stamps every committed file with checkout time, so a stale committed
  report could outrank the one a run just produced. Badges now rank by the
  `generated_at` timestamp inside each report.
- **Badge script rendered `NaN` as a literal red badge.** A metric with no
  successful rows (`MetricResult.empty()`) now degrades to the same gray
  "not yet measured" badge as no report at all, instead of a red `nan`.
- **Badge portfolio URLs were dead.** The printed shields.io URLs hardcoded
  `vjkarthik98/multimodal-rag-assistant` — not this repository — so every
  badge link 404'd. Now derived from `GITHUB_REPOSITORY` on CI, with the
  correct repo slug as the local fallback.
- **Illegal push to a protected branch.** The job's last step committed and
  pushed straight to `main`, which could never have worked: `main` requires
  7 status checks, and the commit was marked `[skip ci]`, so they would
  never run to satisfy it. It had also never executed — always skipped
  behind the earlier failures. It also contradicted this repo's own
  documented policy (`quality-reports/README.md`: "Committing is always a
  manual `git add`, never automatic ... that applies doubly to CI") and
  the "never commit without explicit instruction" rule. Replaced with an
  `actions/upload-artifact` publish (90-day retention) plus a job-summary
  table; the job's `contents` permission dropped from `write` to `read`.
- **Wrong execution environment, entirely.** Even with the above fixed, the
  job ran on a bare `ubuntu-latest` runner, which cannot produce a real
  number: the eval judge (`app/eval/judges/qwen_judge.py`, Qwen2.5-7B-Instruct)
  runs locally by design and needs a GPU and a ~4.7GB GGUF neither of which
  a hosted runner has; there is no `QDRANT_URL`/`QDRANT_API_KEY` so grading
  context falls back to 200-char API snippets instead of full retrieved
  chunks; and `actions/checkout`'s `git clean -ffdx` wipes the gitignored
  BM25 index. The job now runs on `[self-hosted, gpu]` and execs into
  `magik-current`, mirroring `tier2-eval.yml`'s established in-container
  pattern — same GPU, same judge, same real BM25 index, same corpus
  preflight check, and the container's own `JWT_SECRET_KEY` for in-process
  auth (so `EVAL_REPORTER_EMAIL`/`PASSWORD` are no longer needed by this
  job at all).
- **Silent-pass masking.** `continue-on-error: true` on both report steps
  meant every failure above rendered as a 2-second green tick — the gap
  survived a full release undetected. The job now emits an explicit
  `::warning` annotation whenever a report step's outcome isn't `success`.
- **Mode mislabeling.** Running in-container means the call is over
  `127.0.0.1`, which `_mode_tag()` reads as "local" — mislabeling a genuine
  production measurement on both the report filename and the public badge.
  Added an `EVAL_MODE_TAG` override, set to `live` for this job.

## [1.0.0-rc2] - 2026-08-23

v1.0.0-rc1 never reached production. Its own fix was correct, but the tag
was cut from a commit that predated it: both `v0.33.0` and `v1.0.0-rc1`
point at the same pre-fix commit, so CD built and promoted the identical
broken image twice, each time failing `promote-production`'s health check
on `Required models not cached: Qwen/Qwen2-VL-7B-Instruct` and
auto-rolling back to v0.31.0. No code change was needed to fix that —
only tagging a commit that actually contains the rc1 fix. This release is
that tag, plus the one latent bug found while verifying it.

### Fixed
- The three gated pyannote diarization models (`speaker-diarization-3.1`,
  `segmentation-3.0`, `wespeaker-voxceleb-resnet34-LM`) were marked
  `optional: True` in `app/bin/models/download_all_models.py` while
  `startup_validator.py`'s `REQUIRED_MODELS` required all three. `optional`
  means "skip on failure and still exit 0", so a gated download failure
  (an expired/unaccepted `HF_TOKEN` being the likely trigger) would have
  finished provisioning with a green summary, written no manifest entry,
  and left the app crash-looping at startup — the exact failure class as
  the qwen2vl_7b bug above, waiting on a token change. They are no longer
  optional: a failure now stops the provisioning run where the summary
  already prints the `set HF_TOKEN for gated models` hint.

### Added
- `tests/unit/core/test_model_manifest_contract.py` pins the contract
  between the downloader and `startup_validator.py` — the two files have no
  import relationship and drifted apart twice in three releases. Covers both
  directions: every `REQUIRED_MODELS` id must be a non-optional, non-eval-only
  downloader entry, and the already-cached fast path must self-heal a missing
  manifest entry, stay idempotent, preserve sibling entries, and still fail
  on checksum mismatch.

### Known limitations carried forward

Unchanged from [1.0.0-rc1]/[0.33.0]/[0.32.0] below — see README.md's Known
Limitations & Roadmap section.

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
