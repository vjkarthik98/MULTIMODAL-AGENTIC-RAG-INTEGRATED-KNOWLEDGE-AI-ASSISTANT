# Phase 31 — Monitoring & Observability: Deployment Runbook

> Companion to `monitoring/slo.md` (targets/error budgets) and
> `docs/runbooks/phase-30-aws-deployment.md` (the production topology this
> phase adds onto — read that first if you haven't).
>
> This file also covers two CD-safety items explicitly scoped to "Phase 31"
> by project memory when they were deferred at Phase 30's close (Tier-2
> auto-rollback, the SSM secrets migration) — see §6 and §7. Neither is
> "monitoring" in the Prometheus/Grafana sense, but they share this file
> rather than fragmenting into new docs for two items.

Everything the app needs to EMIT telemetry already exists (Prometheus
counters/histograms across ~50 files, OTel spans wrapping the request path,
`structlog` JSON logs). Phase 31 is purely the collection/visualization/
alerting layer on top, plus a few small code fixes that were blocking it —
already applied in this pass (see §1).

## 0. What this phase does NOT change

- Does not touch `.github/workflows/`, `deploy/aws/lambda/`, or
  `deploy/aws/iam/` — Phase 29 CI/CD and the Lambda wake/idle-stop
  automation are unmodified.
- Does not modify the app's `docker run` invocation in `cd.yml` — no new host
  port is published. The monitoring stack reaches the app container by name
  (`magik-current:9464`) on the `magik-net` docker network `cd.yml` already
  creates, not via a published port.
- Does not add Terraform — this deployment (per Phase 30 ADRs) is a
  hand-launched single EC2 host with `docker run`/`docker compose`, not an
  IaC-managed cluster.

## 1. Code fixes applied in this pass (already in the repo, not a manual step)

| File | Fix |
|---|---|
| `app/core/config.py` | `PROMETHEUS_PORT` default 9090→9464 (9090 collides with Prometheus server's own default port on the same docker network) |
| `app/main.py` | The JSON health/model-status route previously at `GET /metrics` renamed to `GET /status` — it was never real Prometheus exposition format (that's served separately via `prometheus_client.start_http_server` on `PROMETHEUS_PORT`); `/metrics` is now free for the actual scrape target |
| `app/api/middleware.py` | `_QUIET_PATHS` updated to match the renamed `/status` route (kept it off the INFO-level access log, as `/metrics` was before) |
| `app/core/infra_registry.py`, `app/core/metrics.py` | Circuit breaker gauge now 3-valued (0=closed, 1=half-open, 2=open) — previously half-open collapsed to the same value (0) as closed, so no alert could ever distinguish "recovering" from "healthy"; a failed probe during half-open now reopens immediately instead of needing `fail_max` fresh failures |
| `app/core/config.py`, `app/core/metrics.py` | New settings (`ONLINE_EVAL_SAMPLE_RATE`, `ONLINE_EVAL_ENABLED`, `ONLINE_EVAL_INTERVAL_SEC`, `MONGO_EVAL_SHADOW_COLLECTION`) and gauges (`magik_eval_online_*`) for the live-quality signal below |
| `app/eval/jobs/shadow_sampler.py` (new) | Deterministic, best-effort sampling of live query/answer/context traces into MongoDB — wired into `app/api/api_routes.py::stream_query`'s existing persistence block |
| `app/eval/jobs/online_eval.py` (new) | Reference-free scoring of sampled traffic (lexical faithfulness/relevancy, numeric-grounding hallucination check, latency, routing distribution) — wired into `app/main.py`'s lifespan as a background task so it shares the app's own Prometheus registry |
| `app/eval/runner.py`, `app/eval/run.py` | `EvalRunner.check_thresholds()` now tracks `last_breached_sections`/`last_error_sections`; `run.py` writes `gate_result.json` alongside `rag_report.json` — originally built for Tier-2 auto-rollback (§6), reused here to feed the Pushgateway push below |
| `.github/workflows/eval-gate.yml`, `docker-compose.monitoring.yml`, `monitoring/prometheus/prometheus.yml` | **Second pass, closes System Design v2 §8's actual ask** ("add RAG-quality SLOs (recall@k, faithfulness) as first-class monitored signals") — the first Phase 31 pass only covered a live-sampled *lexical* proxy, never the real gated numbers. `tier2-full-suite` now pushes every metric in `rag_report.json` to a new `pushgateway` service after every run; see `monitoring/slo.md` §6 and `rag_quality.json`'s "CI Tier-2 (gated, real judge)" panels |
| `docker-compose.monitoring.yml`, `monitoring/loki/*`, `monitoring/grafana/dashboards/logs.json` | **Third pass — log aggregation.** Prior to this, app logs existed only as `docker logs magik-current` / local files on the box — no search, no retention beyond a 150MB rotation cap, no correlation with the metrics/traces above, and CloudWatch was never wired up for the app container (only for the idle-stop Lambda's own execution logs and its `NetworkIn` metric check — unrelated). Added Loki + Promtail instead of CloudWatch Logs, to stay in the same self-hosted, $0-marginal-cost, single-Grafana-pane-of-glass model as the rest of this stack. Promtail tails the *same* Docker json-file logs `docker logs` already reads (no change to how the app container logs, no Docker log-driver swap) and ships them to Loki, with bidirectional click-through to/from Tempo traces via `trace_id` |

## 2. Box `.env` additions (`/opt/magik/.env`, never committed — same file Phase 30 Appendix F describes)

All default OFF/0 — nothing here changes behavior until set:

```bash
# --- Phase 31 monitoring ---
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9464                              # already the code default; explicit for clarity
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317   # NOT localhost:4317 — collector is a separate container
LOG_JSON=true                                      # structured logs -> Loki gets level/trace_id extraction +
                                                     # click-through to the matching Tempo trace. Without this,
                                                     # Promtail still ships raw lines (still searchable), just
                                                     # unparsed — see monitoring/loki/promtail-config.yaml.

# Optional — live RAG-quality dashboard panel (monitoring/grafana/dashboards/rag_quality.json).
# Leave both off (default) to skip live sampling entirely; the offline CI gate
# (eval-gate.yml) is unaffected either way.
ONLINE_EVAL_ENABLED=true
ONLINE_EVAL_SAMPLE_RATE=0.05                       # 5% of sessions
ONLINE_EVAL_INTERVAL_SEC=1800                      # rescoring cadence while the box is up

# --- Grafana (docker-compose.monitoring.yml) ---
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=<generate a real one — compose fails to start without it>

# --- Alerting (monitoring/alerts/contact-points.yml) ---
# NOT SLACK_WEBHOOK_URL (that's a different, CI/CD-only notifier — see
# .github/workflows/cd.yml/tier2-eval.yml — unrelated to Grafana alerting).
# contact-points.yml reads NTFY_WEBHOOK_URL specifically; setting the wrong
# variable here silently leaves Grafana's webhook URL empty and every alert
# fires with nowhere to go.
NTFY_WEBHOOK_URL=<e.g. https://ntfy.sh/<your-topic>, or any other webhook-compatible endpoint (Slack incoming webhook, PagerDuty Events API, etc. all work — "webhook" contact type just POSTs a JSON payload)>
```

## 3. Deploy the monitoring stack (on the box, after Stage 3 of Phase 30 is already done)

```bash
# Persistent dirs on the EBS root volume — same convention as
# /opt/magik/{.hf_cache,data,logs} (Phase 30 Stage 3).
sudo mkdir -p /opt/magik/monitoring/{prometheus,grafana,tempo,loki,promtail}
sudo chown -R ubuntu:ubuntu /opt/magik/monitoring

# magik-net already exists if cd.yml has deployed at least once; idempotent otherwise.
docker network create magik-net 2>/dev/null || true

# Fill in the real values from §2 in /opt/magik/.env, then:
docker compose -f docker-compose.monitoring.yml up -d

# Grafana basic_auth: generate the hash and paste it into deploy/aws/caddy/Caddyfile
# in place of GRAFANA_BASICAUTH_HASH_PLACEHOLDER, then reload Caddy.
docker run --rm caddy:2 caddy hash-password --plaintext '<your password>'
sudo systemctl reload caddy
```

## 4. Validation matrix

| # | Check | Command | Pass criterion |
|---|---|---|---|
| 1 | App Prometheus port reachable from the monitoring network | `docker exec magik-prometheus wget -qO- magik-current:9464/metrics \| head -5` | Prometheus text-exposition output (`# HELP` / `# TYPE` lines), not JSON |
| 2 | `/status` (renamed from `/metrics`) still returns JSON | `curl -sf https://app.yourdomain.com/status` | 200, JSON with `models`/`infra` keys |
| 3 | `/metrics` on the app port is no longer the JSON route | `curl -s https://app.yourdomain.com/metrics` | 404 (nothing registered at that path on the app's own port — the real metrics are on 9464, not proxied through Caddy) |
| 4 | Prometheus scraping the app | Prometheus UI → Status → Targets | `magik-app` job state `UP` |
| 5 | OTel traces flowing | Grafana → Explore → Tempo, run a real query against the app first | A trace appears with spans matching the query pipeline stages |
| 6 | Circuit breaker 3-state gauge | Stop the box's Qdrant reachability (e.g. block egress briefly) and watch `circuit_breaker_state` | Gauge transitions 0 → 2 → 1 (half-open probe) → 0, not stuck at 0 throughout |
| 7 | Grafana not publicly reachable without auth | `curl -s https://app.yourdomain.com/grafana/` from an unauthenticated client | 401 |
| 8 | Grafana dashboards auto-loaded | Log into Grafana → Dashboards → MAGIK folder | `System Health` and `RAG Quality` both present, panels rendering (may be empty until traffic flows) |
| 9 | Alert fires on injected failure | Force a circuit breaker open (test #6) and wait 2 minutes | `magik-circuit-breaker-open` alert fires, webhook message received |
| 10 | Online eval sampling active (if enabled) | Send a few real queries, wait for `ONLINE_EVAL_INTERVAL_SEC`, check `magik_eval_online_sample_count` | Nonzero |
| 11 | Persistent storage survives a stop/start | `docker stop magik-prometheus magik-grafana`, `docker start` them | Grafana dashboards/history and Prometheus data intact (mounted on `/opt/magik/monitoring/*`, not ephemeral) |
| 12 | No secret in the repo | `git grep -iE 'slack_webhook|grafana_admin_password' -- ':!*.example' ':!docs/runbooks/phase-31-monitoring.md'` | nothing real (this runbook's own env-var *names*, not values, are expected matches) |
| 13 | Pushgateway reachable from the runner (host process, not a container) | `curl -sf http://127.0.0.1:9091/-/healthy` on the box | 200 |
| 14 | Tier-2 gated metrics land in Grafana | Trigger `eval-gate.yml` (post-deploy or `workflow_dispatch`), then check the "CI Tier-2 (gated, real judge)" panels in `rag_quality.json` | `magik_eval_gate_metric_value{suite="retrieval",...}` series populated, `magik_eval_gate_exit_code` matches the job's actual result |
| 15 | Promtail discovering containers | `docker exec magik-promtail wget -qO- localhost:9080/targets` | `magik-current`, `magik-redis`, and the monitoring containers all listed as active targets |
| 16 | Logs land in Loki | Open `logs.json`'s "full stream" panel, or Grafana → Explore → Loki, query `{container="magik-current"}` | Recent log lines appear, matching what `docker logs magik-current --tail 20` shows on the box |
| 17 | Structured field extraction (requires `LOG_JSON=true`) | Same Explore query, add `\| json` | `level`, `trace_id`, `event` parse out as fields, not stuck in a single unparsed line |
| 18 | Log → Trace → Log round trip | Find a log line with a non-empty `trace_id` in `logs.json`, click its "View trace" link; from that trace in Tempo, use "Logs for this span" | Lands on the matching trace in Tempo; the reverse link returns to the same log line's neighborhood |
| 19 | `docker logs` still works unaffected | `docker logs magik-current --tail 20` on the box | Unchanged from before this pass — Promtail reads the same files, doesn't touch the container's log driver |
| 20 | Loki survives a stop/start | `docker stop magik-loki`, `docker start magik-loki` | Prior log history still queryable (mounted on `/opt/magik/monitoring/loki`, not ephemeral) |

## 5. Known, deliberate scope limits (see `monitoring/slo.md` §9 for the full list)

- Finance numeric fidelity is not scored on live traffic (offline gate only;
  it IS visible on the CI Tier-2 dashboard panels — see `monitoring/slo.md` §6).
- No Alertmanager — Grafana's built-in unified alerting is the single
  alerting path, deliberately, to keep the container count down on a
  resource-constrained single host.
- **CloudWatch Logs was considered and deliberately not used** for the app
  container, in favor of self-hosted Loki — see the Loki row in §1's table
  for the reasoning (stays in the same $0-marginal-cost, single-Grafana
  model as the rest of this stack, rather than adding a second, separately
  billed, separately viewed log destination). CloudWatch remains in use for
  what it already did before this pass: the idle-stop Lambda's own execution
  logs and its `NetworkIn` metric check — unrelated to app logs.
- `session_id`/`request_id` are extracted by Promtail's JSON pipeline but
  deliberately NOT promoted to indexed Loki labels (only `level`/`trace_id`
  are) — per-session or per-request labels would be effectively unbounded
  cardinality. Both are still queryable via a LogQL line filter
  (`{container="magik-current"} | json | session_id="..."`), just not as a
  fast indexed label.

## 6. Tier-2 auto-rollback (retrieval-section only)

Closes project memory "tier2-autorollback-deferred". Before this change,
`cd.yml`'s `post-deploy-eval` job only *dispatched* Tier-2
(`eval-gate.yml`'s `tier2-full-suite`, via `repository_dispatch`) and never
waited for or reacted to the result — a red Tier-2 run had zero connection
back to the deploy. That gap is now closed, narrowly:

- **Scope, on purpose: `retrieval` only.** `thresholds.yaml` currently gates
  only the `retrieval` section (v5 production baseline, validated
  2026-07-28/30). Every other Tier-2 section (generation, e2e, behavioral,
  routing, ocr/audio/video) is still informational-only against a stale
  pre-Prometheus-judge baseline — a red result there does not reliably
  indicate a real regression, so it must never trigger a rollback. If a
  future re-baseline sets `gate_enabled: true` on another section, this
  mechanism picks it up automatically (see below) — that's a deliberate
  design property, not an oversight to revisit.
- **How it knows which section failed:** `app/eval/runner.py`'s
  `EvalRunner.check_thresholds()` now tracks `last_breached_sections` /
  `last_error_sections` (previously this information existed only as
  ad-hoc printed log lines, never captured structurally). `app/eval/run.py`
  writes it to `app/eval/reports/gate_result.json` alongside the existing
  `rag_report.json`. `eval-gate.yml`'s new "Auto-rollback on retrieval-section
  failure" step reads that file — not the step's raw exit code, which can't
  distinguish "retrieval regressed" from "a different gated section
  regressed" or "an unrelated infra error occurred".
- **How it rolls back:** the exact same sequence `cd.yml`'s own
  health-check-triggered rollback already uses (`docker rm -f` /
  `docker rename magik-previous → magik-current` / `docker start`) — no SSM
  round-trip needed, since Tier-2 already runs *on* the box as the
  self-hosted runner.
- **Only fires on `repository_dispatch`** (i.e. immediately post-deploy),
  never on the nightly schedule or a manual `workflow_dispatch` run days
  later — `magik-previous` is only a meaningful revert target right after a
  fresh deploy; it may be stale or already pruned (see `cd.yml`'s disk-space
  cleanup, which sacrifices `magik-previous` under 40GB free) by the time a
  later run would look at it.
- **A successful rollback still fails the job (exits 1).** A rollback
  happening at all means the deploy was bad — that must surface red on the
  Actions tab and in the Slack notification (now annotated
  "AUTO-ROLLED BACK" when applicable), not read as a quiet pass.
- **If there's no `magik-previous` to roll back to** (first deploy ever, or
  it was pruned), the step fails loudly with a pointer to
  `docs/runbooks/phase-30-aws-deployment.md` Appendix H rather than silently
  doing nothing.

**Validate:** deploy a deliberately regressed retrieval config (e.g. via
`app/eval/run.py --weaken top_k=1`, or by breaking the reranker) to a
staging tag, confirm the Tier-2 run detects it, rolls back, and the job
shows red with the rollback annotation.

## 7. Secrets migration (SSM Parameter Store)

Closes project memory "secrets-management-gap", explicitly scoped to
"when Phase 31 starts" rather than left open-ended. Full details, including
the one-time `aws ssm put-parameter` commands, live in
`deploy/aws/README.md`'s "App secrets in SSM" section — this is a pointer,
not a duplicate.

Summary: `GOOGLE_CLIENT_SECRET`, `SMTP_PASSWORD`, `SECRET_KEY`,
`JWT_SECRET_KEY`, and `MONGO_URI` moved from permanent plaintext in
`/opt/magik/.env` to SSM `SecureString` parameters, fetched fresh by
`cd.yml`'s deploy job on every deploy (mirroring the existing
`/magik/ghcr_pat` pattern) into a `0600`, deploy-scoped, immediately-deleted
env file — not a standing file on disk. New IAM policy:
`deploy/aws/iam/ec2-instance-profile-permissions.json` (previously the
instance profile's permissions weren't captured as a repo file at all, only
the two Lambda policies were — this fixes that adjacent gap too).
`QDRANT_API_KEY`/`QDRANT_URL` (already GitHub encrypted secrets) and
`REDIS_URL`/`REDIS_TOKEN` (Upstash, lower blast radius) were left as-is —
not silently deferred, a deliberate stop-here decision documented in
`deploy/aws/README.md`.

## 8. Request tracing, drift detection, security alerting, Arize Phoenix

A second monitoring pass on top of everything above (§0–§7 unchanged) —
request-level distributed tracing that actually connects into one waterfall
per request, statistical drift detection on live traffic, security-relevant
metrics/alerts (auth, guardrails, GPU), a richer online-eval signal, an
LLM/RAG-native trace UI (Arize Phoenix), and the alert rules and dashboard
panels tying all of it together. Built and landed in 7 phases; this section
documents the result, not the sequence.

### 8.1 What's new, in one table

| Area | What | Files |
|---|---|---|
| Request tracing | Root OTel spans at `query_pipeline()`, `RAGPipeline.run()`, `RAGPipeline.stream()`, `HybridRetriever.search()`, `Reranker.rerank()` — previously the already-instrumented spans deeper in the stack (`agent_controller_handle`, `reasoning_generate_answer`, `qdrant_search`, `prompt_builder`) each opened as their OWN disconnected root trace (no parent span was ever active), so Tempo held fragments, never one coherent per-request waterfall | `app/pipeline/query_pipeline.py`, `app/pipeline/rag_pipeline.py`, `app/retrieval/hybrid_retriever.py`, `app/retrieval/reranker.py` |
| Version correlation | `GIT_SHA` (reads the Dockerfile's own build-time `ARG`/`ENV`, never shells out), `PROMPT_VERSION` (already existed in `prompt_builder.py`, now also stamped on every span) — both tagged on every root span so a quality regression can be correlated back to a specific deploy | `app/core/config.py`, span attributes in the 3 files above |
| Cheap-win metrics | `reranker_latency_seconds` (was untimed entirely), `app/auth/metrics.py` (new — `auth_login_failures_total{reason}`, `auth_mfa_failures_total`, `auth_rate_limit_rejections_total`; `app/auth/` had zero Prometheus metrics before this) | `app/core/metrics.py`, `app/auth/metrics.py`, `app/auth/service.py`/`mfa.py`/`rate_limit.py` |
| GPU/VRAM visibility | `gpu_vram_free_gb` already existed (`app/core/model_loader.py`, refreshed every `MODEL_REAPER_INTERVAL_SEC` by the already-running model reaper) but was never dashboarded or alerted — fixed the actual gap instead of adding a duplicate gauge | `monitoring/grafana/dashboards/system_health.json`, `monitoring/alerts/rules.yml` |
| Richer live sampling | `shadow_sampler.py`'s `_retrieval_stats()` now also captures `retrieval_count`/`top1_score`/`mean_topk_score` from the same `sources` array every caller already builds. A new `sample_and_log()` call was added to the non-streaming `POST /rag/query` route specifically — it's the only route with real agent-decision diversity (`rag`/`direct`/`memory`/`search`); `RAGPipeline.stream()` (the SSE route real users hit) is RAG-only and never routes elsewhere, so without this, `magik_eval_online_route_share` would forever read 100% "rag." Deliberately NOT added inside `query_pipeline()` itself — the eval harness calls that function directly, in-process, bypassing HTTP, and would have contaminated the live-traffic signal with synthetic gold-set queries | `app/eval/jobs/shadow_sampler.py`, `app/api/api_routes.py` |
| Drift detection | `app/eval/jobs/drift_eval.py` (new) — reference-vs-current-window KS-test comparison on `query_length`/`top1_score`/`mean_topk_score`/`latency_ms`. Uses `scipy.stats`, not Evidently — every Evidently version (including the older 0.4.x API) unconditionally pulls in a bundled litestar/uvicorn/plotly web-server stack, ~15 extra packages, for what this needs as a KS-test on 4 columns | `app/eval/jobs/drift_eval.py`, `app/core/config.py` (`DRIFT_*` settings), `app/core/metrics.py` (`magik_drift_*` gauges) |
| Arize Phoenix | Self-hosted LLM/RAG-native trace UI — embedding-space clustering, per-span retrieved-document inspection — on the SAME spans Tempo gets, via a second OTel collector exporter. `app/utils/otel_attrs.py` (new) tags those spans with OpenInference semantic conventions (`openinference-semantic-conventions` — pure constants, no transitive deps) so Phoenix's UI has real data | `docker-compose.monitoring.yml`, `monitoring/otel/collector-config.yaml`, `app/utils/otel_attrs.py` |
| Live retrieval-quality trend | `magik_eval_online_top1_score`/`_mean_topk_score`/`_retrieval_count` — the plain rolling mean of the same fields drift_eval.py compares statistically. A p-value says "did this change"; these say "what is it right now" — a dashboard needs both | `app/eval/jobs/online_eval.py`, `app/core/metrics.py` |
| Agent Health visibility | `agent_controller_duration_seconds`/`_errors_total`/`_fallback_total`/`_active_requests` already existed in `app/agents/agent_controller.py` but were never dashboarded — same "fix the real gap" pattern as GPU/VRAM above | `monitoring/grafana/dashboards/rag_quality.json` |
| Alert rules | 6 new rules: reranker latency breach, application error-LOG spike (first Loki-sourced rule — LogQL, not PromQL), span error-rate spike (first Tempo-sourced rule — TraceQL, see §8.5), guardrail block-rate spike, auth failure spike, drift warning + drift critical | `monitoring/alerts/rules.yml` |
| Dashboard panels | 3 new panels on `system_health.json` (GPU VRAM, reranker latency, security counters), 12 new panels on `rag_quality.json` (Live Retrieval Quality, Agent Health, Drift Detection sections) | `monitoring/grafana/dashboards/*.json` |
| Guardrails hot-path fix | `app/guardrails/metrics.py`'s `record_block`/`record_allow`/`record_scrub` had NO try/except at all, called unconditionally at 7+ call sites on literally every request (input sanitize, output guard, rate limiter). Confirmed by testing, not inspection, that a simulated Prometheus failure propagated straight up. Now wrapped, matching every other metrics module in this codebase | `app/guardrails/metrics.py` |

### 8.2 New `.env` additions (append to §2's block)

```bash
# --- Drift detection (off by default; needs ONLINE_EVAL_SAMPLE_RATE > 0 too) ---
DRIFT_ENABLED=true
DRIFT_WINDOW_SIZE=200                              # rows, not a time window — see config.py's comment
DRIFT_REFERENCE_PATH=app/eval/baselines/drift_reference.jsonl   # code default; override only to relocate it
DRIFT_CHECK_INTERVAL_SEC=3600
DRIFT_WARNING_THRESHOLD=0.25                        # provisional — no measured baseline on this box yet
DRIFT_CRITICAL_THRESHOLD=0.5                        # provisional, same caveat

# GIT_SHA is NOT set here — the Dockerfile bakes it in at build time
# (ARG/ENV GIT_SHA, from ${{ github.sha }} in cd.yml). Only relevant for a
# manual, non-CD build: GIT_SHA=<short-sha> if you want span attribution to
# work outside the normal deploy pipeline.
```

### 8.3 Deploying Phoenix (extends §3)

Phoenix is included in `docker-compose.monitoring.yml` — no separate step,
`docker compose -f docker-compose.monitoring.yml up -d` brings it up with
everything else. Two things §3 doesn't already cover:

- **Grafana version bump.** `grafana/grafana:11.3.1` → `12.1.10` in this same
  compose file, required for the TraceQL alert rule (§8.5) — provision a
  fresh Grafana on this version, don't attempt to upgrade an existing
  `grafana_data` volume in place without checking Grafana's own upgrade
  notes first.
- **UI access.** `127.0.0.1:6006` on the box (SSH tunnel, or `docker exec`)
  — same loopback-only pattern as Prometheus, not yet behind the Caddy
  public path Grafana has. `ssh -L 6006:localhost:6006 <box>` then browse
  `http://localhost:6006`.

### 8.4 Drift reference baseline — build and re-baseline

`drift_eval.py` skips cleanly (`{"skipped": "no_reference_yet"}`) until a
reference file exists — it is never fabricated automatically. Build one
after `DRIFT_ENABLED`/`ONLINE_EVAL_SAMPLE_RATE` have been live long enough
to consider the current window "normal":

```bash
# On the box, inside the app's environment (same one that can reach Mongo):
python -m app.eval.jobs.drift_eval --build-reference
# Writes DRIFT_WINDOW_SIZE (default 200) of the most recent sampled rows to
# DRIFT_REFERENCE_PATH as JSONL. Prints {"rows_written": N, "path": "..."}.
```

**Re-baseline** the same way, any time — it's a plain overwrite, not
versioned. Re-baseline after a deliberate change that should shift what
"normal" means (a new corpus, a model swap, a retrieval-tuning change) —
NOT reactively right after a `magik-drift-critical` alert fires, which would
silently absorb the very regression the alert just caught into the new
"normal" and make it permanently invisible going forward. Confirm the
regression is understood and either accepted or fixed first.

**Debug a single run without affecting the live dashboard:**

```bash
python -m app.eval.jobs.drift_eval
# Scores once, prints the full comparison (per-column p-value/PSI/means,
# severity). Prometheus gauges live in-process — a fresh short-lived
# process like this does NOT update the running app's /metrics. Use it to
# sanity-check the comparison logic against real data, not to refresh the
# dashboard (that only happens inside the live app's own background loop).
```

### 8.5 New alert rules

| Alert | Datasource | Fires on | Notes |
|---|---|---|---|
| `magik-reranker-latency-breach` | Prometheus | p95 `reranker_latency_seconds` > 5s, 5m sustained | Reuses `settings.LATENCY_TARGET_CROSS_MODAL_MS`, not a new number |
| `magik-error-log-spike` | **Loki** (LogQL) | >20 ERROR/CRITICAL log lines, 5m | First non-Prometheus rule in this file. Requires `LOG_JSON=true` |
| `magik-trace-error-spike` | **Tempo** (TraceQL) | >5 error-status spans, 5m | **Requires Grafana 12.1+ and `GF_FEATURE_TOGGLES_ENABLE=tempoAlerting`** (both now set — §8.3, `docker-compose.monitoring.yml`). Grafana documents TraceQL alerting itself as experimental / "should not be used in production environments yet" even at this version — included because it was explicitly requested with that tradeoff accepted. **Verify it actually fires** (force a guardrail block, confirm the alert state changes) on the first real deploy to the new Grafana version — this is the one rule in this file that could not be tested against a live instance before shipping (no Docker daemon in the environment that built it) |
| `magik-guardrail-block-rate-spike` | Prometheus | >10 guardrail blocks, 5m | Provisional threshold |
| `magik-auth-failure-spike` | Prometheus | >10 combined login+MFA failures, 5m | Deliberately excludes rate-limit rejections — those are expected under normal heavy use, would make this noisy rather than a real signal |
| `magik-gpu-vram-critical` | Prometheus | `gpu_vram_free_gb` < 2.0GB, 5m | Below the model reaper's own 6.0GB eviction watermark — means eviction isn't recovering enough headroom |
| `magik-drift-warning` / `magik-drift-critical` | Prometheus | `magik_drift_severity` ≥1 (30m) / ==2 (15m) | Critical requires a QUALITY column (`top1_score`/`mean_topk_score`) to have BOTH drifted AND degraded — high drift volume alone, or a quality column drifting in the IMPROVING direction, is only ever warning-tier at most |

**Known, deliberate gap:** none remaining in this list — Tempo alerting was
initially skipped (metrics-generator config couldn't be verified against a
live instance) and later implemented via TraceQL alerting instead once
Grafana was bumped to 12.1+, per explicit direction. See
`magik-trace-error-spike`'s row above for the residual, unavoidable caveat
(Grafana's own experimental-feature status, not verified live).

**Investigating any alert:** every rule's `annotations.description` in
`monitoring/alerts/rules.yml` names the exact dashboard panel or Explore
query to check next — start there, not with this doc.

### 8.6 New dashboard panels

- `system_health.json`: GPU VRAM free (stat, red<2GB/yellow<6GB/green≥6GB),
  reranker latency p50/p95, security counters (guardrail blocks + auth
  failures + rate-limit rejections, one combined panel).
- `rag_quality.json`, three new sections (each with a markdown divider panel
  explaining scope, same convention as the existing "CI Tier-2" divider):
  **Live Retrieval Quality** (top1/mean-topk score trend, doc count),
  **Agent Health** (controller latency by decision, errors + fallback rate,
  active-request gauge — no "tool calls per request" panel, the
  architecture is single-dispatch, that metric doesn't exist to show),
  **Drift Detection** (severity stat, dataset score, per-column p-value,
  reference/current window size).

### 8.7 Validation matrix additions (extends §4)

| # | Check | Command | Pass criterion |
|---|---|---|---|
| 21 | One request produces one connected trace, not fragments | Send a real query, open Grafana → Explore → Tempo, find the trace | A single trace_id contains `query_pipeline`/`rag_pipeline_stream` as the root with `agent_controller_handle`/`reasoning_generate_answer`/`qdrant_search`/`prompt_builder` nested underneath it — not separate top-level traces |
| 22 | Phoenix receiving spans | Browse `http://localhost:6006` (via SSH tunnel) after sending a few real queries | Recent traces appear, retrieval spans show document lists |
| 23 | Drift job running (if enabled) | `DRIFT_ENABLED=true`, build a reference (§8.4), wait `DRIFT_CHECK_INTERVAL_SEC`, check `magik_drift_severity` | Nonzero-but-present (0/1/2), not absent |
| 24 | TraceQL alert rule provisions without crash-looping Grafana | `docker compose -f docker-compose.monitoring.yml up -d grafana`, `docker logs magik-grafana` | No provisioning error for `magik-trace-error-spike`; if `GF_FEATURE_TOGGLES_ENABLE=tempoAlerting` is missing this WILL fail here — check first |
| 25 | Guardrails still function correctly with a broken Prometheus client | (Test-only — see `tests/unit/api/test_monitoring_outage_resilience.py`) | A malicious query is still blocked even when its own block-counter fails to record; not something to reproduce manually in production |

### 8.8 How to disable this pass safely

Everything in §8 is additive and independently toggleable — none of it
requires touching anything from §0–§7 to turn off:

- **Drift detection only:** `DRIFT_ENABLED=false` (or leave the default).
  `drift_eval.py`'s background loop returns immediately; no gauges update,
  no alerts fire (they read a gauge that stays at its last value, which is
  0/absent if never enabled).
- **Everything live-sampling-related** (drift + the retrieval-quality
  gauges + agent route diversity from §8.1): `ONLINE_EVAL_SAMPLE_RATE=0`
  (already the default) — both `online_eval.py` and `drift_eval.py`'s loops
  check this and return immediately.
- **Phoenix only:** `docker compose -f docker-compose.monitoring.yml stop
  phoenix` — the OTel collector's `otlp/phoenix` exporter will fail to
  connect and log a (harmless, rate-limited) connection error; Tempo export
  on the same pipeline is unaffected, spans still land there normally.
- **The TraceQL alert specifically**, if it turns out to misbehave on a
  real deploy (§8.5's caveat): delete or comment out the
  `magik-trace-error-spike` rule block in `monitoring/alerts/rules.yml` and
  redeploy the monitoring stack — every other rule in that file is
  independent and unaffected.
- **Request tracing itself cannot be selectively disabled** without
  disabling `OTEL_ENABLED` entirely (§2) — the span wrappers are
  unconditional code paths (the OTel SDK no-ops safely when unconfigured,
  per the OTel API's own design, so leaving `OTEL_ENABLED=false` costs
  nothing at runtime beyond an unused no-op tracer).

### 8.9 Escalation: tried and removed (Grafana OnCall, PagerDuty, Opsgenie, Better Stack)

`NTFY_WEBHOOK_URL` (a push notification, no retry/escalation) remains the
*only* alert delivery mechanism. A dedicated real-escalation contact point
was attempted and then removed after every option hit a hard external
blocker not fixable from this codebase, in order:

1. **Grafana OnCall** (self-hosted OSS) — archived 2026-03-24, its phone/
   SMS escalation depends on a Grafana Cloud Connection feature already
   turned off for OSS users, and self-hosting it now would mean 3-4 new
   containers (Postgres, Redis/RabbitMQ, Celery, the Django engine) for a
   product whose core feature is already broken.
2. **PagerDuty** (free tier, 5 users, real escalation, zero new
   containers — this would have been the choice) — trial signup hard-
   rejects consumer email domains; a GoDaddy-forwarded business-style
   address on the project's own domain hit the same wall.
3. **Opsgenie** (same free-tier shape) — Atlassian stopped accepting new
   Opsgenie customers on 2025-06-04; end of life 2027-04-05. Not
   discoverable without trying to sign up — the marketing page reads like
   an active product.
4. **Better Stack** ("PagerDuty alternative," found via a sponsored search
   result) — free tier is explicitly email/Slack only; real phone/SMS
   escalation requires a paid Responder seat (~$29-34/mo/person).

If real 2AM-style paging becomes a requirement again, the next attempt
should start by verifying the candidate service is (a) still accepting new
customers and (b) includes phone/SMS on its actual free tier — both
checked directly against the provider's current site, not secondhand
review/blog content, which was unreliable and contradictory during this
attempt. `monitoring/alerts/contact-points.yml`'s own comment carries the
same summary.
