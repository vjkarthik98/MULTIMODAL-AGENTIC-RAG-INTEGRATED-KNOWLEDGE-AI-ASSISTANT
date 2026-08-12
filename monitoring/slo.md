# MAGIK — Service Level Objectives (Phase 31)

Companion to `docs/runbooks/phase-31-monitoring.md`. Targets below are pulled
from `app/eval/thresholds.yaml` (the CI/offline gate — authoritative) and the
production topology documented in `docs/runbooks/phase-30-aws-deployment.md`.
Where this doc states a number, it is sourced from one of those two files,
not invented for this document.

## 0. Why "availability" is redefined here

MAGIK runs on a **scale-to-zero** EC2 instance (ADR-30-1,
`docs/runbooks/phase-30-aws-deployment.md`): the box intentionally stops
after ~15–20 minutes of no traffic and wakes on the next visit. A standard
"99.9% uptime" SLO is meaningless — the box is *deliberately* down most of
the time by design, not by failure. The real user-facing promise is:

> **A visitor never sees a broken page — they see either an instant response
> (box already warm) or a "waking up" page that resolves within the wake
> budget below.**

That is the SLO this document sets. Steady-state latency/quality SLOs (once
the box is warm) are separate and unaffected by scale-to-zero.

## 1. Wake-latency SLO (scale-to-zero specific)

| Stage | Budget | Source |
|---|---|---|
| API Gateway/CloudFront → Lambda wake gateway responds | < 1s | Lambda cold start is typically <500ms for this handler size (`deploy/aws/lambda/wake_gateway/handler.py`) |
| `StartInstances` called → EC2 `running` | ~30–60s | AWS EC2 typical boot time for this instance type |
| EC2 `running` → app container healthy (`/health` 200) | up to 20 min | `cd.yml`'s own health-check loop budget: "80 x 15s = 20 min... a cold start pages ~18GB of models off EBS" — this is the SAME number the deploy pipeline already uses as its ceiling, not a new invented figure |
| **Total: visitor lands on a broken/stuck page** | **Never** — the wake gateway shows a self-refreshing waking page (`WAKING_HTML`) for the entire window above, then 302s to the app once `/health` returns 200 | `deploy/aws/lambda/wake_gateway/handler.py` |

**Error budget:** the wake gateway itself has no dependency beyond
`ec2:DescribeInstances`/`ec2:StartInstances` and the app's own `/health` —
its failure modes are AWS API errors (extremely rare) or the app failing to
become healthy within 20 minutes (a real incident — see
`docs/runbooks/phase-30-aws-deployment.md` Appendix H, "Wake gateway shows
waking forever"). Budget: **0 unbounded-wake incidents per month** — any
occurrence is an incident, not an accepted rate, because the fix (rollback to
`magik-previous`) is already automatic in `cd.yml`.

## 2. Steady-state latency SLO (box warm)

Directly from `app/eval/thresholds.yaml`'s `latency` section (already CI-gated):

| Metric | Target | Rationale (from thresholds.yaml) |
|---|---|---|
| p95 | ≤ 60.0s | "Current production-observed ceiling on this hardware" |
| p50 | ≤ 20.0s | "Median should be well below ceiling — degradation signal before p95 breaches" |

**Error budget:** thresholds.yaml gates every PR on this; a regression never
reaches production undetected via `eval-gate.yml`. Live confirmation is
`magik_eval_online_latency_p95_ms` / `_p50_ms` (`monitoring/grafana/dashboards/rag_quality.json`)
once `ONLINE_EVAL_ENABLED=true`. Alert: `monitoring/alerts/rules.yml`'s
`magik-p95-latency-breach` fires if the live sample sustains p95 > 60000ms
for 5 minutes.

**Reranker sub-budget** (added in the second Phase 31 pass — request
tracing/drift/Phoenix, see `docs/runbooks/phase-31-monitoring.md` §8): p95
`reranker_latency_seconds` ≤ `settings.LATENCY_TARGET_CROSS_MODAL_MS`
(5.0s default) — the same ceiling `app/retrieval/reranker.py`'s own
`rerank_latency_exceeded` warning log already used, not a new number. Alert:
`magik-reranker-latency-breach`, 5m sustained.

## 3. Retrieval quality SLO (offline, CI-gated — authoritative)

From `app/eval/thresholds.yaml`'s `retrieval` section (`gate_enabled: true`,
v5 production baseline, n=56, measured on the actual production box):

| Metric | Gate (min) | v5 baseline | Margin |
|---|---|---|---|
| recall@5 | 0.4835 | 0.5089 | 5% |
| recall@10 | 0.5259 | 0.5536 | 5% |
| MRR | 0.3380 | 0.3558 | 5% |
| nDCG@10 | 0.3823 | 0.4024 | 5% |
| context_precision | 0.0255 | 0.0268 | 5% (absolute value flagged as low, see thresholds.yaml note — a future retrieval-quality pass, not this SLO) |
| hit_rate | 0.6447 | 0.6786 | 5% |

**Error budget:** zero — any PR that regresses below the gate is
automatically blocked by `eval-gate.yml` (Phase 29, already live). This is
the one SLO in this document with a hard enforcement mechanism rather than an
alert.

## 4. Generation / hallucination quality (offline gate + live drift signal)

| Metric | Offline gate (thresholds.yaml) | Live signal (this phase) |
|---|---|---|
| `generation.faithfulness` | min 0.250 (CrossEncoder+GGUF judge) | `magik_eval_online_faithfulness` — reference-free lexical judge, NOT the same judge; a drift trend matters more than the absolute value being comparable |
| `hallucination.hallucination_rate` | max 0.72 | `magik_eval_online_hallucination_rate` — same detector (`hallucination_flag_single`), applied to live traffic instead of the gold set |
| `finance.numeric_fidelity` | min 0.95 | Not currently sampled online (would need finance-number-specific scoring added to `online_eval.py` — out of scope for this pass, noted as a gap, not silently dropped) |

**Error budget:** the offline gate (0.72 hallucination ceiling) is
enforced pre-merge. The live signal is a **15-minute sustained** breach
before alerting (`magik-eval-hallucination-drift` in `monitoring/alerts/rules.yml`) —
short spikes on a handful of sampled queries are expected noise at low
`ONLINE_EVAL_SAMPLE_RATE`, not an incident.

## 5. Resilience SLO

| Signal | Target | Mechanism |
|---|---|---|
| Circuit breaker open (state=2) duration | Auto-recovers via half-open probe within `{QDRANT,REDIS,MONGO}_CB_RESET_TIMEOUT` (60s/30s/60s defaults, `app/core/config.py`) | `app/core/infra_registry.py`'s `_CircuitBreaker` |
| Time spent in OPEN state per incident, alerted | > 2 minutes sustained | `monitoring/alerts/rules.yml`'s `magik-circuit-breaker-open` (state > 1, i.e. == 2; half-open does not page) |
| Ingestion error rate | < 10% of extracts erroring, 5-min window | `monitoring/alerts/rules.yml`'s `magik-error-rate-spike`, using the verified `magik_{mod}_extracts_total` / `magik_{mod}_extract_errors_total` counters (`app/ingestion/*.py`) |

**Error budget:** an open circuit breaker is expected occasionally (managed
cloud services do have transient blips) — the SLO is about *duration*, not
zero occurrences. 2 minutes sustained open is the alert threshold because the
shortest reset timeout (Redis, 30s) means a healthy dependency should already
be probing half-open well before 2 minutes elapse.

## 6. RAG-quality SLO — the real, gated numbers (CI Tier-2, closes System Design v2 §8's gap)

`eval-gate.yml`'s `tier2-full-suite` job now pushes every numeric metric from
`rag_report.json` to a Prometheus Pushgateway (`docker-compose.monitoring.yml`)
after every post-deploy / manually-dispatched run, visualized in
`monitoring/grafana/dashboards/rag_quality.json`'s "CI Tier-2 (gated, real
judge)" panels. This was the actual System Design v2 §8 ask — "add RAG-quality
SLOs (recall@k, faithfulness) as first-class monitored signals" — and the
first Phase 31 pass had left it out, covering only a live-sampled *lexical*
proxy instead. Same gate floors as §3/§4 above (recall@5 ≥ 0.4835, recall@10
≥ 0.5259, MRR ≥ 0.3380, nDCG@10 ≥ 0.3823, hit_rate ≥ 0.6447), now visible as
a trend, not just a pass/fail in a GitHub Actions log.

**Not pushed from Tier-1** (PR-time gate): it runs on a GitHub-hosted runner
with no network path to the box's Pushgateway (127.0.0.1-bound, same as
Grafana), and scores an ephemeral checkout rather than the production
container — plotting it alongside production numbers on the same dashboard
would be misleading, not just redundant. Tier-1's result stays visible where
it already belongs: the PR's own Actions run.

## 7. Drift detection SLO (statistical, live traffic)

`app/eval/jobs/drift_eval.py` (second Phase 31 pass — see
`docs/runbooks/phase-31-monitoring.md` §8) compares a fixed reference-window
snapshot against the current sampled-traffic window on 4 columns
(`query_length`/`top1_score`/`mean_topk_score`/`latency_ms`) via a KS-test.

| Signal | Target | Mechanism |
|---|---|---|
| Statistical significance per column | KS-test p ≥ 0.05 (not drifted) | `magik_drift_column_pvalue{column}` |
| Dataset-level drift volume | Informational, not itself gated | `magik_drift_dataset_score` (fraction of tracked columns drifted) |
| Severity | 0 (info) sustained | `magik_drift_severity` — CRITICAL requires BOTH high drift volume (≥ `DRIFT_CRITICAL_THRESHOLD`, provisional 0.5) AND a QUALITY column (`top1_score`/`mean_topk_score`) drifting in the DEGRADING direction; high volume alone or a quality column improving is only ever WARNING at most |

**Error budget:** no hard gate — this is a live early-warning signal, not a
CI-blocking one (nothing in `thresholds.yaml` reads these gauges). Alerts:
`magik-drift-warning` (severity ≥1, 30m sustained) / `magik-drift-critical`
(severity ==2, 15m sustained), both in `monitoring/alerts/rules.yml`.
Requires `DRIFT_ENABLED=true`, `ONLINE_EVAL_SAMPLE_RATE > 0`, and a built
reference (`python -m app.eval.jobs.drift_eval --build-reference`) — reads
as `magik_drift_reference_size` / `_current_size` == 0 until all three are
true, not a false "no drift" signal.

## 8. Security signal SLO (guardrails + auth)

| Signal | Target | Mechanism |
|---|---|---|
| Guardrail block rate | No hard target — volume/trend signal | `guardrail_blocks_total{guard_type,surface}` (pre-existing metric, newly alerted — `magik-guardrail-block-rate-spike`, >10 blocks/5m, provisional) |
| Auth failure rate (login + MFA) | No hard target — volume/trend signal | `auth_login_failures_total{reason}` + `auth_mfa_failures_total` (new, `app/auth/metrics.py`) — `magik-auth-failure-spike`, >10 combined/5m, provisional |
| Rate-limit rejections | Not alerted (deliberately) | `auth_rate_limit_rejections_total` — dashboarded only; rate-limiting firing under normal heavy use is expected behavior, alerting on it would be noise, not signal |
| Monitoring itself must not weaken security | Zero regression | `app/guardrails/metrics.py`'s `record_block`/`record_allow`/`record_scrub` previously had no try/except at all — a Prometheus client failure would have propagated into every request's guardrail check. Fixed and pinned by `tests/unit/api/test_monitoring_outage_resilience.py::test_guardrail_still_blocks_correctly_even_with_prometheus_broken`, which asserts a malicious query is still blocked even while its own block-counter fails to record |

**Error budget:** both alert thresholds above are explicitly provisional —
no real production traffic has been measured on this box yet to calibrate
against (same honesty convention as thresholds.yaml's own "informational
until re-baselined" sections). Revisit once real data exists.

## 9. What is explicitly NOT covered yet (documented gap, not silently dropped)

- **Finance numeric fidelity on live traffic** — `online_eval.py` does not
  currently run `compute_finance_fidelity()` against sampled live answers.
  The offline gate (`finance.numeric_fidelity` ≥ 0.95) still applies at merge
  time and IS now visible on the CI Tier-2 dashboard panels (§6) — this
  remaining gap is specifically the *live-sampled* signal, not the offline one.
- **`verification` suite metrics** (`app/eval/thresholds.yaml`'s
  `verification` section — Phase 32 grounding/citation verification loop)
  have no baseline yet upstream, so nothing to monitor here yet either — but
  once they do, they need no new plumbing: the Pushgateway step (§6) pushes
  every numeric metric generically, so a new suite/metric shows up in
  Prometheus automatically the next time it's added to `rag_report.json`.
  Only the dashboard panel itself would need adding.
- **`magik-trace-error-spike`'s TraceQL query is unverified against a live
  Tempo instance** — built from official Grafana/Tempo documentation (no
  Docker daemon was available in the environment that built it), and
  Grafana itself documents TraceQL alerting as experimental / "should not
  be used in production environments yet" even at the v12.1+ this requires.
  Verify it actually fires on the first real deploy to the bumped Grafana
  version (see `docs/runbooks/phase-31-monitoring.md` §8.7, check #24).
- **Arize Phoenix's UI is loopback-only** (`127.0.0.1:6006`), not behind the
  Caddy reverse proxy Grafana has — SSH tunnel required for access. Adding
  a public `/phoenix/` path is a reasonable future step, not done as part
  of this pass to keep it scoped to what was asked.
- **The drift reference is a manual, one-time bootstrap** —
  `python -m app.eval.jobs.drift_eval --build-reference` is never run
  automatically (deliberately: an unattended re-baseline could silently
  absorb a real regression into the new "normal"). Until someone runs it
  once, §7's drift signal reads as `reference_size == 0`, not a false "no
  drift" — check that gauge before trusting an all-clear.
