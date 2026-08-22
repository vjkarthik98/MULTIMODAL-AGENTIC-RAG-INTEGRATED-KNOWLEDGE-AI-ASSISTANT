# k6 Load / Stress / Multi-User Testing

[k6](https://k6.io) (AGPL-3.0, Grafana Labs) — chosen over Locust/Artillery
because it has a native Prometheus remote-write output, so results land in
the *same* Grafana instance MAGIK already runs (`monitoring/grafana/`)
instead of a disconnected tool.

## Scripts

| Script | Purpose | Target |
|---|---|---|
| `smoke.js` | Sanity-check auth + one real query before scaling up | local |
| `stress.js` | Ramp VUs to find where latency/error rate degrades | local only, always |
| `soak.js` | Sustained low load, long duration — leak/drift detection | local only, always |
| `multi_user_tenant.js` | N tenants concurrently — asserts **zero cross-tenant leakage** under load | local (or live) |
| `live_profile.js` | Modest, realistic-traffic profile against the real deployed URL | **live, manual only** |

`stress.js` and `soak.js` never target the live URL — see the approved plan
(`quality-reports/` conventions) and each file's own docstring for why: they're
designed to find breaking points / run for a long time, and the live box is
a single shared GPU instance real recruiters might be looking at.
`multi_user_tenant.js` and `live_profile.js` are safe to point at the live
URL deliberately (see "Live mode" below).

## One-time setup: test tenants

Every script authenticates as one or more dedicated test-tenant accounts —
**never** the shared public demo login (`magikaiassistant@gmail.com`). See
`app/auth/models.py`'s `is_load_test` field and `app/bin/seed_test_tenants.py`
for why.

```bash
# Against local docker-compose Mongo:
docker compose up -d api qdrant redis mongo
python -m app.bin.seed_test_tenants --count 10

# Against the real deployed Mongo (only with an explicit go-ahead — this
# writes real accounts to production):
MONGO_URI=<production connection string> python -m app.bin.seed_test_tenants --count 10

export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
```

## Local mode (default)

```bash
docker compose up -d api qdrant redis mongo
export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"

k6 run perf/k6/smoke.js
k6 run perf/k6/stress.js
k6 run -e SOAK_DURATION=8h perf/k6/soak.js
k6 run --vus 8 --iterations 24 perf/k6/multi_user_tenant.js

# Feed results into the existing Grafana/Prometheus stack instead of just
# the terminal summary:
k6 run --out experimental-prometheus-rw perf/k6/stress.js
```

## Live mode (manual, on-demand only)

```bash
export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
export MAGIK_API_BASE_URL=https://magik.vk-ai.online

k6 run perf/k6/live_profile.js
k6 run --vus 5 --iterations 15 perf/k6/multi_user_tenant.js
```

This wakes the wake-on-demand AWS box (`deploy/aws/lambda/wake_gateway/`) if
it's asleep and holds it awake for the run's duration — a deliberate,
accepted cost for a portfolio project with low real traffic, never something
that should happen automatically (see `.github/workflows/quality-live.yml`,
`workflow_dispatch`-only).

## A real gap this suite surfaced (since fixed)

`RATE_LIMIT_RPM=60/min` (the general API rate limit) used to be enforced
**per client IP** (`app/main.py`'s `rate_limit` middleware, in-memory), not
per authenticated user — a separate per-user Redis-backed limiter existed at
`app/auth/rate_limit.py` (`check_user_rate_limit`) but was unused, since
`_rate_limit_check()`, the hook every mutating route in
`app/api/api_routes.py` called, was a no-op stub. Practical effect: a k6 run
(or any concurrent multi-user traffic from one network location, including
real visitors behind a shared corporate NAT) shared one 60/min bucket
regardless of how many distinct users were authenticated.

This is now fixed: `app/main.py`'s `rate_limit` middleware independently
verifies the Bearer token (it runs before `AuthMiddleware` in the actual
stack — verified empirically, not assumed) and calls the real per-user Redis
limiter when one is present, falling back to per-IP only for unauthenticated
requests. The dead `_rate_limit_check()` stub and its four call sites were
removed. See `app/main.py`'s `rate_limit`/`_rate_limit_user_id` for the full
reasoning.
