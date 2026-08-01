# OWASP ZAP — DAST

[OWASP ZAP](https://www.zaproxy.org/) (Apache-2.0). Complements the existing
SAST/SCA coverage in `.github/workflows/security.yml` (Bandit, pip-audit,
detect-secrets, license scan) with dynamic testing — attacking the *running*
API, not just reading source. Two very different risk profiles:

## Baseline scan — passive, safe, automatic

Spiders the target and inspects real traffic for common issues (missing
security headers, cookie flags, information disclosure) without sending any
attack payloads. Safe against a live, in-use server.

```bash
make zap-baseline                                    # local docker-compose target
TARGET_URL=https://magik.vk-ai.online bash security/zap/run_baseline.sh   # live, safe any time
```

CI: `.github/workflows/quality.yml` runs this on every PR touching
`app/api/`, `app/main.py`, or auth code, against the local docker-compose
stack — informational, non-blocking (see that workflow for why).

## Active scan — real attack traffic, MANUAL / OPT-IN ONLY

`security/zap/run_active_scan.sh` sends real attack payloads (SQLi, XSS,
injection probes) at every discovered endpoint — this is what the guardrail
layer (`app/guardrails/`) and auth brute-force protection exist to catch, but
it's real load and real side effects. Never wired into CI, never scheduled.

Safety properties baked into the script (read its docstring for the full
reasoning):
- Authenticates as a **dedicated test tenant** (`app/bin/seed_test_tenants.py`)
  — never the shared public demo login.
- Excludes `/auth/register`, `/rag/ingest`, `/rag/upload`, `/admin/*` from
  active attack scope (real side effects / LLM-backed and slow).
- `/auth/login` stays in scope deliberately — tripping the brute-force
  limiter (429) is the correct, desired outcome.
- Time-boxed (`SCAN_MINUTES`, default 10).

```bash
export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"
TARGET_URL=http://host.docker.internal:8000 bash security/zap/run_active_scan.sh   # local
TARGET_URL=https://magik.vk-ai.online bash security/zap/run_active_scan.sh          # live — read the script first
```

Live active scans should be run deliberately, off-peak, by a human watching
the output — never as a routine or scheduled task.

## Reports

`quality-reports/security-dast/` — dated `*-baseline.html`/`*-active.html`
(committed when you choose to keep one) plus raw JSON (gitignored — see
`quality-reports/README.md`).
