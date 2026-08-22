#!/usr/bin/env bash
# OWASP ZAP Active Scan — MANUAL, OPT-IN ONLY. Never run this from CI, never
# on a schedule, never against a URL you don't explicitly intend to attack
# right now. Unlike run_baseline.sh (passive, safe, spiders + inspects
# traffic), an active scan actually SENDS attack payloads (SQLi, XSS,
# injection probes, etc.) at every discovered endpoint. That is exactly what
# this project's guardrail layer (app/guardrails/) and auth brute-force
# protection are built to catch — but it is real load, real attack traffic,
# and real side effects, so:
#
#   - Scoped to a DEDICATED TEST TENANT (app/bin/seed_test_tenants.py),
#     authenticated via the Authorization header replacer below — never the
#     shared public demo account (magikaiassistant@gmail.com).
#   - EXCLUDES /auth/register, /rag/ingest, /rag/upload, and other
#     mutating/expensive routes from the active attack context (see
#     EXCLUDE_REGEX below) — these either create real side effects (spam
#     accounts, spam files) or are LLM-backed and slow (up to the documented
#     ~60s p95 per request), and an active scanner fires MANY payloads per
#     endpoint, so leaving them in scope would tie up the one GPU box for a
#     long time.
#   - /auth/login legitimately stays in scope — tripping
#     AUTH_LOGIN_RATE_LIMIT_PER_MIN=5 (429, IP-keyed — see perf/k6/README.md
#     for why this doesn't collaterally affect a different visitor's IP) is
#     the CORRECT, desired outcome, proving the brute-force protection works.
#   - Time-boxed (-m minutes below) so a run has a predictable end.
#
# Usage:
#   export MAGIK_TEST_TENANTS="$(cat .magik_test_tenants.json)"   # from seed_test_tenants.py
#   TARGET_URL=http://host.docker.internal:8000 bash security/zap/run_active_scan.sh   # local
#   TARGET_URL=https://magik.vk-ai.online bash security/zap/run_active_scan.sh          # live — read the file docstring first

set -euo pipefail

TARGET_URL="${TARGET_URL:?Set TARGET_URL explicitly — no default, this script sends real attack traffic}"
REPORT_DIR="${REPORT_DIR:-quality-reports/security-dast}"
DATE_TAG="$(date -u +%Y%m%d-%H%M%S)"
MODE_TAG="local"
[[ "$TARGET_URL" != *"127.0.0.1"* && "$TARGET_URL" != *"host.docker.internal"* ]] && MODE_TAG="live"
SCAN_MINUTES="${SCAN_MINUTES:-10}"

: "${MAGIK_TEST_TENANTS:?Set MAGIK_TEST_TENANTS to dedicated test-tenant credentials (see app/bin/seed_test_tenants.py) — never the shared demo login}"

TENANT_EMAIL="$(python3 -c "import json,os; print(json.loads(os.environ['MAGIK_TEST_TENANTS'])['tenants'][0]['email'])")"
TENANT_PASSWORD="$(python3 -c "import json,os; print(json.loads(os.environ['MAGIK_TEST_TENANTS'])['tenants'][0]['password'])")"

ACCESS_TOKEN="$(curl -sS -X POST "${TARGET_URL}/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"${TENANT_EMAIL}\", \"password\": \"${TENANT_PASSWORD}\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")"

mkdir -p "$REPORT_DIR"

echo "[zap-active] target: ${TARGET_URL} (mode=${MODE_TAG}), tenant: ${TENANT_EMAIL}, time-box: ${SCAN_MINUTES}m"
echo "[zap-active] this sends real attack payloads. Confirming in 5s (Ctrl-C to abort)..."
sleep 5

EXCLUDE_REGEX=".*(auth/register|rag/ingest|rag/upload|admin/).*"

# -z passes raw ZAP CLI options through to the underlying daemon:
#   replacer.full_list(0)  — injects our test tenant's Bearer token into
#                             every request, so the scan runs authenticated.
#   globalexcludeurl.url_list(0) — the documented mutating/expensive-route
#                             exclusion from this file's docstring.
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/${REPORT_DIR}:/zap/wrk/:rw" \
  -t ghcr.io/zaproxy/zaproxy:stable \
  zap-full-scan.py \
  -t "$TARGET_URL" \
  -z "-config replacer.full_list(0).description=auth \
      -config replacer.full_list(0).enabled=true \
      -config replacer.full_list(0).matchtype=REQ_HEADER \
      -config replacer.full_list(0).matchstr=Authorization \
      -config replacer.full_list(0).regex=false \
      -config replacer.full_list(0).replacement='Bearer ${ACCESS_TOKEN}' \
      -config globalexcludeurl.url_list(0).description=excluded_mutating_routes \
      -config globalexcludeurl.url_list(0).enabled=true \
      -config globalexcludeurl.url_list(0).regex='${EXCLUDE_REGEX}'" \
  -m "$SCAN_MINUTES" \
  -r "${DATE_TAG}-${MODE_TAG}-active.html" \
  -J "${DATE_TAG}-${MODE_TAG}-active-raw.json"

echo "[zap-active] NOTE: excluded from active attack scope: ${EXCLUDE_REGEX}"
echo "[zap-active] VERIFY the exclusion actually took effect in the HTML report's URLs-scanned"
echo "[zap-active] list before trusting it blind — ZAP CLI config flag names/behavior can drift"
echo "[zap-active] across versions; this is a manual, human-supervised script, not a CI gate."
echo "[zap-active] report written to ${REPORT_DIR}/${DATE_TAG}-${MODE_TAG}-active.html"
echo "[zap-active] review findings manually — this script does not gate on results (see README.md)."
