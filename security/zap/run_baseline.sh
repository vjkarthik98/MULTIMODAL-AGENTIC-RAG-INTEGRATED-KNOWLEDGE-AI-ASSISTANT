#!/usr/bin/env bash
# OWASP ZAP Baseline scan (Apache-2.0, zaproxy/zaproxy) — PASSIVE only. Spiders
# the target and inspects traffic for common issues (missing security headers,
# cookie flags, information disclosure, etc.) without sending any attack
# payloads. Safe to run against a live, in-use server — this is what CI runs
# on every PR touching app/api/, app/main.py, or auth code, and what's safe to
# run against the real deployed URL any time (see README.md's "Live baseline"
# section). Compare with run_active_scan.sh's docstring — active scanning is a
# completely different risk profile and is never automatic.
#
# Usage:
#   docker compose up -d api qdrant redis mongo
#   bash security/zap/run_baseline.sh                      # local target
#   TARGET_URL=https://magik.vk-ai.online bash security/zap/run_baseline.sh   # live target (read-only crawl, safe)

set -euo pipefail

TARGET_URL="${TARGET_URL:-http://host.docker.internal:8000}"
REPORT_DIR="${REPORT_DIR:-quality-reports/security-dast}"
DATE_TAG="$(date -u +%Y%m%d-%H%M%S)"
MODE_TAG="local"
[[ "$TARGET_URL" != *"127.0.0.1"* && "$TARGET_URL" != *"host.docker.internal"* ]] && MODE_TAG="live"

mkdir -p "$REPORT_DIR"

echo "[zap-baseline] target: ${TARGET_URL} (mode=${MODE_TAG})"

docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$(pwd)/${REPORT_DIR}:/zap/wrk/:rw" \
  -t ghcr.io/zaproxy/zaproxy:stable \
  zap-baseline.py \
  -t "$TARGET_URL" \
  -r "${DATE_TAG}-${MODE_TAG}-baseline.html" \
  -J "${DATE_TAG}-${MODE_TAG}-baseline-raw.json" \
  -I

echo "[zap-baseline] report written to ${REPORT_DIR}/${DATE_TAG}-${MODE_TAG}-baseline.html"
echo "[zap-baseline] -I flag: informational exit code (never fails the build on findings —"
echo "[zap-baseline] this scan is passive/informational per the approved plan, not a blocking gate)."
