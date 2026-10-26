#!/usr/bin/env bash
# Run the RAGAS or DeepEval quality report ON the production box, isolated
# from the live app, and publish the result to the box's Grafana.
#
# Usage (from an SSM session on magik-prod). The box's checkout is synced to
# the promoted tag by cd.yml's monitoring-sync step, so the script is already
# there after a release:
#     cd /home/ubuntu/MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT
#     sudo bash deploy/aws/scripts/run_quality_report.sh ragas 5
#     sudo bash deploy/aws/scripts/run_quality_report.sh ragas 30
#     sudo bash deploy/aws/scripts/run_quality_report.sh deepeval 10
#
# ── Why it is built this way ─────────────────────────────────────────────
# Every earlier attempt ran the eval with `docker exec` INSIDE magik-current:
# a second python process loading its own BGE + SigLIP next to the app's, with
# the 4.7GB Qwen judge spawned on top. The host OOM-killer ended it in
# v1.0.0-rc4 and rc5 and took the runner with it. Three changes fix the cause
# rather than shrinking the workload again:
#
#   1. ISOLATION. The eval gets its OWN container from the same image, under a
#      hard cgroup cap (--memory == --memory-swap). If it outgrows the cap, the
#      kernel kills THIS container only. magik-current, Caddy and the
#      monitoring stack are outside the cgroup and cannot be chosen.
#   2. NO DUPLICATE MODELS. EVAL_REMOTE_MODELS=1 makes the eval fetch grading
#      contexts and embeddings from the app's /internal/eval/* routes
#      (app/api/eval_internal.py), served from models already resident there.
#      The eval process loads only the judge.
#   3. ADMISSION, NOT HOPE. It refuses to start unless the GPU has room for the
#      judge and the host has room for the cap, and QWEN_JUDGE_REQUIRE_GPU=1
#      stops the judge falling back to CPU mid-run.
#
# While it runs, a self-expiring `magik:eval-busy-until` tag keeps the
# idle-stop Lambda from stopping the box (handler.py::_eval_busy). It never
# starts or stops the instance itself.
#
# Results:
#   /opt/magik/quality-reports/<tool>/<timestamp>-production.{json,md}
#   Pushgateway  job=magik_quality_report      — scores (only on success)
#   Pushgateway  job=magik_quality_report_run  — run status (always)
#   Grafana      RAG Quality → "Offline judge — RAGAS / DeepEval"

set -Eeuo pipefail

TOOL="${1:-}"
LIMIT="${2:-}"
case "$TOOL" in
  ragas)    MODULE="app.eval.ragas_report";   LIMIT="${LIMIT:-30}" ;;
  deepeval) MODULE="app.eval.deepeval_suite"; LIMIT="${LIMIT:-10}" ;;
  *) echo "usage: $0 <ragas|deepeval> [row_limit]" >&2; exit 64 ;;
esac
case "$LIMIT" in
  ''|*[!0-9]*|0) echo "row_limit must be a positive integer (got '$LIMIT')" >&2; exit 64 ;;
esac

APP_CONTAINER="${APP_CONTAINER:-magik-current}"
APP_NETWORK="${APP_NETWORK:-magik-net}"
EVAL_CONTAINER="magik-quality-eval"
ENVIRONMENT="${QUALITY_ENVIRONMENT:-production}"
MAGIK_ROOT="${MAGIK_ROOT:-/opt/magik}"
REPORT_DIR="${MAGIK_ROOT}/quality-reports"
JUDGE_GGUF="${MAGIK_ROOT}/.hf_cache/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
PUSHGATEWAY="${PUSHGATEWAY_URL:-http://127.0.0.1:9091}"

# 12g covers python + torch import + ragas/deepeval + the judge worker's host
# side (weights are GPU-offloaded). Admission adds headroom for the live app.
EVAL_MEMORY_LIMIT_GB="${EVAL_MEMORY_LIMIT_GB:-12}"
HOST_HEADROOM_GB="${HOST_HEADROOM_GB:-3}"
# ~4.7GB GGUF + n_ctx=8192 KV cache + buffers — same 6GB qwen_judge.py
# requires, plus 0.5GB so the live app's own allocations are not squeezed.
MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-6656}"
BUSY_TAG="magik:eval-busy-until"
BUSY_LEASE_SECONDS=900
BUSY_REFRESH_SECONDS=300

log()  { printf '[quality-report %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { log "ABORT: $*"; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo) — needs docker and the cgroup files"

# ── Instance identity (IMDSv2) ───────────────────────────────────────────
IMDS_TOKEN="$(curl -sf -X PUT http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 600')" || die "IMDS unreachable — is this the EC2 box?"
INSTANCE_ID="$(curl -sf -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)"
REGION="$(curl -sf -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region)"

# ── Cleanup (always) ─────────────────────────────────────────────────────
ENV_FILE=""
HEARTBEAT_PID=""
LOGS_PID=""
cleanup() {
  local rc=$?
  [ -n "$HEARTBEAT_PID" ] && kill "$HEARTBEAT_PID" 2>/dev/null || true
  [ -n "$LOGS_PID" ] && kill "$LOGS_PID" 2>/dev/null || true
  [ -n "$ENV_FILE" ] && rm -f "$ENV_FILE"
  docker rm -f "$EVAL_CONTAINER" >/dev/null 2>&1 || true
  aws ec2 delete-tags --region "$REGION" --resources "$INSTANCE_ID" \
    --tags "Key=${BUSY_TAG}" >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ── Preflight ────────────────────────────────────────────────────────────
log "tool=$TOOL limit=$LIMIT instance=$INSTANCE_ID environment=$ENVIRONMENT"

docker ps --filter "name=^/${APP_CONTAINER}$" --filter status=running -q | grep -q . \
  || die "$APP_CONTAINER is not running"
docker exec "$APP_CONTAINER" curl -sf http://127.0.0.1:8000/health >/dev/null \
  || die "$APP_CONTAINER /health is not answering — wait for model load to finish"

if docker ps -a --filter "name=^/${EVAL_CONTAINER}$" -q | grep -q .; then
  die "$EVAL_CONTAINER already exists — another report is running (or crashed; 'docker rm -f $EVAL_CONTAINER')"
fi

[ "$(docker exec "$APP_CONTAINER" printenv EVAL_INTERNAL_API_ENABLED 2>/dev/null || true)" = "true" ] \
  || die "EVAL_INTERNAL_API_ENABLED is not true in $APP_CONTAINER — this image/env predates the internal eval API"

EVAL_USER_ID="$(docker exec "$APP_CONTAINER" printenv EVAL_USER_ID 2>/dev/null || true)"
[ -n "$EVAL_USER_ID" ] || die "EVAL_USER_ID unset in $APP_CONTAINER"
docker exec "$APP_CONTAINER" test -f "/app/data/users/${EVAL_USER_ID}/bm25_index/bm25.pkl" \
  || die "no BM25 index for $EVAL_USER_ID — retrieval would be dense-only and scores falsely low. Rebuild: docker exec $APP_CONTAINER python3.12 -m app.retrieval.bm25_retriever --user_id $EVAL_USER_ID"

[ -f "$JUDGE_GGUF" ] || die "judge weights missing: $JUDGE_GGUF"

FREE_VRAM_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
[ "${FREE_VRAM_MB:-0}" -ge "$MIN_FREE_VRAM_MB" ] \
  || die "only ${FREE_VRAM_MB}MB VRAM free, judge needs ${MIN_FREE_VRAM_MB}MB"

AVAIL_KB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
NEED_KB=$(( (EVAL_MEMORY_LIMIT_GB + HOST_HEADROOM_GB) * 1024 * 1024 ))
[ "$AVAIL_KB" -ge "$NEED_KB" ] \
  || die "host MemAvailable $((AVAIL_KB / 1024 / 1024))GB < ${EVAL_MEMORY_LIMIT_GB}GB cap + ${HOST_HEADROOM_GB}GB headroom"

IMAGE="$(docker inspect --format '{{.Config.Image}}' "$APP_CONTAINER")"
log "preflight OK — image=$IMAGE vram_free=${FREE_VRAM_MB}MB mem_available=$((AVAIL_KB / 1024 / 1024))GB"

# ── Keep the idle-stop Lambda off the box ────────────────────────────────
stamp_busy() {
  aws ec2 create-tags --region "$REGION" --resources "$INSTANCE_ID" \
    --tags "Key=${BUSY_TAG},Value=$(( $(date +%s) + BUSY_LEASE_SECONDS ))" >/dev/null
}
stamp_busy || die "could not set the ${BUSY_TAG} tag — apply terraform-new-account/iam.tf (magik_ec2_eval_busy_tag) first, or the idle-stop Lambda may stop the box mid-run"
( while sleep "$BUSY_REFRESH_SECONDS"; do stamp_busy || true; done ) &
HEARTBEAT_PID=$!

# ── Launch the isolated eval container ───────────────────────────────────
# Same env the app was started with (secrets included: EvalAuth mints its JWT
# in-process with JWT_SECRET_KEY). Written 0600, deleted the moment the
# container exists — the same plaintext-window discipline as cd.yml.
umask 077
ENV_FILE="$(mktemp /run/magik-quality-env.XXXXXX)"
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$APP_CONTAINER" \
  | grep -v '^$' > "$ENV_FILE"

ARGS="--limit ${LIMIT}"
CID="$(docker run -d --name "$EVAL_CONTAINER" \
  --gpus all \
  --network "$APP_NETWORK" \
  --memory "${EVAL_MEMORY_LIMIT_GB}g" --memory-swap "${EVAL_MEMORY_LIMIT_GB}g" \
  --oom-score-adj 1000 \
  --shm-size=2g \
  --no-healthcheck \
  --log-opt max-size=50m --log-opt max-file=2 \
  --env-file "$ENV_FILE" \
  -e EVAL_SERVER_URL="http://${APP_CONTAINER}:8000" \
  -e EVAL_REMOTE_MODELS=1 \
  -e QWEN_JUDGE_REQUIRE_GPU=1 \
  -e EVAL_MODE_TAG="$ENVIRONMENT" \
  -e PYTHONUNBUFFERED=1 \
  -v "${MAGIK_ROOT}/.hf_cache:/app/.hf_cache" \
  "$IMAGE" \
  sh -c "python3.12 -m ${MODULE} ${ARGS} && python3.12 -m app.eval.quality_push --tool ${TOOL} --out /app/quality-reports/${TOOL}.prom")"
rm -f "$ENV_FILE"; ENV_FILE=""
log "started $EVAL_CONTAINER (${CID:0:12}), cap ${EVAL_MEMORY_LIMIT_GB}g — streaming its log"

mkdir -p "$REPORT_DIR"
RUN_LOG="${REPORT_DIR}/${TOOL}-$(date -u +%Y%m%d-%H%M%S).log"
docker logs -f "$CID" > >(tee "$RUN_LOG") 2>&1 &
LOGS_PID=$!

# ── Track peak memory from the container's own cgroup ────────────────────
CG="/sys/fs/cgroup/system.slice/docker-${CID}.scope"
PEAK=0
while docker ps -q --filter "id=$CID" --filter status=running | grep -q .; do
  CUR="$(cat "${CG}/memory.current" 2>/dev/null || echo 0)"
  [ "$CUR" -gt "$PEAK" ] && PEAK="$CUR"
  sleep 5
done
KPEAK="$(cat "${CG}/memory.peak" 2>/dev/null || echo 0)"
[ "$KPEAK" -gt "$PEAK" ] && PEAK="$KPEAK"

EXIT_CODE="$(docker inspect --format '{{.State.ExitCode}}' "$CID")"
OOM="$(docker inspect --format '{{.State.OOMKilled}}' "$CID")"
sleep 1; kill "$LOGS_PID" 2>/dev/null || true; LOGS_PID=""
log "eval exited code=$EXIT_CODE oom_killed=$OOM peak_mem=$((PEAK / 1024 / 1024))MB"

# ── Collect reports ──────────────────────────────────────────────────────
docker cp "${CID}:/app/quality-reports/." "${REPORT_DIR}/" 2>/dev/null \
  || log "no report files produced"

# ── Publish ──────────────────────────────────────────────────────────────
OOM_NUM=0; [ "$OOM" = "true" ] && OOM_NUM=1
if ! curl -sf -X PUT --data-binary @- \
  "${PUSHGATEWAY}/metrics/job/magik_quality_report_run/tool/${TOOL}/environment/${ENVIRONMENT}" <<EOF
# TYPE magik_quality_run_exit_code gauge
magik_quality_run_exit_code ${EXIT_CODE}
# TYPE magik_quality_run_oom_killed gauge
magik_quality_run_oom_killed ${OOM_NUM}
# TYPE magik_quality_run_peak_memory_bytes gauge
magik_quality_run_peak_memory_bytes ${PEAK}
# TYPE magik_quality_run_row_limit gauge
magik_quality_run_row_limit ${LIMIT}
# TYPE magik_quality_run_timestamp_seconds gauge
magik_quality_run_timestamp_seconds $(date +%s)
# TYPE magik_quality_run_info gauge
magik_quality_run_info{image="${IMAGE}"} 1
EOF
then
  log "WARNING: Pushgateway unreachable at $PUSHGATEWAY — is the monitoring stack up?"
fi

PROM="${REPORT_DIR}/${TOOL}.prom"
if [ "$EXIT_CODE" = "0" ] && [ -f "$PROM" ]; then
  if curl -sf -X PUT --data-binary "@${PROM}" \
    "${PUSHGATEWAY}/metrics/job/magik_quality_report/tool/${TOOL}/environment/${ENVIRONMENT}"; then
    log "scores pushed to Pushgateway — Grafana: RAG Quality → Offline judge"
  else
    log "WARNING: score push failed; the report is still at $REPORT_DIR/$TOOL/"
  fi
  rm -f "$PROM"
  log "report: $(ls -1t "${REPORT_DIR}/${TOOL}"/*.md 2>/dev/null | head -1)"
  cat "$(ls -1t "${REPORT_DIR}/${TOOL}"/*.md | head -1)"
else
  [ "$OOM" = "true" ] && log "OOM-killed at the ${EVAL_MEMORY_LIMIT_GB}g cap — the live app was not affected. Peak was $((PEAK / 1024 / 1024))MB."
  log "FAILED — scores NOT pushed (previous run's scores stay on the dashboard). Log: $RUN_LOG"
  exit 1
fi
