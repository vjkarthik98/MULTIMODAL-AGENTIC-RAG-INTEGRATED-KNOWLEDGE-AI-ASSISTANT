set -uo pipefail
IMG="ghcr.io/vjkarthik98/multimodal-agentic-rag-integrated-knowledge-ai-assistant:latest"
OWNER="vjkarthik98"
log(){ echo "[deploy] $*"; }

command -v docker >/dev/null 2>&1 || { log "FATAL: docker not installed"; exit 1; }
command -v aws    >/dev/null 2>&1 || { log "FATAL: aws cli not installed"; exit 1; }
docker info >/dev/null 2>&1       || { log "FATAL: docker daemon unreachable"; exit 1; }
for p in /opt/magik/.env /opt/magik/.hf_cache /opt/magik/data /opt/magik/logs; do
  [ -e "$p" ] || { log "FATAL: missing required path $p"; exit 1; }
done
if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  log "FATAL: '--gpus all' not functional — nvidia-container-toolkit missing/broken"
  exit 1
fi
log "preflight OK"

disk_free_gb(){ df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
log "disk free before cleanup: $(disk_free_gb)GB"
docker image prune -f >/dev/null 2>&1 || true
docker builder prune -af >/dev/null 2>&1 || true
log "disk free after safe prune: $(disk_free_gb)GB"
if [ "$(disk_free_gb)" -lt 40 ]; then
  log "under 40GB — dropping the rollback container to reclaim its image"
  docker rm -f magik-previous >/dev/null 2>&1 || true
  docker image prune -af >/dev/null 2>&1 || true
  log "disk free after full prune: $(disk_free_gb)GB"
fi
FREE_GB="$(disk_free_gb)"
if [ "${FREE_GB:-0}" -lt 30 ]; then
  log "FATAL: only ${FREE_GB}GB free on / after cleanup; this image needs ~25GB to unpack."
  df -h / || true
  du -sh /opt/magik/* 2>/dev/null | sort -rh | head -10 || true
  docker system df || true
  exit 1
fi
log "disk OK — ${FREE_GB}GB free"

GHCR_PAT="$(aws ssm get-parameter --name /magik/ghcr_pat \
              --with-decryption --query Parameter.Value --output text)" \
  || { log "FATAL: could not read /magik/ghcr_pat from SSM"; exit 1; }
echo "$GHCR_PAT" | docker login ghcr.io -u "$OWNER" --password-stdin \
  || { log "FATAL: docker login to ghcr.io failed"; exit 1; }
log "pulling $IMG (multi-GB, several minutes expected)"
docker pull "$IMG" || { log "FATAL: docker pull failed"; exit 1; }

echo "__PROD_ENV_B64__" | base64 -d > /opt/magik/.env.prod
log "prod.env config written"

SECRETS_ENV="/opt/magik/.env.secrets"
rm -f "$SECRETS_ENV"
: > "$SECRETS_ENV"
chmod 600 "$SECRETS_ENV"
SECRETS_OK="yes"
for PARAM_KEY in \
  "google_client_secret:GOOGLE_CLIENT_SECRET" \
  "smtp_password:SMTP_PASSWORD" \
  "secret_key:SECRET_KEY" \
  "jwt_secret_key:JWT_SECRET_KEY" \
  "mongo_uri:MONGO_URI" \
  "qdrant_api_key:QDRANT_API_KEY" \
  "redis_token:REDIS_TOKEN" \
  "hf_token:HF_TOKEN" \
  "tavily_api_key:TAVILY_API_KEY"
do
  SSM_NAME="${PARAM_KEY%%:*}"
  ENV_KEY="${PARAM_KEY##*:}"
  VAL="$(aws ssm get-parameter --name "/magik/${SSM_NAME}" \
           --with-decryption --query Parameter.Value --output text 2>/dev/null)" \
    || { log "FATAL: could not read /magik/${SSM_NAME} from SSM"; SECRETS_OK="no"; break; }
  echo "${ENV_KEY}=${VAL}" >> "$SECRETS_ENV"
done
if [ "$SECRETS_OK" != "yes" ]; then
  rm -f "$SECRETS_ENV"
  log "FATAL: app-secrets fetch from SSM failed"
  exit 1
fi
log "app secrets fetched from SSM (9/9)"

docker network create magik-net >/dev/null 2>&1 || true
if ! docker ps --filter name=magik-redis --filter status=running -q | grep -q .; then
  docker rm -f magik-redis >/dev/null 2>&1 || true
  docker run -d --name magik-redis \
    --network magik-net --restart unless-stopped \
    --log-opt max-size=10m --log-opt max-file=2 \
    redis:7-alpine redis-server \
      --save "" --appendonly no \
      --maxmemory 512mb --maxmemory-policy allkeys-lru \
    || { log "FATAL: could not start magik-redis sidecar"; exit 1; }
  log "started magik-redis sidecar"
else
  docker network connect magik-net magik-redis >/dev/null 2>&1 || true
  log "magik-redis already running"
fi

docker rm -f magik-previous >/dev/null 2>&1 || true
docker stop -t 30 magik-current >/dev/null 2>&1 || true
docker rename magik-current magik-previous >/dev/null 2>&1 || true

docker run -d --name magik-current \
  --gpus all --restart unless-stopped \
  --network magik-net \
  --shm-size=2g \
  --log-opt max-size=50m --log-opt max-file=3 \
  -p 8000:8000 \
  --env-file /opt/magik/.env \
  --env-file /opt/magik/.env.prod \
  --env-file "$SECRETS_ENV" \
  -e LOCAL_CACHE_HOST=magik-redis \
  -e TORCH_HOME=/app/.hf_cache/torch \
  -v /opt/magik/.hf_cache:/app/.hf_cache \
  -v /opt/magik/data:/app/data \
  -v /opt/magik/logs:/app/logs \
  "$IMG"
RUN_STATUS=$?
rm -f "$SECRETS_ENV"
if [ "$RUN_STATUS" -ne 0 ]; then
  log "FATAL: docker run failed (port 8000 already held by a non-container process?)"
  docker rename magik-previous magik-current >/dev/null 2>&1 || true
  docker start magik-current >/dev/null 2>&1 || true
  exit 1
fi

log "waiting for /health (up to 20 min for cold model load)"
for i in $(seq 1 80); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    log "HEALTHY after $(( i * 15 ))s"
    docker ps --filter name=magik-current --format '{{.Names}} {{.Status}}'
    exit 0
  fi
  if ! docker ps -q --filter name=magik-current --filter status=running | grep -q .; then
    log "FATAL: container exited during startup — last 80 log lines:"
    docker logs --tail 80 magik-current 2>&1 || true
    break
  fi
  sleep 15
done

log "DEPLOY_HEALTHCHECK_FAILED — last 80 log lines:"
docker logs --tail 80 magik-current 2>&1 || true
docker rm -f magik-current >/dev/null 2>&1 || true
docker rename magik-previous magik-current >/dev/null 2>&1 || true
docker start magik-current >/dev/null 2>&1 || true
exit 1
