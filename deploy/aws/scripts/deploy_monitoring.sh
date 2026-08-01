#!/usr/bin/env bash
# Bring up (or update) the MAGIK monitoring stack — Prometheus, Grafana,
# Tempo, OTel Collector, Loki, Promtail, Pushgateway.
#
# Idempotent: safe to re-run (`docker compose up -d` only recreates what
# actually changed).
#
# Run ON THE BOX, from the repo checkout's root (needs deploy/aws/prod.env
# and docker-compose.monitoring.yml at their normal repo-relative paths):
#
#     bash deploy/aws/scripts/deploy_monitoring.sh
#
# Closes the last piece of project memory "secrets-management-gap"
# (2026-07-31 plan, step 4): GRAFANA_ADMIN_PASSWORD and NTFY_WEBHOOK_URL are
# real secrets but are consumed by THIS stack, not the app container cd.yml
# already handles — they need their own fetch-use-delete lifecycle here,
# same pattern, different script, per this repo's own convention of inline
# per-script SSM fetching rather than a shared abstraction.
#
# Three different delivery mechanisms are in play, not one — see the
# comments at each step below for why.

set -euo pipefail

log(){ echo "[deploy-monitoring] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${AWS_DIR}/../.." && pwd)"

command -v docker >/dev/null 2>&1 || { log "FATAL: docker not installed"; exit 1; }
command -v aws    >/dev/null 2>&1 || { log "FATAL: aws cli not installed"; exit 1; }
[ -f "${AWS_DIR}/prod.env" ] || { log "FATAL: ${AWS_DIR}/prod.env not found"; exit 1; }
[ -f "${REPO_ROOT}/docker-compose.monitoring.yml" ] \
  || { log "FATAL: ${REPO_ROOT}/docker-compose.monitoring.yml not found"; exit 1; }

# ── Fetch this stack's own secrets from SSM ─────────────────────────────────
# Written to a fixed, known path — docker-compose.monitoring.yml's grafana
# service env_file: list points at this exact path (see the companion edit
# there) so NTFY_WEBHOOK_URL lands as a real container environment variable,
# not just compose-level substitution (Grafana expands it itself at runtime
# inside monitoring/alerts/contact-points.yml, a separately mounted file
# compose's own substitution never touches).
SECRETS_ENV="/opt/magik/.env.monitoring.secrets"
rm -f "${SECRETS_ENV}"
: > "${SECRETS_ENV}"
chmod 600 "${SECRETS_ENV}"
FETCH_OK="yes"
for PARAM_KEY in \
  "grafana_admin_password:GRAFANA_ADMIN_PASSWORD" \
  "ntfy_webhook_url:NTFY_WEBHOOK_URL"
do
  SSM_NAME="${PARAM_KEY%%:*}"
  ENV_KEY="${PARAM_KEY##*:}"
  VAL="$(aws ssm get-parameter --name "/magik/${SSM_NAME}" \
           --with-decryption --query Parameter.Value --output text 2>/dev/null)" \
    || { log "FATAL: could not read /magik/${SSM_NAME} from SSM"; FETCH_OK="no"; break; }
  echo "${ENV_KEY}=${VAL}" >> "${SECRETS_ENV}"
done
if [ "${FETCH_OK}" != "yes" ]; then
  rm -f "${SECRETS_ENV}"
  log "FATAL: monitoring-stack secrets fetch from SSM failed — seed them first, see deploy/aws/README.md"
  exit 1
fi
log "monitoring secrets fetched from SSM (2/2)"

# ── Combined env file for Docker Compose's OWN substitution ────────────────
# GRAFANA_ROOT_URL and GRAFANA_ADMIN_PASSWORD are both referenced directly as
# ${VAR} inside docker-compose.monitoring.yml's own YAML (the environment:
# block) — that substitution happens when compose PARSES the file, before
# any container starts, and needs its own --env-file, separate from any
# service's env_file: directive. prod.env supplies GRAFANA_ROOT_URL; the
# SSM-fetched file above supplies GRAFANA_ADMIN_PASSWORD — concatenated here
# since `docker compose --env-file` only accepts one path. Later lines win
# on a key collision, so the SSM secret intentionally comes second even
# though there's no actual overlap today.
COMPOSE_ENV="$(mktemp /tmp/monitoring-compose.XXXXXX.env)"
chmod 600 "${COMPOSE_ENV}"
cat "${AWS_DIR}/prod.env" "${SECRETS_ENV}" > "${COMPOSE_ENV}"

# ── Bring the stack up ───────────────────────────────────────────────────────
cd "${REPO_ROOT}"
docker network create magik-net >/dev/null 2>&1 || true
if docker compose --env-file "${COMPOSE_ENV}" -f docker-compose.monitoring.yml up -d; then
  log "monitoring stack up"
  RC=0
else
  log "FATAL: docker compose up failed"
  RC=1
fi

# ── Cleanup — same fetch-use-delete lifecycle as cd.yml's app secrets.
# Compose bakes env_file/environment values into each container at create
# time; deleting the source files afterward does not affect already-running
# containers, only prevents them from lingering as plaintext on disk.
rm -f "${COMPOSE_ENV}" "${SECRETS_ENV}"
log "throwaway env files removed"

exit "${RC}"
