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
# A dedicated escalation secret (PagerDuty, then Opsgenie) was added here
# and removed — every free real phone/SMS escalation option hit a hard
# external signup/paywall blocker, not something fixable in this repo. See
# monitoring/alerts/contact-points.yml's comment for the details. ntfy is
# the only delivery mechanism again.
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

# Delete the throwaway secret files however this script exits, not just on the
# happy path. The cleanup at the bottom was unreachable from any earlier
# failure, so the GRAFANA_ADMIN_PASSWORD bug below (which aborted under
# `set -u` mid-script on every promote) left the Grafana admin password and the
# ntfy webhook URL sitting in plaintext on disk, and leaked one more
# /tmp/monitoring-compose.*.env per run — the exact opposite of the
# fetch-use-delete lifecycle this file exists to implement. COMPOSE_ENV is
# created further down, hence the `:-` guards: the trap is armed before it
# exists and must not itself trip `set -u`.
cleanup_secrets(){ rm -f "${COMPOSE_ENV:-}" "${SECRETS_ENV:-}" 2>/dev/null || true; }
trap cleanup_secrets EXIT

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
  # Also bind the value into THIS shell. The file is for Docker Compose; the
  # Grafana annotation step near the bottom needs the password in the script's
  # own environment, and nothing ever put it there — so `${GRAFANA_ADMIN_PASSWORD}`
  # was an unbound variable and `set -u` killed the script at that line on
  # every single production promote (v1.0.0-rc3, CD run 33037783625:
  # "line 129: GRAFANA_ADMIN_PASSWORD: unbound variable"). `printf -v` assigns
  # by variable name without an eval.
  printf -v "${ENV_KEY}" '%s' "${VAL}"
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

# ── Ensure persistent EBS-backed host paths exist ───────────────────────────
# docker-compose.monitoring.yml mounts these as `driver: local, o: bind`
# volumes (prometheus/grafana/tempo/loki/promtail/phoenix) — a bind-type local volume
# with a non-existent host directory fails `docker compose up` outright, so
# this must run before the first `up` on a fresh box (or after an EBS/AMI
# rebuild that dropped /opt/magik/monitoring).
for d in prometheus grafana tempo loki promtail phoenix; do
  mkdir -p "/opt/magik/monitoring/${d}"
done
log "host volume directories present under /opt/magik/monitoring"

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

# ── Make sure Grafana actually PICKED UP the alerting webhook ────────────────
# `up -d` alone is not enough to trust here. NTFY_WEBHOOK_URL reaches Grafana
# through `env_file:` (docker-compose.monitoring.yml), and this project has
# already been bitten by Compose not re-reading env_file content for a
# container it considers unchanged — a plain restart definitively does not, and
# the v1.0.0-rc3 promote logged "Container magik-grafana Running" (i.e. NOT
# recreated) on the very run that was supposed to roll out a freshly-rotated
# ntfy topic. Grafana expands ${NTFY_WEBHOOK_URL} itself, at runtime, inside
# the mounted monitoring/alerts/contact-points.yml, so a container holding the
# old value leaves all 11 alert rules provisioned-but-undeliverable: silently
# healthy-looking, with nothing actually paging.
#
# So: compare what the running container HAS against what SSM just gave us, and
# recreate only Grafana when they differ. Idempotent — a correct container is
# left alone, so this costs nothing on the common path.
if [ "${RC}" = "0" ]; then
  RUNNING_NTFY="$(docker exec magik-grafana printenv NTFY_WEBHOOK_URL 2>/dev/null || true)"
  if [ "${RUNNING_NTFY}" != "${NTFY_WEBHOOK_URL:-}" ]; then
    log "grafana is holding a stale NTFY_WEBHOOK_URL — forcing a recreate to pick up the SSM value"
    if docker compose --env-file "${COMPOSE_ENV}" -f docker-compose.monitoring.yml \
         up -d --force-recreate grafana; then
      RECHECK="$(docker exec magik-grafana printenv NTFY_WEBHOOK_URL 2>/dev/null || true)"
      if [ "${RECHECK}" = "${NTFY_WEBHOOK_URL:-}" ]; then
        log "grafana recreated; NTFY_WEBHOOK_URL now matches SSM"
      else
        log "WARN: grafana recreated but NTFY_WEBHOOK_URL still does not match SSM — alerts will not deliver"
      fi
    else
      log "WARN: grafana force-recreate failed — alert delivery may be stale"
    fi
  else
    log "grafana already has the current NTFY_WEBHOOK_URL"
  fi
fi

# ── Grafana deployment annotation ────────────────────────────────────────────
# Without this, a step-change on a dashboard (recall/faithfulness/latency)
# can't be told apart from an unrelated incident at a glance — someone has to
# cross-reference git log by hand. This script already runs at every promote
# (cd.yml's "Sync + redeploy monitoring stack" step) with the repo checked
# out to the exact tag just promoted, so the current HEAD here IS the
# deployed version — no new GIT_REF plumbing needed. Grafana's own port,
# not the Caddy-fronted public URL (GF_SERVER_ROOT_URL): this script runs on
# the box itself, and 127.0.0.1:3001 is the same host-loopback mapping
# Prometheus/Pushgateway already use for local, non-Caddy access (see
# docker-compose.monitoring.yml's grafana `ports:`).
#
# Best-effort only — RC (the actual deploy result) is untouched by this
# block's outcome, matching the rest of this script's monitoring-must-never-
# block-the-real-deploy stance (see cd.yml's `continue-on-error: true` on the
# step that calls this whole script).
if [ "${RC}" = "0" ] && [ -n "${GRAFANA_ADMIN_PASSWORD:-}" ]; then
  DEPLOYED_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  DEPLOYED_TAG="$(git describe --tags --exact-match 2>/dev/null || true)"
  ANNOTATION_TEXT="Deploy: ${DEPLOYED_SHA}${DEPLOYED_TAG:+ (${DEPLOYED_TAG})}"
  # `:-` and the -n guard above are belt-and-braces on top of the printf -v
  # binding at the fetch loop: this block is explicitly best-effort, so it must
  # degrade to a skipped annotation, never abort the deploy under `set -u`.
  if curl -fsS -m 5 -u "admin:${GRAFANA_ADMIN_PASSWORD:-}" \
       -X POST "http://127.0.0.1:3001/api/annotations" \
       -H "Content-Type: application/json" \
       -d "{\"text\": \"${ANNOTATION_TEXT}\", \"tags\": [\"deploy\", \"magik\"]}" \
       >/dev/null 2>&1; then
    log "grafana deployment annotation posted (${DEPLOYED_SHA})"
  else
    log "WARN: grafana deployment annotation failed (non-fatal — Grafana may still be starting up)"
  fi
fi

# ── Cleanup — same fetch-use-delete lifecycle as cd.yml's app secrets.
# Compose bakes env_file/environment values into each container at create
# time; deleting the source files afterward does not affect already-running
# containers, only prevents them from lingering as plaintext on disk.
# The actual removal is the EXIT trap armed at the top, so it also covers every
# early-exit path; this call just keeps the ordering explicit and the log line
# on the success path. `rm -f` is idempotent, so running twice is harmless.
cleanup_secrets
log "throwaway env files removed"

exit "${RC}"
