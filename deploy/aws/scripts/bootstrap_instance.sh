#!/usr/bin/env bash
# One-time bootstrap for a freshly Terraform'd MAGIK box (production or
# staging): formats/mounts the dedicated model EBS volume, creates the
# /opt/magik directory tree, and satisfies cd.yml's deploy-time preflight
# check (which fails fast if /opt/magik/{.env,.hf_cache,data,logs} don't all
# already exist — see deploy-staging's/promote-production's remote script in
# cd.yml).
#
# Run ON THE BOX as root (via SSM Session Manager, or SSH using the key
# Terraform generated at deploy/aws/terraform/magik-admin-key.pem):
#
#     sudo bash bootstrap_instance.sh production   # or: staging
#
# Idempotent: safe to re-run. Does NOT touch an already-formatted/mounted
# model volume (detected via blkid) — this matters for staging, whose model
# volume is cloned from a snapshot and already has ext4 + populated data; only
# a genuinely blank volume (production's first boot) gets mkfs'd.
#
# What this script deliberately does NOT do (separate, later steps — see
# deploy/aws/terraform/README output and docs/runbooks/ci-cd.md):
#   - Download HF models into .hf_cache (~20GB, needs GHCR auth + the pulled
#     app image first)
#   - Rebuild the BM25 index (needs Qdrant reachable + a populated .env)
#   - Register the self-hosted GitHub Actions runner
#   - Write real secrets into .env (cd.yml fetches those fresh from SSM at
#     every deploy into a throwaway file — this script only guarantees the
#     preflight-checked path exists, per the same "fetch-use-delete, never
#     persisted plaintext on disk" convention used everywhere else in this repo)

set -euo pipefail

log(){ echo "[bootstrap] $*"; }

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root: sudo bash $0 <production|staging>" >&2
  exit 1
fi

ENVIRONMENT="${1:-}"
case "${ENVIRONMENT}" in
  production|staging) ;;
  *) echo "Usage: sudo bash $0 <production|staging>" >&2; exit 1 ;;
esac
log "bootstrapping for: ${ENVIRONMENT}"

# ── Find the dedicated model volume ─────────────────────────────────────────
# Nitro-based instances (g6e.xlarge included) expose EBS volumes as NVMe
# devices regardless of the device name given at attach time (Terraform
# requested /dev/sdf, but the OS sees /dev/nvme*n1). CRITICAL: g6e.xlarge (and
# other instance families with local NVMe storage) can have a THIRD nvme
# device besides root: the instance's own local/ephemeral instance-store SSD,
# already formatted and mounted by the AMI's boot scripts (on this DLAMI:
# an LVM PV, mounted at /opt/dlami/nvme). That device is WIPED on every
# stop/start — mounting our persistent model cache on it would silently
# defeat the entire point of this rebuild. A naive "just pick the other
# nvme*n1 device" heuristic can grab exactly that device by accident, so
# instead we explicitly exclude (a) the root device and (b) any device
# already in use as an LVM physical volume or otherwise already mounted
# anywhere, and require EXACTLY ONE unambiguous candidate to remain.
ROOT_PARENT="$(lsblk -no PKNAME "$(findmnt -no SOURCE /)" 2>/dev/null | head -n1)"
if [ -z "${ROOT_PARENT}" ]; then
  # Fallback for filesystems where findmnt reports the special "/dev/root"
  # alias that lsblk --pkname can't resolve directly: find whichever disk
  # has a partition mounted at "/".
  ROOT_PARENT="$(lsblk -rno NAME,MOUNTPOINT,PKNAME | awk '$2=="/"{print $3; exit}')"
fi
log "root device (excluded): ${ROOT_PARENT:-<not found>}"

ALREADY_USED_PVS="$(pvs --noheadings -o pv_name 2>/dev/null | tr -d ' ' || true)"

CANDIDATES=()
for dev in /dev/nvme*n1; do
  [ -e "${dev}" ] || continue
  name="$(basename "${dev}")"
  [ "${name}" = "${ROOT_PARENT}" ] && continue
  echo "${ALREADY_USED_PVS}" | grep -qx "${dev}" && continue
  # Also exclude anything with an existing mountpoint anywhere below it.
  if lsblk -no MOUNTPOINT "${dev}" 2>/dev/null | grep -q '.'; then
    continue
  fi
  CANDIDATES+=("${dev}")
done

if [ "${#CANDIDATES[@]}" -ne 1 ]; then
  echo "FATAL: expected exactly ONE unambiguous model-volume candidate," >&2
  echo "found ${#CANDIDATES[@]}: ${CANDIDATES[*]:-<none>}." >&2
  echo "Root device excluded: ${ROOT_PARENT:-<not found>}; LVM PVs excluded: ${ALREADY_USED_PVS:-<none>}." >&2
  echo "Inspect manually with: lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,PKNAME" >&2
  exit 1
fi
MODEL_DEV="${CANDIDATES[0]}"
log "model volume device: ${MODEL_DEV}"

# ── Format only if blank (never reformat a snapshot-cloned volume) ─────────
FSTYPE="$(blkid -o value -s TYPE "${MODEL_DEV}" 2>/dev/null || true)"
if [ -z "${FSTYPE}" ]; then
  log "no filesystem on ${MODEL_DEV} — formatting ext4 (first boot, blank volume)"
  mkfs.ext4 -F -L magik-models "${MODEL_DEV}"
else
  log "${MODEL_DEV} already has a filesystem (${FSTYPE}) — leaving data intact (snapshot clone or re-run)"
fi

MODEL_UUID="$(blkid -o value -s UUID "${MODEL_DEV}")"
mkdir -p /opt/magik/.hf_cache

# ── /etc/fstab entry, by UUID (not /dev/nvme1n1 — device names are not
# guaranteed stable across reboots on Nitro instances) ─────────────────────
if ! grep -q "${MODEL_UUID}" /etc/fstab 2>/dev/null; then
  echo "UUID=${MODEL_UUID}  /opt/magik/.hf_cache  ext4  defaults,nofail  0  2" >> /etc/fstab
  log "added /etc/fstab entry for ${MODEL_UUID}"
else
  log "/etc/fstab already has an entry for ${MODEL_UUID}"
fi

mountpoint -q /opt/magik/.hf_cache || mount /opt/magik/.hf_cache
log "mounted: $(findmnt -no SOURCE,FSTYPE,SIZE /opt/magik/.hf_cache)"

# ── Rest of /opt/magik (root volume — OS, Docker, git checkout, this
# directory tree — deliberately NOT the model volume, per the explicit
# models-vs-code split this rebuild uses) ──────────────────────────────────
mkdir -p /opt/magik/data /opt/magik/logs

# The app container runs as a non-root user (Dockerfile: `appuser`,
# uid/gid 10001 — both runtime and dev-runtime stages) for defense in depth.
# Bind-mounted host directories keep host ownership inside the container, so
# root:root (the default right after mkdir, running this script as root)
# leaves the container unable to write its own model cache/logs/data —
# first discovered the hard way on the very first post-rebuild deploy
# (PermissionError on /app/.hf_cache/gguf, /app/logs/llama_server.log, every
# model download failing). chown once here, up front, so it's never a
# surprise again.
chown -R 10001:10001 /opt/magik/.hf_cache /opt/magik/data /opt/magik/logs

# Preflight in cd.yml's remote deploy script only checks existence, not
# content — real secrets are fetched fresh from SSM into a throwaway file at
# every deploy (see cd.yml's app-secrets loop), never persisted here in
# plaintext. This is a placeholder that satisfies that check.
if [ ! -f /opt/magik/.env ]; then
  : > /opt/magik/.env
  chmod 600 /opt/magik/.env
  log "created empty /opt/magik/.env placeholder (real secrets are fetched fresh at deploy time, not stored here)"
else
  log "/opt/magik/.env already exists — left untouched"
fi

log "done. Directory tree:"
ls -la /opt/magik

echo
log "NEXT STEPS (not automated by this script):"
log "  1. Authenticate Docker to GHCR (needs /magik/ghcr_pat from SSM):"
log "       aws ssm get-parameter --name /magik/ghcr_pat --with-decryption --query Parameter.Value --output text | docker login ghcr.io -u <gh-username> --password-stdin"
log "  2. Pull the app image once, then run the model-download entrypoint against it (see docs/runbooks/ci-cd.md)."
log "  3. Register the self-hosted GitHub Actions runner (label 'gpu' for production, 'staging-gpu' for staging)."
log "  4. Production only: after models are downloaded, snapshot this volume (aws ec2 create-snapshot --volume-id <prod_models_volume_id>) and feed the snapshot ID into staging_model_snapshot_id + create_staging=true in terraform.tfvars."
