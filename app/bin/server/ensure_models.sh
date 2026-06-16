#!/usr/bin/env bash
# Ensure ALL models are present in the project's permanent .hf_cache folder.
# Models live on the EBS root volume and survive instance stop/start.
# On a fresh clone (no .hf_cache yet) they are downloaded once from HuggingFace.
#
# Called automatically by start_server.sh on every boot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

# Parse key=value from .env (skip comments and blanks)
_env_val() {
    grep -m1 "^${1}=" "${ENV_FILE}" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'"
}

HF_TOKEN="${HF_TOKEN:-$(_env_val HF_TOKEN)}"
HF_CACHE="$(_env_val HF_HOME)"
HF_CACHE="${HF_CACHE:-${REPO_ROOT}/.hf_cache}"
GGUF_DEST="$(_env_val LLM_MODEL_PATH)"
GGUF_DEST="${GGUF_DEST:-${HF_CACHE}/gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf}"

log() { echo "[ensure_models] $*"; }

mkdir -p "${HF_CACHE}"
mkdir -p "$(dirname "${GGUF_DEST}")"

# ── 1. GGUF ──────────────────────────────────────────────────────────────────
GGUF_FILE="$(basename "${GGUF_DEST}")"
if [[ -f "${GGUF_DEST}" ]] && [[ $(stat -c%s "${GGUF_DEST}") -gt 1073741824 ]]; then
    log "GGUF present: ${GGUF_DEST}"
else
    log "GGUF missing — downloading ${GGUF_FILE}..."
    [[ -z "${HF_TOKEN}" ]] && { log "ERROR: HF_TOKEN not set"; exit 1; }
    curl -L --retry 3 --retry-delay 10 --progress-bar \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        -o "${GGUF_DEST}.tmp" \
        "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/${GGUF_FILE}"
    SIZE=$(stat -c%s "${GGUF_DEST}.tmp" 2>/dev/null || echo 0)
    [[ "${SIZE}" -lt 1073741824 ]] && { log "ERROR: download too small (${SIZE} bytes)"; rm -f "${GGUF_DEST}.tmp"; exit 1; }
    mv "${GGUF_DEST}.tmp" "${GGUF_DEST}"
    log "GGUF ready: $(numfmt --to=iec ${SIZE})"
fi

# ── 2. HuggingFace models via Python ─────────────────────────────────────────
source "${REPO_ROOT}/rag_env/bin/activate"

HF_TOKEN="${HF_TOKEN}" python3 "${REPO_ROOT}/app/bin/models/download_all_models.py"

log "All models ready."
