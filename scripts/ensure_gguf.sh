#!/usr/bin/env bash
# Ensure the Mistral GGUF model is present before starting the server.
# Primary location: project models/ dir on persistent EBS volume.
# Fallback download from HuggingFace if somehow missing.
#
# Usage (called automatically by start_server.sh):
#   bash scripts/ensure_gguf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FILENAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEST="${REPO_ROOT}/models/${FILENAME}"

if [[ -f "${DEST}" ]]; then
    SIZE=$(stat -c%s "${DEST}" 2>/dev/null || echo 0)
    if [[ "${SIZE}" -gt 1073741824 ]]; then
        echo "[ensure_gguf] Model present: ${DEST} ($(numfmt --to=iec ${SIZE}))"
        exit 0
    else
        echo "[ensure_gguf] File too small (${SIZE} bytes) — likely corrupt, re-downloading"
        rm -f "${DEST}"
    fi
fi

echo "[ensure_gguf] Model missing at ${DEST}, downloading from HuggingFace..."
mkdir -p "${REPO_ROOT}/models"

# Load HF token from .env if not set in environment
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "${REPO_ROOT}/.env" ]]; then
    HF_TOKEN=$(grep -m1 '^HF_TOKEN=' "${REPO_ROOT}/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ensure_gguf] ERROR: HF_TOKEN not set. Add HF_TOKEN=... to .env"
    exit 1
fi

REPO="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
echo "[ensure_gguf] Downloading: ${URL}"

curl -L --retry 3 --retry-delay 10 --progress-bar \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -o "${DEST}.tmp" \
    "${URL}"

SIZE=$(stat -c%s "${DEST}.tmp" 2>/dev/null || echo 0)
if [[ "${SIZE}" -lt 1073741824 ]]; then
    echo "[ensure_gguf] ERROR: Downloaded file too small (${SIZE} bytes)."
    rm -f "${DEST}.tmp"
    exit 1
fi

mv "${DEST}.tmp" "${DEST}"
echo "[ensure_gguf] Done: ${DEST} ($(numfmt --to=iec ${SIZE}))"
