#!/usr/bin/env bash
# One-shot startup: ensure ALL models exist on nvme, then start uvicorn.
# /opt/dlami/nvme is ephemeral — wiped on every instance stop/start.
# This script re-downloads anything missing before uvicorn loads them.
#
# Usage:
#   bash start_server.sh
#   bash start_server.sh --port 8080

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source rag_env/bin/activate

# Kill any existing uvicorn on the target port so we don't get "address already in use"
PORT="${PORT:-8000}"
OLD_PID=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [[ -n "${OLD_PID}" ]]; then
    echo "[start_server] Stopping existing process on port ${PORT} (PID ${OLD_PID})..."
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 2
fi

# Ensure system dependencies are installed (apt packages survive instance stop
# only if they were installed on the EBS root volume, which they are — but
# this guard makes the script safe to re-run after a fresh AMI launch too).
if ! command -v tesseract &>/dev/null; then
    echo "[start_server] Installing tesseract-ocr..."
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng 2>&1 | grep -E "Setting up|already installed|error" || true
fi

echo "[start_server] Ensuring all models are present on nvme..."
bash scripts/ensure_models.sh

echo "[start_server] Starting uvicorn..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --log-level warning \
    "$@"
