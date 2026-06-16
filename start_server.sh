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

# ── GPU: build LD_LIBRARY_PATH from NVIDIA PyPI packages ─────────────────────
SITE_PKGS="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NVIDIA_LIBS=""
for pkg in cublas cuda_runtime cudnn cufft curand cusolver cusparse nvjitlink; do
    lib_dir="$SITE_PKGS/nvidia/$pkg/lib"
    [ -d "$lib_dir" ] && NVIDIA_LIBS="$lib_dir:$NVIDIA_LIBS"
done
[ -d "/usr/local/cuda/lib64" ] && NVIDIA_LIBS="/usr/local/cuda/lib64:$NVIDIA_LIBS"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"

# Pin cache dirs so models and kernel benchmarks survive OS image changes
export HF_HOME="${SCRIPT_DIR}/.hf_cache"
export TORCH_HOME="${SCRIPT_DIR}/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="${SCRIPT_DIR}/.torch_cache/inductor"

# ── Kill any existing uvicorn on the target port ──────────────────────────────
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

# ffmpeg + ffprobe are required by video/audio ingestion (video_ingest.py
# resolves them via PATH). Guard them the same way as tesseract so a fresh
# AMI launch can never start the server without working media tooling.
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    echo "[start_server] Installing ffmpeg..."
    sudo apt-get install -y ffmpeg 2>&1 | grep -E "Setting up|already installed|error" || true
fi

echo "[start_server] Ensuring all models are present..."
bash app/bin/server/ensure_models.sh

# Reduce CUDA memory fragmentation — must be set before PyTorch imports.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail fast if llama_cpp can't load the CUDA backend
python -c "import llama_cpp; print('[start_server] llama_cpp', llama_cpp.__version__, 'OK')" 2>/dev/null || echo "[start_server] WARN: llama_cpp not found — CPU inference only"

echo "[start_server] Starting uvicorn..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --timeout-keep-alive 30 \
    --limit-concurrency 64 \
    --backlog 256 \
    --log-level warning \
    "$@"
