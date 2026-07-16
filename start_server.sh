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

# ── SWAP: absorb the host-RAM spike during model loading ──────────────────────
# This box has 15 GiB RAM and NO swap by default. Loading ~17 GB of models means
# weights are read into host RAM before they land on the GPU; with TensorFlow +
# PyTorch + loading buffers the transient peak exceeds 15 GiB. Without swap the
# kernel OOM-killer reaps processes — including the SSH / VSCode agent, which is
# why the editor disconnects while the EC2 instance keeps running.
#
# The swapfile lives on the fast ephemeral NVMe (/opt/dlami/nvme, ~217 GB free).
# That volume is wiped on every instance stop/start, so we (re)create the swap
# here on every launch. Idempotent: skipped if already active.
SWAPFILE="/opt/dlami/nvme/swapfile"
SWAP_SIZE_GB="${SWAP_SIZE_GB:-32}"
if ! swapon --show 2>/dev/null | grep -q "${SWAPFILE}"; then
    if [ -d "/opt/dlami/nvme" ]; then
        echo "[start_server] Creating ${SWAP_SIZE_GB}G swap at ${SWAPFILE}..."
        sudo rm -f "${SWAPFILE}" 2>/dev/null || true
        sudo fallocate -l "${SWAP_SIZE_GB}G" "${SWAPFILE}" \
            && sudo chmod 600 "${SWAPFILE}" \
            && sudo mkswap "${SWAPFILE}" >/dev/null \
            && sudo swapon "${SWAPFILE}" \
            && echo "[start_server] Swap active." \
            || echo "[start_server] WARN: swap setup failed — startup may OOM."
    else
        echo "[start_server] WARN: /opt/dlami/nvme not found — no swap, startup may OOM."
    fi
fi

# Prefer dropping reclaimable file cache (the mmap'd GGUF) over swapping the
# LLM's hot anonymous runtime pages. Default swappiness=60 swaps hot pages too
# eagerly on this RAM-tight box, which thrashes generation. 10 keeps active
# model state resident while still allowing swap to absorb the load-time spike.
sudo sysctl -w vm.swappiness=10 >/dev/null 2>&1 || true

source rag_env/bin/activate

# ── GPU: build LD_LIBRARY_PATH from NVIDIA PyPI packages ─────────────────────
SITE_PKGS="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# CUDA LIBRARY PATH ───────────────────────────────────────────────────────────
# System CUDA toolkit: 12.8 (libcudart.so.12). Driver supports up to 13.2.
# Both PyTorch (torch==2.6.0+cu124) and llama-cpp-python (cu124 wheel) use the
# same nvidia-*-cu12 pip packages — single libcudart.so.12 and libcublas.so.12
# in the process. No dual-runtime conflict, no SIGSEGV on model.encode().
NVIDIA_LIBS=""
for pkg in cublas cuda_runtime cudnn cufft curand cusolver cusparse nvjitlink; do
    lib_dir="$SITE_PKGS/nvidia/$pkg/lib"
    [ -d "$lib_dir" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+${NVIDIA_LIBS}:}${lib_dir}"
done
[ -d "/usr/local/cuda/lib64" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+${NVIDIA_LIBS}:}/usr/local/cuda/lib64"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Pin cache dirs so models and kernel benchmarks survive OS image changes
export HF_HOME="${SCRIPT_DIR}/.hf_cache"
export TORCH_HOME="${SCRIPT_DIR}/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="${SCRIPT_DIR}/.torch_cache/inductor"

# ── HOST-RAM SAVINGS ──────────────────────────────────────────────────────────
# transformers/datasets eagerly import TensorFlow if it's installed, which grabs
# several GB of host RAM at startup before any audio is processed. We run on the
# PyTorch backend only; YAMNet still imports TF lazily (inside try/except) when
# an audio file actually needs it. USE_TF=0 stops the eager import.
export USE_TF=0
export USE_TORCH=1
# If TF does load (YAMNet), don't let it pre-grab all GPU memory or spam logs.
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
# HF tokenizers fork-safety: silence the parallelism warning under uvicorn.
export TOKENIZERS_PARALLELISM=false

# ── Redis: required for job-status tracking (IngestJob polling) ──────────────
if ! redis-cli ping >/dev/null 2>&1; then
    echo "[start_server] Starting Redis..."
    sudo systemctl start redis-server 2>/dev/null || sudo redis-server --daemonize yes 2>/dev/null || true
    sleep 1
    redis-cli ping >/dev/null 2>&1 && echo "[start_server] Redis ready." || echo "[start_server] WARN: Redis not available — upload progress won't update in UI."
fi

# ── Kill any existing uvicorn on the target port ──────────────────────────────
PORT="${PORT:-8000}"
OLD_PID=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [[ -n "${OLD_PID}" ]]; then
    echo "[start_server] Stopping existing process on port ${PORT} (PID ${OLD_PID})..."
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 2
fi

# ── Kill any existing llama-server on its port ────────────────────────────────
LLM_PORT="${LLM_SERVER_PORT:-8081}"
OLD_LLM=$(lsof -ti tcp:"${LLM_PORT}" 2>/dev/null || true)
if [[ -n "${OLD_LLM}" ]]; then
    echo "[start_server] Stopping existing llama-server on port ${LLM_PORT} (PID ${OLD_LLM})..."
    kill "${OLD_LLM}" 2>/dev/null || true
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

# ── HF HUB OFFLINE MODE ────────────────────────────────────────────────────
# Set AFTER ensure_models.sh (which still needs the network to fetch
# anything missing), BEFORE any Python model loading. Every from_pretrained()
# call otherwise does a HEAD request to huggingface.co to check for a newer
# revision even when the model is already fully cached in .hf_cache/ — if
# HF Hub is slow/unreachable, huggingface_hub retries with backoff across
# every auxiliary config file (config.json, processor_config.json,
# preprocessor_config.json, adapter_config.json, ...) and can hang startup
# for minutes instead of falling back to cache. Since this project is
# 100% local/offline by design once models are cached, skip the network
# check entirely. Override with HF_HUB_OFFLINE=0 to force an online check
# (e.g. after adding a NEW model not yet in .hf_cache/).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Reduce CUDA memory fragmentation — must be set before PyTorch imports.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail fast if llama_cpp can't load the CUDA backend
python -c "import llama_cpp; print('[start_server] llama_cpp', llama_cpp.__version__, 'GPU offload:', llama_cpp.llama_supports_gpu_offload())" 2>/dev/null || echo "[start_server] WARN: llama_cpp not importable"

# ── LLAMA-SERVER (separate process, own CUDA context) ─────────────────────────
# llama.cpp must NOT run in the uvicorn process: in-process, its CUDA init
# corrupts PyTorch's CUDA context on worker threads → embedding SIGSEGVs at
# ingest time. Running it as a separate llama-server (own CUDA context) keeps
# the LLM on the GPU (fast) while PyTorch owns CUDA in the main process.
# GGUFModel proxies generate()/stream() to this server over HTTP.
LLM_HOST="${LLM_SERVER_HOST:-127.0.0.1}"
GGUF_PATH="${LLM_MODEL_PATH:-${SCRIPT_DIR}/.hf_cache/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf}"
LLM_CTX="${CONTEXT_MAX_TOKENS:-16384}"
echo "[start_server] Launching llama-server on ${LLM_HOST}:${LLM_PORT} (n_gpu_layers=-1)..."
# LATENCY FLAGS:
#  --flash_attn true   : FlashAttention kernel (A10G/Ampere) — faster attention,
#                        lower VRAM for the KV cache.
#  --cache true        : prompt KV cache — reuses the shared RAG system-prompt
#                        prefix across queries so repeat prompts process faster.
#  --use_mlock false   : the server defaults mlock=True, which pins the ~9GB
#                        GGUF in HOST RAM even though weights live on the GPU —
#                        wasted RAM + swap pressure on this 15GB-RAM box. Off.
#  --n_threads_batch   : all cores for prompt (batch) processing.
nohup python -m llama_cpp.server \
    --model "${GGUF_PATH}" \
    --host "${LLM_HOST}" \
    --port "${LLM_PORT}" \
    --n_gpu_layers -1 \
    --n_ctx "${LLM_CTX}" \
    --n_batch "${LLM_N_BATCH:-512}" \
    --n_threads "${LLM_THREADS:-4}" \
    --n_threads_batch "${LLM_THREADS_BATCH:-4}" \
    --flash_attn "${LLM_FLASH_ATTN:-true}" \
    --cache "${LLM_PROMPT_CACHE:-true}" \
    --use_mlock false \
    > "${SCRIPT_DIR}/logs/llama_server.log" 2>&1 &
LLM_SERVER_PID=$!
echo "[start_server] llama-server PID ${LLM_SERVER_PID}; waiting for readiness..."
# Wait up to ~120s for the model to load onto the GPU.
for _i in $(seq 1 60); do
    if curl -sf "http://${LLM_HOST}:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[start_server] llama-server ready."
        break
    fi
    if ! kill -0 "${LLM_SERVER_PID}" 2>/dev/null; then
        echo "[start_server] WARN: llama-server died during startup — see logs/llama_server.log"
        break
    fi
    sleep 2
done

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
