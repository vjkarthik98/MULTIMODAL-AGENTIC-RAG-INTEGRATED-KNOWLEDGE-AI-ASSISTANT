#!/usr/bin/env bash
# =============================================================================
# start_server.sh — GPU-enabled Uvicorn launcher
#
# Sets LD_LIBRARY_PATH to include the CUDA 12 runtime libs installed by pip
# (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12) before starting Uvicorn so
# llama-cpp-python can load libggml-cuda.so and run Mistral on the GPU.
#
# Usage:
#   bash scripts/start_server.sh
#   bash scripts/start_server.sh --workers 2   (multi-worker; model loads once per worker)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtualenv if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$PROJECT_ROOT/rag_env/bin/activate"
fi

# Locate NVIDIA PyPI lib dirs (installed by nvidia-cublas-cu12 etc.)
SITE_PKGS="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NVIDIA_LIBS=""
for pkg in cublas cuda_runtime cudnn cufft curand cusolver cusparse nvjitlink; do
    lib_dir="$SITE_PKGS/nvidia/$pkg/lib"
    if [ -d "$lib_dir" ]; then
        NVIDIA_LIBS="$lib_dir:$NVIDIA_LIBS"
    fi
done

# Add system CUDA lib dir
CUDA_SYS_LIB="/usr/local/cuda/lib64"
[ -d "$CUDA_SYS_LIB" ] && NVIDIA_LIBS="$CUDA_SYS_LIB:$NVIDIA_LIBS"

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"

# Pin HuggingFace cache to project-local dir so it survives OS image changes
export HF_HOME="${PROJECT_ROOT}/.hf_cache"

# Pin PyTorch / cuDNN kernel caches so benchmark results persist across restarts
export TORCH_HOME="${PROJECT_ROOT}/.torch_cache"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_ROOT}/.torch_cache/inductor"

echo "=== Multimodal RAG — GPU Server ==="
echo "LD_LIBRARY_PATH set with $(echo "$NVIDIA_LIBS" | tr ':' '\n' | grep -c .) NVIDIA lib dirs"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU info unavailable')"
echo ""

# Verify llama_cpp imports (fail fast before uvicorn starts)
python -c "import llama_cpp; print('llama_cpp', llama_cpp.__version__, 'OK')"

cd "$PROJECT_ROOT"

exec uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --log-level warning \
    "$@"
