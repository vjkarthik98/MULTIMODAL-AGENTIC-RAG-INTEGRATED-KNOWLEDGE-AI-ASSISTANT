#!/usr/bin/env bash
# install_cuda.sh — Create rag_env and install all dependencies with CUDA support.
# Target: AWS g5.xlarge (NVIDIA A10G 24 GB VRAM, CUDA 13.0 driver, Python 3.12)
#
# Key findings from installation:
#   - Python 3.12 (system) is required — numba does not support Python 3.13 yet
#   - PyTorch cu130 wheels work perfectly (torch 2.12.1+cu130)
#   - llama-cpp-python must be built from source with GGML_CUDA=on;
#     pre-built PyPI wheels are CPU-only for Python 3.12 on this AMI
#   - cuda-toolkit-12-8 (apt) provides a complete cmake-compatible toolkit
#     at /usr/local/cuda-12.8/ for building llama-cpp-python
#   - The Deep Learning AMI has CUDA 13.0 driver — CUDA 12.8 toolkit builds
#     are fully compatible (driver backward-compat)
#
# Usage:
#   bash install_cuda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_TOOLKIT_APT="${CUDA_TOOLKIT_APT:-cuda-toolkit-12-8}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

echo "[install_cuda] Target Python: $($PYTHON_BIN --version)"
echo "[install_cuda] CUDA toolkit: ${CUDA_HOME}"
echo "[install_cuda] PyTorch index: ${TORCH_INDEX_URL}"

# ── System pre-requisites ─────────────────────────────────────────────────────
echo "[install_cuda] Checking system packages..."
MISSING_APT=()
command -v tesseract &>/dev/null || MISSING_APT+=("tesseract-ocr tesseract-ocr-eng")
command -v ffmpeg    &>/dev/null || MISSING_APT+=("ffmpeg")

if [ ! -f "${CUDA_HOME}/bin/nvcc" ]; then
    echo "[install_cuda] Installing CUDA toolkit: ${CUDA_TOOLKIT_APT} ..."
    sudo apt-get update -qq
    sudo apt-get install -y "${CUDA_TOOLKIT_APT}" --fix-missing \
        2>&1 | grep -E "Setting up|Unpacking cuda|already installed" | head -5 || true
fi

if [ "${#MISSING_APT[@]}" -gt 0 ]; then
    sudo apt-get update -qq
    sudo apt-get install -y ${MISSING_APT[*]} 2>&1 | grep -E "Setting up|already installed|error" || true
fi

# Verify nvcc
if [ ! -f "${CUDA_HOME}/bin/nvcc" ]; then
    echo "[install_cuda] ERROR: nvcc not found after apt install at ${CUDA_HOME}/bin/nvcc"
    exit 1
fi
echo "[install_cuda] nvcc: $("${CUDA_HOME}/bin/nvcc" --version | grep release)"

# ── Python virtual environment ────────────────────────────────────────────────
# python3.12-venv may not be pre-installed
if ! $PYTHON_BIN -m venv --help &>/dev/null; then
    echo "[install_cuda] Installing python3.12-venv ..."
    sudo apt-get install -y python3.12-venv python3.12-dev build-essential cmake ninja-build
fi

if [ ! -d "${SCRIPT_DIR}/rag_env" ]; then
    echo "[install_cuda] Creating rag_env ($($PYTHON_BIN --version))..."
    $PYTHON_BIN -m venv "${SCRIPT_DIR}/rag_env"
else
    echo "[install_cuda] rag_env already exists — skipping creation."
fi

source "${SCRIPT_DIR}/rag_env/bin/activate"

echo "[install_cuda] Upgrading pip / setuptools / wheel..."
pip install --upgrade pip setuptools wheel --quiet

# ── PyTorch CUDA 13.0 ─────────────────────────────────────────────────────────
echo "[install_cuda] Installing PyTorch with CUDA 13.0 wheels..."
pip install torch torchvision torchaudio \
    --index-url "${TORCH_INDEX_URL}" \
    --quiet

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — check driver or wheel"
print(f"[install_cuda] PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
EOF

# ── llama-cpp-python with GGML_CUDA via system CUDA 12.8 toolkit ─────────────
echo "[install_cuda] Building llama-cpp-python with GGML_CUDA=on (CUDA ${CUDA_HOME})..."
export CUDA_HOME CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}" \
FORCE_CMAKE=1 \
    pip install "llama-cpp-python>=0.2.60" \
        --upgrade \
        --force-reinstall \
        --no-cache-dir

python - <<'EOF'
import llama_cpp
print(f"[install_cuda] llama_cpp {llama_cpp.__version__} | GPU offload: {llama_cpp.llama_supports_gpu_offload()}")
EOF

# ── Remaining Python requirements ─────────────────────────────────────────────
echo "[install_cuda] Installing requirements.txt (torch already installed — skipped)..."
pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet

# ── spaCy model ───────────────────────────────────────────────────────────────
echo "[install_cuda] Downloading spaCy en_core_web_lg..."
python -m spacy download en_core_web_lg --quiet || \
    echo "[install_cuda] WARN: spaCy model failed — run manually: python -m spacy download en_core_web_lg"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo " rag_env installation complete."
echo " Activate:    source rag_env/bin/activate"
echo " Start server: bash start_server.sh"
echo "══════════════════════════════════════════════════════════════"
