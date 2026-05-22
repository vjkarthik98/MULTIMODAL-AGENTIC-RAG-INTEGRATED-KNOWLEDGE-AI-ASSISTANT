#!/usr/bin/env bash
# =============================================================================
# install_cuda_deps.sh
# Installs the CUDA-enabled llama-cpp-python wheel and configures the shared
# library paths needed to run Mistral 7B fully on the Tesla T4 GPU.
#
# Usage:
#   bash scripts/install_cuda_deps.sh
#
# Requirements:
#   - NVIDIA Tesla T4 (or any Turing/Ampere GPU) with driver ≥ 525
#   - CUDA toolkit ≥ 12.x on PATH (nvcc --version to check)
#   - Active virtualenv (rag_env)
#
# What this script does:
#   1. Installs llama-cpp-python pre-built cu124 wheel (CUDA 12.4, ~1.4 GB)
#   2. Adds CUDA 12 runtime libs (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12)
#   3. Registers the NVIDIA PyPI lib paths via ldconfig so llama_cpp can load them
#   4. Verifies the install with a quick import test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Multimodal RAG — CUDA dependency installer ==="
echo "Project root : $PROJECT_ROOT"
echo "Python       : $(python --version 2>&1)"
echo "CUDA         : $(nvcc --version 2>/dev/null | grep release | head -1 || echo 'nvcc not found')"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
echo ""

# --- Step 1: llama-cpp-python with CUDA 12.4 cuBLAS ---
echo "[1/4] Installing llama-cpp-python (CUDA 12.4 wheel)..."
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    --no-cache-dir \
    --force-reinstall
echo "      Done."

# --- Step 2: NVIDIA CUDA 12 runtime libs from PyPI ---
echo "[2/4] Installing NVIDIA CUDA 12 runtime libs..."
pip install \
    nvidia-cublas-cu12 \
    nvidia-cuda-runtime-cu12 \
    --no-cache-dir
echo "      Done."

# --- Step 3: Register lib paths so the CUDA wheel can find them ---
echo "[3/4] Registering NVIDIA lib paths in ldconfig..."
SITE_PKGS="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUBLAS_LIB="$SITE_PKGS/nvidia/cublas/lib"
CUDART_LIB="$SITE_PKGS/nvidia/cuda_runtime/lib"

if [ -d "$CUBLAS_LIB" ]; then
    echo "$CUBLAS_LIB" | sudo tee /etc/ld.so.conf.d/nvidia-cublas-cu12.conf > /dev/null
    echo "      Registered: $CUBLAS_LIB"
fi
if [ -d "$CUDART_LIB" ]; then
    echo "$CUDART_LIB" | sudo tee /etc/ld.so.conf.d/nvidia-cuda-runtime-cu12.conf > /dev/null
    echo "      Registered: $CUDART_LIB"
fi

# Also register the system CUDA lib dir if present
CUDA_SYSTEM_LIB="/usr/local/cuda/lib64"
if [ -d "$CUDA_SYSTEM_LIB" ]; then
    echo "$CUDA_SYSTEM_LIB" | sudo tee /etc/ld.so.conf.d/cuda-system.conf > /dev/null
    echo "      Registered: $CUDA_SYSTEM_LIB"
fi

sudo ldconfig
echo "      ldconfig refreshed."

# --- Step 4: Verify ---
echo "[4/4] Verifying llama_cpp imports..."
python -c "
import llama_cpp
print('  llama_cpp version :', llama_cpp.__version__)
import os, ctypes
# Quick check: the CUDA backend .so should be present
import pathlib
lib_dir = pathlib.Path(llama_cpp.__file__).parent / 'lib'
cuda_so = list(lib_dir.glob('libggml-cuda*.so*'))
if cuda_so:
    print('  CUDA backend      : OK —', cuda_so[0].name)
else:
    print('  WARNING: libggml-cuda.so not found — GPU offload will not work')
"

echo ""
echo "=== Installation complete. ==="
echo "Restart Uvicorn to apply GPU settings:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"
echo ""
echo "Expected VRAM usage after preload:"
echo "  Mistral 7B Q4_K_M : ~4.1 GB"
echo "  CLIP ViT-B/32     : ~0.6 GB"
echo "  BLIP base         : ~1.0 GB"
echo "  Whisper large-v3  : ~1.5 GB"
echo "  CrossEncoder      : ~0.1 GB"
echo "  MiniLM embedder   : ~0.09 GB"
echo "  Total             : ~7.4 GB  (14.6 GB available on Tesla T4)"
