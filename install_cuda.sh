#!/usr/bin/env bash
# install_cuda.sh — Create rag_env and install all dependencies with CUDA support.
# Target: AWS g6e.xlarge (NVIDIA L40S 48 GB VRAM, CUDA 13.0 driver, Python 3.12)
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
#   - redis-server / ghostscript / clamav-daemon / docker-compose-plugin are
#     system packages, never pip installs — requirements.txt/pyproject.toml
#     can't provide them, so they're provisioned here instead (2026-08-07:
#     confirmed missing after a fresh SSH-only rag_env setup skipped this
#     script's system-package layer entirely and only ran `pip install -r
#     requirements.txt`).
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
# tesseract/ffmpeg = OCR/media fallback binaries (see requirements.txt).
# ghostscript      = camelot-py[cv]'s lattice-flavor table extraction shells out
#                     to `gs` on PATH — silently degrades to pdfplumber-only
#                     tables without it (guarded by try/except, so this is easy
#                     to miss until a lattice-table PDF actually needs it).
# redis-server     = settings.LOCAL_CACHE_HOST/PORT default to localhost:6379
#                     (job-status polling, embedding cache, rate limits) — this
#                     is separate from the Upstash REST client used for
#                     long-term conversation memory. docker-compose.yml runs
#                     this as its own container; on the bare-metal rag_env path
#                     (this script) nothing else provides it.
# clamav-daemon    = the pip packages `clamd`/`pyclamd` (requirements.txt) are
#                     only *clients* — they talk to a running clamd process,
#                     they don't ship the antivirus engine itself. Required by
#                     three independent scan call sites: app/ingestion/router.py
#                     + app/pipeline/ingestion_pipeline.py (TCP socket,
#                     settings.CLAMAV_HOST/PORT, default localhost:3310) and
#                     app/api/api_routes.py (Unix socket, default
#                     /var/run/clamav/clamd.ctl). Both transports are set up
#                     below since both are live call sites.
MISSING_APT=()
command -v tesseract   &>/dev/null || MISSING_APT+=("tesseract-ocr tesseract-ocr-eng")
command -v ffmpeg      &>/dev/null || MISSING_APT+=("ffmpeg")
command -v gs           &>/dev/null || MISSING_APT+=("ghostscript")
command -v redis-server &>/dev/null || MISSING_APT+=("redis-server")
dpkg -s clamav-daemon   &>/dev/null || MISSING_APT+=("clamav clamav-daemon")

echo "[install_cuda] Checking system packages..."

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

# ── redis-server: enable + start (binds 127.0.0.1:6379 by default, matching
# settings.LOCAL_CACHE_HOST default) ──────────────────────────────────────────
if command -v redis-server &>/dev/null; then
    sudo systemctl enable --now redis-server &>/dev/null || \
        echo "[install_cuda] WARN: could not enable/start redis-server — check 'systemctl status redis-server'"
fi

# ── clamav-daemon: virus DB fetch (fresh install ships no DB — clamd refuses
# to serve scans without one) + enable both socket transports the app uses ──
if dpkg -s clamav-daemon &>/dev/null; then
    echo "[install_cuda] Updating ClamAV virus database (freshclam)..."
    sudo systemctl stop clamav-freshclam &>/dev/null || true
    sudo freshclam --quiet || echo "[install_cuda] WARN: freshclam failed — clamd will run with a stale/empty DB until it succeeds"
    sudo systemctl enable --now clamav-freshclam &>/dev/null || true

    CLAMD_CONF="/etc/clamav/clamd.conf"
    if [ -f "${CLAMD_CONF}" ]; then
        # LocalSocket (Unix, /var/run/clamav/clamd.ctl) ships enabled by
        # default in the Debian/Ubuntu package — only TCPSocket needs turning on.
        sudo grep -q '^TCPSocket ' "${CLAMD_CONF}" || \
            (sudo sed -i 's/^#TCPSocket .*/TCPSocket 3310/' "${CLAMD_CONF}"; \
             grep -q '^TCPSocket ' "${CLAMD_CONF}" || echo 'TCPSocket 3310' | sudo tee -a "${CLAMD_CONF}" >/dev/null)
        sudo grep -q '^TCPAddr ' "${CLAMD_CONF}" || \
            (sudo sed -i 's/^#TCPAddr .*/TCPAddr 127.0.0.1/' "${CLAMD_CONF}"; \
             grep -q '^TCPAddr ' "${CLAMD_CONF}" || echo 'TCPAddr 127.0.0.1' | sudo tee -a "${CLAMD_CONF}" >/dev/null)
    fi
    sudo systemctl enable --now clamav-daemon &>/dev/null || \
        echo "[install_cuda] WARN: could not enable/start clamav-daemon — check 'systemctl status clamav-daemon'"
    echo "[install_cuda] ClamAV ready (Unix socket /var/run/clamav/clamd.ctl + TCP 127.0.0.1:3310)."
    echo "[install_cuda] NOTE: scanning is still gated by CLAMAV_ENABLED / MALWARE_SCAN_ENABLED in .env — the daemon being installed does not turn it on."
fi

# ── docker compose plugin: docker-compose.yml / docker-compose.monitoring.yml
# and the Makefile's local-mode targets all invoke `docker compose` (v2 CLI
# plugin), not the deprecated standalone `docker-compose` (v1, Python-based)
# binary. AWS Deep Learning AMIs ship Docker itself already; this only fills
# in the compose plugin (and Docker Engine, on a non-DLAMI box) if missing. ──
if ! docker compose version &>/dev/null; then
    echo "[install_cuda] Installing Docker Engine + compose plugin..."
    if ! command -v docker &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y docker.io 2>&1 | grep -E "Setting up|already installed|error" || true
        sudo usermod -aG docker "$(whoami)" || true
        echo "[install_cuda] NOTE: added $(whoami) to the 'docker' group — log out/in (or 'newgrp docker') for it to take effect."
    fi
    sudo apt-get update -qq
    sudo apt-get install -y docker-compose-plugin 2>&1 | grep -E "Setting up|already installed|error" || true
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
echo " Start server: python start_server.py"
echo "══════════════════════════════════════════════════════════════"
