# syntax=docker/dockerfile:1
#
# MAGIK image — multi-target build, select explicitly with `docker build --target <name>`
# (Docker's implicit "last stage" default is NOT relied on here):
#   `runtime`      — production, CUDA 12.8 + GPU-compiled llama-cpp-python,
#                   matches the validated AWS g6e.xlarge (L40S, 48GB) setup in
#                   install_cuda.sh. Used by `make docker-build` and cd.yml.
#   `dev-runtime`  — local contributor use via docker-compose: CPU-only, no CUDA base
#                   image, no from-source compile — fast to build, matches the default
#                   PyPI llama-cpp-python wheel's documented CPU (BLAS) behavior.
#
# Models are NEVER baked into either target (~17GB+ of resident weights per
# CLAUDE.md's model catalog). .hf_cache/ is always a volume mount.
# app/core/startup_validator.py refuses to serve traffic if the mounted
# cache's manifest is incomplete, so a missing/wrong mount fails loudly.
#
# Building the CUDA wheel is a COMPILE-time step (nvcc cross-compiles for
# sm_89) — it does not require a physical GPU, so `runtime` builds fine on
# ordinary GitHub-hosted runners. Only *running* it with real GPU inference
# needs an actual NVIDIA GPU + nvidia-container-toolkit.

# ---------------------------------------------------------------------------
# Stage 0 — base-deps: shared foundation (system packages, venv, base deps)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base-deps

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential cmake git \
        tesseract-ocr tesseract-ocr-eng ffmpeg libmagic1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements.txt pyproject.toml ./
# Default PyPI llama-cpp-python wheel here is CPU-only (BLAS) — correct
# as-is for dev-runtime; cuda-builder overwrites it below for `runtime`.
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 1a — cuda-builder: GPU-compiled llama-cpp-python (production only)
# ---------------------------------------------------------------------------
FROM base-deps AS cuda-builder

# Flags copied verbatim from requirements.txt's comment block / install_cuda.sh
# (L40S = compute capability 8.9, Ada Lovelace — AWS g6e.xlarge target).
RUN CMAKE_ARGS="-DGGML_CUDA=on -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    CUDA_HOME=/usr/local/cuda \
    pip install "llama-cpp-python==0.3.30" --no-binary llama-cpp-python \
        --force-reinstall --no-cache-dir

# spaCy's Presidio NLP model — small (~587MB) pip-installable package, not
# part of the HF-hub-cached multimodal stack, safe to bake in like any other
# pip dependency (unlike the GB-scale GGUF/embedding/vision models below).
RUN python3.12 -m spacy download en_core_web_lg

# NOTE: a build-time `llama_supports_gpu_offload()` smoke check previously
# lived here, but it doesn't just check compile flags — it probes for an
# actual physical GPU, which GitHub-hosted CI runners never have. That made
# this assertion fail unconditionally in CI regardless of build correctness
# (confirmed: cd.yml build-push failed here with exit code 1 on a build that
# used the exact same CMAKE_ARGS verified working on the real g6e.xlarge box).
# The pip install above already fails loudly on a broken CUDA toolkit/path;
# real GPU-offload verification happens where a GPU actually exists — on the
# box itself (install_cuda.sh's own check, and the deploy health check).

# ---------------------------------------------------------------------------
# Stage 2 — runtime (default target): slim CUDA image, production
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

# Build-time identity, surfaced at runtime by GET /version (Phase 29 §A —
# lets a running instance prove exactly what it's serving without shell/log
# access). Defaults to "unknown" for anyone building locally without
# --build-arg — cd.yml passes the real values on every tagged build.
ARG GIT_SHA=unknown
ARG IMAGE_TAG=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/app/.hf_cache \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    HOST=0.0.0.0 \
    PORT=8000 \
    GIT_SHA=${GIT_SHA} \
    IMAGE_TAG=${IMAGE_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        tesseract-ocr tesseract-ocr-eng ffmpeg libmagic1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --shell /bin/bash --create-home appuser

COPY --from=cuda-builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ app/
COPY start_server.py .
COPY pyproject.toml .

RUN mkdir -p /app/.hf_cache /app/data /app/logs \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Foreground process (uvicorn --workers 1, blocking) — correct as PID 1 for
# a container. NOTE: start_server.py's llama-server child-process cleanup is
# registered via atexit, which is not guaranteed to run on SIGTERM (Docker's
# default `docker stop` signal) — see docs/runbooks/ci-cd.md for the known
# gap. Not patched here: start_server.py is owned by the app, not this layer.
STOPSIGNAL SIGTERM
CMD ["python3.12", "start_server.py"]

# ---------------------------------------------------------------------------
# Stage 2-dev — dev-runtime: CPU-only, no CUDA base image, fast local builds.
# Target of docker-compose.yml. Same app code path as `runtime`; llama.cpp
# and every other model fall back to CPU automatically (device_manager.py
# auto-detects — no code or config difference needed between targets).
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS dev-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/app/.hf_cache \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng ffmpeg libmagic1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --shell /bin/bash --create-home appuser

COPY --from=base-deps /opt/venv /opt/venv

WORKDIR /app
COPY app/ app/
COPY start_server.py .
COPY pyproject.toml .

RUN mkdir -p /app/.hf_cache /app/data /app/logs \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

STOPSIGNAL SIGTERM
CMD ["python3", "start_server.py"]
