"""
GPU startup optimizer for the Multimodal RAG Assistant.

Responsibility: load all GPU models in parallel using the existing
model_registry, then run a CUDA warm-up pass on each model so the first
real request incurs zero JIT / cuBLAS kernel launch penalty.

Design rules:
  - Does NOT import torch or transformers at module level (keep import cheap).
  - All heavy work runs in threads via model_registry._ensure(), which is
    already thread-safe and parallel-load aware.
  - Failures are logged but never crash the server — lazy loading is the
    fallback for any model that fails here.
  - Called once from main.py lifespan as an asyncio background task.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Dedicated executor so the GPU preload's multi-second blocking work
# (Llama() mmap + GPU layer copy + cuBLAS warmup) does NOT compete with
# user-request executors. Sharing the default pool with
# `asyncio.to_thread(process_file, ...)` was causing uploads issued
# during warmup to queue behind the LLM load and time out at 300 s.
_warmup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu_warmup")


async def preload_gpu_models() -> None:
    """
    Load all configured GPU models concurrently, then warm them.
    Runs as a background task — Uvicorn is already accepting requests.
    """
    requested: List[str] = list(settings.WARMUP_MODELS) or ["text_embedder"]

    logger.info(
        event="gpu_preload_started",
        models=requested,
        profile=settings.MODELS_DEVICE_PROFILE,
        vram_budget_gb=settings.VRAM_BUDGET_GB,
    )

    start = time.time()

    loop = asyncio.get_running_loop()

    # Phase 1 — parallel model loads on the dedicated warmup executor
    await loop.run_in_executor(_warmup_executor, _load_models_parallel, requested)

    load_latency = round(time.time() - start, 2)
    logger.info(event="gpu_models_loaded", latency=load_latency, models=requested)

    # Phase 2 — CUDA warm-up: run a tiny forward pass on each GPU model so
    # cuBLAS kernels are compiled and cached before the first real request.
    await loop.run_in_executor(_warmup_executor, _warmup_cuda_kernels)

    total_latency = round(time.time() - start, 2)
    logger.info(event="gpu_preload_complete", total_latency=total_latency)

    # Phase 3 — log VRAM usage after all models are loaded
    await loop.run_in_executor(_warmup_executor, _log_vram_usage)


def _load_models_parallel(requested: List[str]) -> None:
    try:
        from app.core.model_registry import model_registry
        model_registry.ensure(requested)
    except Exception as exc:
        logger.warning(event="gpu_preload_load_failed", error=str(exc))


def _warmup_cuda_kernels() -> None:
    """
    Run a minimal forward pass on each loaded GPU model so cuBLAS / cuDNN
    kernels are compiled and cached. This eliminates the ~200-500 ms
    first-inference penalty.
    """
    _warmup_text_embedder()
    _warmup_llm()
    _warmup_clip()
    _warmup_blip()
    _warmup_reranker()


def _warmup_text_embedder() -> None:
    if not settings.ENABLE_VISION and not settings.WARMUP_AT_STARTUP:
        return
    try:
        from app.core.model_loader import model_loader
        emb = model_loader.get_embedder()
        if emb is None:
            return
        _ = emb.embed_texts(["warmup"])
        logger.debug(event="cuda_warmup_done", model="text_embedder")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="text_embedder", error=str(exc))


def _warmup_llm() -> None:
    try:
        from app.core.model_loader import model_loader
        llm = model_loader.get_llm()
        if llm is None:
            return
        # Use the GGUFModel's own warmup which runs a 5-token generation
        if hasattr(llm, "warmup"):
            llm.warmup()
        logger.debug(event="cuda_warmup_done", model="llm")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="llm", error=str(exc))


def _warmup_clip() -> None:
    if not settings.ENABLE_VISION:
        return
    try:
        from app.core.model_loader import model_loader
        result = model_loader.get_clip()
        if result is None or result[1] is None:
            return
        processor, model, device = result
        if device != "cuda":
            return
        import torch
        dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float16)
        with torch.no_grad():
            model.vision_model(pixel_values=dummy)
        logger.debug(event="cuda_warmup_done", model="clip")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="clip", error=str(exc))


def _warmup_blip() -> None:
    if not settings.ENABLE_VISION:
        return
    try:
        from app.core.model_loader import model_loader
        result = model_loader.get_blip()
        if result is None or result[1] is None:
            return
        processor, model, device = result
        if device != "cuda":
            return
        import torch
        dummy = torch.zeros(1, 3, 384, 384, device=device, dtype=torch.float16)
        with torch.no_grad():
            model.vision_model(pixel_values=dummy)
        logger.debug(event="cuda_warmup_done", model="blip")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="blip", error=str(exc))


def _warmup_reranker() -> None:
    try:
        from app.core.model_loader import model_loader
        reranker = model_loader.get_reranker()
        if reranker is None:
            return
        # Run one tiny prediction to warm cuBLAS
        _ = reranker.predict([("warmup query", "warmup document")])
        logger.debug(event="cuda_warmup_done", model="reranker")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="reranker", error=str(exc))


def _log_vram_usage() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 2)
        reserved  = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 2)
        total     = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        logger.info(
            event="vram_usage_after_preload",
            allocated_gb=allocated,
            reserved_gb=reserved,
            total_gb=total,
            free_gb=round(total - reserved, 2),
        )
    except Exception as exc:
        logger.warning(event="vram_usage_log_failed", error=str(exc))


def set_cuda_performance_flags() -> None:
    """
    Set PyTorch CUDA performance flags. Call this once before any model loads.
    Benchmarks matmul algorithms and enables TF32 for Ampere/Turing (T4).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        # Enable TF32 for matrix multiplications (Turing/Ampere; T4 supports this)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cuDNN benchmark: profile kernels on first batch and cache the fastest
        torch.backends.cudnn.benchmark = True
        # Deterministic is not needed for inference; disable for speed
        torch.backends.cudnn.deterministic = False
        logger.info(event="cuda_performance_flags_set", tf32=True, benchmark=True)
    except Exception as exc:
        logger.warning(event="cuda_flags_failed", error=str(exc))
