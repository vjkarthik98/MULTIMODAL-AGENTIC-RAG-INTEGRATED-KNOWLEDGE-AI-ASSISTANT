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
    """Load models one at a time in priority order with OOM guards between each.

    Parallel loading 12 GPU models simultaneously causes CUDA OOM during
    peak allocation (loading buffers + target tensors coexist briefly).
    Sequential loading is slower but reliable on any VRAM budget.
    """
    # Priority order: highest-priority / most-critical models first so the
    # app is usable for text queries while vision/audio models are still loading.
    _PRIORITY: List[str] = [
        "text_embedder",    # embedding backbone — needed by every modality
        "llm",              # GGUF LLM — needed for all answers
        "reranker",         # cross-encoder — needed for every retrieval
        "siglip",           # vision backbone (image_embedder + siglip_text_embedder depend on it)
        "image_embedder",   # depends on siglip being loaded
        "siglip_text_embedder",
        "qwen2_vl",         # video/chart captioning (INT8, ~2.2 GB)
        "trocr",            # OCR for PDF/image (~1.5 GB)
        "whisper",          # audio transcription (1.55 GB)
        "ner",              # NER entity extraction (0.4 GB)
        "finbert",          # finance sentiment (0.4 GB)
        "diarizer",         # speaker diarization (0.6 GB) — last, needs HF token
    ]

    # Only load models that were requested.
    ordered = [m for m in _PRIORITY if m in requested]
    # Any requested model not in the priority list goes at the end.
    ordered += [m for m in requested if m not in _PRIORITY]

    from app.core.model_registry import model_registry
    import gc
    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except Exception:
        _has_cuda = False

    for name in ordered:
        try:
            logger.info(event="model_preload_start", model=name)
            model_registry.ensure([name])
            # OOM guard: flush temporary loading buffers between each model.
            gc.collect()
            if _has_cuda:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(event="gpu_preload_load_failed", model=name, error=str(exc))


def _warmup_cuda_kernels() -> None:
    """
    Run a minimal forward pass on each loaded GPU model so cuBLAS / cuDNN
    kernels are compiled and cached. This eliminates the ~200-500 ms
    first-inference penalty.
    """
    _warmup_text_embedder()
    _warmup_llm()
    _warmup_siglip()
    _warmup_blip()
    _warmup_qwen2_vl()
    _warmup_trocr()
    _warmup_whisper()
    _warmup_ner()
    _warmup_finbert()
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


def _warmup_siglip() -> None:
    if not settings.ENABLE_VISION:
        return
    try:
        from app.core.model_loader import model_loader
        result = model_loader.get_siglip()
        if result is None or result[1] is None:
            return
        processor, model, device = result
        if device != "cuda":
            return
        import torch
        # SigLIP SO400M uses 384×384 input — match the real resolution.
        dummy = torch.zeros(1, 3, 384, 384, device=device, dtype=torch.float16)
        with torch.no_grad():
            model.vision_model(pixel_values=dummy)
        logger.debug(event="cuda_warmup_done", model="siglip")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="siglip", error=str(exc))


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


def _warmup_qwen2_vl() -> None:
    if not settings.ENABLE_VISION:
        return
    try:
        from app.core.model_loader import model_loader
        result = model_loader.get_qwen2_vl()
        if result is None or result[1] is None:
            return
        processor, model, device = result
        if device != "cuda":
            return
        from PIL import Image as _PIL
        import torch
        dummy_img = _PIL.new("RGB", (224, 224), color=128)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": dummy_img},
            {"type": "text", "text": "warmup"},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy_img], return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
        logger.debug(event="cuda_warmup_done", model="qwen2_vl")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="qwen2_vl", error=str(exc))


def _warmup_trocr() -> None:
    if not settings.ENABLE_VISION:
        return
    try:
        from app.core.model_loader import model_loader
        from PIL import Image
        import numpy as np
        result = model_loader.get_trocr()
        if result is None or result[1] is None:
            return
        processor, model, device = result
        dummy_img = Image.fromarray(np.zeros((32, 128, 3), dtype=np.uint8))
        pixel_values = processor(images=dummy_img, return_tensors="pt").pixel_values.to(device)
        with __import__("torch").no_grad():
            model.generate(pixel_values, max_new_tokens=4)
        logger.debug(event="cuda_warmup_done", model="trocr")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="trocr", error=str(exc))


def _warmup_whisper() -> None:
    if not settings.ENABLE_AUDIO:
        return
    try:
        from app.core.model_loader import model_loader
        import tempfile, struct, wave, os
        whisper = model_loader.get_whisper()
        if whisper is None:
            return
        # Generate a 0.1s silent WAV and transcribe it to warm cuDNN kernels.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack("<" + "h" * 1600, *([0] * 1600)))
        try:
            list(whisper.transcribe(wav_path, beam_size=1)[0])
        finally:
            os.unlink(wav_path)
        logger.debug(event="cuda_warmup_done", model="whisper")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="whisper", error=str(exc))


def _warmup_ner() -> None:
    if not settings.ENABLE_AUDIO:
        return
    try:
        from app.core.model_loader import model_loader
        ner = model_loader.get_ner()
        if ner is None:
            return
        _ = ner("Apple reported quarterly earnings.")
        logger.debug(event="cuda_warmup_done", model="ner")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="ner", error=str(exc))


def _warmup_finbert() -> None:
    try:
        from app.core.model_loader import model_loader
        finbert = model_loader.get_finbert()
        if finbert is None:
            return
        _ = finbert("Revenue increased 12% year over year.")
        logger.debug(event="cuda_warmup_done", model="finbert")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="finbert", error=str(exc))


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
