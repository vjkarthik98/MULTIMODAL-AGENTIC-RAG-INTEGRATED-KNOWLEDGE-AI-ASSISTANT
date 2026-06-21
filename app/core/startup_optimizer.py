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
import queue as _queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Persistent single-thread executor — never kills its worker thread.
#
# Python 3.9+ ThreadPoolExecutor terminates idle workers after 60 seconds.
# That destroys the PyTorch cuBLAS handle we seed during startup warmup.
# The next upload then creates a FRESH OS thread with no CUDA context →
# SIGSEGV when it tries to init cuBLAS after llama.cpp already owns the
# CUDA device.
#
# This executor runs a blocking queue loop on its one daemon thread so the
# thread lives for the entire process lifetime, keeping the cuBLAS handle
# alive across idle gaps of any length.
# ---------------------------------------------------------------------------

class _PersistentSingleThreadExecutor:
    """Single OS thread that never times out, compatible with asyncio run_in_executor."""

    def __init__(self, thread_name: str) -> None:
        self._q: _queue.SimpleQueue = _queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._loop, name=thread_name, daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while True:
            fut, fn, args = self._q.get()
            try:
                fut.set_result(fn(*args))
            except BaseException as exc:  # noqa: BLE001
                try:
                    fut.set_exception(exc)
                except Exception:
                    pass

    def submit(self, fn, *args) -> Future:
        fut: Future = Future()
        self._q.put((fut, fn, args))
        return fut

    def shutdown(self, wait: bool = True) -> None:  # noqa: D401
        pass  # daemon thread — process exit handles cleanup


# Dedicated executor so the GPU preload's multi-second blocking work
# (Llama() mmap + GPU layer copy + cuBLAS warmup) does NOT compete with
# user-request executors. Sharing the default pool with
# `asyncio.to_thread(process_file, ...)` was causing uploads issued
# during warmup to queue behind the LLM load and time out at 300 s.
_warmup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu_warmup")

# Dedicated PERSISTENT single-thread executor for GPU ingestion (embed + store).
# All CUDA ops during file ingestion run here — the same thread that ran
# PyTorch warmup before llama.cpp initialized its CUBLAS handle.  Running
# PyTorch CUDA from the default asyncio thread pool (a brand-new thread
# that has never touched CUDA) after llama.cpp is initialized causes a
# SIGSEGV because the new thread tries to initialize PyTorch's CUBLAS
# handle on a device whose CUDA context is already owned by llama.cpp.
# Uses _PersistentSingleThreadExecutor (not ThreadPoolExecutor) so the
# seeded thread is never killed by Python's 60-second idle timeout.
_gpu_ingest_executor = _PersistentSingleThreadExecutor("gpu_ingest")

# Set to True once _warmup_ingest_thread() has seeded the ingest executor
# thread with a PyTorch CUDA embed call (before llama.cpp initialises its
# CUBLAS handle).  Upload routes must check this before dispatching to the
# executor — if False they call seed_gpu_ingest_thread() to force a lazy
# seed so CUDA context initialisation order is preserved.
_ingest_thread_seeded: threading.Event = threading.Event()


def is_ingest_thread_ready() -> bool:
    """Return True when the ingest executor thread has been CUDA-seeded."""
    return _ingest_thread_seeded.is_set()


def get_gpu_ingest_executor() -> ThreadPoolExecutor:
    """Return the executor that all GPU ingestion work must run on."""
    return _gpu_ingest_executor


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
            # CUDA CONTEXT FENCE — must run BEFORE the GGUF LLM (llama.cpp) loads.
            #
            # llama.cpp creates its own CUDA context the moment the GGUF model is
            # loaded onto the GPU. If PyTorch's primary CUDA context (and the
            # per-thread cuBLAS handle on the gpu_ingest worker) has NOT been
            # fully established first, the later PyTorch CUDA op on that thread —
            # the embedding call during a file upload — SIGSEGVs.
            #
            # Empirically: a real torch CUDA op + torch.cuda.synchronize() on a
            # thread BEFORE llama.cpp loads lets the two coexist on one GPU.
            # We fence BOTH the main warmup thread and the persistent gpu_ingest
            # thread (where request-time embeds run) so neither creates its first
            # cuBLAS handle after llama.cpp owns the device.
            if name == "llm" and _has_cuda:
                _seed_pytorch_cuda_contexts(torch)

            logger.info(event="model_preload_start", model=name)
            model_registry.ensure([name])
            # OOM guard: flush temporary loading buffers between each model.
            gc.collect()
            if _has_cuda:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as exc:
            logger.warning(event="gpu_preload_load_failed", model=name, error=str(exc))


def _seed_pytorch_cuda_contexts(torch) -> None:
    """Fully initialise PyTorch's CUDA kernels on the main thread AND the
    persistent gpu_ingest worker thread before llama.cpp loads.

    IMPORTANT: a bare matmul only initialises cuBLAS. The embedder's BERT
    forward also uses the SDPA / attention kernel path, which must be
    initialised separately. If a thread's FIRST SDPA call happens after
    llama.cpp owns the CUDA device, it SIGSEGVs (observed in
    transformers masking_utils during BGE encode at request time).

    So we run a REAL embedding (full forward: cuBLAS + SDPA + everything) on
    each thread while llama.cpp is still unloaded. After this, every CUDA
    kernel BGE needs is already compiled/cached on both threads and is simply
    reused once the GGUF model is resident — no crash.
    """
    def _real_forward() -> str:
        # Prefer a true embedder forward (covers SDPA); fall back to matmul.
        try:
            from app.core.model_loader import model_loader
            emb = model_loader.get_embedder()
            if emb is not None:
                emb.embed_texts(["cuda context fence"])
                import torch as _t
                _t.cuda.synchronize()
                return "embed"
        except Exception:
            pass
        import torch as _t
        c = _t.mm(_t.randn(64, 64, device="cuda"), _t.randn(64, 64, device="cuda"))
        _t.cuda.synchronize()
        return f"mm:{float(c.sum()):.1f}"

    # Main warmup thread.
    try:
        _real_forward()
        logger.info(event="cuda_context_fenced", thread="main")
    except Exception as exc:
        logger.warning(event="cuda_context_fence_failed", thread="main", error=str(exc))

    # Persistent gpu_ingest thread — the one that runs request-time embeddings.
    try:
        fut = _gpu_ingest_executor.submit(_real_forward)
        fut.result(timeout=60)
        _ingest_thread_seeded.set()
        logger.info(event="cuda_context_fenced", thread="gpu_ingest")
    except Exception as exc:
        logger.warning(event="cuda_context_fence_failed", thread="gpu_ingest", error=str(exc))


def _warmup_cuda_kernels() -> None:
    """
    Run a minimal forward pass on each loaded GPU model so cuBLAS / cuDNN
    kernels are compiled and cached. This eliminates the ~200-500 ms
    first-inference penalty.

    ORDERING RULE: llama.cpp (GGUF LLM) initialises its own CUBLAS handle
    the first time it runs inference. Running it in the middle of PyTorch CUDA
    ops causes llama.cpp's CUBLAS handle to conflict with PyTorch's handle on
    the shared CUDA device → SIGSEGV. Keep _warmup_llm() last so all PyTorch
    CUDA work is done before llama.cpp touches the device.
    """
    # PyTorch CUDA models first
    _warmup_text_embedder()
    _warmup_siglip()
    _warmup_blip()
    _warmup_qwen2_vl()
    _warmup_trocr()
    _warmup_whisper()
    _warmup_ner()
    _warmup_finbert()
    _warmup_reranker()
    # Guardrail corpus embeddings — uses text_embedder (PyTorch CUDA); must run
    # before the LLM warmup so it doesn't fire PyTorch CUDA after llama.cpp.
    _warmup_guardrails_sync()
    # Pre-initialize PyTorch CUDA in the gpu_ingest executor thread BEFORE the
    # LLM warmup.  This seeds the ingest thread's PyTorch cuBLAS handle so
    # subsequent embedding calls during file uploads don't try to create a new
    # cuBLAS handle after llama.cpp already owns the CUDA device.
    _warmup_ingest_thread()
    # llama.cpp (CUBLAS) LAST — nothing that uses PyTorch CUDA may follow
    _warmup_llm()


def _warmup_text_embedder() -> None:
    if not settings.ENABLE_VISION and not settings.WARMUP_AT_STARTUP:
        return
    try:
        from app.core.model_loader import model_loader
        if model_loader._text_embedder is None:
            return
        emb = model_loader.get_embedder()
        if emb is None:
            return
        # Warm with a batch matching the real ingest micro-batch size so the
        # GPU tensor allocator pre-sizes its workspace. A batch-1 warmup leaves
        # the allocator cold; the first real upload then pays a 2-3s cuBLAS
        # workspace resize penalty on every batch transition.
        warmup_size = max(getattr(settings, "INGESTION_MICRO_BATCH", 16), 16)
        _ = emb.embed_texts(["warmup sentence for gpu allocation"] * warmup_size)
        logger.debug(event="cuda_warmup_done", model="text_embedder", batch=warmup_size)
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="text_embedder", error=str(exc))


def _warmup_ingest_thread() -> None:
    """Pre-initialize PyTorch CUDA in the gpu_ingest executor thread.

    Must be called BEFORE _warmup_llm() so the ingest thread's cuBLAS
    handle is created while llama.cpp is still uninitialised — matching
    the same ordering rule applied to all other PyTorch warmup functions.
    After this call the ingest thread already owns its PyTorch cuBLAS
    handle; subsequent embed calls during file upload won't try to create
    a new handle after llama.cpp has taken the CUDA device.
    """
    try:
        from app.core.model_loader import model_loader
        if model_loader._text_embedder is None:
            return
        emb = model_loader.get_embedder()
        if emb is None:
            return
        # Flush all pending PyTorch CUDA ops in THIS thread before the
        # ingest thread starts — avoids any stream ordering surprises.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        # Run inside the ingest executor so THAT thread gets CUDA initialized.
        # Use the same batch size as real ingestion so GPU workspace is pre-sized.
        warmup_size = max(getattr(settings, "INGESTION_MICRO_BATCH", 16), 16)
        warmup_batch = ["ingest thread cuda init"] * warmup_size
        fut = _gpu_ingest_executor.submit(emb.embed_texts, warmup_batch)
        fut.result(timeout=60)
        # Mark the thread as seeded ONLY after success so upload routes see
        # the correct state and don't skip the lazy-seed path on failure.
        _ingest_thread_seeded.set()
        logger.debug(event="cuda_warmup_done", model="ingest_thread")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="ingest_thread", error=str(exc))


def _warmup_guardrails_sync() -> None:
    """Pre-initialize guardrail subsystems (PII engine + jailbreak corpus embeddings).

    Must run BEFORE _warmup_llm() because jailbreak corpus init calls
    model_loader.get_embedder().embed_texts() which is a PyTorch CUDA op.
    Running any PyTorch CUDA work after llama.cpp initialises its CUBLAS handle
    on the same device causes a SIGSEGV.
    """
    try:
        from app.guardrails import warm_up as guardrail_warm_up
        guardrail_warm_up()
        logger.debug(event="cuda_warmup_done", model="guardrails")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="guardrails", error=str(exc))


def _warmup_llm() -> None:
    try:
        from app.core.model_loader import model_loader
        if model_loader._llm is None:
            return
        # Flush all pending PyTorch CUDA ops before llama.cpp initialises its
        # own CUBLAS handle. Without this barrier the two CUBLAS contexts race
        # on the same device and the process segfaults.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
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
        if model_loader._siglip_model is None:
            return
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
        if model_loader._blip_model is None:
            return
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
        if model_loader._qwen2vl_model is None:
            return
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
        if model_loader._trocr_model is None:
            return
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
        if model_loader._whisper is None:
            return
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
        if model_loader._ner is None:
            return
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
        if model_loader._finbert is None:
            return
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
        if model_loader._reranker is None:
            return
        reranker = model_loader.get_reranker()
        if reranker is None:
            return
        # Run one tiny prediction to warm cuBLAS
        _ = reranker.predict([("warmup query", "warmup document")])
        logger.debug(event="cuda_warmup_done", model="reranker")
    except Exception as exc:
        logger.warning(event="cuda_warmup_failed", model="reranker", error=str(exc))


def seed_gpu_ingest_thread() -> None:
    """Lazily seed the ingest executor thread with a PyTorch CUDA embed call.

    Must be called before the first file upload when WARMUP_AT_STARTUP=False
    or when the startup warmup failed to reach _warmup_ingest_thread().
    Blocks the calling thread until the seed completes (≤60 s).
    Idempotent — safe to call multiple times; returns immediately if already seeded.

    ORDERING CONTRACT: this must be called BEFORE get_llm() triggers
    llama.cpp CUBLAS init.  The upload route gate enforces this by checking
    is_ingest_thread_ready() and calling seed_gpu_ingest_thread() when False,
    which happens on lazy model load paths where the LLM hasn't loaded yet.
    """
    if _ingest_thread_seeded.is_set():
        return

    try:
        from app.core.model_loader import model_loader
        emb = model_loader.get_embedder()
        if emb is None:
            logger.warning(event="lazy_ingest_seed_skipped", reason="embedder_unavailable")
            return
        # Flush any pending PyTorch CUDA ops in the calling thread first.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        fut = _gpu_ingest_executor.submit(emb.embed_texts, ["seed"])
        fut.result(timeout=60)
        _ingest_thread_seeded.set()
        logger.info(event="lazy_ingest_thread_seeded")
    except Exception as exc:
        logger.warning(event="lazy_ingest_seed_failed", error=str(exc))


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
