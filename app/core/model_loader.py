"""
Lazy, device-aware model loader for the Multimodal RAG Assistant.

Design goals (Lightning AI hybrid CPU+GPU deploy):
  * Importing this module is cheap — no transformers / faster_whisper /
    sentence-transformers imports happen here. They are pulled lazily on
    the first call to the relevant getter, which means uvicorn cold-start
    no longer pays the price for models it may never load.
  * Each model decides its device and dtype through device_manager,
    not a single global self._device.
  * GGUF Mistral fully offloads to GPU when CUDA is available
    (n_gpu_layers=-1), giving the largest single latency win on Lightning.
  * SigLIP / BLIP / Whisper run on GPU with float16 (or int8 on CPU Whisper)
    so vision/audio ingestion is fast without crowding the LLM's VRAM.
  * Text embedder + cross-encoder reranker stay on CPU under the default
    hybrid profile — both are tiny and that gives back ~1 GB of VRAM.

The Phase-24 contract is preserved: same getter names, same return types,
same Prometheus metrics, same circuit-breaker semantics.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.device_manager import device_manager


def _is_retryable_load_error(exc: BaseException) -> bool:
    """Retry on timeout or transient RuntimeError, but not on interpreter shutdown."""
    if isinstance(exc, FuturesTimeoutError):
        return True
    if isinstance(exc, RuntimeError):
        return "interpreter shutdown" not in str(exc) and "shutdown" not in str(exc).lower()
    return False


logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


# PROMETHEUS METRICS

_model_load_duration = Histogram(
    "model_load_duration_seconds",
    "Model load duration",
    ["model"],
)
_model_load_errors = Counter(
    "model_load_errors_total",
    "Model load errors by model and error type",
    ["model", "error_type"],
)
_model_loaded = Gauge(
    "model_loaded",
    "Whether a model is loaded (1=yes, 0=no)",
    ["model"],
)
_model_evictions = Counter(
    "model_evictions_total",
    "Evictable models unloaded from VRAM, by model and trigger "
    "(idle=TTL sweep, pressure=free-VRAM watermark)",
    ["model", "reason"],
)
_gpu_vram_free = Gauge(
    "gpu_vram_free_gb",
    "Free VRAM in GB as reported by the CUDA driver (all processes on the device)",
)
# embedding_latency_seconds was defined here too, colliding with
# app/pipeline/ingestion_pipeline.py's identical metric name — removed
# rather than unified, since it was dead code: defined but never once
# observed anywhere in this file. ingestion_pipeline.py is the one real
# owner (see app/core/metrics.py for the shared singleton, part of the
# same duplicate-metric audit as file_ingestion_duration_seconds).


# SEMAPHORE — lazy init to avoid missing event loop at import time

_load_semaphore: asyncio.Semaphore | None = None
_load_semaphore_lock = threading.Lock()


def _get_load_semaphore() -> asyncio.Semaphore:
    global _load_semaphore
    if _load_semaphore is None:
        with _load_semaphore_lock:
            if _load_semaphore is None:
                _load_semaphore = asyncio.Semaphore(1)
    return _load_semaphore


# LAZY TORCH IMPORT — only the loaders that need it pay this cost.


def _torch():
    import torch

    return torch


def _torch_dtype(name: str) -> Any:
    """Translate device_manager dtype string into a torch dtype."""
    torch = _torch()
    table = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return table.get(name or "", torch.float32)


class ModelLoader:

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.THREAD_POOL_SIZE,
            thread_name_prefix="model_loader",
        )

        self._llm: Any | None = None
        self._text_embedder: Any | None = None
        self._siglip_text_embedder: Any | None = None
        self._image_embedder: Any | None = None
        self._multimodal: Any | None = None
        self._whisper: Any | None = None
        self._whisper_infer_lock = threading.Lock()
        self._reranker: Any | None = None
        self._nli: Any | None = None
        self._siglip_model = None
        self._siglip_processor = None
        self._siglip_device: str | None = None
        self._blip_model = None
        self._blip_processor = None
        self._blip_device: str | None = None
        self._qwen2vl_model = None
        self._qwen2vl_processor = None
        self._qwen2vl_device: str | None = None
        # Separate, smaller VLM for VIDEO frame captioning (see get_qwen2_vl_video).
        self._qwen2vl_v_model = None
        self._qwen2vl_v_processor = None
        self._qwen2vl_v_device: str | None = None
        self._trocr_model = None
        self._trocr_processor = None
        self._trocr_device: str | None = None
        self._diarizer: Any | None = None
        self._ner: Any | None = None
        self._finbert: Any | None = None

        # Idle-eviction bookkeeping — see _EVICTABLE_MODELS/unload_idle_models
        # below. Only ingestion-only models are tracked; core query-path
        # models (llm, text_embedder, reranker, siglip and its embedder
        # wrappers — siglip_text_embedder is used for cross-modal query-time
        # search, not just ingestion) are deliberately never evicted.
        self._last_used: dict[str, float] = {}

        self._initialized = False

        logger.info(
            "model_loader_initialized",
            profile=device_manager.profile,
            cuda=device_manager.cuda_available,
            vram_total_gb=device_manager.vram_total_gb,
        )

    # PUBLIC: legacy attribute kept for callers that read it directly.

    @property
    def _device(self) -> str:  # noqa: D401 - thin compatibility shim
        """Compat shim: callers historically read this to choose a device."""
        # The "default" device is the LLM's device — the heaviest workload.
        return device_manager.device_for("llm")

    # IDLE EVICTION — ingestion-only models, never the core query path.
    #
    # Added 2026-08-01: this loader never freed a model once loaded, so a
    # long-running process (a full Tier-2 eval run touches every modality
    # in sequence) only ever accumulates VRAM until something OOMs —
    # confirmed live (nvidia-smi: 44.3GB/46.1GB used, ~1.8GB free, still
    # climbing). Eviction is scoped to models that are ONLY ever touched
    # during ingestion/chunking, never during query serving — confirmed by
    # tracing every caller of each getter below. Notably NOT evictable:
    # llm/text_embedder/reranker (every query needs them) and
    # siglip/image_embedder/siglip_text_embedder/multimodal (SigLIP's text-
    # embedding path is used for cross-modal query-time search, not just
    # ingestion — evicting it would silently reload a multi-second model on
    # a live user's query instead of only during the next ingest).
    _EVICTABLE_MODELS: dict[str, tuple[str, ...]] = {
        "whisper": ("_whisper",),
        "blip": ("_blip_model", "_blip_processor", "_blip_device"),
        "qwen2_vl": ("_qwen2vl_model", "_qwen2vl_processor", "_qwen2vl_device"),
        "qwen2_vl_video": ("_qwen2vl_v_model", "_qwen2vl_v_processor", "_qwen2vl_v_device"),
        "trocr": ("_trocr_model", "_trocr_processor", "_trocr_device"),
        "diarizer": ("_diarizer",),
        "ner": ("_ner",),
        "finbert": ("_finbert",),
    }

    def _touch(self, name: str) -> None:
        """Record that an evictable model was just used — called on every
        get_X() call for models in _EVICTABLE_MODELS, cache hit or not, so
        unload_idle_models() only evicts models that are genuinely idle."""
        self._last_used[name] = time.time()

    def _drop(self, name: str, reason: str) -> None:
        """Release one evictable model's references. Caller must hold _lock.

        Only clears THIS loader's references. A job that already called the
        getter still holds its own reference, so refcounting keeps those
        weights alive until it finishes — dropping here is never unsafe, it
        just means the next getter reloads. app/core/model_reaper.py avoids
        the wasteful version of that by skipping while gpu_busy().
        """
        for attr in self._EVICTABLE_MODELS[name]:
            setattr(self, attr, None)
        self._last_used.pop(name, None)
        _model_loaded.labels(model=name).set(0)
        _model_evictions.labels(model=name, reason=reason).inc()

    def unload_idle_models(self, idle_seconds: float = 300.0) -> list[str]:
        """Free any evictable model that hasn't been used in idle_seconds.

        Safe to call anytime: guarded by the same self._lock every getter
        uses, so this never races an in-flight get_X() call — either this
        runs first and the next getter reloads fresh, or the getter's own
        lock acquisition runs first and this simply finds nothing to evict
        for that model this pass (last_used was just updated).
        """
        evicted: list[str] = []
        with self._lock:
            now = time.time()
            for name, attrs in self._EVICTABLE_MODELS.items():
                if getattr(self, attrs[0]) is None:
                    continue  # not loaded — nothing to evict
                last_used = self._last_used.get(name, 0.0)
                if now - last_used < idle_seconds:
                    continue  # used too recently
                idle_for = round(now - last_used, 1)
                self._drop(name, reason="idle")
                evicted.append(name)
                logger.info("model_evicted_idle", model=name, idle_seconds=idle_for)

        if evicted:
            self._oom_guard()  # actually reclaim the freed VRAM
        return evicted

    def unload_until_free(self, target_free_gb: float) -> list[str]:
        """Pressure valve: evict least-recently-used models until free VRAM
        clears `target_free_gb`, regardless of how recently they were used.

        Complements unload_idle_models(). TTL eviction alone is both too slow
        and too eager: too slow because a model idle for 4 minutes is still
        held while a large ingest is about to OOM, and too eager because on a
        card with plenty of headroom it evicts models nothing was competing
        for, paying a reload cost (~25s for a VLM) to free memory no one
        wanted. Watermark eviction only acts when the memory is actually
        contended, which is the signal that matters. This is the same
        least-recently-used-under-pressure policy KServe ModelMesh applies
        across a model pool.

        LRU order comes from the existing `_last_used` bookkeeping; a loaded
        model with no recorded use sorts oldest and is evicted first.

        Re-checks free VRAM after each eviction so it stops as soon as the
        target is met rather than dumping everything. Returns [] and does
        nothing when free VRAM cannot be read (no CUDA / driver query
        failed) — never evict on a guess.
        """
        free_gb = device_manager.free_vram_gb()
        if free_gb is None:
            return []
        _gpu_vram_free.set(free_gb)
        if free_gb >= target_free_gb:
            return []

        evicted: list[str] = []
        with self._lock:
            loaded = [
                name
                for name, attrs in self._EVICTABLE_MODELS.items()
                if getattr(self, attrs[0]) is not None
            ]
            # Oldest last_used first; never-recorded (0.0) sorts oldest.
            loaded.sort(key=lambda n: self._last_used.get(n, 0.0))

            for name in loaded:
                self._drop(name, reason="pressure")
                evicted.append(name)
                logger.warning(
                    "model_evicted_pressure",
                    model=name,
                    free_vram_gb=round(free_gb, 2),
                    target_free_gb=target_free_gb,
                )
                # Reclaim inside the loop — free_vram_gb() reads the driver,
                # which will not reflect the drop until the allocator has
                # actually released it.
                self._oom_guard()
                now_free = device_manager.free_vram_gb()
                if now_free is not None:
                    free_gb = now_free
                    _gpu_vram_free.set(free_gb)
                    if free_gb >= target_free_gb:
                        break

        return evicted

    def refresh_vram_gauge(self) -> float | None:
        """Publish current free VRAM to Prometheus. Returns the value (or
        None if unavailable) so a caller can reuse it without a second
        driver query."""
        free_gb = device_manager.free_vram_gb()
        if free_gb is not None:
            _gpu_vram_free.set(free_gb)
        return free_gb

    # OOM GUARD — flush CUDA cache + Python GC between model loads (MD spec)

    def _oom_guard(self, loading: str | None = None) -> None:
        """Release CUDA memory and run GC before loading the next model.

        `loading`, when given, both marks that model as just-used (so a
        model that was already cached under some other path never evicts
        itself moments before its own getter returns it) and runs the idle-
        eviction sweep first, so a heavy new load gets first crack at VRAM
        another model has been sitting on unused.
        """
        if loading is not None:
            self._touch(loading)
            self.unload_idle_models()

        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    # SAFE LOAD WITH TIMEOUT AND RETRY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception(_is_retryable_load_error),
        reraise=True,
    )
    def _safe_load(self, load_fn, name: str, device: str | None = None) -> Any:
        start = time.time()
        future = self._executor.submit(load_fn)

        # Use the provided device hint; fall back to device_manager only for
        # known MODEL_NAMES — unknown aliases (image_embedder, siglip_text_embedder)
        # are not in MODEL_NAMES so device_for() would wrongly return "cpu".
        from app.core.device_manager import MODEL_NAMES

        resolved_device = device or (
            device_manager.device_for(name.lower()) if name.lower() in MODEL_NAMES else "unknown"
        )

        with tracer.start_as_current_span(f"load_model_{name}") as span:
            span.set_attribute("model.name", name)
            span.set_attribute("device", resolved_device)

            try:
                obj = future.result(timeout=settings.MODEL_LOAD_TIMEOUT_SEC)
                latency = round(time.time() - start, 2)

                _model_load_duration.labels(model=name).observe(latency)
                _model_loaded.labels(model=name).set(1)

                span.set_attribute("load.latency", latency)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "model_loaded",
                    model=name,
                    device=resolved_device,
                    latency=latency,
                )

                return obj

            except FuturesTimeoutError:
                _model_load_errors.labels(model=name, error_type="timeout").inc()
                _model_loaded.labels(model=name).set(0)
                span.set_status(Status(StatusCode.ERROR, "timeout"))
                logger.error("model_load_timeout", model=name)
                raise

            except Exception as exc:
                _model_load_errors.labels(model=name, error_type=type(exc).__name__).inc()
                _model_loaded.labels(model=name).set(0)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                logger.error("model_load_failed", model=name, error=str(exc))
                raise

    # WARMUP — only runs the explicit list, or stays a no-op when disabled.
    # The default lifespan no longer calls this; per-modality loaders do.

    def warmup(self) -> None:
        if self._initialized:
            logger.info("warmup_skipped")
            return

        if not settings.WARMUP_AT_STARTUP:
            self._initialized = True
            logger.info("warmup_skipped_lazy_mode")
            return

        with self._lock:
            if self._initialized:
                return

            from app.core.model_registry import model_registry  # local import

            requested = list(settings.WARMUP_MODELS) or ["text_embedder"]
            logger.info("warmup_started", models=requested)
            model_registry.ensure(requested)

            self._initialized = True
            logger.info("warmup_completed")

    async def warmup_async(self) -> None:
        if self._initialized:
            return
        async with _get_load_semaphore():
            await asyncio.get_running_loop().run_in_executor(self._executor, self.warmup)

    # LLM — GGUF MODEL

    def get_llm(self) -> Any:
        if self._llm:
            return self._llm

        with self._lock:
            if self._llm:
                return self._llm

            if not settings.LLM_MODEL_PATH:
                raise RuntimeError("LLM_MODEL_PATH missing")

            n_gpu_layers = device_manager.llm_gpu_layers()

            def _load_llm():
                from app.llm.gguf_model import GGUFModel  # local import (loads llama_cpp)

                return GGUFModel(
                    settings.LLM_MODEL_PATH,
                    n_gpu_layers=n_gpu_layers,
                )

            self._llm = self._safe_load(_load_llm, "llm")

        return self._llm

    # TEXT EMBEDDER — SENTENCE TRANSFORMERS

    def get_embedder(self):
        if self._text_embedder:
            return self._text_embedder

        with self._lock:
            if self._text_embedder:
                return self._text_embedder

            decision = device_manager.decision_for("text_embedder")

            def _load():
                from app.embeddings.base_embedder import TextEmbedder  # local

                emb = TextEmbedder(
                    model_name=settings.EMBEDDING_MODEL,
                    batch_size=settings.EMBEDDING_BATCH_SIZE,
                    device=decision.device,
                )
                # Truncate sequences to EMBEDDING_MAX_SEQ_LEN tokens — prevents
                # OOM on oversized chunks without changing model weights.
                try:
                    emb.model.max_seq_length = settings.EMBEDDING_MAX_SEQ_LEN
                except Exception:
                    pass
                # FP16 on CUDA shrinks the model and speeds inference.
                if decision.device == "cuda" and decision.dtype == "float16":
                    try:
                        emb.model.half()
                    except Exception as exc:
                        logger.warning(
                            "text_embedder_fp16_failed",
                            error=str(exc),
                        )
                return emb

            self._text_embedder = self._safe_load(_load, "text_embedder")

        return self._text_embedder

    # Alias kept in case any caller spells it this way.
    def get_text_embedder(self):
        return self.get_embedder()

    # SIGLIP MODEL + PROCESSOR

    def get_siglip(self) -> tuple:
        if self._siglip_model:
            return self._siglip_processor, self._siglip_model, self._siglip_device

        with self._lock:
            if self._siglip_model:
                return self._siglip_processor, self._siglip_model, self._siglip_device

            decision = device_manager.decision_for("siglip")

            def _load():
                from transformers import SiglipModel, SiglipProcessor  # local

                kwargs = {}
                if decision.device == "cuda" and decision.dtype == "float16":
                    kwargs["torch_dtype"] = _torch_dtype("float16")
                processor = SiglipProcessor.from_pretrained(settings.SIGLIP_MODEL)
                model = SiglipModel.from_pretrained(settings.SIGLIP_MODEL, **kwargs)
                model.to(decision.device)
                model.eval()
                return processor, model

            self._siglip_processor, self._siglip_model = self._safe_load(_load, "siglip")
            self._siglip_device = decision.device

        return self._siglip_processor, self._siglip_model, self._siglip_device

    # Backward-compat alias — callers that spelled it get_clip() still work.
    def get_clip(self) -> tuple:
        return self.get_siglip()

    # IMAGE EMBEDDER — SIGLIP VISUAL

    def get_image_embedder(self):
        if self._image_embedder:
            return self._image_embedder

        with self._lock:
            if self._image_embedder:
                return self._image_embedder

            processor, model, device = self.get_siglip()

            def _load():
                from app.embeddings.image_embedder import ImageEmbedder  # local

                return ImageEmbedder(model, processor, device)

            self._image_embedder = self._safe_load(_load, "image_embedder", device=device)

        return self._image_embedder

    # SIGLIP TEXT EMBEDDER — CROSS-MODAL

    def get_siglip_text_embedder(self):
        if self._siglip_text_embedder:
            return self._siglip_text_embedder

        with self._lock:
            if self._siglip_text_embedder:
                return self._siglip_text_embedder

            processor, model, device = self.get_siglip()

            def _load():
                from app.embeddings.image_embedder import ClipTextEmbedder  # local

                return ClipTextEmbedder(processor, model, device)

            self._siglip_text_embedder = self._safe_load(
                _load, "siglip_text_embedder", device=device
            )

        return self._siglip_text_embedder

    # Backward-compat alias.
    def get_clip_text_embedder(self):
        return self.get_siglip_text_embedder()

    # MULTIMODAL EMBEDDER — ORCHESTRATES TEXT + IMAGE

    def get_multimodal_embedder(self):
        if self._multimodal:
            return self._multimodal

        with self._lock:
            if self._multimodal:
                return self._multimodal

            from app.embeddings.base_embedder import MultimodalEmbedder  # local

            self._multimodal = MultimodalEmbedder(
                self.get_embedder(),
                self.get_image_embedder(),
            )

        return self._multimodal

    # WHISPER — ASR

    def get_whisper(self):
        self._touch("whisper")
        if self._whisper:
            return self._whisper

        with self._lock:
            if self._whisper:
                return self._whisper

            self._oom_guard(loading="whisper")
            decision = device_manager.decision_for("whisper")
            device = decision.device if decision.device != "mps" else "cpu"
            compute_type = decision.dtype or ("float16" if device == "cuda" else "int8")

            def _load():
                from faster_whisper import WhisperModel  # local

                return WhisperModel(
                    settings.WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                )

            self._whisper = self._safe_load(_load, "whisper")

        return self._whisper

    def get_whisper_lock(self) -> threading.Lock:
        """Serializes concurrent .transcribe() calls on the shared WhisperModel.

        CTranslate2's CUDA backend crashes with `parallel_for failed:
        cudaErrorInvalidDevice: invalid device ordinal` when two Python threads
        call .transcribe() on the same model instance at once — the GIL is
        released during CUDA ops (true concurrency at the Python level), but the
        underlying device context isn't safe to enter from two threads
        simultaneously. audio_chunker.py and video_chunker.py both run
        ThreadPoolExecutor segments against this one singleton, so the lock must
        live here (shared across both modality files) rather than per-file.
        """
        return self._whisper_infer_lock

    # BLIP — IMAGE CAPTIONING

    def get_blip(self) -> tuple:
        self._touch("blip")
        if self._blip_model:
            return self._blip_processor, self._blip_model, self._blip_device

        with self._lock:
            if self._blip_model:
                return self._blip_processor, self._blip_model, self._blip_device

            self._oom_guard(loading="blip")
            decision = device_manager.decision_for("blip")

            def _load():
                from transformers import (  # local
                    BlipForConditionalGeneration,
                    BlipProcessor,
                )

                processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
                kwargs = {}
                if decision.device == "cuda" and decision.dtype == "float16":
                    kwargs["torch_dtype"] = _torch_dtype("float16")
                model = BlipForConditionalGeneration.from_pretrained(settings.BLIP_MODEL, **kwargs)
                model.to(decision.device)
                model.eval()
                return processor, model

            self._blip_processor, self._blip_model = self._safe_load(_load, "blip")
            self._blip_device = decision.device

        return self._blip_processor, self._blip_model, self._blip_device

    # RERANKER — CROSS ENCODER

    def get_reranker(self):
        if self._reranker:
            return self._reranker

        with self._lock:
            if self._reranker:
                return self._reranker

            decision = device_manager.decision_for("reranker")

            def _load():
                from sentence_transformers import CrossEncoder  # local

                ce = CrossEncoder(
                    settings.RERANKER_MODEL,
                    device=decision.device,
                )
                # Cast to float16 on GPU to halve VRAM and speed inference
                if decision.device == "cuda" and decision.dtype == "float16":
                    try:
                        ce.model.half()
                    except Exception as exc:
                        logger.warning("reranker_fp16_failed", error=str(exc))
                return ce

            self._reranker = self._safe_load(_load, "reranker")

        return self._reranker

    # NLI — GPU ENTAILMENT SCORER (GroundednessChecker, Phase 3 of the
    # hallucination-reduction initiative). Same CrossEncoder loading shape as
    # get_reranker() above — deliberately, since a 3-label entailment/
    # neutral/contradiction scorer is the same class of model as the
    # relevance CrossEncoder, just a different checkpoint and label space.

    def get_nli_model(self):
        if self._nli:
            return self._nli

        with self._lock:
            if self._nli:
                return self._nli

            decision = device_manager.decision_for("nli")

            def _load():
                from sentence_transformers import CrossEncoder  # local

                ce = CrossEncoder(
                    settings.NLI_MODEL,
                    device=decision.device,
                )
                if decision.device == "cuda" and decision.dtype == "float16":
                    try:
                        ce.model.half()
                    except Exception as exc:
                        logger.warning("nli_fp16_failed", error=str(exc))
                return ce

            self._nli = self._safe_load(_load, "nli")

        return self._nli

    # QWEN2-VL — VIDEO FRAME + FINANCIAL CHART CAPTIONING (2B INT8, ~2.2 GB)

    def get_qwen2_vl(self) -> tuple:
        self._touch("qwen2_vl")
        if self._qwen2vl_model:
            return self._qwen2vl_processor, self._qwen2vl_model, self._qwen2vl_device

        with self._lock:
            if self._qwen2vl_model:
                return self._qwen2vl_processor, self._qwen2vl_model, self._qwen2vl_device

            self._oom_guard(loading="qwen2_vl")
            decision = device_manager.decision_for("qwen2_vl")

            def _load():
                from transformers import (  # local
                    AutoProcessor,
                    BitsAndBytesConfig,
                    Qwen2VLForConditionalGeneration,
                )

                processor = AutoProcessor.from_pretrained(
                    settings.QWEN2_VL_MODEL, trust_remote_code=True
                )
                # low_cpu_mem_usage + device_map stream each shard straight to its
                # target device as it's read, instead of the plain from_pretrained()
                # default of materializing the FULL fp16 7B state dict (~15GB) in
                # host CPU RAM first and only then .to(device)-copying it. On the
                # g6e.xlarge's 32GB host RAM, already holding ~10 other resident
                # models, that CPU-side spike was enough to push the box into swap
                # thrashing severe enough to hang the SSH session and even
                # `nvidia-smi` itself — reproduced twice, always stalled at shard
                # 1/5 of this exact load. device_map is skipped for CPU/MPS, where
                # there's no separate CPU-then-GPU copy to avoid.
                load_kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}
                if settings.QWEN2_VL_LOAD_IN_8BIT and decision.device == "cuda":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    # Force float16 compute so bitsandbytes INT8 never receives bfloat16
                    # inputs — the bfloat16→float16 cast inside the CUDA kernel segfaults.
                    load_kwargs["torch_dtype"] = _torch_dtype("float16")
                    load_kwargs["device_map"] = {"": decision.device}
                elif decision.device == "cuda":
                    load_kwargs["torch_dtype"] = _torch_dtype("float16")
                    load_kwargs["device_map"] = {"": decision.device}
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    settings.QWEN2_VL_MODEL, **load_kwargs
                )
                if decision.device != "cuda":
                    model.to(decision.device)
                model.eval()
                return processor, model

            self._qwen2vl_processor, self._qwen2vl_model = self._safe_load(_load, "qwen2_vl")
            self._qwen2vl_device = decision.device

        return self._qwen2vl_processor, self._qwen2vl_model, self._qwen2vl_device

    def get_qwen2_vl_video(self) -> tuple:
        """Smaller Qwen2-VL used for VIDEO frame captioning.

        A one-hour earnings/webcast ingest must load Whisper + pyannote + SigLIP +
        TrOCR + BGE at once, alongside the resident llama-server and query models.
        The 7B INT8 captioner (~8 GB) adds substantial VRAM pressure so that, under a
        tight budget, a subsequent GPU allocation (diarization, Whisper cuBLAS, later
        captions) can OOM — the upload fails and nothing reaches Qdrant. Video frames here are a stock
        chart + a ticker overlay whose exact text is already captured verbatim by
        TrOCR, so the VLM only needs a short scene summary — the 2B model is more than
        enough and frees ~6 GB. Image modality keeps the 7B model (chart digitisation
        needs the accuracy); this is a separate singleton so the two never conflict.

        NOTE re: the g6e.xlarge/L40S (48GB) migration — this VRAM constraint likely
        no longer binds on the new card, but VIDEO_QWEN2_VL_MODEL is deliberately
        left at 2B pending real hardware measurement (also needs re-validating the
        separate, latency-motivated 180-token video caption budget alongside any
        model-size change). See docs/runbooks/phase-30-aws-deployment.md.
        """
        self._touch("qwen2_vl_video")
        if self._qwen2vl_v_model:
            return self._qwen2vl_v_processor, self._qwen2vl_v_model, self._qwen2vl_v_device

        with self._lock:
            if self._qwen2vl_v_model:
                return self._qwen2vl_v_processor, self._qwen2vl_v_model, self._qwen2vl_v_device

            # If the video captioner is configured to the same checkpoint as the
            # image one, just reuse that singleton — no point loading it twice.
            if settings.VIDEO_QWEN2_VL_MODEL == settings.QWEN2_VL_MODEL:
                return self.get_qwen2_vl()

            self._oom_guard(loading="qwen2_vl_video")
            decision = device_manager.decision_for("qwen2_vl")

            def _load():
                from transformers import (  # local
                    AutoProcessor,
                    BitsAndBytesConfig,
                    Qwen2VLForConditionalGeneration,
                )

                processor = AutoProcessor.from_pretrained(
                    settings.VIDEO_QWEN2_VL_MODEL, trust_remote_code=True
                )
                # See the identical comment in get_qwen2_vl() above — same fix,
                # same reason (host-RAM spike during from_pretrained() vs. a
                # per-shard device_map load), applied here for consistency in
                # case VIDEO_QWEN2_VL_MODEL is ever bumped off the 2B checkpoint.
                load_kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}
                if settings.QWEN2_VL_LOAD_IN_8BIT and decision.device == "cuda":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    load_kwargs["torch_dtype"] = _torch_dtype("float16")
                    load_kwargs["device_map"] = {"": decision.device}
                elif decision.device == "cuda":
                    load_kwargs["torch_dtype"] = _torch_dtype("float16")
                    load_kwargs["device_map"] = {"": decision.device}
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    settings.VIDEO_QWEN2_VL_MODEL, **load_kwargs
                )
                if decision.device != "cuda":
                    model.to(decision.device)
                model.eval()
                return processor, model

            self._qwen2vl_v_processor, self._qwen2vl_v_model = self._safe_load(
                _load, "qwen2_vl_video"
            )
            self._qwen2vl_v_device = decision.device

        return self._qwen2vl_v_processor, self._qwen2vl_v_model, self._qwen2vl_v_device

    # TROCR — PRINTED OCR FOR FINANCIAL DOCUMENTS

    def get_trocr(self) -> tuple:
        self._touch("trocr")
        if self._trocr_model:
            return self._trocr_processor, self._trocr_model, self._trocr_device

        with self._lock:
            if self._trocr_model:
                return self._trocr_processor, self._trocr_model, self._trocr_device

            self._oom_guard(loading="trocr")
            decision = device_manager.decision_for("trocr")

            def _load():
                import transformers as _tf
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # local
                from transformers.models.trocr.modeling_trocr import (
                    TrOCRSinusoidalPositionalEmbedding,
                )

                processor = TrOCRProcessor.from_pretrained(settings.TROCR_MODEL)
                # Suppress the two known benign weight mismatches:
                #   MISSING encoder.pooler.* — VisionEncoderDecoderModel allocates a
                #     pooler the TrOCR checkpoint never saved; cross-attention generation
                #     never calls it, so we delete it after load.
                #   UNEXPECTED _float_tensor — legacy sinusoidal PE key in checkpoint
                #     with no registered parameter in current transformers.
                _prev = _tf.logging.get_verbosity()
                _tf.logging.set_verbosity_error()
                try:
                    model = VisionEncoderDecoderModel.from_pretrained(
                        settings.TROCR_MODEL,
                        low_cpu_mem_usage=False,
                    )
                finally:
                    _tf.logging.set_verbosity(_prev)
                # Free the randomly-initialised dead pooler so it doesn't waste VRAM.
                if getattr(model.encoder, "pooler", None) is not None:
                    model.encoder.pooler = None
                model.to(decision.device)
                if decision.device == "cuda" and decision.dtype == "float16":
                    model.half()
                model.eval()

                # transformers 5.x fast-init runs module __init__ under a meta-device
                # context. TrOCRSinusoidalPositionalEmbedding stores its sinusoidal
                # table in a plain `.weights` attribute (NOT a registered buffer), so
                # model.to()/.half() never move it — it stays on the meta device and
                # decoder.forward() dies with "Cannot copy out of meta tensor".
                # Rebuild that table with real data on the model's device + dtype.
                _dev = next(model.parameters()).device
                _dtype = next(model.parameters()).dtype
                for _m in model.modules():
                    if isinstance(_m, TrOCRSinusoidalPositionalEmbedding):
                        _w = getattr(_m, "weights", None)
                        if _w is not None and (getattr(_w, "is_meta", False) or _w.device != _dev):
                            _real = _m.get_embedding(
                                int(_w.shape[0]), int(_w.shape[1]), _m.padding_idx
                            )
                            _m.weights = _real.to(device=_dev, dtype=_dtype)
                return processor, model

            self._trocr_processor, self._trocr_model = self._safe_load(_load, "trocr")
            self._trocr_device = decision.device

        return self._trocr_processor, self._trocr_model, self._trocr_device

    # PYANNOTE DIARIZER — SPEAKER DIARIZATION (requires HF_TOKEN + license)

    def get_diarizer(self):
        self._touch("diarizer")
        if self._diarizer:
            return self._diarizer

        with self._lock:
            if self._diarizer:
                return self._diarizer

            self._oom_guard(loading="diarizer")
            if not settings.HF_TOKEN:
                raise RuntimeError(
                    "Diarizer requires HF_TOKEN with pyannote model access. "
                    "Set HF_TOKEN in .env and accept the model license at hf.co."
                )

            decision = device_manager.decision_for("diarizer")

            def _load():
                # Suppress known-benign noise from pyannote's own dependency
                # chain — all confirmed non-fatal (diarization works
                # correctly despite every one of these), traced live
                # 2026-08-01. Same pattern as app/guardrails/pii.py's
                # Presidio suppression: filters set right before the noisy
                # library is imported/used, not globally at app startup.
                import logging as _logging
                import warnings as _warnings

                # pyannote's own deliberate TF32-disabled-for-reproducibility
                # notice — informational, not a problem to fix.
                _warnings.filterwarnings(
                    "ignore",
                    category=Warning,
                    module=r"pyannote\.audio\.utils\.reproducibility",
                )
                # PyTorch's std() warning when a pooling layer sees a
                # sequence too short for a meaningful degrees-of-freedom
                # correction — degrades gracefully, not a crash. Only worth
                # revisiting if real diarization output looks wrong, not
                # from this warning alone.
                _warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module=r"pyannote\.audio\.models\.blocks\.pooling",
                )
                # speechbrain logs its own auto-applied environment
                # workarounds at INFO — not an error, nothing to act on.
                _logging.getLogger("speechbrain.utils.quirks").setLevel(_logging.WARNING)
                try:
                    # The WeSpeaker embedding model (pyannote/wespeaker-
                    # voxceleb-resnet34-LM) runs via onnxruntime, which logs
                    # its own device-enumeration probe via a DRM sysfs path
                    # ("/sys/class/drm/card0") that doesn't exist inside a
                    # Docker container's device namespace — CUDA is used via
                    # the standard NVIDIA driver path regardless, confirmed
                    # by every model loading successfully on device=cuda.
                    import onnxruntime as _ort

                    _ort.set_default_logger_severity(3)  # ERROR and above only
                except ImportError:
                    pass

                # torchaudio ≥2.1 removed AudioMetaData / .info() / .load() /
                # .list_audio_backends() from the public API; patch them back
                # with soundfile-based shims before pyannote.audio is imported.
                from app.utils.torchaudio_compat import patch_torchaudio

                patch_torchaudio()
                import torch
                from pyannote.audio import Pipeline  # local

                pipeline = Pipeline.from_pretrained(
                    settings.DIARIZATION_MODEL,
                    use_auth_token=settings.HF_TOKEN,
                    cache_dir=str(settings.HF_HOME) + "/hub",
                )
                if decision.device == "cuda":
                    pipeline = pipeline.to(torch.device("cuda"))
                return pipeline

            self._diarizer = self._safe_load(_load, "diarizer")

        return self._diarizer

    # BERT-NER — FINANCE ENTITY EXTRACTION

    def get_ner(self):
        self._touch("ner")
        if self._ner:
            return self._ner

        with self._lock:
            if self._ner:
                return self._ner

            self._oom_guard(loading="ner")
            decision = device_manager.decision_for("ner")

            def _load():
                import transformers as _tf
                from transformers import pipeline as hf_pipeline  # local

                # Suppress UNEXPECTED bert.pooler.* — BertForTokenClassification drops
                # the pooler from its architecture, but the dslim checkpoint (derived
                # from full BERT base) still contains those weights in the file.
                _prev = _tf.logging.get_verbosity()
                _tf.logging.set_verbosity_error()
                try:
                    return hf_pipeline(
                        "ner",
                        model=settings.NER_MODEL,
                        aggregation_strategy="simple",
                        device=0 if decision.device == "cuda" else -1,
                    )
                finally:
                    _tf.logging.set_verbosity(_prev)

            self._ner = self._safe_load(_load, "ner")

        return self._ner

    # FINBERT — finance tone/sentiment classifier

    def get_finbert(self):
        self._touch("finbert")
        if self._finbert:
            return self._finbert

        with self._lock:
            if self._finbert:
                return self._finbert

            self._oom_guard(loading="finbert")
            decision = device_manager.decision_for("ner")  # small model, same slot as NER

            def _load():
                from transformers import (
                    BertForSequenceClassification,
                    BertTokenizer,
                )
                from transformers import pipeline as hf_pipeline

                # yiyanghkust/finbert-tone config.json lacks `model_type`;
                # explicitly using BERT classes bypasses auto-detection failure.
                tokenizer = BertTokenizer.from_pretrained(settings.FINBERT_MODEL)
                model = BertForSequenceClassification.from_pretrained(settings.FINBERT_MODEL)
                return hf_pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if decision.device == "cuda" else -1,
                    truncation=True,
                    max_length=512,
                )

            self._finbert = self._safe_load(_load, "finbert")

        return self._finbert

    # HEALTH CHECK

    def health_check(self) -> dict[str, Any]:
        # "llm" (below) only reflects whether the GGUFModel Python wrapper
        # has been constructed, not whether it's actually usable (e.g. in
        # llama_server mode, the separate process could still be loading).
        # llm_ready gives the real signal — see GGUFModel.health_check()'s
        # "ready" field. Additive: doesn't change "llm"'s existing bool
        # meaning, since other health-check consumers already read it.
        llm_ready = False
        if self._llm is not None:
            try:
                llm_ready = bool(self._llm.health_check().get("ready"))
            except Exception:
                llm_ready = False

        health: dict[str, Any] = {
            "llm": self._llm is not None,
            "llm_ready": llm_ready,
            "embedder": self._text_embedder is not None,
            "siglip": self._siglip_model is not None,
            "siglip_text": self._siglip_text_embedder is not None,
            "image_embedder": self._image_embedder is not None,
            "multimodal": self._multimodal is not None,
            "whisper": self._whisper is not None,
            "blip": self._blip_model is not None,
            "qwen2_vl": self._qwen2vl_model is not None,
            "trocr": self._trocr_model is not None,
            "diarizer": self._diarizer is not None,
            "ner": self._ner is not None,
            "finbert": self._finbert is not None,
            "reranker": self._reranker is not None,
            "device": device_manager.device_for("llm"),
            "profile": device_manager.profile,
            "vram_total_gb": device_manager.vram_total_gb,
            "initialized": self._initialized,
            "devices": device_manager.snapshot(),
        }
        if device_manager.cuda_available:
            try:
                torch = _torch()
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                health["vram_reserved_gb"] = round(reserved, 2)
                health["vram_allocated_gb"] = round(allocated, 2)
                health["vram_free_gb"] = round(device_manager.vram_total_gb - reserved, 2)
            except Exception:
                pass
        return health

    # RESET ALL MODELS

    def reset(self) -> None:
        with self._lock:
            self._llm = None
            self._text_embedder = None
            self._siglip_text_embedder = None
            self._image_embedder = None
            self._multimodal = None
            self._whisper = None
            self._reranker = None
            self._siglip_model = None
            self._siglip_processor = None
            self._siglip_device = None
            self._blip_model = None
            self._blip_processor = None
            self._blip_device = None
            self._qwen2vl_model = None
            self._qwen2vl_processor = None
            self._qwen2vl_device = None
            self._trocr_model = None
            self._trocr_processor = None
            self._trocr_device = None
            self._diarizer = None
            self._ner = None
            self._initialized = False

            for name in (
                "LLM",
                "TextEmbedder",
                "SigLIP",
                "SigLIPText",
                "ImageEmbedder",
                "MultimodalEmbedder",
                "Whisper",
                "BLIP",
                "qwen2_vl",
                "trocr",
                "diarizer",
                "ner",
                "Reranker",
            ):
                _model_loaded.labels(model=name).set(0)

            device_manager.empty_cuda_cache()
            logger.warning("models_reset")


# SINGLETON

model_loader = ModelLoader()
