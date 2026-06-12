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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional, Tuple

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _is_retryable_load_error(exc: BaseException) -> bool:
    """Retry on timeout or transient RuntimeError, but not on interpreter shutdown."""
    if isinstance(exc, FuturesTimeoutError):
        return True
    if isinstance(exc, RuntimeError):
        return "interpreter shutdown" not in str(exc) and "shutdown" not in str(exc).lower()
    return False

from app.core.config import settings
from app.core.device_manager import device_manager

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
_embedding_latency = Histogram(
    "embedding_latency_seconds",
    "Embedding latency by model",
    ["model"],
)


# SEMAPHORE — lazy init to avoid missing event loop at import time

_load_semaphore: Optional[asyncio.Semaphore] = None
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

        self._llm:                  Optional[Any] = None
        self._text_embedder:        Optional[Any] = None
        self._siglip_text_embedder: Optional[Any] = None
        self._image_embedder:       Optional[Any] = None
        self._multimodal:           Optional[Any] = None
        self._whisper:              Optional[Any] = None
        self._reranker:             Optional[Any] = None
        self._siglip_model                       = None
        self._siglip_processor                   = None
        self._siglip_device:        Optional[str] = None
        self._blip_model                         = None
        self._blip_processor                     = None
        self._blip_device:          Optional[str] = None
        # New Phase MAGIK models
        self._blip2_model                        = None
        self._blip2_processor                    = None
        self._blip2_device:         Optional[str] = None
        self._llava_model                        = None
        self._llava_processor                    = None
        self._llava_device:         Optional[str] = None
        self._trocr_model                        = None
        self._trocr_processor                    = None
        self._trocr_device:         Optional[str] = None
        self._diarizer:             Optional[Any] = None
        self._ner:                  Optional[Any] = None

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

    # SAFE LOAD WITH TIMEOUT AND RETRY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception(_is_retryable_load_error),
        reraise=True,
    )
    def _safe_load(self, load_fn, name: str, device: Optional[str] = None) -> Any:
        start  = time.time()
        future = self._executor.submit(load_fn)

        # Use the provided device hint; fall back to device_manager only for
        # known MODEL_NAMES — unknown aliases (image_embedder, siglip_text_embedder)
        # are not in MODEL_NAMES so device_for() would wrongly return "cpu".
        from app.core.device_manager import MODEL_NAMES
        resolved_device = device or (
            device_manager.device_for(name.lower())
            if name.lower() in MODEL_NAMES
            else "unknown"
        )

        with tracer.start_as_current_span(f"load_model_{name}") as span:
            span.set_attribute("model.name", name)
            span.set_attribute("device", resolved_device)

            try:
                obj     = future.result(timeout=settings.MODEL_TIMEOUT_SEC)
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
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self.warmup
            )

    # LLM — GGUF MODEL (or MockLLM when LLM_MOCK_MODE=true)

    def get_llm(self) -> Any:
        if self._llm:
            return self._llm

        with self._lock:
            if self._llm:
                return self._llm

            if settings.LLM_MOCK_MODE:
                from app.llm.mock_llm import MockLLM
                self._llm = MockLLM()
                logger.warning(
                    "llm_mock_mode_active",
                    hint="Set LLM_MOCK_MODE=false to use the real GGUF model",
                )
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
                from app.embeddings.text_embedder import TextEmbedder  # local
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

    def get_siglip(self) -> Tuple:
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
    def get_clip(self) -> Tuple:
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
                from app.embeddings.clip_text_embedder import ClipTextEmbedder  # local
                return ClipTextEmbedder(processor, model, device)

            self._siglip_text_embedder = self._safe_load(_load, "siglip_text_embedder", device=device)

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

            from app.embeddings.multimodal_embedder import MultimodalEmbedder  # local
            self._multimodal = MultimodalEmbedder(
                self.get_embedder(),
                self.get_image_embedder(),
            )

        return self._multimodal

    # WHISPER — ASR

    def get_whisper(self):
        if self._whisper:
            return self._whisper

        with self._lock:
            if self._whisper:
                return self._whisper

            decision     = device_manager.decision_for("whisper")
            device       = decision.device if decision.device != "mps" else "cpu"
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

    # BLIP — IMAGE CAPTIONING

    def get_blip(self) -> Tuple:
        if self._blip_model:
            return self._blip_processor, self._blip_model, self._blip_device

        with self._lock:
            if self._blip_model:
                return self._blip_processor, self._blip_model, self._blip_device

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
                model = BlipForConditionalGeneration.from_pretrained(
                    settings.BLIP_MODEL, **kwargs
                )
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

    # BLIP2 — IMAGE CAPTIONING (replaces BLIP-1 in Phase 2+)

    def get_blip2(self) -> Tuple:
        if self._blip2_model:
            return self._blip2_processor, self._blip2_model, self._blip2_device

        with self._lock:
            if self._blip2_model:
                return self._blip2_processor, self._blip2_model, self._blip2_device

            decision = device_manager.decision_for("blip2")

            def _load():
                from transformers import Blip2Processor, Blip2ForConditionalGeneration  # local
                processor = Blip2Processor.from_pretrained(settings.BLIP2_MODEL)
                load_kwargs: dict = {}
                if settings.BLIP2_LOAD_IN_8BIT and decision.device == "cuda":
                    load_kwargs["load_in_8bit"] = True
                elif decision.device == "cuda" and decision.dtype == "float16":
                    load_kwargs["torch_dtype"] = _torch_dtype("float16")
                model = Blip2ForConditionalGeneration.from_pretrained(
                    settings.BLIP2_MODEL, **load_kwargs
                )
                if not settings.BLIP2_LOAD_IN_8BIT:
                    model.to(decision.device)
                model.eval()
                return processor, model

            self._blip2_processor, self._blip2_model = self._safe_load(_load, "blip2")
            self._blip2_device = decision.device

        return self._blip2_processor, self._blip2_model, self._blip2_device

    # LLAVA — VIDEO FRAME CAPTIONING (T4: evict LLM first via _VIDEO_SLOT_LOCK)

    _VIDEO_SLOT_LOCK = threading.Lock()

    def get_llava(self) -> Tuple:
        if self._llava_model:
            return self._llava_processor, self._llava_model, self._llava_device

        with self._lock:
            if self._llava_model:
                return self._llava_processor, self._llava_model, self._llava_device

            decision = device_manager.decision_for("llava")

            def _load():
                from transformers import LlavaForConditionalGeneration, AutoProcessor  # local
                processor = AutoProcessor.from_pretrained(settings.LLAVA_MODEL)
                load_kwargs: dict = {}
                if settings.LLAVA_LOAD_IN_8BIT and decision.device == "cuda":
                    load_kwargs["load_in_8bit"] = True
                elif decision.device == "cuda":
                    load_kwargs["torch_dtype"] = _torch_dtype("float16")
                model = LlavaForConditionalGeneration.from_pretrained(
                    settings.LLAVA_MODEL, **load_kwargs
                )
                if not settings.LLAVA_LOAD_IN_8BIT:
                    model.to(decision.device)
                model.eval()
                return processor, model

            self._llava_processor, self._llava_model = self._safe_load(_load, "llava")
            self._llava_device = decision.device

        return self._llava_processor, self._llava_model, self._llava_device

    # TROCR — PRINTED OCR FOR FINANCIAL DOCUMENTS

    def get_trocr(self) -> Tuple:
        if self._trocr_model:
            return self._trocr_processor, self._trocr_model, self._trocr_device

        with self._lock:
            if self._trocr_model:
                return self._trocr_processor, self._trocr_model, self._trocr_device

            decision = device_manager.decision_for("trocr")

            def _load():
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # local
                processor = TrOCRProcessor.from_pretrained(settings.TROCR_MODEL)
                kwargs = {}
                if decision.device == "cuda" and decision.dtype == "float16":
                    kwargs["torch_dtype"] = _torch_dtype("float16")
                model = VisionEncoderDecoderModel.from_pretrained(settings.TROCR_MODEL, **kwargs)
                model.to(decision.device)
                model.eval()
                return processor, model

            self._trocr_processor, self._trocr_model = self._safe_load(_load, "trocr")
            self._trocr_device = decision.device

        return self._trocr_processor, self._trocr_model, self._trocr_device

    # PYANNOTE DIARIZER — SPEAKER DIARIZATION (requires HF_TOKEN + license)

    def get_diarizer(self):
        if self._diarizer:
            return self._diarizer

        with self._lock:
            if self._diarizer:
                return self._diarizer

            if not settings.HF_TOKEN:
                raise RuntimeError(
                    "Diarizer requires HF_TOKEN with pyannote model access. "
                    "Set HF_TOKEN in .env and accept the model license at hf.co."
                )

            decision = device_manager.decision_for("diarizer")

            def _load():
                from pyannote.audio import Pipeline  # local
                import torch
                pipeline = Pipeline.from_pretrained(
                    settings.DIARIZATION_MODEL,
                    use_auth_token=settings.HF_TOKEN,
                )
                if decision.device == "cuda":
                    pipeline = pipeline.to(torch.device("cuda"))
                return pipeline

            self._diarizer = self._safe_load(_load, "diarizer")

        return self._diarizer

    # BERT-NER — FINANCE ENTITY EXTRACTION

    def get_ner(self):
        if self._ner:
            return self._ner

        with self._lock:
            if self._ner:
                return self._ner

            decision = device_manager.decision_for("ner")

            def _load():
                from transformers import pipeline as hf_pipeline  # local
                return hf_pipeline(
                    "ner",
                    model=settings.NER_MODEL,
                    aggregation_strategy="simple",
                    device=0 if decision.device == "cuda" else -1,
                )

            self._ner = self._safe_load(_load, "ner")

        return self._ner

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {
            "llm":                  self._llm is not None,
            "embedder":             self._text_embedder is not None,
            "siglip":               self._siglip_model is not None,
            "siglip_text":          self._siglip_text_embedder is not None,
            "image_embedder":       self._image_embedder is not None,
            "multimodal":           self._multimodal is not None,
            "whisper":              self._whisper is not None,
            "blip":                 self._blip_model is not None,
            "blip2":                self._blip2_model is not None,
            "llava":                self._llava_model is not None,
            "trocr":                self._trocr_model is not None,
            "diarizer":             self._diarizer is not None,
            "ner":                  self._ner is not None,
            "reranker":             self._reranker is not None,
            "device":               device_manager.device_for("llm"),
            "profile":              device_manager.profile,
            "vram_total_gb":        device_manager.vram_total_gb,
            "initialized":          self._initialized,
            "devices":              device_manager.snapshot(),
        }
        if device_manager.cuda_available:
            try:
                torch = _torch()
                reserved  = torch.cuda.memory_reserved(0)  / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                health["vram_reserved_gb"]  = round(reserved,  2)
                health["vram_allocated_gb"] = round(allocated, 2)
                health["vram_free_gb"]      = round(device_manager.vram_total_gb - reserved, 2)
            except Exception:
                pass
        return health

    # RESET ALL MODELS

    def reset(self) -> None:
        with self._lock:
            self._llm                  = None
            self._text_embedder        = None
            self._siglip_text_embedder = None
            self._image_embedder       = None
            self._multimodal           = None
            self._whisper              = None
            self._reranker             = None
            self._siglip_model         = None
            self._siglip_processor     = None
            self._siglip_device        = None
            self._blip_model           = None
            self._blip_processor       = None
            self._blip_device          = None
            self._blip2_model          = None
            self._blip2_processor      = None
            self._blip2_device         = None
            self._llava_model          = None
            self._llava_processor      = None
            self._llava_device         = None
            self._trocr_model          = None
            self._trocr_processor      = None
            self._trocr_device         = None
            self._diarizer             = None
            self._ner                  = None
            self._initialized          = False

            for name in (
                "LLM", "TextEmbedder", "SigLIP", "SigLIPText",
                "ImageEmbedder", "MultimodalEmbedder", "Whisper",
                "BLIP", "blip2", "llava", "trocr", "diarizer", "ner", "Reranker",
            ):
                _model_loaded.labels(model=name).set(0)

            device_manager.empty_cuda_cache()
            logger.warning("models_reset")


# SINGLETON

model_loader = ModelLoader()
