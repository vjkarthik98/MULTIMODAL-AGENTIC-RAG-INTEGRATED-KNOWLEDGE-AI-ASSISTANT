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
  * CLIP / BLIP / Whisper run on GPU with float16 (or int8 on CPU Whisper)
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
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

        self._llm:                Optional[Any] = None
        self._text_embedder:      Optional[Any] = None
        self._clip_text_embedder: Optional[Any] = None
        self._image_embedder:     Optional[Any] = None
        self._multimodal:         Optional[Any] = None
        self._whisper:            Optional[Any] = None
        self._reranker:           Optional[Any] = None
        self._clip_model                       = None
        self._clip_processor                   = None
        self._clip_device:        Optional[str] = None
        self._blip_model                       = None
        self._blip_processor                   = None
        self._blip_device:        Optional[str] = None

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
        retry=retry_if_exception_type((FuturesTimeoutError, RuntimeError)),
        reraise=True,
    )
    def _safe_load(self, load_fn, name: str) -> Any:
        start  = time.time()
        future = self._executor.submit(load_fn)

        with tracer.start_as_current_span(f"load_model_{name}") as span:
            span.set_attribute("model.name", name)
            span.set_attribute("device", device_manager.device_for(name.lower()))

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
                    device=device_manager.device_for(name.lower()),
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

            self._llm = self._safe_load(_load_llm, "LLM")

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

            self._text_embedder = self._safe_load(_load, "TextEmbedder")

        return self._text_embedder

    # Alias kept in case any caller spells it this way.
    def get_text_embedder(self):
        return self.get_embedder()

    # CLIP MODEL + PROCESSOR

    def get_clip(self) -> Tuple:
        if self._clip_model:
            return self._clip_processor, self._clip_model, self._clip_device

        with self._lock:
            if self._clip_model:
                return self._clip_processor, self._clip_model, self._clip_device

            decision = device_manager.decision_for("clip")

            def _load():
                from transformers import CLIPModel, CLIPProcessor  # local
                processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
                kwargs = {}
                if decision.device == "cuda" and decision.dtype == "float16":
                    kwargs["torch_dtype"] = _torch_dtype("float16")
                model = CLIPModel.from_pretrained(settings.CLIP_MODEL, **kwargs)
                model.to(decision.device)
                model.eval()
                return processor, model

            self._clip_processor, self._clip_model = self._safe_load(_load, "CLIP")
            self._clip_device = decision.device

        return self._clip_processor, self._clip_model, self._clip_device

    # IMAGE EMBEDDER — CLIP VISUAL

    def get_image_embedder(self):
        if self._image_embedder:
            return self._image_embedder

        with self._lock:
            if self._image_embedder:
                return self._image_embedder

            processor, model, device = self.get_clip()

            def _load():
                from app.embeddings.image_embedder import ImageEmbedder  # local
                return ImageEmbedder(model, processor, device)

            self._image_embedder = self._safe_load(_load, "ImageEmbedder")

        return self._image_embedder

    # CLIP TEXT EMBEDDER — CROSS-MODAL

    def get_clip_text_embedder(self):
        if self._clip_text_embedder:
            return self._clip_text_embedder

        with self._lock:
            if self._clip_text_embedder:
                return self._clip_text_embedder

            processor, model, device = self.get_clip()

            def _load():
                from app.embeddings.clip_text_embedder import ClipTextEmbedder  # local
                return ClipTextEmbedder(processor, model, device)

            self._clip_text_embedder = self._safe_load(_load, "ClipTextEmbedder")

        return self._clip_text_embedder

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

            self._whisper = self._safe_load(_load, "Whisper")

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

            self._blip_processor, self._blip_model = self._safe_load(_load, "BLIP")
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
                return CrossEncoder(
                    settings.RERANKER_MODEL,
                    device=decision.device,
                )

            self._reranker = self._safe_load(_load, "Reranker")

        return self._reranker

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "llm":            self._llm is not None,
            "embedder":       self._text_embedder is not None,
            "clip":           self._clip_model is not None,
            "clip_text":      self._clip_text_embedder is not None,
            "image_embedder": self._image_embedder is not None,
            "multimodal":     self._multimodal is not None,
            "whisper":        self._whisper is not None,
            "blip":           self._blip_model is not None,
            "reranker":       self._reranker is not None,
            "device":         device_manager.device_for("llm"),
            "profile":        device_manager.profile,
            "vram_total_gb":  device_manager.vram_total_gb,
            "initialized":    self._initialized,
            "devices":        device_manager.snapshot(),
        }

    # RESET ALL MODELS

    def reset(self) -> None:
        with self._lock:
            self._llm                = None
            self._text_embedder      = None
            self._clip_text_embedder = None
            self._image_embedder     = None
            self._multimodal         = None
            self._whisper            = None
            self._reranker           = None
            self._clip_model         = None
            self._clip_processor     = None
            self._clip_device        = None
            self._blip_model         = None
            self._blip_processor     = None
            self._blip_device        = None
            self._initialized        = False

            for name in (
                "LLM", "TextEmbedder", "CLIP", "ClipTextEmbedder",
                "ImageEmbedder", "MultimodalEmbedder", "Whisper",
                "BLIP", "Reranker",
            ):
                _model_loaded.labels(model=name).set(0)

            device_manager.empty_cuda_cache()
            logger.warning("models_reset")


# SINGLETON

model_loader = ModelLoader()
