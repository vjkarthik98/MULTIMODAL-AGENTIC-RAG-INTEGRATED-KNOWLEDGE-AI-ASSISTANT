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

try:
    import torch
except ImportError:
    torch = None

from faster_whisper import WhisperModel
from sentence_transformers import CrossEncoder
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from app.core.config import settings
from app.embeddings.clip_text_embedder import ClipTextEmbedder
from app.embeddings.image_embedder import ImageEmbedder
from app.embeddings.multimodal_embedder import MultimodalEmbedder
from app.embeddings.text_embedder import TextEmbedder
from app.llm.gguf_model import GGUFModel

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

# SEMAPHORE — PREVENT CONCURRENT HEAVY MODEL LOADS
_load_semaphore = asyncio.Semaphore(1)


class ModelLoader:

    def __init__(self) -> None:
        self._lock     = threading.RLock()
        self._device   = self._detect_device()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.THREAD_POOL_SIZE,
            thread_name_prefix="model_loader",
        )

        self._llm:                Optional[GGUFModel]            = None
        self._text_embedder:      Optional[TextEmbedder]         = None
        self._clip_text_embedder: Optional[ClipTextEmbedder]     = None
        self._image_embedder:     Optional[ImageEmbedder]        = None
        self._multimodal:         Optional[MultimodalEmbedder]   = None
        self._whisper:            Optional[WhisperModel]         = None
        self._reranker:           Optional[CrossEncoder]         = None
        self._clip_model                                         = None
        self._clip_processor                                     = None
        self._blip_model                                         = None
        self._blip_processor                                     = None

        self._initialized = False

        logger.info("model_loader_initialized", device=self._device)

    # DEVICE DETECTION — CUDA > MPS > CPU

    def _detect_device(self) -> str:
        if torch is not None:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

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
            span.set_attribute("device", self._device)

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
                    device=self._device,
                    latency=latency,
                )

                return obj

            except FuturesTimeoutError as exc:
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

    # STAGGERED WARMUP — LOAD ALL MODELS SEQUENTIALLY WITH SLEEP BETWEEN

    def warmup(self) -> None:
        if self._initialized:
            logger.info("warmup_skipped")
            return

        with self._lock:
            if self._initialized:
                return

            logger.info("warmup_started_staggered", device=self._device)

            load_functions = [
                (self.get_llm,          "LLM"),
                (self.get_embedder,     "TextEmbedder"),
                (self.get_clip,         "CLIP"),
                (self.get_whisper,      "Whisper"),
                (self.get_blip,         "BLIP"),
                (self.get_reranker,     "Reranker"),
            ]

            for load_fn, name in load_functions:
                try:
                    future = self._executor.submit(load_fn)
                    future.result(timeout=settings.MODEL_TIMEOUT_SEC)
                    time.sleep(2.0)
                except Exception as exc:
                    logger.warning(
                        "warmup_model_failed",
                        model=name,
                        error=str(exc),
                    )

            self._initialized = True
            logger.info("warmup_completed", device=self._device)

    # ASYNC WARMUP

    async def warmup_async(self) -> None:
        if self._initialized:
            return
        async with _load_semaphore:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, self.warmup
            )

    # LLM — GGUF MODEL

    def get_llm(self) -> GGUFModel:
        if self._llm:
            return self._llm

        with self._lock:
            if self._llm:
                return self._llm

            if not settings.LLM_MODEL_PATH:
                raise RuntimeError("LLM_MODEL_PATH missing")

            self._llm = self._safe_load(
                lambda: GGUFModel(
                    settings.LLM_MODEL_PATH,
                    n_gpu_layers=settings.LLM_GPU_LAYERS,
                ),
                "LLM",
            )

        return self._llm

    # TEXT EMBEDDER — SENTENCE TRANSFORMERS

    def get_embedder(self) -> TextEmbedder:
        if self._text_embedder:
            return self._text_embedder

        with self._lock:
            if self._text_embedder:
                return self._text_embedder

            self._text_embedder = self._safe_load(
                lambda: TextEmbedder(
                    model_name=settings.EMBEDDING_MODEL,
                    batch_size=settings.EMBEDDING_BATCH_SIZE,
                    device=self._device,
                ),
                "TextEmbedder",
            )

        return self._text_embedder

    # MULTILINGUAL EMBEDDER — PHASE 25

    def get_multilingual_embedder(self) -> TextEmbedder:
        multilingual_model = getattr(
            settings, "MULTILINGUAL_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        return self._safe_load(
            lambda: TextEmbedder(
                model_name=multilingual_model,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                device=self._device,
            ),
            "MultilingualEmbedder",
        )

    # CLIP MODEL + PROCESSOR

    def get_clip(self) -> Tuple:
        if self._clip_model:
            return self._clip_processor, self._clip_model, self._device

        with self._lock:
            if self._clip_model:
                return self._clip_processor, self._clip_model, self._device

            def _load():
                processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
                model     = CLIPModel.from_pretrained(settings.CLIP_MODEL)
                model.to(self._device)
                model.eval()
                return processor, model

            self._clip_processor, self._clip_model = self._safe_load(_load, "CLIP")

        return self._clip_processor, self._clip_model, self._device

    # IMAGE EMBEDDER — CLIP VISUAL

    def get_image_embedder(self) -> ImageEmbedder:
        if self._image_embedder:
            return self._image_embedder

        with self._lock:
            if self._image_embedder:
                return self._image_embedder

            processor, model, device = self.get_clip()
            self._image_embedder = self._safe_load(
                lambda: ImageEmbedder(model, processor, device),
                "ImageEmbedder",
            )

        return self._image_embedder

    # CLIP TEXT EMBEDDER — CROSS-MODAL

    def get_clip_text_embedder(self) -> ClipTextEmbedder:
        if self._clip_text_embedder:
            return self._clip_text_embedder

        with self._lock:
            if self._clip_text_embedder:
                return self._clip_text_embedder

            processor, model, device = self.get_clip()
            self._clip_text_embedder = self._safe_load(
                lambda: ClipTextEmbedder(processor, model, device),
                "ClipTextEmbedder",
            )

        return self._clip_text_embedder

    # MULTIMODAL EMBEDDER — ORCHESTRATES TEXT + IMAGE

    def get_multimodal_embedder(self) -> MultimodalEmbedder:
        if self._multimodal:
            return self._multimodal

        with self._lock:
            if self._multimodal:
                return self._multimodal

            self._multimodal = MultimodalEmbedder(
                self.get_embedder(),
                self.get_image_embedder(),
            )

        return self._multimodal

    # WHISPER — ASR

    def get_whisper(self) -> WhisperModel:
        if self._whisper:
            return self._whisper

        with self._lock:
            if self._whisper:
                return self._whisper

            compute_type = (
                "float16" if self._device == "cuda"
                else "float32" if self._device == "mps"
                else "int8"
            )

            self._whisper = self._safe_load(
                lambda: WhisperModel(
                    settings.WHISPER_MODEL,
                    device=self._device if self._device != "mps" else "cpu",
                    compute_type=compute_type if self._device != "mps" else "int8",
                ),
                "Whisper",
            )

        return self._whisper

    # BLIP — IMAGE CAPTIONING

    def get_blip(self) -> Tuple:
        if self._blip_model:
            return self._blip_processor, self._blip_model, self._device

        with self._lock:
            if self._blip_model:
                return self._blip_processor, self._blip_model, self._device

            def _load():
                processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
                model     = BlipForConditionalGeneration.from_pretrained(settings.BLIP_MODEL)
                model.to(self._device)
                model.eval()
                return processor, model

            self._blip_processor, self._blip_model = self._safe_load(_load, "BLIP")

        return self._blip_processor, self._blip_model, self._device

    # RERANKER — CROSS ENCODER

    def get_reranker(self) -> CrossEncoder:
        if self._reranker:
            return self._reranker

        with self._lock:
            if self._reranker:
                return self._reranker

            self._reranker = self._safe_load(
                lambda: CrossEncoder(
                    settings.RERANKER_MODEL,
                    device=self._device,
                ),
                "Reranker",
            )

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
            "device":         self._device,
            "initialized":    self._initialized,
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
            self._blip_model         = None
            self._blip_processor     = None
            self._initialized        = False

            for name in (
                "LLM", "TextEmbedder", "CLIP", "ClipTextEmbedder",
                "ImageEmbedder", "MultimodalEmbedder", "Whisper",
                "BLIP", "Reranker",
            ):
                _model_loaded.labels(model=name).set(0)

            logger.warning("models_reset")


# SINGLETON

model_loader = ModelLoader()

