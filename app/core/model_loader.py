import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Tuple, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import torch
except ImportError:
    torch = None

from transformers import CLIPModel, CLIPProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from faster_whisper import WhisperModel
from sentence_transformers import CrossEncoder

from app.llm.gguf_model import GGUFModel
from app.embeddings.text_embedder import TextEmbedder
from app.embeddings.clip_text_embedder import ClipTextEmbedder
from app.embeddings.image_embedder import ImageEmbedder
from app.embeddings.multimodal_embedder import MultimodalEmbedder

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        self._executor = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)

        # models
        self._llm: Optional[GGUFModel] = None
        self._text_embedder: Optional[TextEmbedder] = None
        self._clip_text_embedder: Optional[ClipTextEmbedder] = None
        self._image_embedder: Optional[ImageEmbedder] = None
        self._whisper: Optional[WhisperModel] = None
        self._blip_processor = None
        self._blip_model = None
        self._reranker: Optional[CrossEncoder] = None
        self._clip_model = None
        self._clip_processor = None
        self._multimodal: Optional[MultimodalEmbedder] = None

        self._initialized = False

        logger.info(event="model_loader_initialized", device=self._device)

    #  SAFE LOAD 
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, RuntimeError)),
        reraise=True
    )
    def _safe_load(self, load_fn, name: str):
        start = time.time()

        future = self._executor.submit(load_fn)
        obj = future.result(timeout=settings.MODEL_TIMEOUT_SEC)

        latency = round(time.time() - start, 2)
        logger.info(event="model_loaded", model=name, latency=latency)

        return obj

    #  WARMUP 
    def warmup(self) -> None:
        if self._initialized:
            logger.info(event="warmup_skipped")
            return

        with self._lock:
            if self._initialized:
                return

            logger.info(event="warmup_started")

            futures = [
                self._executor.submit(self.get_llm),
                self._executor.submit(self.get_embedder),
                self._executor.submit(self.get_clip),
                self._executor.submit(self.get_whisper),
                self._executor.submit(self.get_blip),
                self._executor.submit(self.get_reranker),
            ]

            for f in futures:
                f.result()

            self._initialized = True
            logger.info(event="warmup_completed")

    #  LLM 
    def get_llm(self) -> GGUFModel:
        if self._llm:
            return self._llm

        with self._lock:
            if not self._llm:
                if not settings.LLM_MODEL_PATH:
                    raise RuntimeError("LLM_MODEL_PATH missing")

                def load():
                    return GGUFModel(
                        settings.LLM_MODEL_PATH,
                        n_gpu_layers=settings.LLM_GPU_LAYERS
                    )

                self._llm = self._safe_load(load, "LLM")

        return self._llm

    #  EMBEDDER 
    def get_embedder(self) -> TextEmbedder:
        if self._text_embedder:
            return self._text_embedder

        with self._lock:
            if not self._text_embedder:
                self._text_embedder = self._safe_load(
                    lambda: TextEmbedder(
                        model_name=settings.EMBEDDING_MODEL,
                        batch_size=settings.EMBEDDING_BATCH_SIZE,
                        device=self._device
                    ),
                    "TextEmbedder"
                )

        return self._text_embedder

    #  CLIP 
    def get_clip(self) -> Tuple:
        if self._clip_model:
            return self._clip_processor, self._clip_model, self._device

        with self._lock:
            if not self._clip_model:

                def load():
                    processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
                    model = CLIPModel.from_pretrained(settings.CLIP_MODEL)

                    model.to(self._device)
                    model.eval()
                    return processor, model

                self._clip_processor, self._clip_model = self._safe_load(load, "CLIP")

        return self._clip_processor, self._clip_model, self._device

    #  IMAGE EMBEDDER 
    def get_image_embedder(self) -> ImageEmbedder:
        if self._image_embedder:
            return self._image_embedder

        with self._lock:
            if not self._image_embedder:
                processor, model, device = self.get_clip()

                self._image_embedder = self._safe_load(
                    lambda: ImageEmbedder(model, processor, device),
                    "ImageEmbedder"
                )

        return self._image_embedder

    #  CLIP TEXT 
    def get_clip_text_embedder(self) -> ClipTextEmbedder:
        if self._clip_text_embedder:
            return self._clip_text_embedder

        with self._lock:
            if not self._clip_text_embedder:
                processor, model, device = self.get_clip()

                self._clip_text_embedder = self._safe_load(
                    lambda: ClipTextEmbedder(processor, model, device),
                    "ClipTextEmbedder"
                )

        return self._clip_text_embedder

    #  MULTIMODAL 
    def get_multimodal_embedder(self) -> MultimodalEmbedder:
        if self._multimodal:
            return self._multimodal

        with self._lock:
            if not self._multimodal:
                text = self.get_embedder()
                image = self.get_image_embedder()

                self._multimodal = MultimodalEmbedder(text, image)

        return self._multimodal

    #  WHISPER 
    def get_whisper(self) -> WhisperModel:
        if self._whisper:
            return self._whisper

        with self._lock:
            if not self._whisper:
                compute_type = "float16" if self._device == "cuda" else "int8"

                self._whisper = self._safe_load(
                    lambda: WhisperModel(
                        settings.WHISPER_MODEL,
                        device=self._device,
                        compute_type=compute_type
                    ),
                    "Whisper"
                )

        return self._whisper

    #  BLIP 
    def get_blip(self) -> Tuple:
        if self._blip_model:
            return self._blip_processor, self._blip_model, self._device

        with self._lock:
            if not self._blip_model:

                def load():
                    processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
                    model = BlipForConditionalGeneration.from_pretrained(settings.BLIP_MODEL)

                    model.to(self._device)
                    model.eval()
                    return processor, model

                self._blip_processor, self._blip_model = self._safe_load(load, "BLIP")

        return self._blip_processor, self._blip_model, self._device

    #  RERANKER 
    def get_reranker(self) -> CrossEncoder:
        if self._reranker:
            return self._reranker

        with self._lock:
            if not self._reranker:
                self._reranker = self._safe_load(
                    lambda: CrossEncoder(settings.RERANKER_MODEL, device=self._device),
                    "Reranker"
                )

        return self._reranker

    #  HEALTH 
    def health_check(self) -> Dict[str, bool]:
        return {
            "llm": self._llm is not None,
            "embedder": self._text_embedder is not None,
            "clip": self._clip_model is not None,
            "whisper": self._whisper is not None,
            "reranker": self._reranker is not None,
        }

    #  RESET 
    def reset(self) -> None:
        with self._lock:
            self._llm = None
            self._text_embedder = None
            self._clip_text_embedder = None
            self._image_embedder = None
            self._whisper = None
            self._blip_model = None
            self._reranker = None
            self._clip_model = None
            self._multimodal = None
            self._initialized = False

            logger.warning(event="models_reset")


# SINGLETON
model_loader = ModelLoader()