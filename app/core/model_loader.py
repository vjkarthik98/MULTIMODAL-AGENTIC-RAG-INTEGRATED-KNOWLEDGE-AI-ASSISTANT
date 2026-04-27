from app.core.config import settings
from app.utils.logger import get_logger

import threading
import time

try:
    import torch
except ImportError:
    torch = None


logger = get_logger(__name__)


class ModelLoader:
    def __init__(self):
        self._lock = threading.RLock()
        self._device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        # Core models
        self._llm = None
        self._text_embedder = None
        self._clip_text_embedder = None
        self._image_embedder = None
        self._whisper = None
        self._blip_processor = None
        self._blip_model = None
        self._reranker = None
        self._clip_model = None
        self._clip_processor = None

        logger.info("[ModelLoader] initialized | device=%s", self._device)

    def _safe_load(self, load_fn, name: str):
        start = time.time()
        try:
            obj = load_fn()
            logger.info("[ModelLoader] %s loaded | %.2fs", name, time.time() - start)
            return obj
        except Exception as e:
            logger.error("[ModelLoader] %s load failed | %s", name, str(e))
            raise

    # LLM
    def get_llm(self):
        if self._llm:
            return self._llm

        with self._lock:
            if not self._llm:
                path = getattr(settings, "LLM_MODEL_PATH", None)

                if not path:
                    logger.warning("[ModelLoader] LLM_MODEL_PATH is missing -> LLM disablled")
                    raise ValueError("LLM disabled: no model path provided")
                
                from app.llm.gguf_model import GGUFModel
                self._llm = self._safe_load(
                    lambda: GGUFModel(path),
                    "LLM"
                )
        return self._llm

    # Text Embedder
    def get_embedder(self):
        if self._text_embedder:
            return self._text_embedder

        with self._lock:
            if not self._text_embedder:
                from app.embeddings.text_embedder import TextEmbedder
                self._text_embedder = self._safe_load(
                    lambda: TextEmbedder(
                        model_name=settings.EMBEDDING_MODEL,
                        batch_size=settings.EMBEDDING_BATCH_SIZE,
                        device=self._device
                    ),
                    "TextEmbedder"
                )
        return self._text_embedder

    # CLIP Model
    def get_clip(self):
        if self._clip_model:
            return self._clip_processor, self._clip_model, self._device

        with self._lock:
            if not self._clip_model:
                from transformers import CLIPModel, CLIPProcessor

                def load():
                    processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
                    model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
                    model.to(self._device)
                    model.eval()
                    return processor, model

                self._clip_processor, self._clip_model = self._safe_load(load, "CLIP")

        return self._clip_processor, self._clip_model, self._device

    # CLIP Text Embedder
    def get_clip_text_embedder(self):
        if self._clip_text_embedder:
            return self._clip_text_embedder

        with self._lock:
            if not self._clip_text_embedder:
                from app.embeddings.clip_text_embedder import ClipTextEmbedder
                self._clip_text_embedder = self._safe_load(
                    lambda: ClipTextEmbedder(),
                    "CLIPTextEmbedder"
                )

        return self._clip_text_embedder

    # Image Embedder (NEW)
    def get_image_embedder(self):
        if self._image_embedder:
            return self._image_embedder

        with self._lock:
            if not self._image_embedder:
                from app.embeddings.image_embedder import ImageEmbedder
                self._image_embedder = self._safe_load(
                    lambda: ImageEmbedder(),
                    "ImageEmbedder"
                )

        return self._image_embedder

    # Whisper
    def get_whisper(self):
        if self._whisper:
            return self._whisper

        with self._lock:
            if not self._whisper:
                from faster_whisper import WhisperModel

                compute_type = "float16" if self._device == "cuda" else "int8"

                self._whisper = self._safe_load(
                    lambda: WhisperModel(
                        settings.WHISPER_MODEL,
                        device=self._device,
                        compute_type=compute_type,
                    ),
                    "Whisper"
                )

        return self._whisper

    # BLIP
    def get_blip(self):
        if self._blip_model:
            return self._blip_processor, self._blip_model, self._device

        with self._lock:
            if not self._blip_model:
                from transformers import BlipProcessor, BlipForConditionalGeneration

                def load():
                    processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
                    model = BlipForConditionalGeneration.from_pretrained(settings.BLIP_MODEL)
                    model.to(self._device)
                    model.eval()
                    return processor, model

                self._blip_processor, self._blip_model = self._safe_load(load, "BLIP")

        return self._blip_processor, self._blip_model, self._device

    # Reranker
    def get_reranker(self):
        if self._reranker:
            return self._reranker

        with self._lock:
            if not self._reranker:
                from sentence_transformers import CrossEncoder
                self._reranker = self._safe_load(
                    lambda: CrossEncoder(settings.RERANKER_MODEL, device=self._device),
                    "Reranker"
                )

        return self._reranker


model_loader = ModelLoader()