from app.llm.gguf_model import GGUFModel
from app.embeddings.text_embedder import TextEmbedder
from app.embeddings.clip_text_embedder import ClipTextEmbedder
from transformers import CLIPModel, CLIPProcessor
from app.embeddings.multimodal_embedder import MultimodalEmbedder
from app.embeddings.image_embedder import ImageEmbedder
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.utils.logger import get_logger

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

try:
    import torch
except ImportError:
    torch = None


logger = get_logger(__name__)


class ModelLoader:
    def __init__(self):
        self._lock = threading.RLock()
        self._device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        self._executor = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)

        # MODELS
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
        self._multimodal = None

        # STATE
        self._initialized = False

        logger.info("[ModelLoader] initialized | device=%s", self._device)

    
    # SAFE LOAD WITH TIMEOUT
    def _safe_load(self, load_fn, name: str):
        start = time.time()

        try:
            future = self._executor.submit(load_fn)
            obj = future.result(timeout=settings.MODEL_TIMEOUT_SEC)

            logger.info("[ModelLoader] %s loaded | %.2fs", name, time.time() - start)
            return obj

        except TimeoutError:
            logger.error("[ModelLoader] %s load timeout", name)
            return None

        except Exception as e:
            logger.error("[ModelLoader] %s load failed | %s", name, str(e))
            raise

    
    # WARMUP (PRODUCTION CRITICAL)
    def warmup(self):
        if self._initialized:
            logger.info("[ModelLoader] Warmup already done. Skipping.")
            return

        with self._lock:
            if self._initialized:
                return

            logger.info("[ModelLoader] Warmup started")

            try:
                self.get_llm()
                self.get_embedder()
                self.get_clip()
                self.get_whisper()
                self.get_blip()
                self.get_reranker()

                self._initialized = True

                logger.info("[ModelLoader] Warmup completed")

            except Exception as e:
                logger.error("[ModelLoader] Warmup failed | %s", str(e))
                raise

    
    # LLM
    def get_llm(self):
        if self._llm:
            return self._llm

        with self._lock:
            if not self._llm:
                path = settings.LLM_MODEL_PATH

                if not path:
                    raise ValueError("LLM_MODEL_PATH missing")

                self._llm = self._safe_load(
                    lambda: GGUFModel(path),
                    "LLM"
                )

        return self._llm

    
    # TEXT EMBEDDER
    def get_embedder(self):
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

    
    # MULTIMODAL
    def get_multimodal_embedder(self):
        if self._multimodal:
            return self._multimodal

        with self._lock:
            if not self._multimodal:
                text_embedder = self.get_embedder()
                image_embedder = self.get_image_embedder()

                if not text_embedder or not image_embedder:
                    raise RuntimeError("Multimodal dependencies failed")

                self._multimodal = MultimodalEmbedder(
                    text_embedder,
                    image_embedder
                )

        return self._multimodal

    
    # CLIP
    def get_clip(self):
        if self._clip_model:
            return self._clip_processor, self._clip_model, self._device

        with self._lock:
            if not self._clip_model:

                def load():
                    processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
                    model = CLIPModel.from_pretrained(
                        settings.CLIP_MODEL,
                        low_cpu_mem_usage=True
                    )

                    model.to(self._device)
                    model.eval()
                    return processor, model

                self._clip_processor, self._clip_model = self._safe_load(load, "CLIP")

        return self._clip_processor, self._clip_model, self._device

    
    # IMAGE EMBEDDER
    def get_image_embedder(self):
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

    
    # CLIP TEXT
    def get_clip_text_embedder(self):
        if self._clip_text_embedder:
            return self._clip_text_embedder

        with self._lock:
            if not self._clip_text_embedder:
                processor, model, device = self.get_clip()

                self._clip_text_embedder = self._safe_load(
                    lambda: ClipTextEmbedder(processor, model, device),
                    "CLIPTextEmbedder"
                )

        return self._clip_text_embedder

    
    # WHISPER
    def get_whisper(self):
        if self._whisper:
            return self._whisper

        with self._lock:
            if not self._whisper:
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

                def load():
                    processor = BlipProcessor.from_pretrained(settings.BLIP_MODEL)
                    model = BlipForConditionalGeneration.from_pretrained(settings.BLIP_MODEL)

                    model.to(self._device)
                    model.eval()
                    return processor, model

                self._blip_processor, self._blip_model = self._safe_load(load, "BLIP")

        return self._blip_processor, self._blip_model, self._device

    
    # RERANKER
    def get_reranker(self):
        if self._reranker:
            return self._reranker

        with self._lock:
            if not self._reranker:
                self._reranker = self._safe_load(
                    lambda: CrossEncoder(settings.RERANKER_MODEL, device=self._device),
                    "Reranker"
                )

        return self._reranker

    
    # HEALTH CHECK
    def health_check(self):
        return {
            "llm": self._llm is not None,
            "embedder": self._text_embedder is not None,
            "clip": self._clip_model is not None,
            "whisper": self._whisper is not None,
            "reranker": self._reranker is not None,
        }

    
    # RESET
    def reset(self):
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

            logger.warning("[ModelLoader] all models reset")


# SINGLETON INSTANCE
model_loader = ModelLoader()