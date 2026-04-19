from app.core.config import settings
from app.utils.logger import get_logger
import threading

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


logger = get_logger(__name__)


class ModelLoader:
    MODEL_CONFIG = {
        "llm": "gguf",
        "text_embedding": settings.EMBEDDING_MODEL,
        "whisper": settings.WHISPER_MODEL,
        "blip": settings.BLIP_MODEL,
        "reranker": settings.RERANKER_MODEL,
        "clip": settings.CLIP_MODEL,
    }

    def __init__(self):
        self._llm = None
        self._embedder = None
        self._text_embedder = None
        self._clip_text_embedder = None
        self._whisper = None
        self._blip_processor = None
        self._blip_model = None
        self._reranker = None
        self._clip_model = None
        self._clip_processor = None
        self._lock = threading.Lock()
        
        self._device = "cuda" if torch and torch.cuda.is_available() else "cpu"

        logger.info(f"[ModelLoader] Device={self._device}")

    # LLM
    def get_llm(self):
        if self._llm is not None:
            return self._llm
        
        with self._lock:
            if self._llm is None:
                logger.info("[ModelLoader] Loading LLM...")
            
            if self.MODEL_CONFIG["llm"] != "gguf":
                raise ValueError("Unsupported LLM type")
            
            from app.llm.gguf_model import GGUFModel

            self._llm = GGUFModel()

            logger.info("[ModelLoader] LLM loaded")

        return self._llm

    # GENERATION WRAPPER
    def generate(self, prompt: str) -> str:
        llm = self.get_llm()

        try:
            return llm.generate(prompt)
        except Exception as e:
            logger.error(f"[ModelLoader] Generation failed: {str(e)}")
            raise

    # EMBEDDERS
    def get_text_embedding_model(self):
        if self._embedder is not None:
            return self._embedder 
        
        with self._lock:
            if self._embedder is None: 
                logger.info("[ModelLoader] Loading SentenceTransformer...")
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(
                    self.MODEL_CONFIG["text_embedding"],
                    device=self._device,
                )
            return self._embedder

    def get_embedder(self):
        if self._text_embedder is not None:
            return self._text_embedder 
            
        with self._lock: 
            if self._text_embedder is None:
                from app.embeddings.text_embedder import TextEmbedder
                self._text_embedder = TextEmbedder()
            return self._text_embedder

    # WHISPER
    def get_whisper(self):
        if self._whisper is not None:
            return self._whisper
        
        with self._lock:
            if self._whisper is None: 
                logger.info("[ModelLoader] Loading Whisper...")
                from faster_whisper import WhisperModel

                compute_type = "float16" if self._device == "cuda" else "int8"
                
                self._whisper = WhisperModel(
                    self.MODEL_CONFIG["whisper"],
                    device=self._device,
                    compute_type=compute_type,
                )
            return self._whisper

    # BLIP
    def get_blip(self):
        if self._blip_model is not None:
            return self._blip_processor, self._blip_model, self._device
        
        with self._lock:
            if self._blip_model is None:
                logger.info("[ModelLoader] Loading BLIP...")
                from transformers import BlipForConditionalGeneration, BlipProcessor

                self._blip_processor = BlipProcessor.from_pretrained(self.MODEL_CONFIG["blip"])
                self._blip_model = BlipForConditionalGeneration.from_pretrained(
                self.MODEL_CONFIG["blip"]
                ).to(self._device)
                self._blip_model.eval()

            return self._blip_processor, self._blip_model, self._device

    # CLIP
    def get_clip(self):
        if self._clip_model is not None:
            return self._clip_processor, self._clip_model, self._device 
        
        with self._lock:
            if self._clip_model is None:
                logger.info("[ModelLoader] Loading CLIP...")
                from transformers import CLIPModel, CLIPProcessor

                self._clip_processor = CLIPProcessor.from_pretrained(self.MODEL_CONFIG["clip"])
                self._clip_model = CLIPModel.from_pretrained(self.MODEL_CONFIG["clip"]).to(
                    self._device
                )
                self._clip_model.eval()

            return self._clip_processor, self._clip_model, self._device

    def get_clip_text_embedder(self):
        if self._clip_text_embedder is not None:
            return self._clip_text_embedder
        
        with self._lock:
            if self._clip_text_embedder is None: 

                from app.embeddings.clip_text_embedder import ClipTextEmbedder

                self._clip_text_embedder = ClipTextEmbedder()
            return self._clip_text_embedder
    
    # RERANKER
    def get_reranker(self):
        if self._reranker is not None:
            return self._reranker 
        
        with self._lock: 
            if self._reranker is None:  
                logger.info("[ModelLoader] Loading Reranker...")
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    self.MODEL_CONFIG["reranker"],
                    device=self._device,
                )
            return self._reranker

    def rerank(self, query: str, documents: list[str], top_k: int = 5):
        reranker = self.get_reranker()

        pairs = [(query, document) for document in documents]

        scores = reranker.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x:x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


model_loader = ModelLoader()

