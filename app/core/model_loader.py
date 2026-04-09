from app.llm.gguf_model import GGUFModel
from app.embeddings.text_embedder import TextEmbedder
from app.embeddings.clip_text_embedder import ClipTextEmbedder
from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import CrossEncoder
import torch
import logging

# Logger
logger = logging.getLogger(__name__)


class ModelLoader:

    MODEL_CONFIG = {
        "llm": "gguf",
        "text_embedding": "all-MiniLM-L6-v2",
        "clip_text": "openai/clip-vit-large-patch14",
        "whisper": "base",
        "blip": "Salesforce/blip-image-captioning-large",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }

    def __init__(self):
        self.MODEL_CONFIG = {
            "llm": "gguf",
            "text_embedding": "all-MiniLM-L6-v2",
            "clip_text": "openai/clip-vit-large-patch14",
            "whisper": "base",
            "blip": "Salesforce/blip-image-captioning-large",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }

        self._llm = None
        self._embedder = None
        self._clip = None
        self._whisper = None
        self._blip_processor = None
        self._blip_model = None
        self._reranker = None

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # LLM
    def get_llm(self):
        if self._llm is None:
            logger.info("[ModelLoader] Loading LLM...")
            self._llm = self._load_llm()
        return self._llm

    def _load_llm(self):
        if self.MODEL_CONFIG["llm"] == "gguf":
            return GGUFModel()
        raise ValueError("Unsupported LLM type")

    # LLM Wrapper
    def generate(self, prompt: str) -> str:
        llm = self.get_llm()

        # Case 1: Langchain/OpenAI style
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            return response.content

        # Case 2: GGUF / llama.cpp style
        elif hasattr(llm, "__call__"):
            output = llm(prompt)

            if isinstance(output, dict):
                if "choices" in output:
                    return output["choices"][0]["text"]
                if "content" in output:
                    return output["content"]

            return str(output)

        # Case 3: custom models
        elif hasattr(llm, "generate"):
            return llm.generate(prompt)

        else:
            raise ValueError("Unsupported LLM interface")

    # TEXT EMBEDDING
    def get_embedder(self):
        if self._embedder is None:
            logger.info("[ModelLoader] Loading Text Embedder...")
            self._embedder = TextEmbedder(
                model_name=self.MODEL_CONFIG["text_embedding"]
            )
        return self._embedder

    # CLIP TEXT EMBEDDER
    def get_clip_text_embedder(self):
        if self._clip is None:
            logger.info("[ModelLoader] Loading CLIP Text Embedder...")
            self._clip = ClipTextEmbedder(
                model_name=self.MODEL_CONFIG["clip_text"]
            )
        return self._clip

    # WHISPER
    def get_whisper(self):
        if self._whisper is None:
            logger.info("[ModelLoader] Loading Whisper...")
            self._whisper = WhisperModel(
                self.MODEL_CONFIG["whisper"]
            )
        return self._whisper

    # BLIP
    def get_blip(self):
        if self._blip_model is None:
            logger.info("[ModelLoader] Loading BLIP...")
            self._blip_processor = BlipProcessor.from_pretrained(
                self.MODEL_CONFIG["blip"]
            )

            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                self.MODEL_CONFIG["blip"]
            ).to(self._device)

        return self._blip_processor, self._blip_model, self._device

    # Reranker
    def get_reranker(self):
        if self._reranker is None:
            logger.info("[ModelLoader] Loading Reranker...")
            self._reranker = CrossEncoder(
                self.MODEL_CONFIG["reranker"]
            )
        return self._reranker


model_loader = ModelLoader()