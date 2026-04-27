from typing import List, Union

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


logger = get_logger(__name__)


class ClipTextEmbedder:
    def __init__(self):
        if torch is None or F is None:
            raise ImportError("torch is required for CLIP embeddings")

        processor, model, device = model_loader.get_clip()

        self.processor = processor
        self.model = model
        self.device = device

        self.max_length = getattr(settings, "CLIP_MAX_LENGTH", 77)

        logger.info("[ClipTextEmbedder] initialized | device=%s", self.device)

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        texts = self._prepare_inputs(text)

        try:
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = F.normalize(features, p=2, dim=-1)

            embeddings = features.detach().cpu().numpy().tolist()

            # Validate dimension
            expected_dim = settings.VISION_EMBEDDING_DIM
            if embeddings and len(embeddings[0]) != expected_dim:
                logger.warning(
                    "[ClipTextEmbedder] dimension mismatch | expected=%s got=%s",
                    expected_dim,
                    len(embeddings[0])
                )

            return embeddings

        except Exception as e:
            logger.error("[ClipTextEmbedder][FAILED] %s", str(e))
            raise

    def embed_single(self, text: str) -> List[float]:
        return self.embed(text)[0]

    def _prepare_inputs(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            text = [text]

        cleaned = []
        for t in text:
            if not t or not str(t).strip():
                continue

            t = str(t).strip()

            # Safe truncation
            if len(t) > settings.MAX_PROMPT_CHARS:
                t = t[:settings.MAX_PROMPT_CHARS]

            cleaned.append(t)

        if not cleaned:
            raise ValueError("No valid text inputs provided")

        return cleaned