from app.core.model_loader import model_loader
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as functional
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    functional = None


logger = get_logger(__name__)


class ClipTextEmbedder:
    def __init__(self):
        processor, model, device = model_loader.get_clip()
        self.processor = processor
        self.model = model
        self.device = device
        logger.info("[ClipTextEmbedder] CLIP model loaded")

    def embed(self, text: str):
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("text cannot be empty")
        if torch is None or functional is None:
            raise ImportError("torch is required for CLIP text embeddings")

        try:
            inputs = self.processor(
                text=[clean_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = functional.normalize(text_features, p=2, dim=-1)

            embedding = text_features[0].detach().cpu().numpy()
            logger.debug("[ClipTextEmbedder] embedding_dim=%s", len(embedding))
            return embedding.tolist()

        except Exception as exc:
            logger.error("[ClipTextEmbedder][FAILED] error=%s", exc)
            raise
