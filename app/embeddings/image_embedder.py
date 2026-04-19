from PIL import Image

from app.core.model_loader import model_loader
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as functional
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    functional = None


logger = get_logger(__name__)


class ImageEmbedder:
    def __init__(self):
        processor, model, device = model_loader.get_clip()
        self.processor = processor
        self.model = model
        self.device = device
        logger.info("[ImageEmbedder] CLIP model loaded")

    def embed(self, image_path: str):
        if not image_path:
            raise ValueError("image_path is required")
        if torch is None or functional is None:
            raise ImportError("torch is required for image embeddings")

        try:
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = functional.normalize(image_features, p=2, dim=-1)

            embedding = image_features[0].detach().cpu().numpy()
            logger.debug("[ImageEmbedder] embedding_dim=%s", len(embedding))
            return embedding.tolist()

        except Exception as exc:
            logger.error("[ImageEmbedder][FAILED] image=%s | error=%s", image_path, exc)
            raise
