from typing import List, Union
from pathlib import Path

from PIL import Image

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


class ImageEmbedder:
    def __init__(self):
        if torch is None or F is None:
            raise ImportError("torch is required for image embeddings")

        processor, model, device = model_loader.get_clip()

        self.processor = processor
        self.model = model
        self.device = device

        self.max_image_size = getattr(settings, "MAX_IMAGE_DIM", 1024)

        logger.info("[ImageEmbedder] initialized | device=%s", self.device)

    def embed(self, image_path: Union[str, Path]) -> List[float]:
        return self.embed_batch([image_path])[0]

    def embed_batch(self, image_paths: List[Union[str, Path]]) -> List[List[float]]:
        images = self._load_images(image_paths)

        try:
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = F.normalize(features, p=2, dim=-1)

            embeddings = features.detach().cpu().numpy().tolist()

            # Validate embedding dimension
            expected_dim = settings.VISION_EMBEDDING_DIM
            if embeddings and len(embeddings[0]) != expected_dim:
                logger.warning(
                    "[ImageEmbedder] dimension mismatch | expected=%s got=%s",
                    expected_dim,
                    len(embeddings[0])
                )

            return embeddings

        except Exception as e:
            logger.error("[ImageEmbedder][FAILED] %s", str(e))
            raise

    def _load_images(self, image_paths: List[Union[str, Path]]) -> List[Image.Image]:
        loaded_images = []

        for path in image_paths:
            try:
                path = Path(path)

                if not path.exists():
                    logger.warning("[ImageEmbedder] file not found: %s", path)
                    continue

                if path.stat().st_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                    logger.warning("[ImageEmbedder] file too large: %s", path)
                    continue

                with Image.open(path) as img:
                    img = img.convert("RGB")

                    # Resize if too large
                    if max(img.size) > self.max_image_size:
                        img.thumbnail((self.max_image_size, self.max_image_size))

                    loaded_images.append(img.copy())

            except Exception as e:
                logger.warning("[ImageEmbedder] failed to load image=%s | %s", path, str(e))
                continue

        if not loaded_images:
            raise ValueError("No valid images to embed")

        return loaded_images