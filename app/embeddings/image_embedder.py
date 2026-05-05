import hashlib
import time
from typing import List, Union, Dict
from pathlib import Path

from PIL import Image, ImageOps

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


logger = get_logger(__name__)


class ImageEmbedder:

    def __init__(self, model, processor, device):

        if torch is None or F is None:
            raise ImportError("TORCH_REQUIRED")

        self.processor = processor
        self.model = model
        self.device = device

        self.max_image_size = getattr(settings, "MAX_IMAGE_DIM", 1024)
        self.expected_dim = settings.VISION_EMBEDDING_DIM
        self.batch_size = getattr(settings, "INGESTION_BATCH_SIZE", 32)

        logger.info(
            event="image_embedder_initialized",
            device=device
        )

    #  HASH 
    def _hash(self, path: Path) -> str:
        return hashlib.sha256(str(path.resolve()).encode()).hexdigest()

    #  VALIDATE 
    def _valid_embedding(self, emb: List[float]) -> bool:
        return isinstance(emb, list) and len(emb) == self.expected_dim

    #  LOAD 
    def _load_images(self, paths: List[Union[str, Path]]):

        images = []
        seen: Dict[str, bool] = {}

        for p in paths:
            try:
                path = Path(p)

                if not path.exists():
                    continue

                if path.stat().st_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                    continue

                h = self._hash(path)
                if h in seen:
                    continue
                seen[h] = True

                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")

                    if img.size[0] < 32 or img.size[1] < 32:
                        continue

                    if max(img.size) > self.max_image_size:
                        img.thumbnail((self.max_image_size, self.max_image_size))

                    images.append(img.copy())

            except Exception as e:
                logger.warning(event="image_load_failed", error=str(e))

        if not images:
            raise ValueError("NO_VALID_IMAGES")

        limit = getattr(settings, "MAX_PARALLEL_REQUESTS", 100)
        return images[:limit]

    #  SINGLE 
    def embed(self, image_path: Union[str, Path]) -> List[float]:
        return self.embed_batch([image_path])[0]

    #  BATCH 
    def embed_batch(self, image_paths: List[Union[str, Path]]) -> List[List[float]]:

        start = time.time()

        images = self._load_images(image_paths)
        results = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]

            try:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                    features = F.normalize(features, p=2, dim=-1)

                embeddings = features.detach().cpu().numpy().tolist()

                for emb in embeddings:
                    if self._valid_embedding(emb):
                        results.append(emb)

            except Exception as e:
                logger.error(event="image_batch_failed", error=str(e))

        if not results:
            raise ValueError("NO_EMBEDDINGS")

        logger.info(
            event="image_embed_success",
            count=len(results),
            latency=round(time.time() - start, 3)
        )

        return results