import time
from typing import List, Union, Dict
from pathlib import Path

from PIL import Image

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
            raise ImportError("TORCH REQUIRED FOR IMAGE EMBEDDING")

        self.processor = processor
        self.model = model
        self.device = device

        self.max_image_size = getattr(settings, "MAX_IMAGE_DIM", 1024)
        self.expected_dim = settings.VISION_EMBEDDING_DIM

        logger.info("[ImageEmbedder] initialized | device=%s", self.device)

    # SINGLE EMBEDDING
    def embed(self, image_path: Union[str, Path]) -> List[float]:
        return self.embed_batch([image_path])[0]

    # BATCH EMBEDDING (SAFE)
    def embed_batch(self, image_paths: List[Union[str, Path]]) -> List[List[float]]:

        start = time.time()

        images, valid_paths = self._load_images(image_paths)

        results = []

        # SAFE BATCH PROCESSING
        for i in range(0, len(images), settings.INGESTION_BATCH_SIZE):

            batch_images = images[i:i + settings.INGESTION_BATCH_SIZE]

            try:
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                    features = F.normalize(features, p=2, dim=-1)

                embeddings = features.detach().cpu().numpy().tolist()

                for emb in embeddings:
                    if self._validate_embedding(emb):
                        results.append(emb)

            except Exception as e:
                logger.error("[ImageEmbedder][BATCH_FAIL] %s", str(e))
                continue

        if not results:
            raise ValueError("NO VALID IMAGE EMBEDDINGS")

        logger.info(
            "[ImageEmbedder][SUCCESS] count=%s | latency=%.2fs",
            len(results),
            time.time() - start
        )

        return results

    # VALIDATE EMBEDDING
    def _validate_embedding(self, emb: List[float]) -> bool:

        if not emb or not isinstance(emb, list):
            return False

        if len(emb) != self.expected_dim:
            logger.warning(
                "[ImageEmbedder] DIM MISMATCH expected=%s got=%s",
                self.expected_dim,
                len(emb)
            )
            return False

        return True

    # LOAD AND PREPROCESS IMAGES
    def _load_images(self, image_paths: List[Union[str, Path]]):

        loaded_images = []
        valid_paths = []

        seen: Dict[str, int] = {}

        for path in image_paths:
            try:
                path = Path(path)

                if not path.exists():
                    logger.warning("[ImageEmbedder] NOT FOUND: %s", path)
                    continue

                # DEDUPLICATION
                key = str(path.resolve())
                if key in seen:
                    continue
                seen[key] = 1

                # SIZE CHECK
                if path.stat().st_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                    logger.warning("[ImageEmbedder] TOO LARGE: %s", path)
                    continue

                with Image.open(path) as img:

                    # NORMALIZE IMAGE
                    img = img.convert("RGB")

                    # RESIZE
                    if max(img.size) > self.max_image_size:
                        img.thumbnail((self.max_image_size, self.max_image_size))

                    loaded_images.append(img.copy())
                    valid_paths.append(str(path))

            except Exception as e:
                logger.warning("[ImageEmbedder] LOAD FAIL %s | %s", path, str(e))
                continue

        if not loaded_images:
            raise ValueError("NO VALID IMAGES")

        # HARD LIMIT
        max_batch = getattr(settings, "MAX_PARALLEL_REQUESTS", 100)

        return loaded_images[:max_batch], valid_paths[:max_batch]