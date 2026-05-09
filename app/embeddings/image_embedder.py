import hashlib
import math
import time
from pathlib import Path
from typing import Dict, List, Union

from PIL import Image, ImageOps

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F     = None

logger = get_logger(__name__)


class ImageEmbedder:

    def __init__(self, model, processor, device: str) -> None:

        if torch is None or F is None:
            raise ImportError("TORCH_REQUIRED_FOR_IMAGE_EMBEDDER")

        self.processor    = processor
        self.model        = model
        self.device       = device

        self.max_image_dim = settings.MAX_IMAGE_DIM
        self.expected_dim  = settings.VISION_EMBEDDING_DIM
        self.batch_size    = settings.INGESTION_BATCH_SIZE

        logger.info(
            event="image_embedder_initialized",
            device=device,
            max_dim=self.max_image_dim,
            dim=self.expected_dim,
        )

    # HASH

    def _hash(self, path: Path) -> str:
        return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()

    # EMBEDDING VALIDATION

    def _valid_embedding(self, emb: List[float]) -> bool:
        if not isinstance(emb, list):
            return False
        if len(emb) != self.expected_dim:
            return False
        if any(math.isnan(v) or math.isinf(v) for v in emb):
            return False
        return True

    # IMAGE LOADING

    def _load_images(
        self,
        paths: List[Union[str, Path]],
        session_id: str = "default",
    ) -> List[Image.Image]:

        images:              List[Image.Image] = []
        seen:    Dict[str, bool]               = {}

        for p in paths:
            try:
                path = Path(p)

                if not path.exists():
                    logger.warning(
                        event="image_path_not_found",
                        path=str(path),
                        session_id=session_id,
                    )
                    continue

                if path.stat().st_size > settings.MAX_FILE_SIZE_IMAGE:
                    logger.warning(
                        event="image_too_large_skipped",
                        path=path.name,
                        size=path.stat().st_size,
                        session_id=session_id,
                    )
                    continue

                h = self._hash(path)
                if h in seen:
                    continue
                seen[h] = True

                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)

                    if img.mode != "RGB":
                        logger.debug(
                            event="image_mode_converted",
                            mode=img.mode,
                            path=path.name,
                            session_id=session_id,
                        )
                        img = img.convert("RGB")

                    w, h_px = img.size

                    if w < 32 or h_px < 32:
                        logger.warning(
                            event="image_too_small_skipped",
                            width=w,
                            height=h_px,
                            path=path.name,
                            session_id=session_id,
                        )
                        continue

                    if max(w, h_px) > self.max_image_dim:
                        img.thumbnail(
                            (self.max_image_dim, self.max_image_dim),
                            Image.LANCZOS,
                        )

                    images.append(img.copy())

            except Exception as e:
                logger.warning(
                    event="image_load_failed",
                    path=str(p),
                    error=str(e),
                    session_id=session_id,
                )

        if not images:
            raise ValueError("NO_VALID_IMAGES_TO_EMBED")

        cap = settings.INGESTION_BATCH_SIZE * 10
        return images[:cap]

    # SINGLE EMBED

    def embed(
        self,
        image_path: Union[str, Path],
        session_id: str = "default",
    ) -> List[float]:
        return self.embed_batch([image_path], session_id=session_id)[0]

    # BATCH EMBED

    def embed_batch(
        self,
        image_paths: List[Union[str, Path]],
        session_id: str = "default",
    ) -> List[List[float]]:

        start   = time.time()
        images  = self._load_images(image_paths, session_id=session_id)
        results: List[List[float]] = []

        t_target_sec = settings.LATENCY_TARGET_IMAGE_MS / 1000.0

        for i in range(0, len(images), self.batch_size):
            batch   = images[i:i + self.batch_size]
            t_batch = time.time()

            try:
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                    features = F.normalize(features, p=2, dim=-1)

                embeddings    = features.detach().cpu().numpy().tolist()
                batch_latency = time.time() - t_batch

                if batch_latency > t_target_sec:
                    logger.warning(
                        event="image_embed_batch_latency_exceeded",
                        latency=round(batch_latency, 3),
                        target=t_target_sec,
                        batch_size=len(batch),
                        session_id=session_id,
                    )

                for emb in embeddings:
                    if self._valid_embedding(emb):
                        results.append(emb)

            except Exception as e:
                logger.error(
                    event="image_embed_batch_failed",
                    batch_start=i,
                    error=str(e),
                    session_id=session_id,
                )

        if not results:
            raise ValueError("NO_IMAGE_EMBEDDINGS_PRODUCED")

        total_latency = round(time.time() - start, 3)
        throughput    = round(len(results) / max(total_latency, 1e-6), 1)

        logger.info(
            event="image_embed_success",
            count=len(results),
            throughput_per_sec=throughput,
            latency=total_latency,
            session_id=session_id,
        )

        return results