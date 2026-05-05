import hashlib
import time
from typing import List, Union, Dict

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


logger = get_logger(__name__)


class ClipTextEmbedder:

    def __init__(self, processor, model, device):

        if torch is None or F is None:
            raise ImportError("TORCH_REQUIRED")

        self.processor = processor
        self.model = model
        self.device = device

        self.max_length = getattr(settings, "CLIP_MAX_LENGTH", 77)
        self.expected_dim = settings.VISION_EMBEDDING_DIM
        self.batch_size = getattr(settings, "INGESTION_BATCH_SIZE", 32)

        logger.info(
            event="clip_text_embedder_initialized",
            device=device
        )

    #  HASH 
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    #  NORMALIZE 
    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().split())

    #  PREPARE 
    def _prepare_inputs(self, text: Union[str, List[str]]) -> List[str]:

        if isinstance(text, str):
            text = [text]

        cleaned = []
        seen: Dict[str, bool] = {}

        for t in text:
            t = str(t).strip()
            if not t:
                continue

            t = self._normalize(t)

            # soft limit
            if len(t) > settings.MAX_PROMPT_CHARS:
                t = t[:settings.MAX_PROMPT_CHARS]

            h = self._hash(t)
            if h in seen:
                continue

            seen[h] = True
            cleaned.append(t)

        if not cleaned:
            raise ValueError("NO_VALID_INPUT")

        limit = getattr(settings, "MAX_PARALLEL_REQUESTS", 100)
        return cleaned[:limit]

    #  VALIDATE 
    def _valid(self, emb: List[float]) -> bool:
        return isinstance(emb, list) and len(emb) == self.expected_dim

    #  MAIN 
    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:

        start = time.time()

        texts = self._prepare_inputs(text)
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                    features = F.normalize(features, p=2, dim=-1)

                embeddings = features.detach().cpu().numpy().tolist()

                for emb in embeddings:
                    if self._valid(emb):
                        results.append(emb)

            except Exception as e:
                logger.error(event="clip_batch_failed", error=str(e))

        if not results:
            raise ValueError("NO_EMBEDDINGS")

        logger.info(
            event="clip_embed_success",
            count=len(results),
            latency=round(time.time() - start, 3)
        )

        return results

    #  SINGLE 
    def embed_single(self, text: str) -> List[float]:
        return self.embed(text)[0]