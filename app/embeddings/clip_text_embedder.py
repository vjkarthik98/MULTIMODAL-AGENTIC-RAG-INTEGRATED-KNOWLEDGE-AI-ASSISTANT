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
            raise ImportError("TORCH REQUIRED FOR CLIP")

    
        self.processor = processor
        self.model = model
        self.device = device

        self.max_length = getattr(settings, "CLIP_MAX_LENGTH", 77)
        self.expected_dim = settings.VISION_EMBEDDING_DIM

        logger.info("[ClipTextEmbedder] initialized | device=%s", self.device)

    # NORMALIZE TEXT
    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().split())

    # PREPARE INPUTS
    def _prepare_inputs(self, text: Union[str, List[str]]) -> List[str]:

        if isinstance(text, str):
            text = [text]

        cleaned = []
        seen: Dict[str, int] = {}

        for t in text:
            if not t or not str(t).strip():
                continue

            t = self._normalize(str(t))

            # SAFE TRUNCATION
            if len(t) > settings.MAX_PROMPT_CHARS:
                t = t[:settings.MAX_PROMPT_CHARS]

            key = t[:100]

            # DEDUPLICATION
            if key in seen:
                continue

            seen[key] = 1
            cleaned.append(t)

        if not cleaned:
            raise ValueError("NO VALID TEXT INPUTS")

        # HARD LIMIT
        max_batch = getattr(settings, "MAX_PARALLEL_REQUESTS", 100)
        return cleaned[:max_batch]

    # VALIDATE EMBEDDING
    def _validate_embedding(self, emb: List[float]) -> bool:

        if not emb or not isinstance(emb, list):
            return False

        if len(emb) != self.expected_dim:
            logger.warning(
                "[ClipTextEmbedder] DIM MISMATCH expected=%s got=%s",
                self.expected_dim,
                len(emb)
            )
            return False

        return True

    # MAIN EMBED FUNCTION
    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:

        start = time.time()

        texts = self._prepare_inputs(text)

        results = []

        # SAFE BATCH PROCESSING
        for i in range(0, len(texts), settings.INGESTION_BATCH_SIZE):

            batch = texts[i:i + settings.INGESTION_BATCH_SIZE]

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
                    if self._validate_embedding(emb):
                        results.append(emb)

            except Exception as e:
                logger.error("[ClipTextEmbedder][BATCH_FAIL] %s", str(e))
                continue

        if not results:
            raise ValueError("NO VALID EMBEDDINGS GENERATED")

        logger.info(
            "[ClipTextEmbedder][SUCCESS] count=%s | latency=%.2fs",
            len(results),
            time.time() - start
        )

        return results

    # SINGLE EMBEDDING
    def embed_single(self, text: str) -> List[float]:
        return self.embed(text)[0]