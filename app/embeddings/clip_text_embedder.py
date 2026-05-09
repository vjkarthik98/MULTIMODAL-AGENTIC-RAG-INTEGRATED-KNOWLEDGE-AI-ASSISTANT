import hashlib
import math
import time
from typing import Dict, List, Optional, Union

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F     = None

logger = get_logger(__name__)


# CLIP TOKEN LIMIT (hard limit from OpenAI CLIP architecture)
_CLIP_MAX_TOKEN_LENGTH = 77


class ClipTextEmbedder:

    def __init__(self, processor, model, device: str) -> None:

        if torch is None or F is None:
            raise ImportError("TORCH_REQUIRED_FOR_CLIP_TEXT_EMBEDDER")

        self.processor    = processor
        self.model        = model
        self.device       = device

        self.max_length   = min(
            getattr(settings, "CLIP_MAX_LENGTH", _CLIP_MAX_TOKEN_LENGTH),
            _CLIP_MAX_TOKEN_LENGTH,
        )
        self.expected_dim = settings.VISION_EMBEDDING_DIM
        self.batch_size   = settings.INGESTION_BATCH_SIZE

        logger.info(
            event="clip_text_embedder_initialized",
            device=device,
            max_length=self.max_length,
            dim=self.expected_dim,
        )

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # NORMALIZE

    def _normalize(self, text: str) -> str:
        import unicodedata
        text = unicodedata.normalize("NFC", text)
        return " ".join(text.strip().split())

    # PREPARE INPUTS

    def _prepare_inputs(self, text: Union[str, List[str]]) -> List[str]:

        if isinstance(text, str):
            text = [text]

        cleaned: List[str]       = []
        seen:    Dict[str, bool] = {}

        for t in text:
            t = str(t).strip()
            if not t:
                continue

            t = self._normalize(t)

            if len(t) > settings.MAX_PROMPT_CHARS:
                t = t[:settings.MAX_PROMPT_CHARS]

            h = self._hash(t)
            if h in seen:
                continue

            seen[h] = True
            cleaned.append(t)

        if not cleaned:
            raise ValueError("NO_VALID_CLIP_TEXT_INPUT")

        cap = settings.INGESTION_BATCH_SIZE * 10
        return cleaned[:cap]

    # EMBEDDING VALIDATION

    def _valid(self, emb: List[float]) -> bool:
        if not isinstance(emb, list):
            return False
        if len(emb) != self.expected_dim:
            return False
        if any(math.isnan(v) or math.isinf(v) for v in emb):
            return False
        return True

    # EMBED

    def embed(self, text: Union[str, List[str]], session_id: str = "default") -> List[List[float]]:

        start  = time.time()
        texts  = self._prepare_inputs(text)
        results: List[List[float]] = []

        t_target_sec = settings.LATENCY_TARGET_IMAGE_MS / 1000.0

        for i in range(0, len(texts), self.batch_size):
            batch   = texts[i:i + self.batch_size]
            t_batch = time.time()

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

                embeddings    = features.detach().cpu().numpy().tolist()
                batch_latency = time.time() - t_batch

                if batch_latency > t_target_sec:
                    logger.warning(
                        event="clip_text_batch_latency_exceeded",
                        latency=round(batch_latency, 3),
                        target=t_target_sec,
                        batch_size=len(batch),
                        session_id=session_id,
                    )

                for emb in embeddings:
                    if self._valid(emb):
                        results.append(emb)

            except Exception as e:
                logger.error(
                    event="clip_text_batch_failed",
                    batch_start=i,
                    error=str(e),
                    session_id=session_id,
                )

        if not results:
            raise ValueError("NO_CLIP_TEXT_EMBEDDINGS_PRODUCED")

        total_latency = round(time.time() - start, 3)
        throughput    = round(len(results) / max(total_latency, 1e-6), 1)

        logger.info(
            event="clip_text_embed_success",
            count=len(results),
            throughput_per_sec=throughput,
            latency=total_latency,
            session_id=session_id,
        )

        return results

    # SINGLE

    def embed_single(self, text: str, session_id: str = "default") -> List[float]:
        return self.embed(text, session_id=session_id)[0]

    # QUERY ALIAS (used by hybrid retriever vision path)

    def embed_query(self, query: str, session_id: str = "default") -> List[float]:
        return self.embed_single(query, session_id=session_id)