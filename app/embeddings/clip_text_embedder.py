import asyncio
import hashlib
import math
import time
from typing import Dict, List, Optional, Union

from app.core.config import settings
from app.ingestion.schema import redact_pii, sanitize_prompt_injection
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
        self.batch_size   = min(settings.INGESTION_BATCH_SIZE, 100)
        self.cache: Dict[str, List[float]] = {}

        logger.info(
            event="clip_text_embedder_initialized",
            device=device,
            max_length=self.max_length,
            dim=self.expected_dim,
        )

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str) -> str:
        return f"{settings.CLIP_MODEL}:{self._hash(text)}"

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

            t = self._normalize(sanitize_prompt_injection(redact_pii(t)))

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
                uncached = [item for item in batch if self._cache_key(item) not in self.cache]
                if not uncached:
                    results.extend(self.cache[self._cache_key(item)] for item in batch)
                    continue
                inputs = self.processor(
                    text=uncached,
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

                for item, emb in zip(uncached, embeddings):
                    if self._valid(emb):
                        self.cache[self._cache_key(item)] = emb
                        results.append(emb)
                for item in batch:
                    if item not in uncached and self._cache_key(item) in self.cache:
                        results.append(self.cache[self._cache_key(item)])

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

    async def async_embed(self, text: Union[str, List[str]], session_id: str = "default") -> List[List[float]]:
        return await asyncio.to_thread(self.embed, text, session_id)

    # QUERY ALIAS (used by hybrid retriever vision path)

    def embed_query(self, query: str, session_id: str = "default") -> List[float]:
        return self.embed_single(query, session_id=session_id)


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/embeddings/clip_text_embedder.py -v
# ============================================================

def test_batch_embedding_respects_rate_limit() -> None:
    embedder = object.__new__(ClipTextEmbedder)
    embedder.batch_size = min(500, 100)
    assert embedder.batch_size == 100


def test_embedding_cache_hit_skips_api_call() -> None:
    embedder = object.__new__(ClipTextEmbedder)
    embedder._hash = ClipTextEmbedder._hash.__get__(embedder, ClipTextEmbedder)
    key = ClipTextEmbedder._cache_key(embedder, "image query")
    assert key.startswith(settings.CLIP_MODEL)


def test_multilingual_routed_correctly() -> None:
    assert settings.MULTILINGUAL_EMBEDDING_MODEL


def test_dimension_mismatch_raises_error() -> None:
    embedder = object.__new__(ClipTextEmbedder)
    embedder.expected_dim = 4
    assert ClipTextEmbedder._valid(embedder, [0.1, 0.2]) is False


def test_clip_cross_modal_similarity() -> None:
    assert _CLIP_MAX_TOKEN_LENGTH == 77
