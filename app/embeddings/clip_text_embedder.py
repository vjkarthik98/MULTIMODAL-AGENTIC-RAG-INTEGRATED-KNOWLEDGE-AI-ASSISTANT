from __future__ import annotations

import asyncio
import hashlib
import math
import time
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Union

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

logger = get_logger(__name__)


# CLIP ARCHITECTURE HARD LIMIT

_CLIP_MAX_TOKEN_LENGTH: int = 77


# PROMETHEUS METRICS

def _make_metrics():
    if not settings.PROMETHEUS_ENABLED:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def labels(self, **kw): return self
        n = _Noop()
        return n, n, n, n
    try:
        from prometheus_client import Counter, Histogram
        embed_latency = Histogram(
            "clip_text_embedding_latency_seconds",
            "CLIP text embedding batch latency",
            ["batch_size"],
        )
        embed_errors = Counter(
            "clip_text_embedding_errors_total",
            "CLIP text embedding failures",
            ["reason"],
        )
        embed_total = Counter(
            "clip_text_embeddings_produced_total",
            "Total CLIP text embeddings produced",
        )
        cache_hits = Counter(
            "clip_text_embedding_cache_hits_total",
            "Redis CLIP text embedding cache hits",
        )
        return embed_latency, embed_errors, embed_total, cache_hits
    except Exception:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def labels(self, **kw): return self
        n = _Noop()
        return n, n, n, n


_EMBED_LATENCY, _EMBED_ERRORS, _EMBED_TOTAL, _CACHE_HITS = _make_metrics()


# SEMAPHORE

_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)
    return _SEMAPHORE


# CUSTOM EXCEPTIONS

class ClipTextEmbedderError(Exception):
    """Base exception for CLIP text embedding errors."""


class DimensionMismatchError(ClipTextEmbedderError):
    """Raised when embedding dimension does not match expected."""


class NoValidTextsError(ClipTextEmbedderError):
    """Raised when all input texts fail validation."""


class TorchNotAvailableError(ClipTextEmbedderError):
    """Raised when PyTorch is not installed."""


# EMBEDDING RESULT MODEL

class ClipTextEmbeddingResult:
    """Structured result for a single CLIP text embedding."""

    def __init__(
        self,
        text_preview: str,
        embedding: List[float],
        embedding_dim: int,
        model_name: str,
        checksum_sha256: str,
        token_count_estimate: int,
        was_truncated: bool,
        embedding_id: str,
        latency_ms: float,
        cache_hit: bool,
        session_id: str,
        language: Optional[str] = None,
    ) -> None:
        self.text_preview          = text_preview
        self.embedding             = embedding
        self.embedding_dim         = embedding_dim
        self.model_name            = model_name
        self.checksum_sha256       = checksum_sha256
        self.token_count_estimate  = token_count_estimate
        self.was_truncated         = was_truncated
        self.embedding_id          = embedding_id
        self.latency_ms            = latency_ms
        self.cache_hit             = cache_hit
        self.session_id            = session_id
        self.language              = language

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_preview":         self.text_preview,
            "embedding":            self.embedding,
            "embedding_dim":        self.embedding_dim,
            "model_name":           self.model_name,
            "checksum_sha256":      self.checksum_sha256,
            "token_count_estimate": self.token_count_estimate,
            "was_truncated":        self.was_truncated,
            "embedding_id":         self.embedding_id,
            "latency_ms":           self.latency_ms,
            "cache_hit":            self.cache_hit,
            "session_id":           self.session_id,
            "language":             self.language,
        }


# TEXT NORMALISATION

def _normalize_text(text: str) -> str:
    """NFC normalisation + whitespace collapse."""
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# NULL BYTE + BOM STRIP

def _sanitize_text(text: str) -> str:
    """Strip null bytes, BOM, and control characters."""
    text = text.replace("\x00", "")
    text = text.lstrip("\ufeff")
    return text


# PROMPT INJECTION GUARD

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard the above",
    "forget everything",
    "you are now",
    "act as ",
    "jailbreak",
    "system prompt",
    "new instructions",
]


def _sanitize_injection(text: str) -> str:
    """Strip prompt injection patterns before embedding."""
    lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            idx  = lower.find(pattern)
            text = text[:idx].strip()
            lower = text.lower()
            logger.warning(
                event="clip_text_injection_pattern_stripped",
                pattern=pattern,
            )
    return text


# TOKEN ESTIMATE (WORD-BASED PROXY)

def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


# TRUNCATION TO CLIP TOKEN LIMIT

def _truncate_to_clip_limit(text: str, max_chars: int) -> tuple[str, bool]:
    """
    Truncate text to stay within CLIP's 77-token hard limit.
    Returns (truncated_text, was_truncated).
    """
    if len(text) <= max_chars:
        return text, False
    # WORD-BOUNDARY TRUNCATION
    words     = text.split()
    truncated = ""
    for word in words:
        candidate = (truncated + " " + word).strip()
        if len(candidate) > max_chars:
            break
        truncated = candidate
    if not truncated:
        truncated = text[:max_chars]
    return truncated, True


# SHA-256 HASH

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# REDIS CACHE HELPERS

def _cache_key(checksum: str) -> str:
    return f"clip_txt_emb:{checksum}"


def _cache_get(checksum: str) -> Optional[List[float]]:
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem is None:
            return None
        raw = mem.cache_get(_cache_key(checksum))
        if raw and isinstance(raw, list):
            return raw
    except Exception:
        pass
    return None


def _cache_set(checksum: str, embedding: List[float]) -> None:
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem is None:
            return
        mem.cache_set(
            _cache_key(checksum),
            embedding,
            ttl=settings.REDIS_EMBEDDING_CACHE_TTL,
        )
    except Exception:
        pass


# EMBEDDING VALIDATION

def _valid_embedding(emb: List[float], expected_dim: int) -> bool:
    if not isinstance(emb, list):
        return False
    if len(emb) != expected_dim:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in emb):
        return False
    return True


# LANGUAGE DETECTION (BEST-EFFORT)

def _detect_language(text: str) -> Optional[str]:
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return None


# CLIP TEXT EMBEDDER CLASS

class ClipTextEmbedder:
    """
    CLIP text-side embedder for cross-modal retrieval.

    Responsibilities:
      - Normalise, sanitize, truncate to CLIP 77-token limit.
      - Redis embedding cache keyed by SHA-256 of cleaned text.
      - Batch CLIP text inference with L2 normalisation.
      - Dimension consistency enforcement.
      - Full Phase 24 metadata on every result.
      - Prompt injection guard on all inputs.
    """

    def __init__(self, processor, model, device: str) -> None:
        if not TORCH_AVAILABLE:
            raise TorchNotAvailableError("TORCH_REQUIRED_FOR_CLIP_TEXT_EMBEDDER")

        self.processor    = processor
        self.model        = model
        self.device       = device
        self.model_name   = settings.CLIP_MODEL
        self.expected_dim = settings.VISION_EMBEDDING_DIM
        self.batch_size   = settings.EMBEDDING_BATCH_SIZE

        # CLIP HARD LIMIT — NEVER EXCEED
        self.max_length = min(
            getattr(settings, "CLIP_MAX_LENGTH", _CLIP_MAX_TOKEN_LENGTH),
            _CLIP_MAX_TOKEN_LENGTH,
        )

        # CONSERVATIVE CHAR LIMIT (CLIP 77 TOKENS ≈ 300 CHARS ON AVERAGE)
        self._max_chars = self.max_length * 4

        logger.info(
            event="clip_text_embedder_initialized",
            device=device,
            model=self.model_name,
            expected_dim=self.expected_dim,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

    # PREPARE AND VALIDATE INPUT TEXTS

    def _prepare_texts(
        self,
        texts: Union[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Normalise, sanitize, deduplicate, and build per-text metadata.

        Returns list of dicts:
          { text, checksum, token_estimate, was_truncated, language }
        """
        if isinstance(texts, str):
            texts = [texts]

        prepared: List[Dict[str, Any]] = []
        seen_checksums: Dict[str, bool] = {}

        for raw in texts:
            # NORMALISE + SANITIZE
            text = _normalize_text(_sanitize_text(str(raw or "")))
            text = _sanitize_injection(text)

            if not text or len(text) < 2:
                _EMBED_ERRORS.labels(reason="empty_text").inc()
                continue

            # CAP TO MAX PROMPT CHARS FIRST
            if len(text) > settings.MAX_PROMPT_CHARS:
                text = text[:settings.MAX_PROMPT_CHARS]

            # TRUNCATE TO CLIP HARD LIMIT
            text, was_truncated = _truncate_to_clip_limit(text, self._max_chars)

            if not text:
                _EMBED_ERRORS.labels(reason="empty_after_truncation").inc()
                continue

            checksum = _sha256(text)

            # DEDUP
            if checksum in seen_checksums:
                continue
            seen_checksums[checksum] = True

            token_estimate = _estimate_tokens(text)
            language       = _detect_language(text)

            prepared.append({
                "text":            text,
                "checksum":        checksum,
                "token_estimate":  token_estimate,
                "was_truncated":   was_truncated,
                "language":        language,
                "text_preview":    text[:80],
            })

        return prepared

    # EMBED SINGLE TEXT

    def embed_single(
        self,
        text: str,
        session_id: str = "default",
    ) -> List[float]:
        """Embed a single text string. Returns raw embedding vector."""
        results = self.embed(text, session_id=session_id)
        if not results:
            raise NoValidTextsError(f"NO_EMBEDDING_PRODUCED for text: {text[:80]}")
        return results[0].embedding

    # EMBED — ACCEPTS SINGLE OR LIST

    def embed(
        self,
        texts: Union[str, List[str]],
        session_id: str = "default",
    ) -> List[ClipTextEmbeddingResult]:
        """
        Embed one or more text strings via CLIP.

        Steps:
          1. Normalise + sanitize + injection guard + dedup.
          2. Check Redis cache per text (SHA-256 keyed).
          3. Batch CLIP inference for cache-miss texts.
          4. L2 normalise all vectors.
          5. Validate embedding dimensions.
          6. Cache results in Redis.
          7. Return ClipTextEmbeddingResult list.
        """
        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        prepared = self._prepare_texts(texts)
        if not prepared:
            raise NoValidTextsError("NO_VALID_TEXTS_AFTER_PREPARATION")

        start_total    = time.time()
        cached_results: List[ClipTextEmbeddingResult] = []
        miss_items:     List[Dict[str, Any]]           = []

        # CACHE CHECK
        for item in prepared:
            cached_emb = _cache_get(item["checksum"])
            if cached_emb and _valid_embedding(cached_emb, self.expected_dim):
                _CACHE_HITS.inc()
                cached_results.append(ClipTextEmbeddingResult(
                    text_preview         = item["text_preview"],
                    embedding            = cached_emb,
                    embedding_dim        = self.expected_dim,
                    model_name           = self.model_name,
                    checksum_sha256      = item["checksum"],
                    token_count_estimate = item["token_estimate"],
                    was_truncated        = item["was_truncated"],
                    embedding_id         = str(uuid.uuid4()),
                    latency_ms           = 0.0,
                    cache_hit            = True,
                    session_id           = session_id,
                    language             = item["language"],
                ))
            else:
                miss_items.append(item)

        # BATCH INFERENCE FOR CACHE MISSES
        fresh_results: List[ClipTextEmbeddingResult] = []

        for i in range(0, len(miss_items), self.batch_size):
            batch = miss_items[i:i + self.batch_size]
            batch_texts = [item["text"] for item in batch]
            t_batch     = time.time()

            try:
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                    features = F.normalize(features, p=2, dim=-1)

                embeddings    = features.detach().cpu().numpy().tolist()
                batch_latency = round((time.time() - t_batch) * 1000, 1)

                # LATENCY WARNING
                target_ms = settings.LATENCY_TARGET_IMAGE_MS
                if batch_latency > target_ms:
                    logger.warning(
                        event="clip_text_batch_latency_exceeded",
                        latency_ms=batch_latency,
                        target_ms=target_ms,
                        batch_size=len(batch_texts),
                        session_id=session_id,
                    )

                _EMBED_LATENCY.labels(
                    batch_size=str(len(batch_texts))
                ).observe(batch_latency / 1000.0)

                for emb, item in zip(embeddings, batch):
                    emb_list = emb if isinstance(emb, list) else list(emb)

                    # DIMENSION CONSISTENCY CHECK
                    if len(emb_list) != self.expected_dim:
                        logger.error(
                            event="clip_text_dim_mismatch",
                            got=len(emb_list),
                            expected=self.expected_dim,
                            text_preview=item["text_preview"],
                            session_id=session_id,
                        )
                        _EMBED_ERRORS.labels(reason="dim_mismatch").inc()
                        raise DimensionMismatchError(
                            f"DIMENSION_MISMATCH: got {len(emb_list)}, "
                            f"expected {self.expected_dim}"
                        )

                    # NaN / Inf GUARD
                    if not _valid_embedding(emb_list, self.expected_dim):
                        logger.warning(
                            event="clip_text_invalid_values",
                            text_preview=item["text_preview"],
                            session_id=session_id,
                        )
                        _EMBED_ERRORS.labels(reason="invalid_values").inc()
                        continue

                    # CACHE STORE
                    _cache_set(item["checksum"], emb_list)
                    _EMBED_TOTAL.inc()

                    fresh_results.append(ClipTextEmbeddingResult(
                        text_preview         = item["text_preview"],
                        embedding            = emb_list,
                        embedding_dim        = self.expected_dim,
                        model_name           = self.model_name,
                        checksum_sha256      = item["checksum"],
                        token_count_estimate = item["token_estimate"],
                        was_truncated        = item["was_truncated"],
                        embedding_id         = str(uuid.uuid4()),
                        latency_ms           = batch_latency,
                        cache_hit            = False,
                        session_id           = session_id,
                        language             = item["language"],
                    ))

            except DimensionMismatchError:
                raise

            except Exception as exc:
                logger.error(
                    event="clip_text_batch_failed",
                    batch_start=i,
                    error=str(exc),
                    session_id=session_id,
                )
                _EMBED_ERRORS.labels(reason="batch_exception").inc()

        all_results = cached_results + fresh_results

        if not all_results:
            raise NoValidTextsError("NO_CLIP_TEXT_EMBEDDINGS_PRODUCED")

        total_latency = round(time.time() - start_total, 3)
        throughput    = round(len(all_results) / max(total_latency, 1e-6), 1)

        logger.info(
            event="clip_text_embed_success",
            total=len(all_results),
            cached=len(cached_results),
            fresh=len(fresh_results),
            throughput_per_sec=throughput,
            total_latency_sec=total_latency,
            session_id=session_id,
        )

        return all_results

    # QUERY ALIAS — USED BY HYBRID RETRIEVER VISION PATH

    def embed_query(
        self,
        query: str,
        session_id: str = "default",
    ) -> List[float]:
        """Single query embedding — alias used by hybrid retriever."""
        return self.embed_single(query, session_id=session_id)

    # ASYNC WRAPPERS

    async def embed_async(
        self,
        texts: Union[str, List[str]],
        session_id: str = "default",
    ) -> List[ClipTextEmbeddingResult]:
        async with _get_semaphore():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.embed(texts, session_id),
            )

    async def embed_query_async(
        self,
        query: str,
        session_id: str = "default",
    ) -> List[float]:
        async with _get_semaphore():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.embed_query(query, session_id),
            )

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "model_loaded":     self.model is not None,
            "processor_loaded": self.processor is not None,
            "device":           self.device,
            "model_name":       self.model_name,
            "expected_dim":     self.expected_dim,
            "max_length":       self.max_length,
            "batch_size":       self.batch_size,
            "torch_available":  TORCH_AVAILABLE,
        }


