"""base_embedder.py

Shared base for all per-modality embedders (Phase 3).

Every modality embedder extends BaseEmbedder and overrides _build_embed_text()
to return the modality-specific enriched string for encoding. All encoding,
caching, batching, and validation logic lives here once.

text_embedder.py (legacy pipeline path) is kept untouched alongside this.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FINANCE NUMBER NORMALIZATION
# Identical to the copy in text_embedder.py — kept here so base_embedder has
# no import dependency on text_embedder (avoids circular risk).
# ══════════════════════════════════════════════════════════════════════════════

_SCALE_EXPAND: Dict[str, Tuple[float, str, str]] = {
    "b":  (1e9,  "billion",  "million"),
    "bn": (1e9,  "billion",  "million"),
    "t":  (1e12, "trillion", "billion"),
    "tn": (1e12, "trillion", "billion"),
    "m":  (1e6,  "million",  "billion"),
    "mn": (1e6,  "million",  "billion"),
    "k":  (1e3,  "thousand", "million"),
}

_FIN_NUM_RE = re.compile(
    r'([$€£¥₹]?)'
    r'([\d,]+\.?\d*)'
    r'\s*([BMKTbmkt]n?)\b'
)
_PCT_RE = re.compile(r'([\d.]+)\s*%')
_BPS_RE = re.compile(r'([\d.]+)\s*bps\b', re.IGNORECASE)
_QTR_RE = re.compile(r'\bQ([1-4])\s*(?:FY)?\s*(\d{2,4})\b', re.IGNORECASE)
_HY_RE  = re.compile(r'\bH([12])\s+(\d{4})\b', re.IGNORECASE)

_QTR_WORDS = {"1": "first", "2": "second", "3": "third", "4": "fourth"}
_HY_WORDS  = {"1": "first", "2": "second"}


def normalize_finance_numbers(text: str) -> str:
    """Append expanded forms of finance numbers so scale variants match at query time."""
    extras: List[str] = []
    for m in _FIN_NUM_RE.finditer(text):
        raw_num = m.group(2).replace(",", "")
        suffix  = m.group(3).lower().rstrip("n")
        if suffix not in _SCALE_EXPAND:
            continue
        try:
            num_val = float(raw_num)
        except ValueError:
            continue
        mult, full_word, cross_word = _SCALE_EXPAND[suffix]
        cross_divisor = 1e9 if cross_word == "billion" else 1e6 if cross_word == "million" else 1e3
        cross_val = num_val * mult / cross_divisor
        extras.append(f"{num_val} {full_word}")
        extras.append(f"{num_val} {full_word} dollars")
        cross_str = f"{cross_val:.1f}".rstrip("0").rstrip(".")
        extras.append(f"{cross_str} {cross_word}")
    for m in _PCT_RE.finditer(text):
        extras.append(f"{m.group(1)} percent")
    for m in _BPS_RE.finditer(text):
        extras.append(f"{m.group(1)} basis points")
    for m in _QTR_RE.finditer(text):
        q, yr = m.group(1), m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        extras.append(f"Q{q} fiscal year {yr}")
        extras.append(f"{_QTR_WORDS.get(q, 'Q'+q)} quarter {yr}")
    for m in _HY_RE.finditer(text):
        h, yr = m.group(1), m.group(2)
        extras.append(f"{_HY_WORDS.get(h, 'H'+h)} half {yr}")
    return (text + " " + " ".join(extras)) if extras else text


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def valid_embedding(emb: List[float], expected_dim: int) -> bool:
    if not isinstance(emb, list) or len(emb) != expected_dim:
        return False
    return not any(math.isnan(v) or math.isinf(v) for v in emb)


def sanitize_text(text: str) -> Optional[str]:
    if not text:
        return None
    text = unicodedata.normalize("NFC", text.strip())
    if settings.STRIP_NULL_BYTES and "\x00" in text:
        text = text.replace("\x00", "")
    if settings.STRIP_BOM:
        text = text.lstrip("﻿￾")
    if not text:
        return None
    try:
        from app.guardrails.input_guard import sanitize as _guard
        text = _guard(text, surface="embedder") or text
    except Exception:
        pass
    if len(text) > settings.MAX_PROMPT_CHARS:
        text = text[:settings.MAX_PROMPT_CHARS]
    return text or None


# ══════════════════════════════════════════════════════════════════════════════
# REDIS + LRU EMBEDDING CACHE
# ══════════════════════════════════════════════════════════════════════════════

class _EmbeddingCache:

    def __init__(self) -> None:
        self._local: Dict[str, List[float]] = {}
        self._redis = None

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.core.infra_registry import infra
            self._redis = infra.get_memory()
        except Exception:
            self._redis = None
        return self._redis

    def _key(self, text: str, model: str, dim: int) -> str:
        raw = f"{model}:{dim}:{text[:500]}"
        return "emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str, dim: int) -> Optional[List[float]]:
        key = self._key(text, model, dim)
        if key in self._local:
            return self._local[key]
        redis = self._get_redis()
        if redis:
            try:
                cached = redis.cache_get(key)
                if cached and isinstance(cached, list):
                    self._local[key] = cached
                    return cached
            except Exception:
                pass
        return None

    def set(self, text: str, model: str, dim: int, embedding: List[float]) -> None:
        key = self._key(text, model, dim)
        self._local[key] = embedding
        redis = self._get_redis()
        if redis:
            try:
                redis.cache_set(key, embedding, ttl=settings.EMBEDDING_CACHE_TTL)
            except Exception:
                pass


_shared_cache = _EmbeddingCache()


# ══════════════════════════════════════════════════════════════════════════════
# BASE EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════

class BaseEmbedder(ABC):
    """Abstract base for all per-modality embedders.

    Subclasses override _build_embed_text(doc, cleaned_text) to return the
    modality-specific enriched string. All encoding, caching, and batching
    is handled here.
    """

    def _get_model(self):
        """Lazy accessor for the shared TextEmbedder singleton."""
        from app.core.model_loader import model_loader
        return model_loader.get_embedder()

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        """Build the full enriched string to encode for this document.

        Args:
            doc: IngestedDocument instance.
            cleaned_text: Already sanitized + finance-normalized text.

        Returns:
            Final string to pass to the encoder (max MAX_PROMPT_CHARS).
        """
        ...

    # ── Public API ──────────────────────────────────────────────────────────

    def embed_documents(
        self,
        docs: List[Any],
        session_id: str = "default",
    ) -> List[Any]:
        """Embed a list of IngestedDocument objects.

        Sets doc.embedding on each successful doc. Returns list of docs that
        received a valid embedding.
        """
        if not docs:
            return []
        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        embedder = self._get_model()
        start    = time.time()

        texts:      List[str] = []
        valid_docs: List[Any] = []
        seen: Dict[str, bool] = {}

        for doc in docs:
            try:
                raw   = getattr(doc, "text", "") or ""
                clean = sanitize_text(raw)
                if not clean:
                    continue
                clean    = normalize_finance_numbers(clean)
                enriched = self._build_embed_text(doc, clean)
                enriched = enriched[:settings.MAX_PROMPT_CHARS]
                h = hashlib.sha256(enriched.encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen[h] = True
                texts.append(enriched)
                valid_docs.append(doc)
            except Exception as exc:
                logger.warning(event="embed_doc_prep_skip", error=str(exc), session_id=session_id)

        if not valid_docs:
            return []

        cap        = settings.INGESTION_BATCH_SIZE * 10
        texts      = texts[:cap]
        valid_docs = valid_docs[:cap]

        results: List[Any] = []

        for i in range(0, len(texts), embedder.batch_size):
            batch_texts = texts[i:i + embedder.batch_size]
            batch_docs  = valid_docs[i:i + embedder.batch_size]
            try:
                embs = embedder._encode_with_retry(embedder.model, batch_texts)
                for doc, emb, txt in zip(batch_docs, embs, batch_texts):
                    if not valid_embedding(emb, embedder.expected_dim):
                        continue
                    doc.embedding = emb
                    struct = dict(getattr(doc, "structure", {}) or {})
                    struct["embedding_space"] = "text"
                    struct["embedding_model"] = embedder.model_name
                    doc.structure = struct
                    _shared_cache.set(txt, embedder.model_name, embedder.expected_dim, emb)
                    results.append(doc)
            except Exception as exc:
                logger.error(
                    event="embed_batch_failed", batch_start=i,
                    error=str(exc), session_id=session_id,
                )

        latency    = round(time.time() - start, 3)
        throughput = round(len(results) / max(latency, 1e-6), 1)
        logger.info(
            event="embed_documents_done",
            modality=self.__class__.__name__,
            embedded=len(results),
            total=len(valid_docs),
            throughput_per_sec=throughput,
            latency=latency,
            session_id=session_id,
        )
        return results

    def embed_query(self, query: str, session_id: str = "default") -> List[float]:
        """Embed a query string (uses BGE query instruction prefix)."""
        return self._get_model().embed_text(query, session_id)
