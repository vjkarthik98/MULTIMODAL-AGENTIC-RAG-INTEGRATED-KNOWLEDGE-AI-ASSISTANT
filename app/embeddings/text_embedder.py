from __future__ import annotations

import asyncio
import hashlib
import math
import time
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# REDIS EMBEDDING CACHE — SECTION 4.3 (SHA-256 key, TTL 30 days)

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

        # LOCAL LRU FIRST
        if key in self._local:
            return self._local[key]

        # REDIS CACHE — SECTION 4.3
        redis = self._get_redis()
        if redis:
            try:
                cached = redis.cache_get(key)
                if cached and isinstance(cached, list):
                    # POPULATE LOCAL
                    self._local[key] = cached
                    return cached
            except Exception:
                pass

        return None

    def set(self, text: str, model: str, dim: int, embedding: List[float]) -> None:
        key = self._key(text, model, dim)
        self._local[key] = embedding

        # REDIS STORE
        redis = self._get_redis()
        if redis:
            try:
                redis.cache_set(
                    key,
                    embedding,
                    ttl=settings.EMBEDDING_CACHE_TTL,
                )
            except Exception:
                pass


_cache = _EmbeddingCache()


# EMBEDDING VALIDATION

def _valid_embedding(emb: List[float], expected_dim: int) -> bool:
    if not isinstance(emb, list):
        return False
    if len(emb) != expected_dim:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in emb):
        return False
    return True


# SANITIZE TEXT — SECTION 2.3

def _sanitize(text: str) -> Optional[str]:
    if not text:
        return None

    # NFC NORMALIZATION
    text = unicodedata.normalize("NFC", text.strip())

    # STRIP NULL BYTES
    if settings.STRIP_NULL_BYTES and "\x00" in text:
        text = text.replace("\x00", "")

    # STRIP BOM
    if settings.STRIP_BOM:
        text = text.lstrip("\ufeff\ufffe")

    if not text:
        return None

    # PROMPT INJECTION SANITIZATION — SECTION 5
    _INJECTION_PATTERNS = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard the above",
        "forget everything",
        "you are now",
        "act as",
        "jailbreak",
    ]
    lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            idx = lower.find(pattern)
            text = text[:idx].strip()
            break

    if len(text) > settings.MAX_PROMPT_CHARS:
        text = text[:settings.MAX_PROMPT_CHARS]

    return text if text else None


# MODALITY PREFIX — context enrichment for embedding quality

def _prefix(doc: Any) -> str:
    m  = getattr(doc, "modality", "")
    st = getattr(doc, "subtype", "")
    s  = getattr(doc, "structure", {}) or {}

    if m == "table":
        return "Table: "
    if m == "image":
        return "OCR: " if st == "ocr" else "Image: "
    if m == "audio":
        ts = s.get("timestamp_start", "")
        return f"Audio {ts}s: " if ts else "Audio: "
    if m == "video":
        if st == "speech":
            return "Video speech: "
        if st == "frame":
            return "Video frame: "
        if st == "ocr":
            return "Video OCR: "
    if m == "text" and st == "heading":
        return "Heading: "
    return ""


# CONTEXT ENRICHMENT

def _enrich(doc: Any, text: str) -> str:
    context = []
    if getattr(doc, "source_type", None):
        context.append(f"[{doc.source_type.upper()}]")
    if getattr(doc, "modality", None):
        context.append(f"[{doc.modality.upper()}]")
    if getattr(doc, "page", None):
        context.append(f"[Page {doc.page}]")
    if getattr(doc, "source", None):
        context.append(f"[{doc.source}]")

    enriched = " ".join(context) + " " + _prefix(doc) + text
    return enriched.strip()[:settings.MAX_PROMPT_CHARS]


# TEXT EMBEDDER CLASS

class TextEmbedder:

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        device: str,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name    = model_name
        self.batch_size    = batch_size
        self.device        = device
        self.expected_dim  = settings.TEXT_EMBEDDING_DIM
        self.max_text_len  = settings.MAX_PROMPT_CHARS

        self.model = SentenceTransformer(model_name, device=device)

        logger.info(
            event="text_embedder_initialized",
            model=model_name,
            device=device,
            dim=self.expected_dim,
        )

    # ENCODE WITH TENACITY RETRY

    def _encode_with_retry(
        self,
        model,
        texts: List[str],
        max_retries: int = 3,
    ) -> List[List[float]]:
        wait = 1.0
        for attempt in range(max_retries):
            try:
                t_start = time.time()
                embs = model.encode(
                    texts,
                    batch_size=len(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                elapsed_ms = round((time.time() - t_start) * 1000, 1)

                if elapsed_ms > settings.LATENCY_TARGET_EMBED_BATCH_MS:
                    logger.warning(
                        event="embed_batch_latency_exceeded",
                        latency_ms=elapsed_ms,
                        target_ms=settings.LATENCY_TARGET_EMBED_BATCH_MS,
                        batch_size=len(texts),
                    )

                return [e.tolist() for e in embs]

            except Exception as e:
                logger.warning(
                    event="embed_encode_retry",
                    attempt=attempt,
                    error=str(e),
                )
                if attempt >= max_retries - 1:
                    raise
                time.sleep(wait)
                wait = min(wait * 2, 10.0)

        return []

    # SINGLE TEXT EMBED

    def embed_text(
        self,
        text: str,
        session_id: str = "default",
    ) -> List[float]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        clean = _sanitize(text)
        if not clean:
            raise ValueError("EMPTY_TEXT")

        # CACHE CHECK — SECTION 4.3
        cached = _cache.get(clean, self.model_name, self.expected_dim)
        if cached is not None:
            logger.debug(event="embed_cache_hit", session_id=session_id)
            return cached

        t_start = time.time()
        emb = self.model.encode(
            clean,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        if not _valid_embedding(emb, self.expected_dim):
            raise ValueError("INVALID_EMBEDDING_SINGLE")

        _cache.set(clean, self.model_name, self.expected_dim, emb)

        logger.debug(
            event="embed_single_success",
            latency_ms=round((time.time() - t_start) * 1000, 1),
            session_id=session_id,
        )

        return emb

    # RAW STRING LIST EMBED — BATCH WITH CACHE — SECTION 4.3

    def embed_texts(
        self,
        texts: List[str],
        session_id: str = "default",
    ) -> List[List[float]]:

        if not texts:
            return []

        results: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            raw_batch = texts[i:i + self.batch_size]
            clean_batch: List[str] = []
            cached_results: List[Optional[List[float]]] = []

            for t in raw_batch:
                c = _sanitize(t)
                if not c:
                    cached_results.append(None)
                    clean_batch.append("")
                    continue

                hit = _cache.get(c, self.model_name, self.expected_dim)
                if hit is not None:
                    cached_results.append(hit)
                    clean_batch.append("")
                else:
                    cached_results.append(None)
                    clean_batch.append(c)

            # ENCODE ONLY UNCACHED
            to_encode = [(idx, t) for idx, t in enumerate(clean_batch) if t]
            encoded: Dict[int, List[float]] = {}

            if to_encode:
                indices, batch_texts = zip(*to_encode)
                try:
                    embs = self._encode_with_retry(self.model, list(batch_texts))
                    for idx, emb in zip(indices, embs):
                        if _valid_embedding(emb, self.expected_dim):
                            encoded[idx] = emb
                            _cache.set(clean_batch[idx], self.model_name, self.expected_dim, emb)
                except Exception as e:
                    logger.error(
                        event="embed_texts_batch_failed",
                        error=str(e),
                        session_id=session_id,
                    )

            for idx, cached_emb in enumerate(cached_results):
                if cached_emb is not None:
                    results.append(cached_emb)
                elif idx in encoded:
                    results.append(encoded[idx])

        return results

    # DOCUMENT BATCH EMBED — SECTION 4.3

    def embed_documents(
        self,
        documents: List[Any],
        session_id: str = "default",
    ) -> List[Any]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        if not documents:
            return []

        start = time.time()
        texts:     List[str] = []
        valid_docs:  List[Any] = []
        seen: Dict[str, bool] = {}

        for doc in documents:
            try:
                raw  = getattr(doc, "text", "") or ""
                clean = _sanitize(raw)
                if not clean:
                    continue

                enriched = _enrich(doc, clean)
                h = hashlib.sha256(enriched.encode("utf-8")).hexdigest()

                if h in seen:
                    continue

                seen[h] = True
                texts.append(enriched)
                valid_docs.append(doc)

            except Exception as e:
                logger.warning(
                    event="embed_doc_skip",
                    error=str(e),
                    session_id=session_id,
                )

        if not valid_docs:
            return []

        # SAFETY CAP
        cap        = settings.INGESTION_BATCH_SIZE * 10
        texts      = texts[:cap]
        valid_docs = valid_docs[:cap]

        results: List[Any] = []
        t_target = settings.LATENCY_TARGET_EMBED_BATCH_MS / 1000.0

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_docs  = valid_docs[i:i + self.batch_size]

            t_batch = time.time()

            try:
                embs = self._encode_with_retry(self.model, batch_texts)

                batch_latency = time.time() - t_batch

                if batch_latency > t_target:
                    logger.warning(
                        event="embed_batch_latency_exceeded",
                        latency=round(batch_latency, 3),
                        target=t_target,
                        batch_size=len(batch_texts),
                        session_id=session_id,
                    )

                for doc, emb in zip(batch_docs, embs):
                    if not _valid_embedding(emb, self.expected_dim):
                        continue

                    doc.embedding = emb

                    structure = dict(doc.structure or {})
                    structure["embedding_space"] = "text"
                    structure["embedding_model"] = self.model_name
                    doc.structure = structure

                    # CACHE SET — SECTION 4.3
                    text_for_cache = _sanitize(getattr(doc, "text", "") or "")
                    if text_for_cache:
                        _cache.set(text_for_cache, self.model_name, self.expected_dim, emb)

                    results.append(doc)

            except Exception as e:
                logger.error(
                    event="embed_batch_failed",
                    batch_start=i,
                    error=str(e),
                    session_id=session_id,
                )

        total_latency = round(time.time() - start, 3)
        throughput    = round(len(results) / max(total_latency, 1e-6), 1)

        logger.info(
            event="embed_documents_success",
            embedded=len(results),
            total=len(valid_docs),
            throughput_per_sec=throughput,
            latency=total_latency,
            session_id=session_id,
        )

        return results

    # QUERY EMBED — ALIAS

    def embed_query(
        self,
        query: str,
        session_id: str = "default",
    ) -> List[float]:
        return self.embed_text(query, session_id)

    # MATRYOSHKA EMBED — SECTION 4.3 (short dim + full dim)

    def embed_matryoshka(
        self,
        text: str,
        session_id: str = "default",
    ) -> Tuple[List[float], List[float]]:
        full = self.embed_text(text, session_id)
        short = full[:settings.MATRYOSHKA_SHORT_DIM]
        # L2 NORMALIZE SHORT
        norm = math.sqrt(sum(v * v for v in short)) + 1e-10
        short = [v / norm for v in short]
        return short, full

    # SPARSE VECTOR (BM25-STYLE PROXY) — SECTION 4.3

    def embed_sparse(
        self,
        text: str,
        vocab_size: int = 30000,
    ) -> Dict[int, float]:
        clean = _sanitize(text)
        if not clean:
            return {}
        import re
        tokens = re.findall(r"\b[a-z0-9]+\b", clean.lower())
        tf: Dict[int, float] = {}
        for tok in tokens:
            idx = hash(tok) % vocab_size
            tf[idx] = tf.get(idx, 0.0) + 1.0
        total = sum(tf.values()) + 1e-10
        return {k: round(v / total, 6) for k, v in tf.items()}

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "model":       self.model_name,
            "device":      self.device,
            "dim":         self.expected_dim,
            "batch_size":  self.batch_size,
            "cache_local": len(_cache._local),
        }


