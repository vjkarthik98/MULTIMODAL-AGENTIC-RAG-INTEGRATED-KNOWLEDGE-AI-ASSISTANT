from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS

def _make_metrics():
    if not settings.PROMETHEUS_ENABLED:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def labels(self, **kw): return self
        n = _Noop()
        return n, n, n, n, n
    try:
        from prometheus_client import Counter, Histogram, Gauge
        embed_latency = Histogram(
            "multimodal_embedding_latency_seconds",
            "Total multimodal embedding pipeline latency",
            ["modality"],
        )
        embed_errors = Counter(
            "multimodal_embedding_errors_total",
            "Multimodal embedding failures",
            ["modality", "reason"],
        )
        embed_total = Counter(
            "multimodal_embeddings_produced_total",
            "Total embeddings produced by modality",
            ["modality"],
        )
        cache_hits = Counter(
            "multimodal_embedding_cache_hits_total",
            "Cache hits by modality",
            ["modality"],
        )
        active_jobs = Gauge(
            "multimodal_embedding_active_jobs",
            "In-progress multimodal embedding jobs",
        )
        return embed_latency, embed_errors, embed_total, cache_hits, active_jobs
    except Exception:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def labels(self, **kw): return self
            def set(self, *a, **kw): pass
        n = _Noop()
        return n, n, n, n, n


_EMBED_LATENCY, _EMBED_ERRORS, _EMBED_TOTAL, _CACHE_HITS, _ACTIVE = _make_metrics()


# SEMAPHORE

_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)
    return _SEMAPHORE


# CUSTOM EXCEPTIONS

class MultimodalEmbedderError(Exception):
    """Base exception for multimodal embedding errors."""


class NoValidDocumentsError(MultimodalEmbedderError):
    """Raised when no documents survive validation."""


class EmbeddingDimensionError(MultimodalEmbedderError):
    """Raised when embedding dimension does not match expected."""


class SessionIdRequiredError(MultimodalEmbedderError):
    """Raised when session_id is missing."""


# EMBEDDING SUMMARY MODEL

class MultimodalEmbeddingReport:
    """
    Summary report for a full multimodal embed_documents() call.
    Carries counts, latencies, and per-modality breakdowns.
    """

    def __init__(
        self,
        session_id: str,
        total_input: int,
        text_embedded: int,
        vision_embedded: int,
        failed_text: int,
        failed_vision: int,
        deduped_skipped: int,
        text_latency_sec: float,
        vision_latency_sec: float,
        total_latency_sec: float,
        modality_breakdown: Dict[str, int],
        embedding_model_text: str,
        embedding_model_vision: str,
    ) -> None:
        self.session_id            = session_id
        self.total_input           = total_input
        self.text_embedded         = text_embedded
        self.vision_embedded       = vision_embedded
        self.failed_text           = failed_text
        self.failed_vision         = failed_vision
        self.deduped_skipped       = deduped_skipped
        self.text_latency_sec      = text_latency_sec
        self.vision_latency_sec    = vision_latency_sec
        self.total_latency_sec     = total_latency_sec
        self.modality_breakdown    = modality_breakdown
        self.embedding_model_text  = embedding_model_text
        self.embedding_model_vision = embedding_model_vision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":             self.session_id,
            "total_input":            self.total_input,
            "text_embedded":          self.text_embedded,
            "vision_embedded":        self.vision_embedded,
            "failed_text":            self.failed_text,
            "failed_vision":          self.failed_vision,
            "deduped_skipped":        self.deduped_skipped,
            "text_latency_sec":       self.text_latency_sec,
            "vision_latency_sec":     self.vision_latency_sec,
            "total_latency_sec":      self.total_latency_sec,
            "modality_breakdown":     self.modality_breakdown,
            "embedding_model_text":   self.embedding_model_text,
            "embedding_model_vision": self.embedding_model_vision,
        }


# HELPERS

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_embedding(emb: Any, expected_dim: int) -> bool:
    if not isinstance(emb, list):
        return False
    if len(emb) != expected_dim:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in emb):
        return False
    return True


def _modality_counts(docs: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in docs:
        m = getattr(d, "modality", "unknown")
        counts[m] = counts.get(m, 0) + 1
    return counts


def _resolve_asset_path(doc: Any) -> Optional[str]:
    """
    Resolve physical image path from document structure.
    Checks asset_path, frame_path, source_path keys in order.
    """
    structure = getattr(doc, "structure", {}) or {}
    for key in ("asset_path", "frame_path", "source_path"):
        path = structure.get(key)
        if not path:
            continue
        p = Path(str(path))
        if p.exists():
            return str(p)
        logger.warning(
            event="multimodal_asset_path_not_found",
            key=key,
            path=str(path),
        )
    return None


# ROUTING LOGIC

def _route_documents(docs: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Route documents into text-path and vision-path lists.

    TEXT PATH:
      - modality: text, table
      - modality: audio  (transcript text)
      - modality: image  (caption / ocr subtypes)
      - modality: video, subtype: speech | ocr

    VISION PATH:
      - modality: image,  subtype: caption  (has asset_path → CLIP image embed)
      - modality: video,  subtype: frame    (has asset_path → CLIP image embed)

    A document can appear in BOTH lists (text embed + vision embed).
    """
    text_docs:   List[Any] = []
    vision_docs: List[Any] = []

    for doc in docs:
        try:
            modality = getattr(doc, "modality", "")
            subtype  = getattr(doc, "subtype",  "") or ""

            if modality in ("text", "table"):
                text_docs.append(doc)

            elif modality == "audio":
                text_docs.append(doc)

            elif modality == "image":
                # ALWAYS TEXT-EMBED (caption or OCR text)
                text_docs.append(doc)
                # ADDITIONALLY VISION-EMBED IF ASSET PATH EXISTS
                if subtype == "caption" and _resolve_asset_path(doc):
                    vision_docs.append(doc)

            elif modality == "video":
                if subtype in ("speech", "ocr"):
                    text_docs.append(doc)
                elif subtype == "frame":
                    vision_docs.append(doc)
                    # ALSO TEXT-EMBED FRAME CAPTION
                    text_docs.append(doc)

        except Exception as exc:
            logger.warning(
                event="multimodal_route_failed",
                error=str(exc),
                modality=getattr(doc, "modality", "unknown"),
            )

    return text_docs, vision_docs


# MULTIMODAL EMBEDDER CLASS

class MultimodalEmbedder:
    """
    Orchestrates text + image + CLIP embedding pipelines.

    Responsibilities:
      - Route IngestedDocument objects to correct embedder (text vs vision).
      - SHA-256 deduplication before embedding.
      - Batch text embedding via TextEmbedder.
      - Batch vision embedding via ImageEmbedder (CLIP).
      - Per-modality Prometheus metrics.
      - Full Phase 24 metadata stamped on every embedded document.
      - Graceful degradation: partial failure returns partial result.
      - Async wrapper for pipeline integration.
    """

    def __init__(self, text_embedder, image_embedder) -> None:
        self.text_embedder  = text_embedder
        self.image_embedder = image_embedder
        self.batch_size     = settings.EMBEDDING_BATCH_SIZE
        self.max_docs       = settings.INGESTION_BATCH_SIZE * 10

        self._text_model_name   = settings.EMBEDDING_MODEL
        self._vision_model_name = settings.CLIP_MODEL

        logger.info(
            event="multimodal_embedder_initialized",
            text_model=self._text_model_name,
            vision_model=self._vision_model_name,
            batch_size=self.batch_size,
            max_docs=self.max_docs,
        )

    # MAIN EMBED DOCUMENTS

    def embed_documents(
        self,
        documents: List[Any],
        session_id: str,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Embed all documents — returns (text_embedded, vision_embedded).

        Args:
            documents:  List of IngestedDocument objects.
            session_id: Caller session ID — required.

        Returns:
            Tuple of (text_embedded_docs, vision_embedded_docs).
            Each doc has .embedding populated and structure metadata stamped.

        Raises:
            SessionIdRequiredError: if session_id is empty.
            NoValidDocumentsError:  if all documents fail dedup/validation.
        """
        if not session_id:
            raise SessionIdRequiredError("SESSION_ID_REQUIRED")

        if not documents:
            return [], []

        start_total = time.time()
        _ACTIVE.set(1)

        # CAP INPUT
        docs = documents[:self.max_docs]
        original_count = len(docs)

        # SHA-256 DEDUP ON TEXT CONTENT
        seen_hashes: Dict[str, bool] = {}
        unique_docs: List[Any]       = []
        deduped_skipped              = 0

        for doc in docs:
            text = getattr(doc, "text", "") or ""
            h    = _sha256_text(text)
            if h in seen_hashes:
                deduped_skipped += 1
                continue
            seen_hashes[h] = True
            unique_docs.append(doc)

        if not unique_docs:
            _ACTIVE.set(0)
            raise NoValidDocumentsError(
                f"ALL_DOCUMENTS_DEDUPED: {deduped_skipped} duplicates from {original_count} inputs"
            )

        modality_breakdown = _modality_counts(unique_docs)

        logger.info(
            event="multimodal_embed_start",
            total_input=original_count,
            after_dedup=len(unique_docs),
            deduped_skipped=deduped_skipped,
            modality_breakdown=modality_breakdown,
            session_id=session_id,
        )

        # ROUTE
        text_docs, vision_docs = _route_documents(unique_docs)

        # TEXT EMBEDDING
        t_text        = time.time()
        embedded_text: List[Any] = []
        failed_text               = 0

        if text_docs:
            try:
                embedded_text = self._embed_text_batched(text_docs, session_id)
                _EMBED_TOTAL.labels(modality="text").inc(len(embedded_text))
                _EMBED_LATENCY.labels(modality="text").observe(
                    time.time() - t_text
                )
            except Exception as exc:
                failed_text = len(text_docs)
                _EMBED_ERRORS.labels(modality="text", reason="batch_failed").inc()
                logger.error(
                    event="multimodal_text_embed_failed",
                    count=len(text_docs),
                    error=str(exc),
                    session_id=session_id,
                )

        text_latency = round(time.time() - t_text, 3)

        # VISION EMBEDDING
        t_vision        = time.time()
        embedded_vision: List[Any] = []
        failed_vision               = 0

        if vision_docs:
            try:
                embedded_vision = self._embed_vision_batched(vision_docs, session_id)
                _EMBED_TOTAL.labels(modality="vision").inc(len(embedded_vision))
                _EMBED_LATENCY.labels(modality="vision").observe(
                    time.time() - t_vision
                )
            except Exception as exc:
                failed_vision = len(vision_docs)
                _EMBED_ERRORS.labels(modality="vision", reason="batch_failed").inc()
                logger.error(
                    event="multimodal_vision_embed_failed",
                    count=len(vision_docs),
                    error=str(exc),
                    session_id=session_id,
                )

        vision_latency    = round(time.time() - t_vision, 3)
        total_latency     = round(time.time() - start_total, 3)
        _ACTIVE.set(0)

        # REPORT
        report = MultimodalEmbeddingReport(
            session_id            = session_id,
            total_input           = original_count,
            text_embedded         = len(embedded_text),
            vision_embedded       = len(embedded_vision),
            failed_text           = failed_text,
            failed_vision         = failed_vision,
            deduped_skipped       = deduped_skipped,
            text_latency_sec      = text_latency,
            vision_latency_sec    = vision_latency,
            total_latency_sec     = total_latency,
            modality_breakdown    = modality_breakdown,
            embedding_model_text  = self._text_model_name,
            embedding_model_vision= self._vision_model_name,
        )

        logger.info(
            event="multimodal_embed_success",
            **report.to_dict(),
        )

        return embedded_text, embedded_vision

    # TEXT BATCH EMBEDDING

    def _embed_text_batched(
        self,
        docs: List[Any],
        session_id: str,
    ) -> List[Any]:
        """
        Embed text documents in sub-batches.
        Stamps embedding_space = 'text' on structure.
        Validates embedding dimension before accepting.
        """
        results: List[Any] = []

        for i in range(0, len(docs), self.batch_size):
            batch = docs[i:i + self.batch_size]
            try:
                embedded = self.text_embedder.embed_documents(
                    batch,
                    session_id=session_id,
                )

                for doc in embedded:
                    emb = getattr(doc, "embedding", None)
                    if not _valid_embedding(emb, settings.TEXT_EMBEDDING_DIM):
                        _EMBED_ERRORS.labels(modality="text", reason="invalid_dim").inc()
                        continue

                    # STAMP METADATA
                    structure                    = dict(getattr(doc, "structure", {}) or {})
                    structure["embedding_space"] = "text"
                    structure["embedding_model"] = self._text_model_name
                    structure["embedding_dim"]   = settings.TEXT_EMBEDDING_DIM
                    structure["embedded_at"]     = time.time()
                    doc.structure                = structure

                    results.append(doc)

            except Exception as exc:
                logger.error(
                    event="multimodal_text_batch_failed",
                    batch_start=i,
                    error=str(exc),
                    session_id=session_id,
                )
                _EMBED_ERRORS.labels(modality="text", reason="batch_exception").inc()

        return results

    # VISION BATCH EMBEDDING

    def _embed_vision_batched(
        self,
        docs: List[Any],
        session_id: str,
    ) -> List[Any]:
        """
        Embed vision documents (images/video frames) in sub-batches via CLIP.
        Resolves asset path, validates embedding, stamps embedding_space = 'vision'.
        Skips documents with missing or invalid asset paths.
        """
        results: List[Any]          = []
        seen_paths: Dict[str, bool] = {}

        for i in range(0, len(docs), self.batch_size):
            batch       = docs[i:i + self.batch_size]
            paths:      List[str] = []
            valid_docs: List[Any] = []

            for doc in batch:
                asset_path = _resolve_asset_path(doc)

                if not asset_path:
                    logger.warning(
                        event="multimodal_vision_no_asset_path",
                        modality=getattr(doc, "modality", ""),
                        subtype=getattr(doc, "subtype", ""),
                        session_id=session_id,
                    )
                    _EMBED_ERRORS.labels(modality="vision", reason="no_asset_path").inc()
                    continue

                # PATH-LEVEL DEDUP
                path_hash = _sha256_text(asset_path)
                if path_hash in seen_paths:
                    continue
                seen_paths[path_hash] = True

                paths.append(asset_path)
                valid_docs.append(doc)

            if not paths:
                continue

            try:
                emb_results = self.image_embedder.embed_batch(
                    paths,
                    session_id=session_id,
                )

                for doc, emb_result in zip(valid_docs, emb_results):
                    emb_list = emb_result.embedding if hasattr(emb_result, "embedding") else emb_result

                    if not _valid_embedding(emb_list, settings.VISION_EMBEDDING_DIM):
                        _EMBED_ERRORS.labels(modality="vision", reason="invalid_dim").inc()
                        logger.warning(
                            event="multimodal_vision_invalid_embedding",
                            expected_dim=settings.VISION_EMBEDDING_DIM,
                            got_dim=len(emb_list) if isinstance(emb_list, list) else -1,
                            session_id=session_id,
                        )
                        continue

                    doc.embedding = emb_list

                    # STAMP METADATA
                    structure                    = dict(getattr(doc, "structure", {}) or {})
                    structure["embedding_space"] = "vision"
                    structure["embedding_model"] = self._vision_model_name
                    structure["embedding_dim"]   = settings.VISION_EMBEDDING_DIM
                    structure["embedded_at"]     = time.time()

                    if hasattr(emb_result, "checksum_sha256"):
                        structure["asset_checksum_sha256"] = emb_result.checksum_sha256
                    if hasattr(emb_result, "width"):
                        structure["asset_width"]  = emb_result.width
                        structure["asset_height"] = emb_result.height

                    doc.structure = structure
                    results.append(doc)

            except Exception as exc:
                logger.error(
                    event="multimodal_vision_batch_failed",
                    batch_start=i,
                    error=str(exc),
                    session_id=session_id,
                )
                _EMBED_ERRORS.labels(modality="vision", reason="batch_exception").inc()

        return results

    # ASYNC WRAPPERS

    async def embed_documents_async(
        self,
        documents: List[Any],
        session_id: str,
    ) -> Tuple[List[Any], List[Any]]:
        """Async entry point for pipeline integration."""
        async with _get_semaphore():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.embed_documents(documents, session_id),
            )

    # CROSS-MODAL SIMILARITY — USED BY HYBRID RETRIEVER

    def cross_modal_similarity(
        self,
        query_text: str,
        image_embedding: List[float],
        session_id: str = "default",
    ) -> float:
        """
        Cosine similarity between a text query and a CLIP image embedding.
        Both vectors are projected into CLIP's shared 512-dim space by encoding
        the query with the CLIP text encoder rather than the MiniLM text encoder.
        """
        if not query_text or not image_embedding:
            return 0.0
        try:
            import numpy as np
            from app.core.model_loader import model_loader
            clip_text_emb = model_loader.get_clip_text_embedder().embed_query(
                query_text, session_id=session_id
            )
            if not clip_text_emb or len(clip_text_emb) != len(image_embedding):
                return 0.0
            a     = np.array(clip_text_emb, dtype=float)
            b     = np.array(image_embedding, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
            val   = float(np.dot(a, b) / denom)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return round(val, 6)
        except Exception:
            return 0.0

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        text_ok   = self.text_embedder is not None
        vision_ok = self.image_embedder is not None

        text_health   = {}
        vision_health = {}

        try:
            if hasattr(self.text_embedder, "health_check"):
                text_health = self.text_embedder.health_check()
        except Exception:
            pass

        try:
            if hasattr(self.image_embedder, "health_check"):
                vision_health = self.image_embedder.health_check()
        except Exception:
            pass

        return {
            "text_embedder_ok":    text_ok,
            "vision_embedder_ok":  vision_ok,
            "text_model":          self._text_model_name,
            "vision_model":        self._vision_model_name,
            "batch_size":          self.batch_size,
            "max_docs":            self.max_docs,
            "text_health":         text_health,
            "vision_health":       vision_health,
        }


