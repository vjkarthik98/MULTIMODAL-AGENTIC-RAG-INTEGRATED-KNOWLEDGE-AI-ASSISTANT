from __future__ import annotations

import asyncio
import hashlib
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NP_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: List[List[str]]) -> None:
            self.corpus = corpus

        def get_scores(self, tokens: List[str]) -> List[float]:
            query = set(tokens)
            return [float(len(query & set(doc))) for doc in self.corpus]

try:
    import pybreaker
    _breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

    class _DummyBreaker:
        def __call__(self, fn):
            return fn

    _breaker = _DummyBreaker()  # type: ignore[assignment]


# INDEX PATHS
_INDEX_DIR = Path(settings.BM25_INDEX_DIR) if hasattr(settings, "BM25_INDEX_DIR") else settings.DATA_DIR / "bm25_index"
_INDEX_FILE = _INDEX_DIR / "bm25_index.pkl"

# STOPWORDS
_STOPWORDS: Set[str] = {
    "the", "is", "and", "a", "an", "of", "to", "in", "on", "for",
    "at", "by", "with", "from", "this", "that", "it", "be", "as",
    "are", "was", "were", "has", "have", "had", "do", "does", "did",
    "but", "or", "not", "so", "if", "its", "our", "we", "he", "she",
    "they", "you", "i", "me", "my", "your", "their", "what", "which",
    "who", "when", "where", "how", "all", "been", "will", "would",
    "could", "should", "may", "might", "can", "any", "some", "no",
}


class BM25Retriever:

    def __init__(self) -> None:
        self.documents: List[Any] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.modality_filter: Optional[str] = None
        self.max_docs: int = settings.BM25_MAX_DOCS
        self._index_loaded: bool = False

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # TOKENIZE

    def _tokenize(self, text: str) -> List[str]:
        text = str(text or "").lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
        return tokens[:settings.BM25_MAX_TOKENS]

    # METADATA EXTRACTION

    def _metadata(self, doc: Any) -> Dict[str, Any]:
        s = dict(getattr(doc, "structure", {}) or {})
        return {
            "modality": getattr(doc, "modality", "text"),
            "subtype": getattr(doc, "subtype", None),
            "source": getattr(doc, "source", None),
            "source_type": getattr(doc, "source_type", None),
            "doc_id": s.get("doc_id"),
            "chunk_id": getattr(doc, "chunk_id", None),
            "session_id": s.get("session_id"),
            "content_type": s.get("content_type"),
            "page": getattr(doc, "page", None),
            "language": s.get("language"),
            "timestamp_start": s.get("timestamp_start"),
            "ingestion_time": s.get("ingestion_time"),
            "checksum_sha256": s.get("checksum_sha256"),
        }

    # MODALITY FILTER SETTER

    def set_modality_filter(self, modality: Optional[str]) -> None:
        self.modality_filter = modality

    # CIRCUIT-BROKEN SAVE

    def _save_index(self) -> None:
        def _do_save() -> None:
            _INDEX_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
                "saved_at": time.time(),
                "doc_count": len(self.documents),
            }
            tmp_path = _INDEX_FILE.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(_INDEX_FILE)
            logger.info(
                event="bm25_index_saved",
                path=str(_INDEX_FILE),
                docs=len(self.documents),
            )

        try:
            if _PYBREAKER_AVAILABLE:
                _breaker(_do_save)()
            else:
                _do_save()
        except Exception as exc:
            logger.error(event="bm25_index_save_failed", error=str(exc))

    # CIRCUIT-BROKEN LOAD

    def _load_index(self) -> None:
        if self._index_loaded:
            return

        if not _INDEX_FILE.exists():
            logger.info(event="bm25_no_saved_index")
            return

        def _do_load() -> None:
            with open(_INDEX_FILE, "rb") as f:
                payload = pickle.load(f)
            self.documents = payload.get("documents", [])
            self.tokenized_corpus = payload.get("tokenized_corpus", [])
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(
                    event="bm25_index_loaded",
                    docs=len(self.documents),
                    path=str(_INDEX_FILE),
                )
            else:
                logger.warning(event="bm25_saved_index_empty")
            self._index_loaded = True

        try:
            if _PYBREAKER_AVAILABLE:
                _breaker(_do_load)()
            else:
                _do_load()
        except Exception as exc:
            logger.error(event="bm25_index_load_failed", error=str(exc))
            self.documents = []
            self.tokenized_corpus = []
            self.bm25 = None

    # BUILD INDEX — FULL REBUILD

    def build_index(self, documents: List[Any]) -> None:
        if not documents:
            logger.warning(event="bm25_empty_input")
            return

        start = time.time()
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        seen: Set[str] = set()

        for doc in documents[:self.max_docs]:
            try:
                text = getattr(doc, "text", "")
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue

                if structure.get("embedding_space", "text") != "text":
                    continue

                text = text[:settings.BM25_MAX_TEXT_CHARS]
                h = self._hash(text)

                if h in seen:
                    continue
                seen.add(h)

                tokens = self._tokenize(text)
                if not tokens:
                    continue

                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)

            except Exception as exc:
                logger.warning(event="bm25_doc_skip", error=str(exc))

        if not self.tokenized_corpus:
            logger.warning(event="bm25_no_corpus")
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()

        logger.info(
            event="bm25_index_built",
            docs=len(self.documents),
            latency=round(time.time() - start, 2),
        )

    # ADD DOCUMENT — SINGLE DOC INCREMENTAL (called from ingestion pipeline)

    def add_document(self, text: str, metadata: Dict[str, Any]) -> None:
        if not text or not text.strip():
            return
        text = text[:settings.BM25_MAX_TEXT_CHARS]
        h = self._hash(text)
        seen_existing: Set[str] = {
            self._hash(getattr(d, "text", "")[:settings.BM25_MAX_TEXT_CHARS])
            for d in self.documents
        }
        if h in seen_existing:
            return
        tokens = self._tokenize(text)
        if not tokens:
            return

        class _Doc:
            pass

        doc = _Doc()
        doc.text = text  # type: ignore[attr-defined]
        doc.structure = metadata  # type: ignore[attr-defined]
        doc.modality = metadata.get("modality", "text")  # type: ignore[attr-defined]
        doc.subtype = metadata.get("subtype")  # type: ignore[attr-defined]
        doc.source = metadata.get("source")  # type: ignore[attr-defined]
        doc.source_type = metadata.get("source_type")  # type: ignore[attr-defined]
        doc.chunk_id = metadata.get("chunk_id")  # type: ignore[attr-defined]
        doc.page = metadata.get("page")  # type: ignore[attr-defined]

        self.documents.append(doc)
        self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
        logger.info(
            event="bm25_document_added",
            session_id=metadata.get("session_id"),
            total=len(self.documents),
        )

    # ADD DOCUMENTS — INCREMENTAL

    def add_documents(self, documents: List[Any], session_id: str = "") -> None:
        if not documents:
            return

        start = time.time()
        added = 0

        seen_existing: Set[str] = {
            self._hash(getattr(d, "text", "")[:settings.BM25_MAX_TEXT_CHARS])
            for d in self.documents
        }

        for doc in documents:
            try:
                text = getattr(doc, "text", "")
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue

                if structure.get("embedding_space", "text") != "text":
                    continue

                text = text[:settings.BM25_MAX_TEXT_CHARS]
                h = self._hash(text)

                if h in seen_existing:
                    continue

                tokens = self._tokenize(text)
                if not tokens:
                    continue

                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)
                seen_existing.add(h)
                added += 1

                if len(self.documents) >= self.max_docs:
                    logger.warning(
                        event="bm25_max_docs_reached",
                        max=self.max_docs,
                        session_id=session_id,
                    )
                    break

            except Exception as exc:
                logger.warning(event="bm25_add_doc_skip", error=str(exc))

        if added == 0:
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()

        logger.info(
            event="bm25_documents_added",
            added=added,
            total=len(self.documents),
            latency=round(time.time() - start, 2),
            session_id=session_id,
        )

    # SCORE NORMALIZATION

    def _normalize_scores(self, raw_scores: Any) -> Any:
        if _NP_AVAILABLE:
            scores = np.asarray(raw_scores, dtype=float)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            max_s = float(scores.max()) if scores.size > 0 else 1e-6
            if max_s > 1e-6:
                scores = scores / max_s
            return scores
        else:
            scores = [float(s) for s in raw_scores]
            max_s = max(scores) if scores else 1e-6
            if max_s > 1e-6:
                scores = [s / max_s for s in scores]
            return scores

    # TOP-K INDICES

    def _topk_indices(self, norm_scores: Any, top_k: int) -> List[int]:
        if _NP_AVAILABLE:
            if len(norm_scores) <= top_k:
                idxs = list(range(len(norm_scores)))
            else:
                idxs = list(np.argpartition(norm_scores, -top_k)[-top_k:])
            idxs = sorted(idxs, key=lambda i: norm_scores[i], reverse=True)
            return idxs
        else:
            idxs = sorted(range(len(norm_scores)), key=lambda i: norm_scores[i], reverse=True)
            return idxs[:top_k]

    # FILTER APPLY

    def _passes_filters(
        self,
        meta: Dict[str, Any],
        session_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        # SESSION ISOLATION — always enforce when session_id is provided
        if session_id and meta.get("session_id") != session_id:
            return False

        if self.modality_filter and meta.get("modality") != self.modality_filter:
            return False

        if filters:
            if filters.get("modality") and meta.get("modality") != filters["modality"]:
                return False
            if filters.get("language") and meta.get("language") != filters["language"]:
                return False
            if filters.get("source_type") and meta.get("source_type") != filters["source_type"]:
                return False

        return True

    # SEARCH

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        if not self.bm25:
            self._load_index()

        if not self.bm25:
            logger.info(event="bm25_empty_index_returning_empty_list", session_id=session_id)
            return []

        if not query:
            return []

        start = time.time()
        top_k = min(top_k or settings.BM25_TOP_K, len(self.documents))

        if top_k <= 0:
            return []

        query = query[:settings.MAX_PROMPT_CHARS]
        tokens = self._tokenize(query)

        if not tokens:
            return []

        try:
            raw_scores = self.bm25.get_scores(tokens)
        except Exception as exc:
            logger.error(event="bm25_score_failed", error=str(exc), session_id=session_id)
            return []

        norm_scores = self._normalize_scores(raw_scores)
        idxs = self._topk_indices(norm_scores, top_k)

        results: List[Dict[str, Any]] = []
        modality_weights: Dict[str, float] = getattr(settings, "BM25_MODALITY_WEIGHTS", {
            "text": 1.0, "table": 1.1, "image": 0.9,
            "audio": 1.0, "video": 1.0,
        })

        for idx in idxs:
            if len(results) >= top_k:
                break

            if idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            meta = self._metadata(doc)

            if not self._passes_filters(meta, session_id, filters):
                continue

            text = getattr(doc, "text", "").strip()
            if not text:
                continue

            if _NP_AVAILABLE:
                raw_score = float(norm_scores[idx])
            else:
                raw_score = float(norm_scores[idx])

            modality_boost = modality_weights.get(meta.get("modality", "text"), 1.0)
            final_score = raw_score * modality_boost

            if final_score < settings.BM25_MIN_SCORE:
                continue

            results.append({
                "id": f"bm25_{idx}",
                "text": text[:settings.RAG_DOC_MAX_CHARS],
                "score": round(final_score, 5),
                "metadata": meta,
            })

        logger.info(
            event="bm25_search_success",
            query_len=len(query),
            results=len(results),
            top_k=top_k,
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )

        return results

    # ASYNC SEARCH WRAPPER

    async def async_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, session_id, top_k, filters)

    # GDPR PURGE — REMOVE ALL DOCS BY SESSION OR USER

    def purge_by_session(self, session_id: str) -> int:
        before = len(self.documents)
        filtered_docs = []
        filtered_corpus = []
        for doc, tokens in zip(self.documents, self.tokenized_corpus):
            s = getattr(doc, "structure", {}) or {}
            if s.get("session_id") != session_id:
                filtered_docs.append(doc)
                filtered_corpus.append(tokens)
        self.documents = filtered_docs
        self.tokenized_corpus = filtered_corpus
        removed = before - len(self.documents)
        if removed > 0:
            self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
            self._save_index()
            logger.info(
                event="bm25_purge_session",
                session_id=session_id,
                removed=removed,
            )
        return removed

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "ready": self.bm25 is not None,
            "doc_count": len(self.documents),
            "index_exists": _INDEX_FILE.exists(),
            "index_size_bytes": _INDEX_FILE.stat().st_size if _INDEX_FILE.exists() else 0,
            "index_path": str(_INDEX_FILE),
            "modality_filter": self.modality_filter,
            "bm25_available": _BM25_AVAILABLE,
            "numpy_available": _NP_AVAILABLE,
            "circuit_breaker": _PYBREAKER_AVAILABLE,
        }

    # CLEAR

    def clear(self) -> None:
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._index_loaded = False
        if _INDEX_FILE.exists():
            try:
                _INDEX_FILE.unlink()
                logger.info(event="bm25_index_cleared")
            except Exception as exc:
                logger.error(event="bm25_index_clear_failed", error=str(exc))


