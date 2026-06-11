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
from app.utils.paths import user_bm25_path

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


# Legacy global index (kept for backward-compat during migration; per-user path used by default)
_LEGACY_INDEX_DIR = Path(settings.BM25_INDEX_DIR) if hasattr(settings, "BM25_INDEX_DIR") else settings.DATA_DIR / "bm25_index"
_LEGACY_INDEX_FILE = _LEGACY_INDEX_DIR / "bm25_index.pkl"

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

# FINANCIAL ABBREVIATION EXPANSION — applied at both index-time and query-time
# so "EPS" and "earnings per share" match each other even when neither phrase
# appears verbatim in the other's text.
_FIN_ABBR: Dict[str, List[str]] = {
    "eps":              ["earnings", "per", "share"],
    "ebitda":           ["earnings", "before", "interest", "taxes", "depreciation", "amortization"],
    "revenue":          ["net", "sales"],
    "net sales":        ["revenue"],
    "yoy":              ["year", "over", "year"],
    "fy":               ["fiscal", "year"],
    "q1":               ["first", "quarter"],
    "q2":               ["second", "quarter"],
    "q3":               ["third", "quarter"],
    "q4":               ["fourth", "quarter"],
    "ceo":              ["chief", "executive", "officer"],
    "cfo":              ["chief", "financial", "officer"],
    "capex":            ["capital", "expenditure", "expenditures"],
    "r&d":              ["research", "development"],
    "gm":               ["gross", "margin"],
    "op":               ["operating"],
    "ttm":              ["trailing", "twelve", "months"],
    "pe":               ["price", "earnings"],
    "pb":               ["price", "book"],
    "roe":              ["return", "equity"],
    "roa":              ["return", "assets"],
    "fcf":              ["free", "cash", "flow"],
    "gaap":             ["generally", "accepted", "accounting", "principles"],
    "non-gaap":         ["non", "gaap", "adjusted"],
    "diluted":          ["diluted", "per", "share"],
    "buyback":          ["share", "repurchase"],
    "repurchase":       ["buyback", "share", "repurchase"],
}

# HIGH-VALUE FINANCIAL BIGRAMS — kept as single tokens in BM25 corpus
# so phrases like "net income" score higher than individual words.
_FIN_BIGRAMS: Set[str] = {
    "net income", "net sales", "gross margin", "operating income",
    "earnings per share", "diluted eps", "basic eps", "per share",
    "cash flow", "free cash", "total revenue", "total assets",
    "total liabilities", "shareholders equity", "return on equity",
    "return on assets", "research development", "capital expenditure",
    "year over year", "fiscal year", "first quarter", "second quarter",
    "third quarter", "fourth quarter", "annual report", "form 10k",
    "income statement", "balance sheet", "cash flow statement",
    "interest expense", "tax rate", "effective tax", "share repurchase",
    "stock buyback", "dividend per share", "book value",
}


class BM25Document:
    """Picklable document wrapper for BM25 index storage."""
    __slots__ = [
        "text", "structure", "modality", "subtype",
        "source", "source_type", "chunk_id", "page",
    ]

    @classmethod
    def from_payload(cls, p: Dict[str, Any]) -> "BM25Document":
        obj = cls()
        obj.text = p.get("text") or p.get("content") or ""
        obj.structure = {
            "doc_id": p.get("doc_id"),
            "chunk_id": p.get("chunk_id"),
            "session_id": p.get("session_id"),
            "content_type": p.get("content_type"),
            "language": p.get("language"),
            "timestamp_start": p.get("timestamp_start"),
            "timestamp_end": p.get("timestamp_end"),
            "ingestion_time": p.get("ingestion_time"),
            "checksum_sha256": p.get("checksum_sha256"),
            "section_number": p.get("section_number"),
            "is_forward_looking": p.get("is_forward_looking", False),
            "embedding_space": "text",
        }
        obj.modality = p.get("modality", "text")
        obj.subtype = p.get("subtype")
        obj.source = p.get("source")
        obj.source_type = p.get("source_type")
        obj.chunk_id = p.get("chunk_id")
        obj.page = p.get("page")
        return obj


class BM25Retriever:

    def __init__(self, user_id: Optional[str] = None) -> None:
        self.user_id: Optional[str] = user_id
        self.documents: List[Any] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.modality_filter: Optional[str] = None
        self.max_docs: int = settings.BM25_MAX_DOCS
        self._index_loaded: bool = False
        self._loaded_user_id: Optional[str] = None  # tracks which user's index is loaded

    def _index_file(self, user_id: Optional[str] = None) -> Path:
        uid = user_id or self.user_id
        if uid:
            return user_bm25_path(uid)
        return _LEGACY_INDEX_FILE

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # TOKENIZE

    def _tokenize(self, text: str) -> List[str]:
        text = str(text or "").lower()

        # ── Bigram extraction ─────────────────────────────────────────────
        # Before splitting into unigrams, scan for high-value financial
        # bigrams and add them as single tokens. This makes "net income"
        # score higher than the product of "net" × "income" individually.
        bigram_tokens: List[str] = []
        for bigram in _FIN_BIGRAMS:
            if bigram in text:
                bigram_tokens.append(bigram.replace(" ", "_"))

        # ── Unigram extraction ────────────────────────────────────────────
        # Capture decimal/dollar figures as single tokens BEFORE splitting on
        # word boundaries so "$6.13" and "6.13" index as one token, not
        # ["6", "13"].
        tokens = re.findall(r'\d+\.\d+|\b[a-z0-9]+\b', text)
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

        # ── Abbreviation expansion ────────────────────────────────────────
        # For every known abbreviation found in the token stream, add its
        # expanded form too. Both forms are indexed so abbreviated queries
        # match full-form text and vice-versa.
        expanded: List[str] = []
        for tok in tokens:
            expanded.append(tok)
            if tok in _FIN_ABBR:
                expanded.extend(_FIN_ABBR[tok])

        all_tokens = bigram_tokens + expanded
        return all_tokens[:settings.BM25_MAX_TOKENS]

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
            "timestamp_end": s.get("timestamp_end"),
            "ingestion_time": s.get("ingestion_time"),
            "checksum_sha256": s.get("checksum_sha256"),
            "section_number": s.get("section_number"),
            # section_id / section_title are the locators the UI uses for the
            # source chip (DOCX has no page numbers). Without these, a chunk
            # retrieved via BM25 loses its heading and the chip shows only the
            # filename — even though the vector path carries it.
            "section_id": s.get("section_id"),
            "section_title": s.get("section_title"),
            "is_forward_looking": s.get("is_forward_looking", False),
            # Image caption — used as locator in source chip (same role as section_title for DOCX)
            "caption": s.get("caption"),
        }

    # MODALITY FILTER SETTER

    def set_modality_filter(self, modality: Optional[str]) -> None:
        self.modality_filter = modality

    # CIRCUIT-BROKEN SAVE

    def _save_index(self, user_id: Optional[str] = None) -> None:
        index_file = self._index_file(user_id)

        def _do_save() -> None:
            index_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
                "saved_at": time.time(),
                "doc_count": len(self.documents),
            }
            tmp_path = index_file.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(index_file)
            logger.info(
                event="bm25_index_saved",
                path=str(index_file),
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

    def _load_index(self, user_id: Optional[str] = None) -> None:
        effective_uid = user_id or self.user_id
        # Reload if a different user's index is requested
        if self._index_loaded and self._loaded_user_id == effective_uid:
            return
        if self._index_loaded and self._loaded_user_id != effective_uid:
            self._index_loaded = False
            self.documents = []
            self.tokenized_corpus = []
            self.bm25 = None

        index_file = self._index_file(user_id)

        if not index_file.exists():
            logger.info(event="bm25_no_saved_index", path=str(index_file))
            return

        def _do_load() -> None:
            with open(index_file, "rb") as f:
                payload = pickle.load(f)
            self.documents = payload.get("documents", [])
            self.tokenized_corpus = payload.get("tokenized_corpus", [])
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(
                    event="bm25_index_loaded",
                    docs=len(self.documents),
                    path=str(index_file),
                )
            else:
                logger.warning(event="bm25_saved_index_empty")
            self._index_loaded = True
            self._loaded_user_id = effective_uid

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

    def build_index(self, documents: List[Any], user_id: Optional[str] = None) -> None:
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
        self._save_index(user_id)

        logger.info(
            event="bm25_index_built",
            docs=len(self.documents),
            latency=round(time.time() - start, 2),
        )

    # ADD DOCUMENT — SINGLE DOC INCREMENTAL (called from ingestion pipeline)

    def add_document(self, text: str, metadata: Dict[str, Any], user_id: Optional[str] = None) -> None:
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

        doc = BM25Document()
        doc.text = text
        doc.structure = metadata
        doc.modality = metadata.get("modality", "text")
        doc.subtype = metadata.get("subtype")
        doc.source = metadata.get("source")
        doc.source_type = metadata.get("source_type")
        doc.chunk_id = metadata.get("chunk_id")
        doc.page = metadata.get("page")

        self.documents.append(doc)
        self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index(user_id)
        logger.info(
            event="bm25_document_added",
            session_id=metadata.get("session_id"),
            user_id=user_id or self.user_id,
            total=len(self.documents),
        )

    # ADD DOCUMENTS — INCREMENTAL

    def add_documents(self, documents: List[Any], session_id: str = "", user_id: Optional[str] = None) -> None:
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
        self._save_index(user_id)

        logger.info(
            event="bm25_documents_added",
            added=added,
            total=len(self.documents),
            latency=round(time.time() - start, 2),
            session_id=session_id,
            user_id=user_id or self.user_id,
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
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if not self.bm25:
            self._load_index(user_id)

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
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, session_id, top_k, filters, user_id)

    # GDPR PURGE — REMOVE ALL DOCS BY SESSION OR USER

    def delete_by_source(self, filename: str, user_id: Optional[str] = None) -> int:
        """Remove all BM25 entries whose source filename contains `filename`."""
        before = len(self.documents)
        filtered_docs = []
        filtered_corpus = []
        for doc, tokens in zip(self.documents, self.tokenized_corpus):
            source = getattr(doc, "source", "") or ""
            if filename not in source:
                filtered_docs.append(doc)
                filtered_corpus.append(tokens)
        self.documents = filtered_docs
        self.tokenized_corpus = filtered_corpus
        removed = before - len(self.documents)
        if removed > 0:
            self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
            self._save_index(user_id)
            logger.info(event="bm25_delete_by_source", filename=filename, removed=removed)
        return removed

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

    def health_check(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        index_file = self._index_file(user_id)
        return {
            "ready": self.bm25 is not None,
            "doc_count": len(self.documents),
            "index_exists": index_file.exists(),
            "index_size_bytes": index_file.stat().st_size if index_file.exists() else 0,
            "index_path": str(index_file),
            "user_id": user_id or self.user_id,
            "modality_filter": self.modality_filter,
            "bm25_available": _BM25_AVAILABLE,
            "numpy_available": _NP_AVAILABLE,
            "circuit_breaker": _PYBREAKER_AVAILABLE,
        }

    # CLEAR

    def clear(self, user_id: Optional[str] = None) -> None:
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._index_loaded = False
        index_file = self._index_file(user_id)
        if index_file.exists():
            try:
                index_file.unlink()
                logger.info(event="bm25_index_cleared", path=str(index_file))
            except Exception as exc:
                logger.error(event="bm25_index_clear_failed", error=str(exc))


