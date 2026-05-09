import hashlib
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# INDEX PERSISTENCE PATH

_INDEX_DIR  = settings.DATA_DIR / "bm25_index"
_INDEX_FILE = _INDEX_DIR / "bm25_index.pkl"


class BM25Retriever:

    def __init__(self) -> None:
        self.documents:         List      = []
        self.tokenized_corpus:  List      = []
        self.bm25:              Optional[BM25Okapi] = None
        self.modality_filter:   Optional[str]       = None
        self.max_docs:          int       = settings.BM25_MAX_DOCS

        self.stopwords: Set[str] = {
            "the", "is", "and", "a", "an", "of", "to", "in", "on", "for",
            "at", "by", "with", "from", "this", "that", "it", "be", "as",
        }

        # LOAD PERSISTED INDEX ON STARTUP
        self._load_index()

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # TOKENIZE

    def _tokenize(self, text: str) -> List[str]:
        text   = str(text or "").lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        return tokens[:settings.BM25_MAX_TOKENS]

    # METADATA

    def _metadata(self, doc) -> Dict:
        s = dict(getattr(doc, "structure", {}) or {})

        return {
            "modality":     getattr(doc, "modality", "text"),
            "subtype":      getattr(doc, "subtype", None),
            "source":       getattr(doc, "source", None),
            "source_type":  getattr(doc, "source_type", None),
            "doc_id":       s.get("doc_id"),
            "chunk_id":     getattr(doc, "chunk_id", None),
            "session_id":   s.get("session_id"),
            "content_type": s.get("content_type"),
            "page":         getattr(doc, "page", None),
        }

    # MODALITY FILTER

    def set_modality_filter(self, modality: Optional[str]) -> None:
        self.modality_filter = modality

    # PERSISTENCE: SAVE

    def _save_index(self) -> None:
        try:
            _INDEX_DIR.mkdir(parents=True, exist_ok=True)

            payload = {
                "documents":        self.documents,
                "tokenized_corpus": self.tokenized_corpus,
            }

            with open(_INDEX_FILE, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(
                event="bm25_index_saved",
                path=str(_INDEX_FILE),
                docs=len(self.documents),
            )

        except Exception as e:
            logger.error(event="bm25_index_save_failed", error=str(e))

    # PERSISTENCE: LOAD

    def _load_index(self) -> None:
        if not _INDEX_FILE.exists():
            logger.info(event="bm25_no_saved_index")
            return

        try:
            with open(_INDEX_FILE, "rb") as f:
                payload = pickle.load(f)

            self.documents        = payload.get("documents", [])
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

        except Exception as e:
            logger.error(event="bm25_index_load_failed", error=str(e))
            # RESET TO CLEAN STATE ON CORRUPT INDEX
            self.documents        = []
            self.tokenized_corpus = []
            self.bm25             = None

    # BUILD INDEX (full rebuild)

    def build_index(self, documents: List) -> None:

        if not documents:
            logger.warning(event="bm25_empty_input")
            return

        start = time.time()

        self.documents        = []
        self.tokenized_corpus = []
        self.bm25             = None

        seen: Set[str] = set()

        for doc in documents[:self.max_docs]:
            try:
                text      = getattr(doc, "text", "")
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue

                if structure.get("embedding_space", "text") != "text":
                    continue

                text = text[:settings.BM25_MAX_TEXT_CHARS]
                h    = self._hash(text)

                if h in seen:
                    continue
                seen.add(h)

                tokens = self._tokenize(text)
                if not tokens:
                    continue

                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)

            except Exception as e:
                logger.warning(event="bm25_doc_skip", error=str(e))

        if not self.tokenized_corpus:
            logger.warning(event="bm25_no_corpus")
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # PERSIST TO DISK
        self._save_index()

        logger.info(
            event="bm25_index_built",
            docs=len(self.documents),
            latency=round(time.time() - start, 2),
        )

    # ADD DOCUMENTS (incremental — used by ingestion_pipeline)

    def add_documents(self, documents: List, session_id: str = "") -> None:

        if not documents:
            return

        start    = time.time()
        added    = 0
        seen_existing: Set[str] = {
            self._hash(getattr(d, "text", "")[:settings.BM25_MAX_TEXT_CHARS])
            for d in self.documents
        }

        for doc in documents:
            try:
                text      = getattr(doc, "text", "")
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue

                if structure.get("embedding_space", "text") != "text":
                    continue

                text = text[:settings.BM25_MAX_TEXT_CHARS]
                h    = self._hash(text)

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
                    logger.warning(event="bm25_max_docs_reached", max=self.max_docs)
                    break

            except Exception as e:
                logger.warning(event="bm25_add_doc_skip", error=str(e))

        if added == 0:
            return

        # REBUILD BM25 WITH NEW DOCUMENTS
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # PERSIST UPDATED INDEX
        self._save_index()

        logger.info(
            event="bm25_documents_added",
            added=added,
            total=len(self.documents),
            latency=round(time.time() - start, 2),
            session_id=session_id,
        )

    # SEARCH

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        if not self.bm25:
            logger.warning(event="bm25_not_ready", session_id=session_id)
            return []

        if not query:
            return []

        start  = time.time()
        top_k  = min(top_k or settings.BM25_TOP_K, len(self.documents))

        if top_k <= 0:
            return []

        query  = query[:settings.MAX_PROMPT_CHARS]
        tokens = self._tokenize(query)

        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)

        if scores.size == 0:
            return []

        max_score  = max(float(scores.max()), 1e-6)
        norm_scores = scores / max_score

        idxs = np.argpartition(norm_scores, -top_k)[-top_k:]
        idxs = idxs[np.argsort(norm_scores[idxs])[::-1]]

        results: List[Dict] = []
        weights = settings.BM25_MODALITY_WEIGHTS

        for idx in idxs:
            if len(results) >= top_k:
                break

            doc  = self.documents[idx]
            meta = self._metadata(doc)

            if session_id and meta.get("session_id") != session_id:
                continue

            if self.modality_filter and meta.get("modality") != self.modality_filter:
                continue

            text = getattr(doc, "text", "").strip()
            if not text:
                continue

            score = float(norm_scores[idx])
            score *= weights.get(meta.get("modality", "text"), 1.0)

            if score < settings.BM25_MIN_SCORE:
                continue

            results.append({
                "id":       f"bm25_{idx}",
                "text":     text[:settings.RAG_DOC_MAX_CHARS],
                "score":    round(score, 5),
                "metadata": meta,
            })

        logger.info(
            event="bm25_search_success",
            results=len(results),
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )

        return results

    # HEALTH CHECK

    def health_check(self) -> Dict:
        return {
            "ready":       self.bm25 is not None,
            "doc_count":   len(self.documents),
            "index_size":  _INDEX_FILE.stat().st_size if _INDEX_FILE.exists() else 0,
            "index_path":  str(_INDEX_FILE),
            "modality_filter": self.modality_filter,
        }

    # CLEAR

    def clear(self) -> None:
        self.documents        = []
        self.tokenized_corpus = []
        self.bm25             = None

        if _INDEX_FILE.exists():
            try:
                _INDEX_FILE.unlink()
                logger.info(event="bm25_index_cleared")
            except Exception as e:
                logger.error(event="bm25_index_clear_failed", error=str(e))