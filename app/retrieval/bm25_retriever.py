import re
import time
import hashlib
from typing import List, Dict, Set

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:

    def __init__(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        self.modality_filter = None
        self.max_docs = settings.BM25_MAX_DOCS

        self.stopwords = {
            "the", "is", "and", "a", "an", "of", "to", "in", "on", "for"
        }

    #  HASH 
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    #  TOKENIZE 
    def _tokenize(self, text: str) -> List[str]:

        text = str(text or "").lower()

        tokens = re.findall(r"\b[a-z0-9]+\b", text)

        tokens = [t for t in tokens if t not in self.stopwords]

        return tokens[:settings.BM25_MAX_TOKENS]

    #  METADATA 
    def _metadata(self, doc):

        s = dict(getattr(doc, "structure", {}) or {})

        return {
            "modality": getattr(doc, "modality", "text"),
            "subtype": getattr(doc, "subtype", None),
            "source": getattr(doc, "source", None),
            "doc_id": s.get("doc_id"),
            "chunk_id": getattr(doc, "chunk_id", None),
            "session_id": s.get("session_id"),
            "content_type": s.get("content_type"),
        }

    #  FILTER 
    def set_modality_filter(self, modality: str):
        self.modality_filter = modality

    #  BUILD 
    def build_index(self, documents: List):

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

                # only text space
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

            except Exception as e:
                logger.warning(event="bm25_doc_skip", error=str(e))

        if not self.tokenized_corpus:
            logger.warning(event="bm25_no_corpus")
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(
            event="bm25_index_built",
            docs=len(self.documents),
            latency=round(time.time() - start, 2)
        )

    #  SEARCH 
    def search(self, query: str, session_id: str = None, top_k: int = None):

        if not self.bm25:
            logger.warning(event="bm25_not_ready")
            return []

        if not query:
            return []

        start = time.time()

        top_k = top_k or settings.BM25_TOP_K

        query = query[:settings.MAX_PROMPT_CHARS]
        tokens = self._tokenize(query)

        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)

        if scores.size == 0:
            return []

        # normalization (important for fusion)
        max_score = max(scores.max(), 1e-6)
        norm_scores = scores / max_score

        idxs = np.argpartition(norm_scores, -top_k)[-top_k:]
        idxs = idxs[np.argsort(norm_scores[idxs])[::-1]]

        results = []

        weights = getattr(settings, "BM25_MODALITY_WEIGHTS", {})

        for idx in idxs:
            if len(results) >= top_k:
                break

            doc = self.documents[idx]
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

            results.append({
                "id": f"bm25_{idx}",
                "text": text[:settings.RAG_DOC_MAX_CHARS],
                "score": score,
                "metadata": meta,
            })

        logger.info(
            event="bm25_retrieved",
            results=len(results),
            latency=round(time.time() - start, 2)
        )

        return results