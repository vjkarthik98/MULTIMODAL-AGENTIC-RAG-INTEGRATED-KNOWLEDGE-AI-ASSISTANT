import re
import time
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

        # SIMPLE STOPWORDS (LIGHTWEIGHT)
        self.stopwords = {
            "the", "is", "and", "a", "an", "of", "to", "in", "on", "for"
        }

    #  TOKENIZATION 
    def _tokenize(self, text: str) -> List[str]:

        text = str(text or "").lower()

        # KEEP ALPHANUMERIC WORDS
        tokens = re.findall(r"\b[a-z0-9]+\b", text)

        # REMOVE STOPWORDS
        tokens = [t for t in tokens if t not in self.stopwords]

        return tokens[:settings.BM25_MAX_TOKENS]

    #  METADATA 
    def _doc_metadata(self, document):

        structure = dict(getattr(document, "structure", {}) or {})

        return {
            "modality": getattr(document, "modality", "text"),
            "subtype": getattr(document, "subtype", None),
            "source": getattr(document, "source", None),
            "doc_id": structure.get("doc_id"),
            "chunk_id": getattr(document, "chunk_id", None),
            "session_id": structure.get("session_id"),
            "content_type": structure.get("content_type"),
        }

    #  FILTER 
    def set_modality_filter(self, modality: str):
        self.modality_filter = modality

    #  BUILD INDEX 
    def build_index(self, documents: List):

        if not documents:
            logger.warning("[BM25] empty input")
            return

        start = time.time()

        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        seen: Set[str] = set()

        for doc in documents[:self.max_docs]:
            try:
                text = getattr(doc, "text", None)
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue

                if structure.get("embedding_space", "text") != "text":
                    continue

                text = str(text)[:settings.BM25_MAX_TEXT_CHARS]

                # DEDUPLICATION
                key = text[:100]
                if key in seen:
                    continue
                seen.add(key)

                tokens = self._tokenize(text)

                if not tokens:
                    continue

                self.documents.append(doc)
                self.tokenized_corpus.append(tokens)

            except Exception:
                continue

        if not self.tokenized_corpus:
            logger.warning("[BM25] no valid corpus")
            return

        # FIX: PASS TOKENIZED CORPUS
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(
            "[BM25] index built | docs=%s latency=%ss",
            len(self.documents),
            round(time.time() - start, 2)
        )

    #  SEARCH 
    def search(self, query: str, session_id: str = None, top_k: int = None):

        if not self.bm25:
            logger.warning("[BM25] index not built")
            return []

        if not query or not query.strip():
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

        # NORMALIZATION
        max_score = max(scores.max(), 1e-6)
        normalized = scores / max_score

        indices = np.argsort(normalized)[::-1]

        results = []

        for idx in indices:

            if len(results) >= top_k:
                break

            doc = self.documents[idx]
            metadata = self._doc_metadata(doc)

            # EARLY FILTERING
            if session_id and metadata.get("session_id") != session_id:
                continue

            if self.modality_filter and metadata.get("modality") != self.modality_filter:
                continue

            text = getattr(doc, "text", "").strip()
            if not text:
                continue

            score = float(normalized[idx])

            # MODALITY WEIGHT
            weights = getattr(settings, "BM25_MODALITY_WEIGHTS", {})
            score *= weights.get(metadata.get("modality", "text"), 1.0)

            results.append({
                "id": f"bm25_{idx}",
                "text": text[:settings.RAG_DOC_MAX_CHARS],
                "score": score,
                "metadata": metadata,
            })

        logger.info(
            "[BM25] retrieved | results=%s latency=%ss",
            len(results),
            round(time.time() - start, 2)
        )

        return results