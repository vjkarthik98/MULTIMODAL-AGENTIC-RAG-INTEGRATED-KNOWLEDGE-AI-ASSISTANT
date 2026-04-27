import re
import time
from typing import List

import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    from rank_bm25 import BM25Okapi  


logger = get_logger(__name__)


class BM25Retriever:

    def __init__(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.modality_filter = None

        self.max_docs = settings.BM25_MAX_DOCS

    # TOKENIZATION 
    def _tokenize(self, text: str):
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()

        return tokens[:settings.BM25_MAX_TOKENS]

    # METADATA 
    def _doc_metadata(self, document):
        structure = dict(getattr(document, "structure", {}) or {})

        return {
            "modality": getattr(document, "modality", "text"),
            "subtype": getattr(document, "subtype", None),
            "source": getattr(document, "source", None),
            "source_type": getattr(document, "source_type", None),
            "page": getattr(document, "page", None),
            "doc_id": structure.get("doc_id"),
            "chunk_id": getattr(document, "chunk_id", None),
            "session_id": structure.get("session_id"),
            "structure": structure,
            "embedding_space": structure.get("embedding_space", "text"),
            "content_type": structure.get("content_type"),
            "timestamp": structure.get("timestamp"),
        }

    # FILTER 
    def set_modality_filter(self, modality: str):
        self.modality_filter = modality

    # BUILD INDEX 
    def build_index(self, documents: List):

        if not documents:
            logger.warning("[BM25] empty input")
            return

        start = time.time()

        # RESET STATE
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None

        filtered = []
        corpus = []

        for doc in documents[:self.max_docs]:
            try:
                text = getattr(doc, "text", None)
                structure = getattr(doc, "structure", {}) or {}

                if not text:
                    continue

                if structure.get("embedding_space", "text") != "text":
                    continue

                # Truncate text
                text = str(text)[:settings.BM25_MAX_TEXT_CHARS]

                tokens = self._tokenize(text)

                if not tokens:
                    continue

                filtered.append(doc)
                corpus.append(text)

            except Exception:
                continue

        if not corpus:
            logger.warning("[BM25] no valid corpus after filtering")
            return

        self.documents = filtered
        self.tokenized_corpus = corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        latency = round(time.time() - start, 2)

        logger.info(
            "[BM25] index built | docs=%s latency=%ss",
            len(self.documents),
            latency
        )

    # SEARCH 
    def search(self, query: str, session_id: str = None, top_k: int = None):

        if not self.bm25:
            logger.warning("[BM25] search skipped (index not built)")
            return []

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        start = time.time()

        top_k = top_k or settings.BM25_TOP_K

        # Query safety
        query = query[:settings.MAX_PROMPT_CHARS]

        tokens = self._tokenize(query)

        if not tokens:
            logger.warning("[BM25] empty tokens for query")
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)

        if scores.size == 0:
            return []

        max_score = float(scores.max()) or 1.0


        normalized = scores / (max_score + 1e-6)

        indices = np.argsort(normalized)[::-1][: top_k * settings.BM25_CANDIDATE_MULTIPLIER]

        results = []

        for idx in indices:
            doc = self.documents[idx]
            metadata = self._doc_metadata(doc)

            if session_id and metadata.get("session_id") != session_id:
                continue

            if self.modality_filter and metadata.get("modality") != self.modality_filter:
                continue

            score = float(normalized[idx])


            # Config-driven modality boost
            modality = metadata.get("modality", "text")
            weights = getattr(settings, "BM25_MODALITY_WEIGHTS", {})

            score *= weights.get(modality, 1.0)

            results.append(
                {
                    "id": f"bm25_{idx}",
                    "text": doc.text[:settings.RAG_DOC_MAX_CHARS],
                    "score": score,
                    "metadata": metadata,
                }
            )

        # FALLBACK if session filter killed everything
        if not results and session_id:
            logger.warning("[BM25] fallback without session filter")

            for idx in indices[:top_k]:
                doc = self.documents[idx]
                metadata = self._doc_metadata(doc)

                results.append(
                    {
                        "id": f"bm25_{idx}",
                        "text": doc.text[:settings.RAG_DOC_MAX_CHARS],
                        "score": float(normalized[idx]),
                        "metadata": metadata,
                    }
                )


        latency = round(time.time() - start, 2)

        logger.info(
            "[BM25] retrieved | session_id=%s results=%s latency=%ss",
            session_id,
            len(results),
            latency
        )

        return results[:top_k]