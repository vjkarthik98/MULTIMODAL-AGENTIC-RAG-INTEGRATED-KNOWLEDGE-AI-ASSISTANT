import re

import numpy as np

from app.utils.logger import get_logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - lightweight fallback
    class BM25Okapi:  # type: ignore[override]
        def __init__(self, tokenized_corpus):
            self.corpus = tokenized_corpus
            self.avgdl = sum(len(document) for document in tokenized_corpus) / max(len(tokenized_corpus), 1)
            self.doc_freqs = []
            self.idf = {}
            self.k1 = 1.5
            self.b = 0.75

            frequencies = {}
            for document in tokenized_corpus:
                doc_frequency = {}
                for token in document:
                    doc_frequency[token] = doc_frequency.get(token, 0) + 1
                self.doc_freqs.append(doc_frequency)
                for token in doc_frequency:
                    frequencies[token] = frequencies.get(token, 0) + 1

            total_documents = len(tokenized_corpus)
            for token, frequency in frequencies.items():
                self.idf[token] = np.log(1 + (total_documents - frequency + 0.5) / (frequency + 0.5))

        def get_scores(self, query_tokens):
            scores = np.zeros(len(self.corpus), dtype=float)
            for index, document in enumerate(self.corpus):
                doc_length = len(document) or 1
                frequencies = self.doc_freqs[index]
                for token in query_tokens:
                    frequency = frequencies.get(token, 0)
                    if not frequency:
                        continue
                    idf = self.idf.get(token, 0.0)
                    numerator = frequency * (self.k1 + 1)
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * doc_length / max(self.avgdl, 1.0)
                    )
                    scores[index] += idf * numerator / denominator
            return scores


logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.modality_filter = None

    def _tokenize(self, text: str):
        normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        return normalized.split()

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
            "segment_index": structure.get("segment_index"),
            "linked_segment_index": structure.get("linked_segment_index"),
            "start_time": structure.get("start_time"),
            "end_time": structure.get("end_time"),
        }

    def set_modality_filter(self, modality: str):
        self.modality_filter = modality

    def build_index(self, documents):
        if not documents:
            logger.warning("[BM25] No documents provided")
            return

        filtered_documents = []
        corpus = []

        for document in documents:
            text = getattr(document, "text", None)
            structure = getattr(document, "structure", {}) or {}
            embedding_space = structure.get("embedding_space", "text")

            if embedding_space != "text" or not text:
                continue

            filtered_documents.append(document)
            corpus.append(text)

        if not corpus:
            logger.warning("[BM25] No valid text corpus after filtering")
            return

        self.documents = filtered_documents
        self.tokenized_corpus = [self._tokenize(text) for text in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("[BM25] Index built | docs=%s", len(self.documents))

    def search(self, query, session_id=None, top_k=10):
        if not self.bm25:
            logger.warning("[BM25] Index not built")
            return []
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        scores = np.asarray(self.bm25.get_scores(self._tokenize(query)), dtype=float)
        if scores.size == 0:
            return []

        max_score = float(scores.max()) if scores.size else 1.0
        if max_score <= 0:
            max_score = 1.0
        normalized_scores = scores / max_score

        top_indices = np.argsort(normalized_scores)[::-1][: top_k * 3]
        results = []

        for index in top_indices:
            document = self.documents[index]
            metadata = self._doc_metadata(document)

            if session_id and metadata.get("session_id") != session_id:
                continue
            if self.modality_filter and metadata.get("modality") != self.modality_filter:
                continue

            score = float(normalized_scores[index])
            modality = metadata.get("modality")
            content_type = metadata.get("content_type")
            chunk_index = metadata.get("structure", {}).get("chunk_index", 0)
            timestamp = metadata.get("timestamp")

            if modality == "audio":
                score *= 1.05
            elif modality == "image":
                score *= 1.1 if metadata.get("subtype") == "caption" else 0.95
            elif modality == "text" and metadata.get("subtype") == "heading":
                score *= 1.1
            elif modality == "video":
                if content_type == "video_speech":
                    score *= 1.15
                elif content_type == "video_frame":
                    score *= 1.05
                if timestamp is not None and timestamp < 10:
                    score *= 1.05

            if chunk_index == 0:
                score *= 1.05

            results.append(
                {
                    "id": f"bm25_{index}",
                    "text": document.text,
                    "score": score,
                    "metadata": metadata,
                }
            )

            if len(results) >= top_k:
                break

        logger.info("[BM25] Retrieved | session_id=%s | results=%s", session_id, len(results))
        return results
