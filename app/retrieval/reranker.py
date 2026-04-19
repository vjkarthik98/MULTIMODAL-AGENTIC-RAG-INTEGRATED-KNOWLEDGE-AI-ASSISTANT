import time

import numpy as np

from app.core.model_loader import model_loader
from app.utils.logger import get_logger


logger = get_logger(__name__)


class Reranker:
    def __init__(self):
        self.model = model_loader.get_reranker()
        logger.info("[Reranker] Loaded from ModelLoader")

    def _build_context(self, document):
        metadata = document.get("metadata", {}) or {}
        structure = metadata.get("structure", {}) or {}
        modality = metadata.get("modality", "text")
        page = metadata.get("page")
        source = metadata.get("source_type") or metadata.get("source")
        embedding_space = metadata.get("embedding_space", structure.get("embedding_space", "text"))
        content_type = metadata.get("content_type", structure.get("content_type"))

        context = ""
        if source:
            context += f"[{str(source).upper()}]"
        if page:
            context += f"[Page {page}]"

        if modality == "table":
            context += "[Structured Table Data]"
        elif modality == "image":
            context += "[Visual Content]"
        elif modality == "audio":
            start = metadata.get("start_time", structure.get("start_time"))
            end = metadata.get("end_time", structure.get("end_time"))
            context += f"[Spoken from {start}s to {end}s]" if start is not None and end is not None else "[Audio Speech]"
        elif modality == "video":
            if content_type == "video_speech":
                start = metadata.get("start_time", structure.get("start_time"))
                end = metadata.get("end_time", structure.get("end_time"))
                context += f"[Video speech from {start}s to {end}s]" if start is not None and end is not None else "[Video speech]"
            elif content_type == "video_frame":
                timestamp = metadata.get("timestamp", structure.get("timestamp"))
                context += f"[Video frame at {timestamp}s]" if timestamp is not None else "[Video visual content]"

        if embedding_space == "vision":
            context += "[Visual Similarity Match]"

        return context + document.get("text", "")

    def rerank(self, query, documents, top_k=5):
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if not documents:
            logger.warning("[Reranker] No documents to rerank")
            return []

        start_time = time.time()

        try:
            pairs = []
            valid_documents = []
            for document in documents:
                text = document.get("text", "")
                if not text:
                    continue
                pairs.append((query, self._build_context(document)))
                valid_documents.append(document)

            if not pairs:
                logger.warning("[Reranker] No valid text found")
                return []

            scores = np.asarray(self.model.predict(pairs)).reshape(-1)
            if scores.size != len(valid_documents):
                logger.warning("[Reranker] Score length mismatch, adjusting")
                scores = scores[: len(valid_documents)]
                valid_documents = valid_documents[: len(scores)]

            scored_documents = []
            for document, score in zip(valid_documents, scores):
                metadata = document.get("metadata", {}) or {}
                structure = metadata.get("structure", {}) or {}
                modality = metadata.get("modality", "text")
                embedding_space = metadata.get("embedding_space", structure.get("embedding_space", "text"))
                content_type = metadata.get("content_type", structure.get("content_type"))
                chunk_index = structure.get("chunk_index", 0)
                timestamp = metadata.get("timestamp", structure.get("timestamp"))

                position_boost = 1.0 + (0.2 / (chunk_index + 1))
                modality_boost = 1.0
                if modality == "table":
                    modality_boost = 1.1
                elif modality == "image":
                    modality_boost = 1.15
                elif modality == "audio":
                    modality_boost = 1.1
                elif modality == "video":
                    if content_type == "video_speech":
                        modality_boost = 1.3
                    elif content_type == "video_frame":
                        modality_boost = 1.2

                space_boost = 1.1 if embedding_space == "vision" else 1.0
                temporal_boost = 1.05 if timestamp is not None and timestamp < 10 else 1.0
                fusion_score = document.get("score", 0.0)

                final_score = (
                    float(score) * 0.7 + fusion_score * 0.3
                ) * position_boost * modality_boost * space_boost * temporal_boost

                scored_documents.append(
                    {
                        "text": document["text"],
                        "metadata": metadata,
                        "score": final_score,
                    }
                )

            scored_documents.sort(key=lambda item: item["score"], reverse=True)

            final_results = []
            seen_keys = set()
            for document in scored_documents:
                metadata = document.get("metadata", {}) or {}
                key = (
                    metadata.get("doc_id"),
                    metadata.get("chunk_id"),
                    metadata.get("source"),
                    document.get("text"),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                final_results.append(document)
                if len(final_results) >= top_k:
                    break

            latency = time.time() - start_time
            logger.info("[Reranker] SUCCESS | docs=%s | latency=%.2fs", len(final_results), latency)
            return final_results

        except Exception as exc:
            logger.error("[Reranker] FAILED | error=%s", exc)
            return [
                {
                    "text": document.get("text", ""),
                    "metadata": document.get("metadata", {}),
                    "score": document.get("score", 0.0),
                }
                for document in documents[:top_k]
            ]
