import hashlib

from app.core.model_loader import model_loader
from app.utils.logger import get_logger


logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, bm25_retriever, vector_store, embedder, clip_text_embedder=None):
        self.bm25 = bm25_retriever
        self.vector_store = vector_store
        self.embedder = embedder
        self.clip_text_embedder = clip_text_embedder

    def _make_key(self, text, metadata):
        base = "|".join(
            [
                text or "",
                str(metadata.get("doc_id") or ""),
                str(metadata.get("chunk_id") or ""),
                str(metadata.get("source") or ""),
                str(metadata.get("embedding_space") or ""),
            ]
        )
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    def _modality_boost(self, metadata):
        modality = metadata.get("modality")
        content_type = metadata.get("content_type")

        if modality == "table":
            return 1.1
        if modality == "image":
            return 1.2
        if modality == "audio":
            return 1.15
        if modality == "video":
            if content_type == "video_speech":
                return 1.25
            if content_type == "video_frame":
                return 1.15
        return 1.0

    def _normalize(self, results):
        if not results:
            return []

        max_score = max(result.get("score", 0.0) for result in results) or 1.0
        normalized = []
        for result in results:
            result = dict(result)
            result["norm_score"] = result.get("score", 0.0) / (max_score + 1e-6)
            normalized.append(result)
        return normalized

    def search(self, query, session_id, top_k=10):
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if not session_id:
            raise ValueError("session_id required")

        try:
            logger.info("[HybridRetriever][START] session_id=%s", session_id)

            bm25_results = self._normalize(
                self.bm25.search(query, session_id=session_id, top_k=top_k * 2)
            )

            text_vector_results = []
            try:
                query_vector = self.embedder.embed_query(query, session_id=session_id)
                text_vector_results = self._normalize(
                    self.vector_store.search_text(
                        query_vector=query_vector,
                        limit=top_k * 2,
                        session_id=session_id,
                    )
                )
            except Exception as exc:
                logger.error("[HybridRetriever][TEXT_VECTOR_FAIL] session_id=%s | error=%s", session_id, exc)

            vision_results = []
            try:
                clip_embedder = self.clip_text_embedder or model_loader.get_clip_text_embedder()
                clip_query_vector = clip_embedder.embed(query)
                vision_results = self._normalize(
                    self.vector_store.search_vision(
                        query_vector=clip_query_vector,
                        limit=top_k * 2,
                        session_id=session_id,
                    )
                )
            except Exception as exc:
                logger.error("[HybridRetriever][VISION_FAIL] session_id=%s | error=%s", session_id, exc)

            combined = {}

            for result in bm25_results:
                metadata = result["metadata"]
                key = self._make_key(result["text"], metadata)
                boost = self._modality_boost(metadata)
                combined[key] = {
                    "text": result["text"],
                    "metadata": metadata,
                    "bm25_score": result["norm_score"],
                    "vector_score": 0.0,
                    "vision_score": 0.0,
                    "score": result["norm_score"] * 0.35 * boost,
                }

            for result in text_vector_results:
                metadata = result["metadata"]
                key = self._make_key(result["text"], metadata)
                boost = self._modality_boost(metadata)
                score = result["norm_score"]

                if key in combined:
                    combined[key]["vector_score"] = score
                    combined[key]["score"] += score * boost
                else:
                    combined[key] = {
                        "text": result["text"],
                        "metadata": metadata,
                        "bm25_score": 0.0,
                        "vector_score": score,
                        "vision_score": 0.0,
                        "score": score * boost,
                    }

            for result in vision_results:
                metadata = result["metadata"]
                key = self._make_key(result["text"], metadata)
                boost = self._modality_boost(metadata)
                score = result["norm_score"] * 1.15

                timestamp = metadata.get("timestamp")
                if timestamp is not None and timestamp < 10:
                    score *= 1.05

                if key in combined:
                    combined[key]["vision_score"] = score
                    combined[key]["score"] += score * boost
                else:
                    combined[key] = {
                        "text": result["text"],
                        "metadata": metadata,
                        "bm25_score": 0.0,
                        "vector_score": 0.0,
                        "vision_score": score,
                        "score": score * boost,
                    }

            results = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
            logger.info("[HybridRetriever][SUCCESS] session_id=%s | results=%s", session_id, len(results))
            return results[:top_k]

        except Exception as exc:
            logger.error("[HybridRetriever][FAILED] session_id=%s | error=%s", session_id, exc)
            return []
