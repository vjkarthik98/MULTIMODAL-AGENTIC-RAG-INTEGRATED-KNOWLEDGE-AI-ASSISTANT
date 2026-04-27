import hashlib
from typing import List, Dict

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger


logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, bm25_retriever, vector_store, embedder, clip_text_embedder=None):
        self.bm25 = bm25_retriever
        self.vector_store = vector_store
        self.embedder = embedder
        self.clip_text_embedder = clip_text_embedder

        self.candidate_multiplier = settings.HYBRID_CANDIDATES_MULTIPLIER

        # Explicit weights 
        self.w_bm25 = getattr(settings, "HYBRID_WEIGHT_BM25", 0.5)
        self.w_vector = getattr(settings, "HYBRID_WEIGHT_VECTOR", 0.8)
        self.w_vision = getattr(settings, "HYBRID_WEIGHT_VISION", 0.9)

    def _make_key(self, text: str, metadata: Dict) -> str:
        base = "|".join([
            text[:200] if text else "",
            str(metadata.get("doc_id", "")),
            str(metadata.get("chunk_id", "")),
            str(metadata.get("source", "")),
            str(metadata.get("embedding_space", "")),
        ])
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    def _modality_boost(self, metadata: Dict) -> float:
        modality = metadata.get("modality")
        content_type = metadata.get("content_type")

        if modality == "table":
            return 1.1
        if modality == "image":
            return 1.15
        if modality == "audio":
            return 1.1
        if modality == "video":
            if content_type == "video_speech":
                return 1.2
            if content_type == "video_frame":
                return 1.1
        return 1.0

    def _normalize(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return []

        max_score = max(r.get("score", 0.0) for r in results) or 1.0

        normalized = []
        for r in results:
            r = dict(r)
            r["norm_score"] = r.get("score", 0.0) / (max_score + 1e-6)
            normalized.append(r)

        return normalized
    
    
    def search(self, query: str, session_id: str, top_k: int = None) -> List[Dict]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not session_id:
            raise ValueError("session_id required")

        top_k = top_k or settings.DEFAULT_TOP_K
        candidate_k = top_k * self.candidate_multiplier

        try:
            logger.info("[HybridRetriever][START] session_id=%s", session_id)

            # BM25 
            bm25_raw = self.bm25.search(query, session_id=session_id, top_k=candidate_k)
            bm25_results = self._normalize(bm25_raw)

            # VECTOR 
            text_vector_results = []
            try:
                query_vector = self.embedder.embed_query(query)
                text_vector_results = self._normalize(
                    self.vector_store.search_text(
                        query_vector=query_vector,
                        limit=candidate_k,
                        session_id=session_id,
                    )
                )
            except Exception as e:
                logger.warning("[HybridRetriever][TEXT_VECTOR_FAIL] %s", str(e))

            # VISION 
            vision_results = []
            try:
                clip_embedder = self.clip_text_embedder or model_loader.get_clip_text_embedder()
                clip_query_vector = clip_embedder.embed_single(query)

                vision_results = self._normalize(
                    self.vector_store.search_vision(
                        query_vector=clip_query_vector,
                        limit=candidate_k,
                        session_id=session_id,
                    )
                )
            except Exception as e:
                logger.warning("[HybridRetriever][VISION_FAIL] %s", str(e))

            logger.info(
                "[HybridRetriever] sources | bm25=%s vector=%s vision=%s",
                len(bm25_results),
                len(text_vector_results),
                len(vision_results)
            )

            # FIX 5: VISION PRIORITY MODE
            if vision_results and not text_vector_results and not bm25_results:
                logger.info("[HybridRetriever] using vision-only results")

                results = [
                    {
                        "text": r.get("text", ""),
                        "metadata": r.get("metadata", {}),
                        "score": r.get("norm_score", 0.0)
                    }
                    for r in vision_results
                ]

                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]

            # NORMAL HYBRID 
            combined = {}

            def merge(results, key_name, weight):
                for r in results:
                    metadata = r.get("metadata", {})
                    key = self._make_key(r.get("text", ""), metadata)
                    boost = self._modality_boost(metadata)

                    score = r.get("norm_score", 0.0) * weight * boost

                    if key not in combined:
                        combined[key] = {
                            "text": r.get("text", ""),
                            "metadata": metadata,
                            "bm25_score": 0.0,
                            "vector_score": 0.0,
                            "vision_score": 0.0,
                            "score": 0.0,
                        }

                    combined[key][key_name] = score
                    combined[key]["score"] += score

            merge(bm25_results, "bm25_score", self.w_bm25)
            merge(text_vector_results, "vector_score", self.w_vector)
            merge(vision_results, "vision_score", self.w_vision)

            if not combined:
                logger.warning("[HybridRetriever] no results from any source")
                return []

            results = list(combined.values())

            results.sort(key=lambda x: x["score"], reverse=True)

            logger.info(
                "[HybridRetriever][SUCCESS] session_id=%s | final=%s",
                session_id,
                len(results),
            )

            return results[:top_k]

        except Exception as e:
            logger.error("[HybridRetriever][FAILED] session_id=%s | error=%s", session_id, e)
            return []

        