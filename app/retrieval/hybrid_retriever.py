import hashlib
import time
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

        self.w_bm25 = settings.HYBRID_WEIGHT_BM25
        self.w_vector = settings.HYBRID_WEIGHT_VECTOR
        self.w_vision = settings.HYBRID_WEIGHT_VISION

    #  NORMALIZE QUERY 
    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  UNIQUE KEY 
    def _make_key(self, text: str, metadata: Dict) -> str:
        base = "|".join([
            text[:200],
            str(metadata.get("doc_id")),
            str(metadata.get("chunk_id")),
            str(metadata.get("source")),
            str(metadata.get("embedding_space")),
        ])
        return hashlib.md5(base.encode()).hexdigest()

    #  MODALITY BOOST (QUERY-AWARE) 
    def _modality_boost(self, metadata: Dict, query: str) -> float:
        modality = metadata.get("modality")

        if "image" in query.lower():
            if modality == "image":
                return 1.2

        if "video" in query.lower():
            if modality == "video":
                return 1.2

        if modality == "table":
            return 1.1

        return 1.0

    #  GLOBAL NORMALIZATION 
    def _global_normalize(self, results: List[Dict]):

        if not results:
            return results

        max_score = max(r.get("score", 0.0) for r in results) or 1.0

        for r in results:
            r["norm_score"] = r.get("score", 0.0) / (max_score + 1e-6)

        return results

    #  SEARCH 
    def search(self, query: str, session_id: str, top_k: int = None) -> List[Dict]:

        start = time.time()

        if not query or not query.strip():
            return []

        if not session_id:
            raise ValueError("session_id required")

        query = self._normalize_query(query)

        top_k = top_k or settings.DEFAULT_TOP_K
        candidate_k = min(
            top_k * self.candidate_multiplier,
            settings.MAX_CHUNKS
        )

        try:
            logger.info("[HybridRetriever][START]")

            #  BM25 
            t1 = time.time()
            bm25_results = self.bm25.search(query, session_id, candidate_k)
            logger.info("[HybridRetriever] bm25 latency=%.2fs", time.time() - t1)

            #  VECTOR 
            text_vector_results = []
            try:
                t2 = time.time()
                q_vec = self.embedder.embed_query(query)

                text_vector_results = self.vector_store.search_text(
                    q_vec, candidate_k, session_id
                )

                logger.info("[HybridRetriever] vector latency=%.2fs", time.time() - t2)

            except Exception as e:
                logger.warning("[HybridRetriever] vector fail | %s", str(e))

            #  VISION 
            vision_results = []
            try:
                t3 = time.time()
                clip = self.clip_text_embedder or model_loader.get_clip_text_embedder()
                v_vec = clip.embed_single(query)

                vision_results = self.vector_store.search_vision(
                    v_vec, candidate_k, session_id
                )

                logger.info("[HybridRetriever] vision latency=%.2fs", time.time() - t3)

            except Exception as e:
                logger.warning("[HybridRetriever] vision fail | %s", str(e))

            # COMBINE ALL FOR GLOBAL NORMALIZATION
            all_results = bm25_results + text_vector_results + vision_results

            if not all_results:
                return []

            all_results = self._global_normalize(all_results)

            combined = {}

            for r in all_results:

                text = r.get("text", "")
                metadata = r.get("metadata", {})

                if not text:
                    continue

                key = self._make_key(text, metadata)

                boost = self._modality_boost(metadata, query)

                score = r.get("norm_score", 0.0) * boost

                if key not in combined:
                    combined[key] = {
                        "text": text,
                        "metadata": metadata,
                        "score": 0.0
                    }

                # WEIGHT BASED ON SOURCE
                if r in bm25_results:
                    score *= self.w_bm25
                elif r in text_vector_results:
                    score *= self.w_vector
                else:
                    score *= self.w_vision

                combined[key]["score"] += score

            results = list(combined.values())

            # FILTER LOW QUALITY
            results = [r for r in results if r["score"] > 0.05]

            results.sort(key=lambda x: x["score"], reverse=True)

            latency = round(time.time() - start, 2)

            logger.info(
                "[HybridRetriever][SUCCESS] results=%s latency=%ss",
                len(results),
                latency
            )

            return results[:top_k]

        except Exception as e:
            logger.error("[HybridRetriever][FAILED] %s", str(e))
            return []