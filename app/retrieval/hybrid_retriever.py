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

        self.candidate_multiplier = min(settings.HYBRID_CANDIDATES_MULTIPLIER, 3)

        self.w_bm25 = settings.HYBRID_WEIGHT_BM25
        self.w_vector = settings.HYBRID_WEIGHT_VECTOR
        self.w_vision = settings.HYBRID_WEIGHT_VISION

        # Embedding cache
        self._query_embedding_cache = {}

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    def _make_key(self, text: str, metadata: Dict) -> str:
        base = "|".join([
            text[:150],
            str(metadata.get("doc_id")),
            str(metadata.get("chunk_id")),
        ])
        return hashlib.md5(base.encode()).hexdigest()

    def _is_vision_query(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ["image", "photo", "diagram", "visual"])

    # Cached Embedding
    def _get_query_embedding(self, query: str):
        if query in self._query_embedding_cache:
            return self._query_embedding_cache[query]

        vec = self.embedder.embed_query(query)
        self._query_embedding_cache[query] = vec
        return vec

    def search(self, query: str, session_id: str, top_k: int = None) -> List[Dict]:

        if not query or not session_id:
            return []

        start = time.time()
        query = self._normalize_query(query)

        top_k = top_k or settings.DEFAULT_TOP_K
        candidate_k = min(top_k * self.candidate_multiplier, 15)

        try:
            # BM25
            bm25_results = self.bm25.search(query, session_id, candidate_k)

            # VECTOR (cached embedding)
            text_vector_results = []
            try:
                q_vec = self._get_query_embedding(query)
                text_vector_results = self.vector_store.search_text(
                    q_vec, candidate_k, session_id
                )
            except Exception:
                pass

            # VISION 
            vision_results = []
            if self._is_vision_query(query):
                try:
                    clip = self.clip_text_embedder or model_loader.get_clip_text_embedder()
                    v_vec = clip.embed_single(query)

                    vision_results = self.vector_store.search_vision(
                        v_vec, candidate_k, session_id
                    )
                except Exception:
                    pass

            combined = {}

            def _add_results(results, weight):
                for r in results:
                    text = r.get("text")
                    metadata = r.get("metadata", {})

                    if not text:
                        continue

                    key = self._make_key(text, metadata)
                    score = r.get("score", 0.0) * weight

                    if key not in combined:
                        combined[key] = {
                            "text": text,
                            "metadata": metadata,
                            "score": score
                        }
                    else:
                        combined[key]["score"] += score

            _add_results(bm25_results, self.w_bm25)
            _add_results(text_vector_results, self.w_vector)

            if vision_results:
                _add_results(vision_results, self.w_vision)

            results = list(combined.values())

            if not results:
                return []

            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

            logger.info(
                "[HybridRetriever] results=%s latency=%.2fs",
                len(results),
                time.time() - start
            )

            return results

        except Exception as e:
            logger.error("[HybridRetriever][FAILED] %s", str(e))
            return []