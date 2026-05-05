import hashlib
import time
from typing import List, Dict

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:

    def __init__(self, bm25, vector_store, embedder, clip_text_embedder=None):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.clip_text_embedder = clip_text_embedder

        self.w_bm25 = settings.HYBRID_WEIGHT_BM25
        self.w_vector = settings.HYBRID_WEIGHT_VECTOR
        self.w_vision = settings.HYBRID_WEIGHT_VISION

        self.candidate_multiplier = min(settings.HYBRID_CANDIDATES_MULTIPLIER, 3)

        # embedding cache
        self._cache: Dict[str, List[float]] = {}

    #  HASH 
    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:200]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode()).hexdigest()

    #  QUERY 
    def _normalize(self, q: str) -> str:
        return " ".join(q.strip().split())

    def _is_vision(self, q: str) -> bool:
        q = q.lower()
        return any(k in q for k in ["image", "photo", "diagram", "visual", "figure"])

    #  EMBEDDING 
    def _embed_query(self, q: str):

        if q in self._cache:
            return self._cache[q]

        vec = self.embedder.embed_query(q)
        self._cache[q] = vec
        return vec

    #  SCORE NORMALIZATION 
    def _normalize_scores(self, results: List[Dict]):

        if not results:
            return results

        max_score = max(r.get("score", 0.0) for r in results) or 1e-6

        for r in results:
            r["score"] = r.get("score", 0.0) / max_score

        return results

    #  SEARCH 
    def search(self, query: str, session_id: str, top_k: int = None) -> List[Dict]:

        if not query or not session_id:
            return []

        start = time.time()

        query = self._normalize(query)
        top_k = top_k or settings.DEFAULT_TOP_K
        candidate_k = min(top_k * self.candidate_multiplier, 20)

        try:
            #  BM25 
            bm25_res = self.bm25.search(query, session_id, candidate_k)
            bm25_res = self._normalize_scores(bm25_res)

            #  VECTOR 
            vec_res = []
            try:
                q_vec = self._embed_query(query)
                vec_res = self.vector_store.search_text(q_vec, candidate_k, session_id)
                vec_res = self._normalize_scores(vec_res)
            except Exception as e:
                logger.warning(event="vector_failed", error=str(e))

            #  VISION 
            vis_res = []
            if self._is_vision(query):
                try:
                    clip = self.clip_text_embedder or model_loader.get_clip_text_embedder()
                    v_vec = clip.embed_single(query)

                    vis_res = self.vector_store.search_vision(v_vec, candidate_k, session_id)
                    vis_res = self._normalize_scores(vis_res)
                except Exception as e:
                    logger.warning(event="vision_failed", error=str(e))

            #  FUSION 
            combined = {}

            def add(results, weight):
                for r in results:
                    text = r.get("text")
                    meta = r.get("metadata", {})

                    if not text:
                        continue

                    h = self._hash(text, meta)
                    score = r.get("score", 0.0) * weight

                    if h not in combined:
                        combined[h] = {
                            "text": text,
                            "metadata": meta,
                            "score": score,
                        }
                    else:
                        combined[h]["score"] += score

            add(bm25_res, self.w_bm25)
            add(vec_res, self.w_vector)

            if vis_res:
                add(vis_res, self.w_vision)

            results = list(combined.values())

            if not results:
                return []

            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

            logger.info(
                event="hybrid_success",
                results=len(results),
                latency=round(time.time() - start, 3)
            )

            return results

        except Exception as e:
            logger.error(event="hybrid_failed", error=str(e))
            return []