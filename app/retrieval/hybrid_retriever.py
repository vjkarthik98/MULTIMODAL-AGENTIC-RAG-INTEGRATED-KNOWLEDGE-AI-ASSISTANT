import hashlib
import math
import time
from collections import OrderedDict
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)


# VISION KEYWORDS

_VISION_KEYWORDS = {
    "image", "photo", "diagram", "visual", "figure",
    "chart", "graph", "screenshot", "picture", "illustration",
}


class HybridRetriever:

    def __init__(self, bm25, vector_store, embedder, clip_text_embedder=None) -> None:
        self.bm25               = bm25
        self.vector_store       = vector_store
        self.embedder           = embedder
        self.clip_text_embedder = clip_text_embedder

        self.w_bm25   = settings.HYBRID_WEIGHT_BM25
        self.w_vector = settings.HYBRID_WEIGHT_VECTOR
        self.w_vision = settings.HYBRID_WEIGHT_VISION

        self.candidate_multiplier = settings.HYBRID_CANDIDATES_MULTIPLIER
        self.min_score            = settings.HYBRID_MIN_SCORE

        # LRU EMBEDDING CACHE
        self._cache: OrderedDict = OrderedDict()
        self._cache_max: int     = settings.LRU_CACHE_MAXSIZE

    # HASH

    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:200]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # NORMALIZE

    def _normalize_query(self, q: str) -> str:
        return " ".join(q.strip().split())

    # VISION DETECTION

    def _is_vision_query(self, q: str) -> bool:
        tokens = set(q.lower().split())
        return bool(tokens & _VISION_KEYWORDS)

    # LRU EMBEDDING CACHE

    def _embed_query(self, q: str, session_id: str = "") -> List[float]:
        if q in self._cache:
            self._cache.move_to_end(q)
            return self._cache[q]

        try:
            vec = self.embedder.embed_query(q, session_id=session_id)
        except Exception as e:
            logger.error(event="embed_query_failed", error=str(e), session_id=session_id)
            raise

        self._cache[q] = vec
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

        return vec

    # SCORE NORMALIZATION

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results

        scores    = [r.get("score", 0.0) for r in results]
        max_score = max(scores) if scores else 0.0

        if max_score <= 0.0:
            return results

        for r in results:
            r["score"] = r.get("score", 0.0) / max_score

        return results

    # SCORE VALID

    def _valid_score(self, score: float) -> bool:
        return not (math.isnan(score) or math.isinf(score))

    # FUSION

    def _fuse(self, combined: Dict, results: List[Dict], weight: float) -> None:
        for r in results:
            text  = r.get("text")
            meta  = r.get("metadata", {})
            score = r.get("score", 0.0) * weight

            if not text:
                continue

            if not self._valid_score(score):
                continue

            if score < self.min_score * weight:
                continue

            h = self._hash(text, meta)

            if h not in combined:
                combined[h] = {
                    "text":     text,
                    "metadata": meta,
                    "score":    score,
                }
            else:
                combined[h]["score"] += score

    # SEARCH

    def search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        if not query or not session_id:
            return []

        start       = time.time()
        query       = self._normalize_query(query)
        top_k       = top_k or settings.DEFAULT_TOP_K
        candidate_k = min(top_k * self.candidate_multiplier, 50)

        try:
            # BM25
            bm25_res: List[Dict] = []
            try:
                bm25_res = self.bm25.search(query, session_id, candidate_k)
                bm25_res = self._normalize_scores(bm25_res)
            except Exception as e:
                logger.warning(event="bm25_search_failed", error=str(e), session_id=session_id)

            # VECTOR
            vec_res: List[Dict] = []
            try:
                q_vec   = self._embed_query(query, session_id=session_id)
                vec_res = self.vector_store.search_text(q_vec, candidate_k, session_id)
                vec_res = self._normalize_scores(vec_res)
            except Exception as e:
                logger.warning(event="vector_search_failed", error=str(e), session_id=session_id)

            # EARLY EXIT IF BOTH EMPTY
            if not bm25_res and not vec_res:
                logger.warning(
                    event="hybrid_no_results",
                    query_len=len(query),
                    session_id=session_id,
                )
                return []

            # VISION
            vis_res: List[Dict] = []
            if self._is_vision_query(query):
                try:
                    clip  = self.clip_text_embedder or model_loader.get_clip_text_embedder()
                    v_vec = clip.embed_single(query, session_id=session_id)
                    vis_res = self.vector_store.search_vision(v_vec, candidate_k, session_id)
                    vis_res = self._normalize_scores(vis_res)
                except Exception as e:
                    logger.warning(
                        event="vision_search_failed",
                        error=str(e),
                        session_id=session_id,
                    )

            # FUSION
            combined: Dict = {}
            self._fuse(combined, bm25_res,  self.w_bm25)
            self._fuse(combined, vec_res,   self.w_vector)

            if vis_res:
                self._fuse(combined, vis_res, self.w_vision)

            if not combined:
                return []

            results = sorted(
                combined.values(),
                key=lambda x: x["score"],
                reverse=True,
            )[:top_k]

            logger.info(
                event="hybrid_search_success",
                results=len(results),
                bm25_count=len(bm25_res),
                vector_count=len(vec_res),
                vision_count=len(vis_res),
                latency=round(time.time() - start, 3),
                session_id=session_id,
            )

            return results

        except Exception as e:
            logger.error(
                event="hybrid_search_failed",
                error=str(e),
                session_id=session_id,
            )
            return []