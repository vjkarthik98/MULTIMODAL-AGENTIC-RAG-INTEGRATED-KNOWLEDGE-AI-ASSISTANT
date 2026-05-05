import time
import hashlib
from typing import List, Dict

from app.core.config import settings
from app.utils.logger import get_logger

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.core.model_loader import model_loader
from app.core.infra_registry import infra

logger = get_logger(__name__)


class Retriever:

    def __init__(self):
        self.vector_store = infra.get_vector_store()
        self.bm25 = BM25Retriever()
        self.reranker = Reranker()  # FIXED (previously wrong object)

        self.embedder = model_loader.get_embedder()

        self.max_candidates = settings.RAG_TOP_K * 4

    #  HASH 
    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:150]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode()).hexdigest()

    #  NORMALIZE 
    def _normalize(self, q: str) -> str:
        return " ".join(q.strip().split())

    #  QUERY EXPANSION 
    def _expand_query(self, query: str) -> List[str]:

        if not settings.AGENT_QUERY_EXPANSION_ENABLED:
            return [query]

        try:
            llm = model_loader.get_llm()

            prompt = f"""
Generate 2 alternative search queries.
Keep meaning same. No explanation.

Query: {query}
"""

            response = llm.generate(prompt)

            variations = [
                v.strip("- ").strip()
                for v in response.split("\n")
                if v.strip()
            ]

            return [query] + variations[:2]

        except Exception as e:
            logger.warning(event="query_expand_failed", error=str(e))
            return [query]

    #  BM25 SYNC 
    def _ensure_bm25(self):
        if not getattr(self.bm25, "documents", None):
            logger.warning(event="bm25_empty")

    #  VECTOR 
    def _vector_search(self, q: str, session_id: str):

        try:
            vec = self.embedder.embed_query(q)

            return self.vector_store.search_text(
                query_vector=vec,
                session_id=session_id,
                limit=self.max_candidates
            )

        except Exception as e:
            logger.error(event="vector_failed", error=str(e))
            return []

    #  BM25 
    def _bm25_search(self, q: str, session_id: str):

        try:
            return self.bm25.search(
                q,
                session_id=session_id,
                top_k=self.max_candidates
            )
        except Exception as e:
            logger.error(event="bm25_failed", error=str(e))
            return []

    #  MERGE 
    def _merge(self, vector_res, bm25_res):

        combined = {}

        def add(results, weight):
            for r in results:
                text = r.get("text")
                meta = r.get("metadata", {})

                if not text:
                    continue

                h = self._hash(text, meta)
                score = float(r.get("score", 0.0)) * weight

                if h not in combined:
                    combined[h] = {
                        "text": text,
                        "metadata": meta,
                        "score": score,
                    }
                else:
                    combined[h]["score"] += score

        add(vector_res, settings.HYBRID_WEIGHT_VECTOR)
        add(bm25_res, settings.HYBRID_WEIGHT_BM25)

        return list(combined.values())

    #  FINAL FILTER 
    def _filter(self, results: List[Dict], top_k: int):

        if not results:
            return []

        results = [
            r for r in results
            if r.get("text") and r.get("score", 0.0) > 0.01
        ]

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    #  MAIN 
    def retrieval(self, query: str, session_id: str = "default", top_k: int = 5):

        if not query:
            return []

        start = time.time()

        try:
            query = self._normalize(query)

            self._ensure_bm25()

            queries = self._expand_query(query)

            vector_res = []
            bm25_res = []

            for q in queries:
                vector_res.extend(self._vector_search(q, session_id))
                bm25_res.extend(self._bm25_search(q, session_id))

            merged = self._merge(vector_res, bm25_res)

            if not merged:
                return []

            reranked = self.reranker.rerank(query, merged)

            final = self._filter(reranked, top_k)

            logger.info(
                event="retrieval_success",
                results=len(final),
                latency=round(time.time() - start, 2)
            )

            return final

        except Exception as e:
            logger.error(event="retrieval_failed", error=str(e))
            return []