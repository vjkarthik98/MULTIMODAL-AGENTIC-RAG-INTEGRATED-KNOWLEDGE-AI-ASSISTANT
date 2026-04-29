import time
from typing import List, Dict

from app.core.config import settings
from app.utils.logger import get_logger


from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker

from app.core.model_loader import model_loader
from app.pipeline.ingestion_pipeline import pipeline
from app.core.infra_registry import infra



logger = get_logger(__name__)


class Retriever:

    # INIT
    def __init__(self):
        self.vector_store = infra.get_vector_store()
        self.bm25 = BM25Retriever()
        self.reranker = model_loader.get_reranker()

        self.embedder = model_loader.get_embedder()

    # MAIN RETRIEVAL
    def retrieval(self, query: str, session_id: str = "default", top_k: int = 5) -> List[Dict]:

        start = time.time()

        logger.info("[HybridRetriever][START]")

        try:
            # STEP 1: ENSURE BM25 READY
            self._ensure_bm25()

            # STEP 2: QUERY EXPANSION (FIXED - NOW USED)
            queries = self._expand_query(query)

            vector_results = []
            bm25_results = []

            for q in queries:
                vector_results.extend(self._vector_search(q, session_id))
                bm25_results.extend(self._bm25_search(q))

            # STEP 3: MERGE RESULTS
            merged = self._merge_results(vector_results, bm25_results)

            if not merged:
                logger.warning("[HybridRetriever] no candidates")
                return []

            # STEP 4: RERANK
            reranked = self._rerank(query, merged)

            # STEP 5: FINAL FILTER
            final = self._final_filter(reranked, top_k)

            logger.info(
                "[HybridRetriever][SUCCESS] results=%s latency=%ss",
                len(final),
                round(time.time() - start, 2)
            )

            return final

        except Exception as e:
            logger.error("[HybridRetriever][FAIL] %s", str(e))
            return []

    # QUERY EXPANSION
    def _expand_query(self, query: str) -> List[str]:

        try:
            prompt = f"""
Generate 3 alternative search queries for the following user query.
Keep them short and semantically similar.

Query: {query}
"""

            llm = model_loader.get_llm()
            response = llm.generate(prompt)

            variations = [
                q.strip("- ").strip()
                for q in response.split("\n")
                if q.strip()
            ]

            expanded = [query] + variations[:3]

            logger.info("[Retriever] expanded queries=%s", len(expanded))

            return expanded

        except Exception as e:
            logger.warning("[Retriever] query expansion failed | %s", str(e))
            return [query]

    # BM25 SYNC
    def _ensure_bm25(self):

        try:
            if not getattr(self.bm25, "documents", None):
                logger.warning("[HybridRetriever] BM25 empty → syncing from pipeline")

                if hasattr(pipeline, "bm25") and getattr(pipeline.bm25, "documents", None):
                    self.bm25 = pipeline.bm25
                    logger.info("[HybridRetriever] BM25 synced from pipeline")
                else:
                    logger.warning("[HybridRetriever] BM25 still empty")

        except Exception as e:
            logger.error("[HybridRetriever] BM25 sync failed | %s", str(e))

    # VECTOR SEARCH
    def _vector_search(self, query: str, session_id: str):

        try:
            vector = self.embedder.embed_query(query)

            results = self.vector_store.search_text(
                query_vector=vector,
                session_id=session_id,
                limit=settings.RAG_TOP_K * 4
            )

            return results

        except Exception as e:
            logger.error("[HybridRetriever] vector search failed | %s", str(e))
            return []

    # BM25 SEARCH
    def _bm25_search(self, query: str):

        try:
            if not getattr(self.bm25, "documents", None):
                return []

            return self.bm25.search(query, top_k=settings.RAG_TOP_K * 4)

        except Exception as e:
            logger.error("[HybridRetriever] bm25 failed | %s", str(e))
            return []

    # MERGE RESULTS
    def _merge_results(self, vector_results: List[Dict], bm25_results: List[Dict]):

        merged = {}

        for r in vector_results:
            key = self._get_key(r)
            merged[key] = r

        for r in bm25_results:
            key = self._get_key(r)

            if key in merged:
                merged[key]["score"] += 0.2
            else:
                merged[key] = r

        return list(merged.values())

    # KEY GENERATOR
    def _get_key(self, r: Dict):
        return (
            str(r.get("metadata", {}).get("doc_id")),
            str(r.get("metadata", {}).get("chunk_id")),
            r.get("text", "")[:100]
        )

    # RERANK
    def _rerank(self, query: str, results: List[Dict]):

        try:
            return self.reranker.rerank(query, results)
        except Exception as e:
            logger.error("[Reranker] failed | %s", str(e))
            return results

    # FINAL FILTER
    def _final_filter(self, results: List[Dict], top_k: int):

        if not results:
            return []

        filtered = [
            r for r in results
            if r.get("text") and r.get("score", 0.0) > 0.01
        ]

        if len(filtered) < top_k:
            return results[:top_k]

        return filtered[:top_k]