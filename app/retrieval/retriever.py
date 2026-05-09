import hashlib
import math
import time
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.infra_registry import infra
from app.core.model_loader import model_loader
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:

    def __init__(self) -> None:
        self.vector_store   = infra.get_vector_store()
        self.bm25           = BM25Retriever()
        self.reranker       = Reranker()
        self.embedder       = model_loader.get_embedder()
        self.max_candidates = min(
            settings.RAG_TOP_K * settings.HYBRID_CANDIDATES_MULTIPLIER,
            50,
        )

    # HASH

    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:150]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # NORMALIZE

    def _normalize(self, q: str) -> str:
        return " ".join(q.strip().split())

    # SCORE VALID

    def _valid_score(self, score: float) -> bool:
        return not (math.isnan(score) or math.isinf(score))

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

    # QUERY EXPANSION

    def _expand_query(self, query: str, session_id: str) -> List[str]:

        if not settings.AGENT_QUERY_EXPANSION_ENABLED:
            return [query]

        try:
            llm    = model_loader.get_llm()
            prompt = (
                f"Generate 2 alternative search queries.\n"
                f"Keep meaning same. No explanation.\n\n"
                f"Query: {query[:settings.MAX_PROMPT_CHARS // 4]}"
            )

            t_start  = time.time()
            response = llm.generate(prompt, max_tokens=80, temperature=0.0)
            elapsed  = time.time() - t_start

            if elapsed > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    event="query_expansion_timeout",
                    elapsed=round(elapsed, 2),
                    session_id=session_id,
                )
                return [query]

            variations = [
                v.strip("- ").strip()
                for v in response.split("\n")
                if v.strip() and len(v.strip()) > 5
            ]

            # DEDUP EXPANDED QUERIES
            seen:    set        = {query.lower()}
            unique:  List[str]  = [query]

            for v in variations[:2]:
                if v.lower() not in seen:
                    seen.add(v.lower())
                    unique.append(v)

            return unique

        except Exception as e:
            logger.warning(
                event="query_expand_failed",
                error=str(e),
                session_id=session_id,
            )
            return [query]

    # BM25 CHECK

    def _ensure_bm25(self, session_id: str) -> None:
        if not getattr(self.bm25, "documents", None):
            logger.warning(event="bm25_index_empty", session_id=session_id)

    # VECTOR SEARCH

    def _vector_search(self, q: str, session_id: str) -> List[Dict]:
        try:
            vec = self.embedder.embed_query(q, session_id=session_id)
            return self.vector_store.search_text(
                query_vector=vec,
                session_id=session_id,
                limit=self.max_candidates,
            )
        except Exception as e:
            logger.error(
                event="vector_search_failed",
                error=str(e),
                session_id=session_id,
            )
            return []

    # BM25 SEARCH

    def _bm25_search(self, q: str, session_id: str) -> List[Dict]:
        try:
            return self.bm25.search(
                q,
                session_id=session_id,
                top_k=self.max_candidates,
            )
        except Exception as e:
            logger.error(
                event="bm25_search_failed",
                error=str(e),
                session_id=session_id,
            )
            return []

    # MERGE

    def _merge(self, vector_res: List[Dict], bm25_res: List[Dict]) -> List[Dict]:
        combined: Dict = {}

        def _add(results: List[Dict], weight: float) -> None:
            for r in results:
                text  = r.get("text")
                meta  = r.get("metadata", {})
                score = float(r.get("score", 0.0)) * weight

                if not text:
                    continue

                if not self._valid_score(score):
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

        _add(vector_res, settings.HYBRID_WEIGHT_VECTOR)
        _add(bm25_res,   settings.HYBRID_WEIGHT_BM25)

        return list(combined.values())

    # FILTER

    def _filter(self, results: List[Dict], top_k: int) -> List[Dict]:
        if not results:
            return []

        results = [
            r for r in results
            if r.get("text")
            and r.get("score", 0.0) > settings.HYBRID_MIN_SCORE
        ]

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    # MAIN

    def retrieval(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 5,
    ) -> List[Dict]:

        if not query:
            return []

        start = time.time()

        try:
            query = self._normalize(query)

            self._ensure_bm25(session_id)

            queries = self._expand_query(query, session_id)

            vector_res: List[Dict] = []
            bm25_res:   List[Dict] = []

            for q in queries:
                v_results = self._vector_search(q, session_id)
                b_results = self._bm25_search(q, session_id)

                vector_res.extend(v_results)
                bm25_res.extend(b_results)

            # NORMALIZE BEFORE MERGE
            vector_res = self._normalize_scores(vector_res)
            bm25_res   = self._normalize_scores(bm25_res)

            merged = self._merge(vector_res, bm25_res)

            if not merged:
                logger.warning(
                    event="retrieval_no_results",
                    queries=len(queries),
                    session_id=session_id,
                )
                return []

            reranked = self.reranker.rerank(
                query,
                merged,
                top_k=top_k,
                session_id=session_id,
            )

            final = self._filter(reranked, top_k)

            logger.info(
                event="retrieval_success",
                results=len(final),
                queries=len(queries),
                vector_total=len(vector_res),
                bm25_total=len(bm25_res),
                merged=len(merged),
                latency=round(time.time() - start, 2),
                session_id=session_id,
            )

            return final

        except Exception as e:
            logger.error(
                event="retrieval_failed",
                error=str(e),
                session_id=session_id,
            )
            return []