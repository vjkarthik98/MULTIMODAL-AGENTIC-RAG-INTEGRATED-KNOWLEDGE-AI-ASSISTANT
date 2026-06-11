import asyncio
import hashlib
import math
import time
from typing import Any, Dict, List, Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.infra_registry import infra
from app.core.model_loader import model_loader
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_retrieval_duration = Histogram(
    "retrieval_latency_seconds",
    "Retrieval latency by retriever type",
    ["retriever_type"],
)
_retrieval_errors = Counter(
    "retrieval_errors_total",
    "Retrieval errors by retriever type and error type",
    ["retriever_type", "error_type"],
)
_retrieval_results = Histogram(
    "retrieval_results_count",
    "Number of results returned per retrieval",
    ["retriever_type"],
)

# SEMAPHORE — CAP CONCURRENT RETRIEVAL WORKERS
_semaphore = asyncio.Semaphore(5)


# SHA-256 HASH

def _hash(text: str, meta: Dict) -> str:
    base = f"{text[:150]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# NORMALIZE QUERY

def _normalize(q: str) -> str:
    import unicodedata
    q = unicodedata.normalize("NFC", str(q or ""))
    return " ".join(q.strip().split())


# SCORE VALID CHECK

def _valid_score(score: float) -> bool:
    return not (math.isnan(score) or math.isinf(score))


# SCORE NORMALIZATION — min-max to [0,1]
# Pure /max collapses the score range when all scores have a high floor
# (e.g. BM25Plus always gives non-zero scores). Min-max preserves the spread.

def _normalize_scores(results: List[Dict]) -> List[Dict]:
    if not results:
        return results
    scores = [r.get("score", 0.0) for r in results]
    min_s  = min(scores)
    max_s  = max(scores)
    spread = max_s - min_s
    if spread > 1e-8:
        for r in results:
            r["score"] = (r.get("score", 0.0) - min_s) / spread
    elif max_s > 1e-8:
        for r in results:
            r["score"] = 1.0
    return results


# MMR — MAXIMAL MARGINAL RELEVANCE FOR DIVERSITY

def _mmr(
    results: List[Dict],
    top_k: int,
    lambda_param: float = None,
) -> List[Dict]:
    """
    SELECTS DIVERSE RESULTS BY PENALISING SIMILARITY TO ALREADY-SELECTED ITEMS.
    LAMBDA=1 → PURE RELEVANCE. LAMBDA=0 → PURE DIVERSITY.
    """
    if not results:
        return []

    lam = lambda_param if lambda_param is not None else getattr(settings, "MMR_LAMBDA", 0.5)

    import numpy as np

    def _cosine(v1: List[float], v2: List[float]) -> float:
        a     = np.nan_to_num(np.array(v1, dtype=float))
        b     = np.nan_to_num(np.array(v2, dtype=float))
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _has_embedding(r: Dict) -> bool:
        emb = r.get("embedding") or (r.get("metadata", {}) or {}).get("embedding")
        return isinstance(emb, list) and len(emb) > 0

    def _get_embedding(r: Dict) -> Optional[List[float]]:
        emb = r.get("embedding")
        if not emb:
            emb = (r.get("metadata", {}) or {}).get("embedding")
        return emb

    # IF NO EMBEDDINGS AVAILABLE FALL BACK TO SCORE-BASED SELECTION
    if not any(_has_embedding(r) for r in results):
        return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]

    selected:   List[Dict] = []
    candidates: List[Dict] = list(results)

    while candidates and len(selected) < top_k:
        best_score = -float("inf")
        best_idx   = 0

        for i, cand in enumerate(candidates):
            relevance = cand.get("score", 0.0)

            if not selected:
                mmr_score = relevance
            else:
                cand_emb = _get_embedding(cand)
                if cand_emb is None:
                    mmr_score = relevance
                else:
                    max_sim = max(
                        _cosine(cand_emb, _get_embedding(s))
                        for s in selected
                        if _get_embedding(s) is not None
                    ) if selected else 0.0

                    mmr_score = lam * relevance - (1.0 - lam) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        selected.append(candidates.pop(best_idx))

    return selected


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

    # QUERY EXPANSION VIA LLM

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
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
                    "query_expansion_timeout",
                    elapsed=round(elapsed, 2),
                    session_id=session_id,
                )
                return [query]

            variations = [
                v.strip("- ").strip()
                for v in response.split("\n")
                if v.strip() and len(v.strip()) > 5
            ]

            seen:   set       = {query.lower()}
            unique: List[str] = [query]

            for v in variations[:2]:
                if v.lower() not in seen:
                    seen.add(v.lower())
                    unique.append(v)

            return unique

        except Exception as exc:
            logger.warning(
                "query_expand_failed",
                error=str(exc),
                session_id=session_id,
            )
            return [query]

    # BM25 INDEX CHECK

    def _ensure_bm25(self, session_id: str) -> None:
        if not getattr(self.bm25, "documents", None):
            logger.warning("bm25_index_empty", session_id=session_id)

    # VECTOR SEARCH WITH RETRY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    def _vector_search(self, q: str, session_id: str, user_id: Optional[str] = None) -> List[Dict]:
        try:
            vec = self.embedder.embed_query(q, session_id=session_id)
            return self.vector_store.search_text(
                query_vector=vec,
                session_id=session_id,
                limit=self.max_candidates,
                user_id=user_id,
            )
        except Exception as exc:
            logger.error(
                "vector_search_failed",
                error=str(exc),
                session_id=session_id,
            )
            return []

    # BM25 SEARCH WITH RETRY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    def _bm25_search(self, q: str, session_id: str, user_id: Optional[str] = None) -> List[Dict]:
        try:
            return self.bm25.search(
                q,
                session_id=session_id,
                top_k=self.max_candidates,
                user_id=user_id,
            )
        except Exception as exc:
            logger.error(
                "bm25_search_failed",
                error=str(exc),
                session_id=session_id,
            )
            return []

    # METADATA FILTER — BY MODALITY, DATE RANGE, LANGUAGE

    def _apply_metadata_filter(
        self,
        results: List[Dict],
        modality: Optional[str] = None,
        language: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
    ) -> List[Dict]:
        filtered = []
        for r in results:
            meta = r.get("metadata", {}) or {}

            if modality and meta.get("modality") != modality:
                continue

            if language and meta.get("language") and meta.get("language") != language:
                continue

            if date_from:
                ingestion = meta.get("ingestion_time")
                if ingestion and float(ingestion) < date_from:
                    continue

            if date_to:
                ingestion = meta.get("ingestion_time")
                if ingestion and float(ingestion) > date_to:
                    continue

            filtered.append(r)

        return filtered

    # MERGE VECTOR + BM25 RESULTS — RECIPROCAL RANK FUSION

    def _rrf_merge(
        self,
        vector_res: List[Dict],
        bm25_res: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """
        RECIPROCAL RANK FUSION — COMBINES DENSE AND SPARSE RANKINGS.
        SCORE(D) = SUM_OVER_LISTS(1 / (K + RANK(D)))
        """
        combined: Dict[str, Dict] = {}

        def _add(results: List[Dict], weight: float) -> None:
            for rank, r in enumerate(results, start=1):
                text  = r.get("text")
                meta  = r.get("metadata", {})
                score = weight * (1.0 / (k + rank))

                if not text:
                    continue

                if not _valid_score(score):
                    continue

                h = _hash(text, meta)

                if h not in combined:
                    combined[h] = {
                        "text":      text,
                        "metadata":  meta,
                        "score":     score,
                        "embedding": r.get("embedding"),
                    }
                else:
                    combined[h]["score"] += score

        _add(vector_res, settings.HYBRID_WEIGHT_VECTOR)
        _add(bm25_res,   settings.HYBRID_WEIGHT_BM25)

        return list(combined.values())

    # WEIGHTED SCORE MERGE — FALLBACK WHEN RRF NOT PREFERRED

    def _merge(
        self,
        vector_res: List[Dict],
        bm25_res: List[Dict],
    ) -> List[Dict]:
        combined: Dict[str, Dict] = {}

        def _add(results: List[Dict], weight: float) -> None:
            for r in results:
                text  = r.get("text")
                meta  = r.get("metadata", {})
                score = float(r.get("score", 0.0)) * weight

                if not text:
                    continue

                if not _valid_score(score):
                    continue

                h = _hash(text, meta)

                if h not in combined:
                    combined[h] = {
                        "text":      text,
                        "metadata":  meta,
                        "score":     score,
                        "embedding": r.get("embedding"),
                    }
                else:
                    combined[h]["score"] += score

        _add(vector_res, settings.HYBRID_WEIGHT_VECTOR)
        _add(bm25_res,   settings.HYBRID_WEIGHT_BM25)

        return list(combined.values())

    # FILTER BY SCORE THRESHOLD

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

    # MAIN SYNC RETRIEVAL

    def retrieval(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 5,
        modality: Optional[str] = None,
        language: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        use_mmr: bool = True,
        use_rrf: bool = True,
        user_id: Optional[str] = None,
    ) -> List[Dict]:

        if not query:
            return []

        start = time.time()

        with tracer.start_as_current_span("retrieval") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("session.id", session_id)
            span.set_attribute("top_k", top_k)

            try:
                query = _normalize(query)

                self._ensure_bm25(session_id)

                try:
                    queries = self._expand_query(query, session_id) or [query]
                except Exception as exc:
                    logger.warning(
                        "query_expansion_failed",
                        error=str(exc),
                        session_id=session_id,
                    )
                    queries = [query]

                vector_res: List[Dict] = []
                bm25_res:   List[Dict] = []

                for q in queries:
                    try:
                        v_results = self._vector_search(q, session_id, user_id)
                        vector_res.extend(v_results)
                    except Exception as exc:
                        logger.warning(
                            "vector_search_query_failed",
                            query=q[:50],
                            error=str(exc),
                            session_id=session_id,
                        )

                    try:
                        b_results = self._bm25_search(q, session_id, user_id)
                        bm25_res.extend(b_results)
                    except Exception as exc:
                        logger.warning(
                            "bm25_search_query_failed",
                            query=q[:50],
                            error=str(exc),
                            session_id=session_id,
                        )

                if not vector_res and not bm25_res:
                    logger.warning(
                        "retrieval_no_results",
                        queries=len(queries),
                        session_id=session_id,
                    )
                    return []

                # NORMALIZE SCORES BEFORE MERGE
                vector_res = _normalize_scores(vector_res)
                bm25_res   = _normalize_scores(bm25_res)

                # METADATA FILTER
                if modality or language or date_from or date_to:
                    vector_res = self._apply_metadata_filter(
                        vector_res, modality, language, date_from, date_to
                    )
                    bm25_res = self._apply_metadata_filter(
                        bm25_res, modality, language, date_from, date_to
                    )

                # MERGE — RRF OR WEIGHTED
                if use_rrf:
                    merged = self._rrf_merge(vector_res, bm25_res)
                else:
                    merged = self._merge(vector_res, bm25_res)

                if not merged:
                    logger.warning(
                        "retrieval_empty_after_merge",
                        session_id=session_id,
                    )
                    return []

                # RERANK
                reranked = self.reranker.rerank(
                    query,
                    merged,
                    top_k=top_k,
                    session_id=session_id,
                )

                # MMR DIVERSITY
                if use_mmr and reranked:
                    reranked = _mmr(reranked, top_k)

                # FINAL SCORE FILTER
                final = self._filter(reranked, top_k)

                latency = round(time.time() - start, 2)

                _retrieval_duration.labels(retriever_type="hybrid").observe(latency)
                _retrieval_results.labels(retriever_type="hybrid").observe(len(final))

                span.set_attribute("results.count", len(final))
                span.set_attribute("vector.count", len(vector_res))
                span.set_attribute("bm25.count", len(bm25_res))
                span.set_attribute("merged.count", len(merged))
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "retrieval_success",
                    results=len(final),
                    queries=len(queries),
                    vector_total=len(vector_res),
                    bm25_total=len(bm25_res),
                    merged=len(merged),
                    latency=latency,
                    session_id=session_id,
                )

                return final

            except Exception as exc:
                latency    = round(time.time() - start, 2)
                error_type = type(exc).__name__

                _retrieval_errors.labels(
                    retriever_type="hybrid",
                    error_type=error_type,
                ).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "retrieval_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )
                return []

    # ASYNC RETRIEVAL WRAPPER

    async def retrieval_async(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = 5,
        modality: Optional[str] = None,
        language: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        use_mmr: bool = True,
        use_rrf: bool = True,
        user_id: Optional[str] = None,
    ) -> List[Dict]:

        async with _semaphore:
            return await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.retrieval(
                    query,
                    session_id,
                    top_k,
                    modality,
                    language,
                    date_from,
                    date_to,
                    use_mmr,
                    use_rrf,
                    user_id,
                ),
            )


