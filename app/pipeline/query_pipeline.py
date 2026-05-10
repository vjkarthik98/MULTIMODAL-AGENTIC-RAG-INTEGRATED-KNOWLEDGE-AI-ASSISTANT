import hashlib
import time
import unicodedata
from typing import Any, Dict, Optional

from app.agents.agent_controller import AgentController
from app.core.config import settings
from app.core.infra_registry import infra
from app.core.model_loader import model_loader
from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.result_fusion import ResultFusion
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.utils.logger import get_logger

logger = get_logger(__name__)


# GLOBALS

_agent        = AgentController()
_bm25         = infra.get_bm25()
_vector_store = infra.get_vector_store()
_memory       = infra.get_memory()

_fusion   = ResultFusion()
_reranker = Reranker()

_hybrid:     Optional[HybridRetriever] = None
_reasoning:  Optional[ReasoningEngine] = None
_decomposer: Optional[QueryDecomposer] = None


# CACHE KEY

def _cache_key(session_id: str, query: str) -> str:
    base = f"{session_id}:{query.strip().lower()}"
    return "qresp:" + hashlib.sha256(base.encode("utf-8")).hexdigest()


def _cache_get(session_id: str, query: str) -> Optional[Dict]:
    if not _memory:
        return None
    try:
        return _memory.cache_get(_cache_key(session_id, query))
    except Exception:
        return None


def _cache_set(session_id: str, query: str, data: Dict) -> None:
    if not _memory:
        return
    try:
        _memory.cache_set(
            _cache_key(session_id, query),
            data,
            ttl=settings.REDIS_QUERY_CACHE_TTL,
        )
    except Exception:
        pass


# SINGLETON LAZY INIT

def _get_hybrid(embedder) -> HybridRetriever:
    global _hybrid
    if not _hybrid:
        _hybrid = HybridRetriever(_bm25, _vector_store, embedder)
    return _hybrid


def _get_reasoning(llm):
    global _reasoning, _decomposer
    if not _reasoning:
        _reasoning = ReasoningEngine(llm)
    if not _decomposer:
        _decomposer = QueryDecomposer(llm)
    return _reasoning, _decomposer


# NORMALIZE

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    return " ".join(query.strip().split())


# MAIN

def query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:

    start = time.time()

    if not query or not query.strip():
        return {"answer": "Query cannot be empty."}

    query = _normalize(query)[:settings.MAX_PROMPT_CHARS]

    # CACHE HIT
    cached = _cache_get(session_id, query)
    if cached:
        cached["latency"] = round(time.time() - start, 3)
        cached["cache_hit"] = True
        logger.debug(event="query_cache_hit", session_id=session_id)
        return cached

    try:
        llm      = model_loader.get_llm()
        embedder = model_loader.get_embedder()

        reasoning, decomposer = _get_reasoning(llm)
        hybrid                = _get_hybrid(embedder)

        # AGENT DECISION
        t_agent = time.time()
        agent   = _agent.handle(query, session_id)
        agent_latency = round(time.time() - t_agent, 3)

        decision = agent.get("decision", "")

        # SHORT-CIRCUIT FOR NON-RAG DECISIONS
        if decision in {"direct", "search", "memory"}:
            resp = {
                "answer":        agent.get("response", "No answer generated."),
                "confidence":    agent.get("confidence", 0.5),
                "decision":      decision,
                "agent_latency": agent_latency,
                "latency":       round(time.time() - start, 3),
                "metadata":      {"source": "agent"},
            }
            _cache_set(session_id, query, resp)
            return resp

        # MEMORY CONTEXT
        memory_context = ""
        if _memory and len(query) > 20:
            try:
                history = _memory.get_history(session_id)
                if history:
                    filtered       = filter_relevant_history(
                        query, history, embedder, session_id=session_id
                    )
                    memory_context = build_memory_context(
                        "", filtered, session_id=session_id
                    )
            except Exception as e:
                logger.warning(
                    event="memory_fetch_failed",
                    error=str(e),
                    session_id=session_id,
                )

        # QUERY DECOMPOSITION
        queries = [query]
        if len(query.split()) > settings.DECOMPOSITION_MIN_WORDS:
            try:
                sub = decomposer.decompose(query, session_id=session_id)
                if sub:
                    queries = sub[:settings.DECOMPOSITION_MAX_SUBQUERIES]
            except Exception as e:
                logger.warning(
                    event="decompose_failed",
                    error=str(e),
                    session_id=session_id,
                )

        # HYBRID RETRIEVAL
        t_ret     = time.time()
        retrieved = []

        for q in queries:
            try:
                results = hybrid.search(q, session_id=session_id, top_k=settings.DEFAULT_TOP_K)
                retrieved.extend(results)
            except Exception as e:
                logger.warning(
                    event="hybrid_search_failed",
                    query=q[:50],
                    error=str(e),
                    session_id=session_id,
                )

        retrieval_latency = round(time.time() - t_ret, 3)

        if not retrieved:
            logger.warning(
                event="query_pipeline_no_results",
                queries=len(queries),
                session_id=session_id,
            )
            return {
                "answer":  "No relevant information found.",
                "latency": round(time.time() - start, 3),
            }

        # FUSION
        fused = _fusion.fuse(retrieved, session_id=session_id)

        if not fused:
            return {
                "answer":  "No relevant information found.",
                "latency": round(time.time() - start, 3),
            }

        # RERANK
        reranked = _reranker.rerank(
            query, fused, top_k=settings.RERANK_TOP_K, session_id=session_id
        )

        if not reranked:
            return {
                "answer":  "No relevant information found.",
                "latency": round(time.time() - start, 3),
            }

        final_docs = reranked[:settings.RAG_TOP_K]

        # REASONING
        t_reason = time.time()
        output   = reasoning.generate_answer(
            query=query,
            retrieved_docs=final_docs,
            memory_context=memory_context,
            session_id=session_id,
        )
        reasoning_latency = round(time.time() - t_reason, 3)

        answer = (output.get("answer", "") or "").strip()
        if not answer:
            answer = "No answer generated."

        total_latency = round(time.time() - start, 3)

        response = {
            "answer":       answer,
            "confidence":   output.get("confidence", 0.5),
            "sources_used": output.get("sources_used", 0),
            "decision":     decision,
            "latency":      total_latency,
            "metadata": {
                "agent_latency":     agent_latency,
                "retrieval_latency": retrieval_latency,
                "reasoning_latency": reasoning_latency,
                "docs_used":         len(final_docs),
                "queries_expanded":  len(queries),
            },
        }

        _cache_set(session_id, query, response)

        logger.info(
            event="query_pipeline_success",
            decision=decision,
            docs_used=len(final_docs),
            queries=len(queries),
            latency=total_latency,
            session_id=session_id,
        )

        return response

    except Exception as e:
        logger.error(
            event="query_pipeline_failed",
            error=str(e),
            session_id=session_id,
        )
        return {
            "answer":  "Something went wrong. Please try again.",
            "latency": round(time.time() - start, 3),
        }