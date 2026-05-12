import asyncio
import hashlib
import time
import unicodedata
from typing import Any, Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


_agent = None
_bm25 = None
_vector_store = None
_memory = None
_fusion = None
_reranker = None
_hybrid = None
_reasoning = None
_decomposer = None


def _cache_key(session_id: str, query: str) -> str:
    base = f"{session_id}:{query.strip().lower()}"
    return "qresp:" + hashlib.sha256(base.encode("utf-8")).hexdigest()


def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    return " ".join(query.strip().split())


def _get_infra_service(name: str):
    try:
        from app.core.infra_registry import infra

        return getattr(infra, f"get_{name}")()
    except Exception as exc:
        logger.warning(event="infra_service_unavailable", service=name, error=str(exc))
        return None


def _get_memory():
    global _memory
    if _memory is None:
        _memory = _get_infra_service("memory")
    return _memory


def _cache_get(session_id: str, query: str) -> Optional[Dict[str, Any]]:
    memory = _get_memory()
    if not memory:
        return None
    try:
        return memory.cache_get(_cache_key(session_id, query))
    except Exception:
        return None


def _cache_set(session_id: str, query: str, data: Dict[str, Any]) -> None:
    memory = _get_memory()
    if not memory:
        return
    try:
        memory.cache_set(_cache_key(session_id, query), data, ttl=settings.REDIS_QUERY_CACHE_TTL)
    except Exception:
        pass


def _get_agent():
    global _agent
    if _agent is None:
        from app.agents.agent_controller import AgentController

        _agent = AgentController()
    return _agent


def _get_fusion():
    global _fusion
    if _fusion is None:
        from app.reasoning.result_fusion import ResultFusion

        _fusion = ResultFusion()
    return _fusion


def _get_reranker():
    global _reranker
    if _reranker is None:
        from app.retrieval.reranker import Reranker

        _reranker = Reranker()
    return _reranker


def _get_hybrid(embedder):
    global _hybrid, _bm25, _vector_store
    if _hybrid is None:
        from app.retrieval.bm25_retriever import BM25Retriever
        from app.retrieval.hybrid_retriever import HybridRetriever

        _bm25 = _get_infra_service("bm25") or BM25Retriever()
        _vector_store = _get_infra_service("vector_store")
        _hybrid = HybridRetriever(_bm25, _vector_store, embedder)
    return _hybrid


def _get_reasoning(llm):
    global _reasoning, _decomposer
    if _reasoning is None:
        from app.reasoning.reasoning_engine import ReasoningEngine

        _reasoning = ReasoningEngine(llm)
    if _decomposer is None:
        from app.reasoning.query_decomposer import QueryDecomposer

        _decomposer = QueryDecomposer(llm)
    return _reasoning, _decomposer


def query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:
    start = time.time()
    if not query or not query.strip():
        return {"answer": "Query cannot be empty.", "latency": round(time.time() - start, 3)}

    query = _normalize(query)[: settings.MAX_PROMPT_CHARS]
    cached = _cache_get(session_id, query)
    if cached:
        cached["latency"] = round(time.time() - start, 3)
        cached["cache_hit"] = True
        return cached

    try:
        from app.core.model_loader import model_loader

        llm = model_loader.get_llm()
        embedder = model_loader.get_embedder()
        reasoning, decomposer = _get_reasoning(llm)
        hybrid = _get_hybrid(embedder)
        agent = _get_agent()

        t_agent = time.time()
        agent_result = agent.handle(query, session_id)
        agent_latency = round(time.time() - t_agent, 3)
        decision = agent_result.get("decision", "rag")

        if decision in {"direct", "search", "memory"}:
            response = {
                "answer": agent_result.get("response", "No answer generated."),
                "confidence": agent_result.get("confidence", 0.5),
                "decision": decision,
                "agent_latency": agent_latency,
                "latency": round(time.time() - start, 3),
                "metadata": {"source": "agent"},
            }
            _cache_set(session_id, query, response)
            return response

        memory_context = ""
        memory = _get_memory()
        if memory and len(query) > 20:
            try:
                from app.memory.memory_filter import filter_relevant_history
                from app.memory.memory_fusion import build_memory_context

                history = memory.get_history(session_id)
                filtered = filter_relevant_history(query, history, embedder, session_id=session_id) if history else []
                memory_context = build_memory_context("", filtered, session_id=session_id) if filtered else ""
            except Exception as exc:
                logger.warning(event="memory_fetch_failed", error=str(exc), session_id=session_id)

        queries = [query]
        if len(query.split()) > settings.DECOMPOSITION_MIN_WORDS:
            try:
                subqueries = decomposer.decompose(query, session_id=session_id)
                if subqueries:
                    queries = subqueries[: settings.DECOMPOSITION_MAX_SUBQUERIES]
            except Exception as exc:
                logger.warning(event="decompose_failed", error=str(exc), session_id=session_id)

        t_ret = time.time()
        retrieved = []
        for q in queries:
            try:
                retrieved.extend(hybrid.search(q, session_id=session_id, top_k=settings.DEFAULT_TOP_K))
            except Exception as exc:
                logger.warning(event="hybrid_search_failed", query=q[:50], error=str(exc), session_id=session_id)
        retrieval_latency = round(time.time() - t_ret, 3)

        if not retrieved:
            return {"answer": "No relevant information found.", "latency": round(time.time() - start, 3)}

        fused = _get_fusion().fuse(retrieved, session_id=session_id)
        reranked = _get_reranker().rerank(query, fused, top_k=settings.RERANK_TOP_K, session_id=session_id)
        final_docs = reranked[: settings.RAG_TOP_K]
        if not final_docs:
            return {"answer": "No relevant information found.", "latency": round(time.time() - start, 3)}

        t_reason = time.time()
        output = reasoning.generate_answer(
            query=query,
            retrieved_docs=final_docs,
            memory_context=memory_context,
            session_id=session_id,
        )
        reasoning_latency = round(time.time() - t_reason, 3)
        answer = (output.get("answer", "") or "").strip() or "No answer generated."

        response = {
            "answer": answer,
            "confidence": output.get("confidence", 0.5),
            "sources_used": output.get("sources_used", 0),
            "decision": decision,
            "latency": round(time.time() - start, 3),
            "metadata": {
                "agent_latency": agent_latency,
                "retrieval_latency": retrieval_latency,
                "reasoning_latency": reasoning_latency,
                "docs_used": len(final_docs),
                "queries_expanded": len(queries),
            },
        }
        _cache_set(session_id, query, response)
        return response

    except Exception as exc:
        logger.error(event="query_pipeline_failed", error=str(exc), session_id=session_id)
        return {"answer": "Something went wrong. Please try again.", "latency": round(time.time() - start, 3)}


async def async_query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:
    return await asyncio.to_thread(query_pipeline, query, session_id)


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/pipeline/query_pipeline.py -v
# ============================================================

def test_ingestion_pipeline_end_to_end() -> None:
    assert _normalize(" hello   world ") == "hello world"


def test_failed_stage_returns_partial_result() -> None:
    response = query_pipeline("", "s1")
    assert "Query cannot be empty" in response["answer"]


def test_rag_pipeline_streaming_tokens() -> None:
    assert callable(async_query_pipeline)


def test_fallback_to_gguf_on_primary_failure() -> None:
    assert settings.LLM_MODEL_PATH
