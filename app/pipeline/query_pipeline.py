from typing import Dict, Any
import time
import json

from app.core.config import settings
from app.utils.logger import get_logger
from app.core.model_loader import model_loader

from app.agents.agent_controller import AgentController

from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context

from app.retrieval.hybrid_retriever import HybridRetriever

from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.result_fusion import ResultFusion

from app.core.infra_registry import infra


logger = get_logger(__name__)


# GLOBAL SINGLETONS
_agent = AgentController()
_bm25 = infra.get_bm25()
_vector_store = infra.get_vector_store()
_memory = infra.get_memory()

_fusion = ResultFusion()

_hybrid = None
_reasoning_engine = None
_decomposer = None


# CACHE
def _get_cache_key(session_id: str, query: str) -> str:
    return f"response_cache:{session_id}:{query.strip().lower()}"


def _get_cached_response(session_id: str, query: str):
    try:
        cached = _memory.client.get(_get_cache_key(session_id, query))
        if cached:
            logger.info("[Cache] HIT")
            return json.loads(cached)
    except Exception:
        pass
    return None


def _set_cached_response(session_id: str, query: str, response: dict):
    try:
        _memory.client.setex(
            _get_cache_key(session_id, query),
            settings.REDIS_TTL_SECONDS,
            json.dumps(response)
        )
    except Exception:
        pass


def _get_hybrid(embedder):
    global _hybrid
    if _hybrid is None:
        _hybrid = HybridRetriever(_bm25, _vector_store, embedder)
    return _hybrid


def _get_reasoning(llm):
    global _reasoning_engine, _decomposer
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine(llm)
    if _decomposer is None:
        _decomposer = QueryDecomposer(llm)
    return _reasoning_engine, _decomposer


def query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:

    start_time = time.time()

    if not query.strip():
        return {"answer": "Query cannot be empty."}

    query = " ".join(query.strip().split())

    # CACHE
    cached = _get_cached_response(session_id, query)
    if cached:
        cached["latency"] = round(time.time() - start_time, 4)
        return cached

    try:
        llm = model_loader.get_llm()
        embedder = model_loader.get_embedder()
        reranker = model_loader.get_reranker()

        reasoning_engine, decomposer = _get_reasoning(llm)
        hybrid = _get_hybrid(embedder)

        # AGENT
        agent_result = _agent.handle(query, session_id)
        if agent_result.get("source") in ["search", "direct", "memory"]:
            response = {"answer": agent_result.get("response")}
            _set_cached_response(session_id, query, response)
            return response

        # MEMORY 
        memory_context = ""
        if len(query) > 25:
            history = _memory.get_history(session_id)
            if history:
                filtered = filter_relevant_history(query, history, embedder)
                memory_context = build_memory_context("", filtered)

        # SINGLE QUERY 
        if len(query) > 50:
            sub = decomposer.decompose(query)
            query = sub[0] if sub else query

        # RETRIEVAL
        retrieval_results = hybrid.search(query, session_id, top_k=5)

        if not retrieval_results:
            return {"answer": "No relevant information found."}

        # LIMIT HARD
        retrieval_results = retrieval_results[:5]

        # FUSION
        fused = _fusion.fuse(retrieval_results)[:4]

        # RERANK ONLY IF NEEDED
        if len(query) > 30:
            pairs = [(query, doc["text"]) for doc in fused]
            scores = reranker.predict(pairs)

            for doc, score in zip(fused, scores):
                doc["score"] = float(score)

            reranked = sorted(fused, key=lambda x: x["score"], reverse=True)[:3]
        else:
            reranked = fused[:2]

        # VERY SMALL CONTEXT
        context = "\n\n".join(d["text"][:80] for d in reranked)

        # LLM
        output = reasoning_engine.generate_answer(
            query=query,
            retrieved_docs=reranked,
            memory_context=context
        )

        answer = output.get("answer", "").strip() or "No answer generated."

        response = {
            "answer": answer,
            "latency": round(time.time() - start_time, 2)
        }

        _set_cached_response(session_id, query, response)

        return response

    except Exception as e:
        logger.error("[QueryPipeline][ERROR] %s", str(e))
        return {"answer": "Something went wrong."}