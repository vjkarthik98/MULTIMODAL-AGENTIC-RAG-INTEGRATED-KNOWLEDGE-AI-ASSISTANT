from typing import Dict, Any
import time
import json
import hashlib

from app.core.config import settings
from app.utils.logger import get_logger
from app.core.model_loader import model_loader

from app.agents.agent_controller import AgentController

from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker

from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.result_fusion import ResultFusion

from app.core.infra_registry import infra

logger = get_logger(__name__)


#  GLOBAL 
_agent = AgentController()
_bm25 = infra.get_bm25()
_vector_store = infra.get_vector_store()
_memory = infra.get_memory()

_fusion = ResultFusion()
_reranker = Reranker()

_hybrid = None
_reasoning = None
_decomposer = None


#  CACHE 
def _cache_key(session_id: str, query: str):
    base = f"{session_id}:{query.strip().lower()}"
    return "resp:" + hashlib.sha256(base.encode()).hexdigest()


def _cache_get(session_id, query):
    try:
        val = _memory.client.get(_cache_key(session_id, query))
        return json.loads(val) if val else None
    except Exception:
        return None


def _cache_set(session_id, query, data):
    try:
        _memory.client.setex(
            _cache_key(session_id, query),
            settings.REDIS_TTL_SECONDS,
            json.dumps(data)
        )
    except Exception:
        pass


#  SINGLETON 
def _get_hybrid(embedder):
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


#  MAIN 
def query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:

    start = time.time()

    if not query or not query.strip():
        return {"answer": "Query cannot be empty."}

    query = " ".join(query.strip().split())

    #  CACHE 
    cached = _cache_get(session_id, query)
    if cached:
        cached["latency"] = round(time.time() - start, 3)
        return cached

    try:
        llm = model_loader.get_llm()
        embedder = model_loader.get_embedder()

        reasoning, decomposer = _get_reasoning(llm)
        hybrid = _get_hybrid(embedder)

        #  AGENT 
        agent = _agent.handle(query, session_id)

        if agent.get("decision") in {"direct", "search", "memory"}:
            resp = {
                "answer": agent.get("response", ""),
                "latency": round(time.time() - start, 3)
            }
            _cache_set(session_id, query, resp)
            return resp

        #  MEMORY 
        memory_context = ""
        if len(query) > 25:
            history = _memory.get_history(session_id)
            if history:
                filtered = filter_relevant_history(query, history, embedder)
                memory_context = build_memory_context("", filtered)

        #  DECOMPOSE 
        queries = [query]
        if len(query) > 60:
            sub = decomposer.decompose(query)
            if sub:
                queries = sub[:2]

        #  RETRIEVAL 
        retrieved = []

        for q in queries:
            retrieved.extend(
                hybrid.search(q, session_id=session_id, top_k=5)
            )

        if not retrieved:
            return {"answer": "No relevant information found."}

        #  FUSION 
        fused = _fusion.fuse(retrieved, session_id=session_id)

        if not fused:
            return {"answer": "No relevant information found."}

        #  RERANK 
        reranked = _reranker.rerank(query, fused, top_k=5)

        if not reranked:
            return {"answer": "No relevant information found."}

        #  FINAL CONTEXT 
        final_docs = reranked[:settings.RAG_TOP_K]

        #  REASONING 
        output = reasoning.generate_answer(
            query=query,
            retrieved_docs=final_docs,
            memory_context=memory_context,
            session_id=session_id
        )

        answer = output.get("answer", "").strip() or "No answer generated."

        response = {
            "answer": answer,
            "confidence": output.get("confidence", 0.5),
            "sources_used": output.get("sources_used", 0),
            "latency": round(time.time() - start, 3)
        }

        _cache_set(session_id, query, response)

        return response

    except Exception as e:
        logger.error(event="query_pipeline_failed", error=str(e))
        return {"answer": "Something went wrong."}