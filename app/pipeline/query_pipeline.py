from __future__ import annotations

import asyncio
import hashlib
import time
import unicodedata
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS — SECTION 6

def _get_metrics():
    try:
        from prometheus_client import Counter, Histogram
        retrieval_latency = Histogram(
            "retrieval_latency_seconds",
            "Retrieval latency by retriever type",
            ["retriever_type"],
        )
        llm_latency = Histogram(
            "llm_call_latency_seconds",
            "LLM call latency by model",
            ["model"],
        )
        query_errors = Counter(
            "query_pipeline_errors_total",
            "Query pipeline errors by stage",
            ["stage"],
        )
        return {
            "retrieval_latency": retrieval_latency,
            "llm_latency":       llm_latency,
            "query_errors":      query_errors,
        }
    except Exception:
        return {}


_METRICS: Dict[str, Any] = {}

if settings.PROMETHEUS_ENABLED:
    try:
        _METRICS = _get_metrics()
    except Exception:
        pass


def _record_retrieval_latency(retriever_type: str, latency: float) -> None:
    try:
        if "retrieval_latency" in _METRICS:
            _METRICS["retrieval_latency"].labels(retriever_type=retriever_type).observe(latency)
    except Exception:
        pass


def _record_llm_latency(model: str, latency: float) -> None:
    try:
        if "llm_latency" in _METRICS:
            _METRICS["llm_latency"].labels(model=model).observe(latency)
    except Exception:
        pass


def _record_query_error(stage: str) -> None:
    try:
        if "query_errors" in _METRICS:
            _METRICS["query_errors"].labels(stage=stage).inc()
    except Exception:
        pass


# NORMALIZE QUERY — SECTION 2.3

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    # STRIP NULL BYTES
    query = query.replace("\x00", "")
    # STRIP BOM
    query = query.lstrip("\ufeff\ufffe")
    return " ".join(query.strip().split())


# PROMPT INJECTION SANITIZATION — SECTION 5

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard the above",
    "forget everything",
    "you are now",
    "act as",
    "jailbreak",
]


def _sanitize_query(query: str) -> str:
    lower = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            idx   = lower.find(pattern)
            query = query[:idx].strip()
            logger.warning(event="query_injection_stripped", pattern=pattern)
            break
    return query


# CACHE KEY — SECTION 4.6

def _cache_key(session_id: str, query: str) -> str:
    base = f"{session_id}:{query.strip().lower()}"
    return "qresp:" + hashlib.sha256(base.encode("utf-8")).hexdigest()


def _cache_get(session_id: str, query: str) -> Optional[Dict[str, Any]]:
    try:
        from app.core.infra_registry import infra
        memory = infra.get_memory()
        if not memory:
            return None
        return memory.cache_get(_cache_key(session_id, query))
    except Exception:
        return None


def _cache_set(session_id: str, query: str, data: Dict[str, Any]) -> None:
    try:
        from app.core.infra_registry import infra
        memory = infra.get_memory()
        if not memory:
            return
        memory.cache_set(
            _cache_key(session_id, query),
            data,
            ttl=settings.REDIS_QUERY_CACHE_TTL,
        )
    except Exception:
        pass


# LAZY SINGLETONS

_agent        = None
_bm25         = None
_vector_store = None
_memory       = None
_fusion       = None
_reranker     = None
_hybrid       = None
_reasoning    = None
_decomposer   = None


def _get_agent():
    global _agent
    if _agent is None:
        from app.agents.agent_controller import AgentController
        _agent = AgentController()
    return _agent


def _get_infra():
    global _bm25, _vector_store, _memory
    if _bm25 is None or _vector_store is None or _memory is None:
        from app.core.infra_registry import infra
        _bm25         = infra.get_bm25()
        _vector_store = infra.get_vector_store()
        _memory       = infra.get_memory()
    return _bm25, _vector_store, _memory


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


def _get_hybrid(embedder: Any) -> Any:
    global _hybrid
    if _hybrid is None:
        from app.retrieval.hybrid_retriever import HybridRetriever
        bm25, vector_store, _ = _get_infra()
        _hybrid = HybridRetriever(bm25, vector_store, embedder)
    return _hybrid


def _get_reasoning_components(llm: Any):
    global _reasoning, _decomposer
    if _reasoning is None:
        from app.reasoning.reasoning_engine import ReasoningEngine
        _reasoning = ReasoningEngine(llm)
    if _decomposer is None:
        from app.reasoning.query_decomposer import QueryDecomposer
        _decomposer = QueryDecomposer(llm)
    return _reasoning, _decomposer


# MEMORY CONTEXT BUILDER — SECTION 4.7

def _build_memory_context(
    query: str,
    session_id: str,
    embedder: Any,
    memory: Any,
) -> str:
    if not memory or len(query) < 20:
        return ""
    try:
        from app.memory.memory_filter import filter_relevant_history
        from app.memory.memory_fusion import build_memory_context
        history = memory.get_history(session_id)
        if not history:
            return ""
        filtered = filter_relevant_history(
            query, history, embedder, session_id=session_id
        )
        return build_memory_context("", filtered, session_id=session_id)
    except Exception as e:
        logger.warning(event="memory_context_failed", error=str(e), session_id=session_id)
        return ""


# STORE INTERACTION — SECTION 4.7

def _store_interaction(
    session_id: str,
    query: str,
    answer: str,
    memory: Any,
) -> None:
    if not memory or not answer.strip():
        return
    try:
        from app.memory.memory_manager import MemoryManager
        mgr = MemoryManager()
        mgr.add_interaction(session_id, query, answer)
    except Exception as e:
        logger.warning(event="memory_store_failed", error=str(e), session_id=session_id)


# STREAMING RESPONSE — SECTION 4.6

async def stream_query(
    query: str,
    session_id: str = "default",
) -> AsyncIterator[str]:
    query = _normalize(query)
    query = _sanitize_query(query)

    if not query:
        yield "Query cannot be empty."
        return

    query = query[:settings.MAX_PROMPT_CHARS]

    try:
        from app.core.model_loader import model_loader
        from app.pipeline.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        gen = rag.stream(query, session_id=session_id)

        async def _wrap_sync_gen():
            loop = asyncio.get_event_loop()
            for token in await loop.run_in_executor(None, list, gen):
                yield token

        async for token in _wrap_sync_gen():
            yield token

        yield "[DONE]"

    except Exception as e:
        logger.error(event="stream_query_failed", error=str(e), session_id=session_id)
        yield f"[ERROR]: {e}"


# MAIN QUERY PIPELINE

def query_pipeline(
    query: str,
    session_id: str = "default",
) -> Dict[str, Any]:

    start      = time.time()
    trace_id   = str(uuid.uuid4())
    # OTEL SPAN STUB 
    span_ctx   = {"trace_id": trace_id}

    if not query or not query.strip():
        return {
            "answer":   "Query cannot be empty.",
            "latency":  0.0,
            "trace_id": trace_id,
        }

    # NORMALIZE AND SANITIZE — SECTION 2.3 / SECTION 5
    query = _normalize(query)
    query = _sanitize_query(query)
    query = query[:settings.MAX_PROMPT_CHARS]

    # CACHE HIT — SECTION 4.6
    cached = _cache_get(session_id, query)
    if cached:
        cached["latency"]   = round(time.time() - start, 3)
        cached["cache_hit"] = True
        cached["trace_id"]  = trace_id
        logger.debug(event="query_cache_hit", session_id=session_id)
        return cached

    try:
        from app.core.model_loader import model_loader
        llm      = model_loader.get_llm()
        embedder = model_loader.get_embedder()

        reasoning, decomposer = _get_reasoning_components(llm)
        hybrid                = _get_hybrid(embedder)
        fusion                = _get_fusion()
        reranker              = _get_reranker()
        _, _, memory          = _get_infra()

        # AGENT DECISION — SECTION 4.9
        t_agent = time.time()
        agent   = _get_agent()

        try:
            agent_result  = agent.handle(query, session_id)
            agent_latency = round(time.time() - t_agent, 3)
        except Exception as e:
            logger.warning(event="agent_failed", error=str(e), session_id=session_id)
            _record_query_error("agent")
            agent_result  = {"decision": "rag", "response": "", "confidence": 0.5}
            agent_latency = round(time.time() - t_agent, 3)

        decision = agent_result.get("decision", "rag")

        # SHORT-CIRCUIT FOR NON-RAG DECISIONS — SECTION 4.9
        if decision in {"direct", "search", "memory"}:
            answer = agent_result.get("response", "No answer generated.")
            resp = {
                "answer":        answer,
                "confidence":    agent_result.get("confidence", 0.5),
                "decision":      decision,
                "agent_latency": agent_latency,
                "latency":       round(time.time() - start, 3),
                "trace_id":      trace_id,
                "metadata":      {"source": "agent"},
            }
            _cache_set(session_id, query, resp)
            _store_interaction(session_id, query, answer, memory)
            return resp

        # MEMORY CONTEXT — SECTION 4.7
        memory_context = _build_memory_context(query, session_id, embedder, memory)

        # QUERY DECOMPOSITION — SECTION 4.8
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
                _record_query_error("decompose")

        # HYBRID RETRIEVAL — SECTION 4.5
        t_ret     = time.time()
        retrieved: List[Dict[str, Any]] = []

        for q in queries:
            try:
                results = hybrid.search(
                    q,
                    session_id=session_id,
                    top_k=settings.DEFAULT_TOP_K,
                )
                retrieved.extend(results)
            except Exception as e:
                logger.warning(
                    event="hybrid_search_failed",
                    query=q[:50],
                    error=str(e),
                    session_id=session_id,
                )
                _record_query_error("retrieval")

        retrieval_latency = round(time.time() - t_ret, 3)
        _record_retrieval_latency("hybrid", retrieval_latency)

        if not retrieved:
            logger.warning(
                event="query_pipeline_no_results",
                queries=len(queries),
                session_id=session_id,
            )
            return {
                "answer":   "No relevant information found.",
                "latency":  round(time.time() - start, 3),
                "trace_id": trace_id,
            }

        # RESULT FUSION — SECTION 4.8
        fused = fusion.fuse(retrieved, session_id=session_id)
        if not fused:
            return {
                "answer":   "No relevant information found.",
                "latency":  round(time.time() - start, 3),
                "trace_id": trace_id,
            }

        # RERANK — SECTION 4.5
        reranked = reranker.rerank(
            query,
            fused,
            top_k=settings.RERANK_TOP_K,
            session_id=session_id,
        )

        if not reranked:
            return {
                "answer":   "No relevant information found.",
                "latency":  round(time.time() - start, 3),
                "trace_id": trace_id,
            }

        final_docs = reranked[:settings.RAG_TOP_K]

        # REASONING — SECTION 4.8
        t_reason = time.time()

        try:
            output = reasoning.generate_answer(
                query=query,
                retrieved_docs=final_docs,
                memory_context=memory_context,
                session_id=session_id,
            )
            reasoning_latency = round(time.time() - t_reason, 3)
            _record_llm_latency(settings.EMBEDDING_MODEL, reasoning_latency)
        except Exception as e:
            logger.error(
                event="reasoning_failed",
                error=str(e),
                session_id=session_id,
            )
            _record_query_error("reasoning")
            # FALLBACK CHAIN — SECTION 4.6: LOCAL GGUF
            try:
                t_fallback = time.time()
                context    = "\n\n".join(
                    d.get("text", "")[:settings.RAG_DOC_MAX_CHARS]
                    for d in final_docs
                )
                prompt = (
                    f"Answer based on context only.\n\n"
                    f"CONTEXT:\n{context[:settings.MAX_CONTEXT_CHARS]}\n\n"
                    f"QUERY:\n{query}\n\nAnswer:"
                )
                raw = llm.generate(
                    prompt,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=0.2,
                    session_id=session_id,
                )
                reasoning_latency = round(time.time() - t_fallback, 3)
                output = {
                    "answer":       raw.strip() or "Unable to generate answer.",
                    "confidence":   0.4,
                    "sources_used": len(final_docs),
                }
                logger.info(
                    event="fallback_gguf_used",
                    latency=reasoning_latency,
                    session_id=session_id,
                )
            except Exception as fe:
                logger.error(
                    event="fallback_gguf_failed",
                    error=str(fe),
                    session_id=session_id,
                )
                output = {
                    "answer":       "Something went wrong generating the answer.",
                    "confidence":   0.1,
                    "sources_used": 0,
                }
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
            "trace_id":     trace_id,
            "metadata": {
                "agent_latency":     agent_latency,
                "retrieval_latency": retrieval_latency,
                "reasoning_latency": reasoning_latency,
                "docs_used":         len(final_docs),
                "queries_expanded":  len(queries),
                "memory_injected":   bool(memory_context),
                "cache_hit":         False,
            },
        }

        _cache_set(session_id, query, response)
        _store_interaction(session_id, query, answer, memory)

        logger.info(
            event="query_pipeline_success",
            decision=decision,
            docs_used=len(final_docs),
            queries=len(queries),
            latency=total_latency,
            session_id=session_id,
            trace_id=trace_id,
        )

        return response

    except Exception as e:
        _record_query_error("pipeline")
        logger.error(
            event="query_pipeline_failed",
            error=str(e),
            session_id=session_id,
            trace_id=trace_id,
        )
        return {
            "answer":   "Something went wrong. Please try again.",
            "latency":  round(time.time() - start, 3),
            "trace_id": trace_id,
            "error":    str(e),
        }


# ASYNC WRAPPER — SECTION 4.6

async def query_pipeline_async(
    query: str,
    session_id: str = "default",
) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, query_pipeline, query, session_id),
        timeout=settings.REQUEST_TIMEOUT_SEC,
    )

