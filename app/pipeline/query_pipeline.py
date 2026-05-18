from __future__ import annotations

import asyncio
import hashlib
import threading
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


# LAZY SINGLETONS — all guarded by a per-singleton lock to prevent double-init

_agent        = None
_bm25         = None
_vector_store = None
_memory       = None
_fusion       = None
_reranker     = None
_hybrid       = None
_reasoning    = None
_decomposer   = None

_lock_agent     = threading.Lock()
_lock_infra     = threading.Lock()
_lock_fusion    = threading.Lock()
_lock_reranker  = threading.Lock()
_lock_hybrid    = threading.Lock()
_lock_reasoning = threading.Lock()


def _get_agent():
    global _agent
    if _agent is None:
        with _lock_agent:
            if _agent is None:
                from app.agents.agent_controller import AgentController
                _agent = AgentController()
    return _agent


def _get_infra():
    global _bm25, _vector_store, _memory
    if _bm25 is None or _vector_store is None or _memory is None:
        with _lock_infra:
            if _bm25 is None or _vector_store is None or _memory is None:
                from app.core.infra_registry import infra
                _bm25         = infra.get_bm25()
                _vector_store = infra.get_vector_store()
                _memory       = infra.get_memory()
    return _bm25, _vector_store, _memory


def _get_fusion():
    global _fusion
    if _fusion is None:
        with _lock_fusion:
            if _fusion is None:
                from app.reasoning.result_fusion import ResultFusion
                _fusion = ResultFusion()
    return _fusion


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _lock_reranker:
            if _reranker is None:
                from app.retrieval.reranker import Reranker
                _reranker = Reranker()
    return _reranker


def _get_hybrid(embedder: Any) -> Any:
    global _hybrid
    if _hybrid is None:
        with _lock_hybrid:
            if _hybrid is None:
                from app.retrieval.hybrid_retriever import HybridRetriever
                bm25, vector_store, _ = _get_infra()
                _hybrid = HybridRetriever(bm25, vector_store, embedder)
    return _hybrid


def _get_reasoning_components(llm: Any):
    global _reasoning, _decomposer
    if _reasoning is None or _decomposer is None:
        with _lock_reasoning:
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


# STORE INTERACTION + AUTO-SUMMARIZE EVERY N TURNS — SECTION 4.7

def _store_interaction(
    session_id: str,
    query: str,
    answer: str,
    memory: Any,
) -> None:
    if not answer.strip():
        return
    try:
        from app.memory.memory_manager import MemoryManager
        from app.core.infra_registry import infra
        mgr = MemoryManager()
        mgr.add_interaction(session_id, query, answer)

        # AUTO-SUMMARIZE AFTER EVERY N TURNS — fires in background thread
        size = mgr.get_memory_size(session_id)
        every_n = settings.MEMORY_SUMMARY_EVERY_N_TURNS * 2  # each turn = 2 messages
        if every_n > 0 and size > 0 and size % every_n == 0:
            import threading
            def _run_summary():
                try:
                    from app.core.model_loader import model_loader
                    llm = model_loader.get_llm()
                    mgr.summarize_and_compress(session_id, llm)
                    logger.info(
                        event="auto_summary_triggered",
                        session_id=session_id,
                        turn=size // 2,
                    )
                except Exception as exc:
                    logger.warning(
                        event="auto_summary_failed",
                        error=str(exc),
                        session_id=session_id,
                    )
            threading.Thread(target=_run_summary, daemon=True).start()

    except Exception as e:
        logger.warning(event="memory_store_failed", error=str(e), session_id=session_id)


# SOURCES ARRAY BUILDER — PHASE 24.8

def _build_sources_array(docs: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    """Build the Phase 24.8 standardised sources array from reranked docs (top min(max_items, len))."""
    import os as _os
    out: List[Dict[str, Any]] = []
    for doc in docs[:max_items]:
        meta  = doc.get("metadata") or {}
        text  = doc.get("text") or ""
        score = doc.get("final_score") if doc.get("final_score") is not None else doc.get("score")
        try:
            score = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        src_raw = (
            meta.get("source")
            or meta.get("filename")
            or meta.get("file_path")
            or "unknown"
        )
        source_name = _os.path.basename(str(src_raw)) if src_raw != "unknown" else "unknown"

        modality = str(meta.get("modality") or "text")

        page_number: Optional[int] = None
        raw_page = meta.get("page_number") if meta.get("page_number") is not None else meta.get("page")
        if isinstance(raw_page, int):
            page_number = raw_page
        elif raw_page is not None:
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                pass

        start_time: Optional[float] = None
        end_time:   Optional[float] = None
        raw_start = meta.get("start_time") if meta.get("start_time") is not None else meta.get("timestamp_start")
        raw_end   = meta.get("end_time")   if meta.get("end_time")   is not None else meta.get("timestamp_end")
        if raw_start is not None:
            try:
                start_time = float(raw_start)
            except (TypeError, ValueError):
                pass
        if raw_end is not None:
            try:
                end_time = float(raw_end)
            except (TypeError, ValueError):
                pass

        doc_id = str(meta.get("doc_id") or meta.get("chunk_id") or "")

        out.append({
            "text":        str(text)[:200],
            "score":       round(score, 6),
            "source":      source_name,
            "page_number": page_number,
            "start_time":  start_time,
            "end_time":    end_time,
            "modality":    modality,
            "doc_id":      doc_id,
        })
    return out


def _confidence_from_sources(sources: List[Dict[str, Any]]) -> float:
    """Mean score of top-3 sources, clamped to [0.0, 1.0]. Returns 0.0 if no sources."""
    scores = [s["score"] for s in sources[:3] if isinstance(s.get("score"), (int, float))]
    if not scores:
        return 0.0
    return round(max(0.0, min(sum(scores) / len(scores), 1.0)), 6)


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
        from app.pipeline.rag_pipeline import RAGPipeline
        import queue as _queue

        rag = RAGPipeline()

        # Run the sync generator in a thread and forward tokens via a queue
        token_queue: _queue.Queue = _queue.Queue()
        _SENTINEL = object()

        def _producer():
            try:
                for tok in rag.stream(query, session_id=session_id):
                    token_queue.put(tok)
            except Exception as exc:
                token_queue.put(exc)
            finally:
                token_queue.put(_SENTINEL)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _producer)

        while True:
            item = await loop.run_in_executor(None, token_queue.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                logger.error(event="stream_query_failed", error=str(item), session_id=session_id)
                yield f"[ERROR]: {item}"
                break
            if item:
                yield item

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
            "answer":               "Query cannot be empty.",
            "confidence":           0.0,
            "decision":             "reject",
            "source":               "validation",
            "session_id":           session_id,
            "request_id":           trace_id,
            "latency":              0.0,
            "sources":              [],
            "is_fallback":          False,
            "hallucination_warning": True,
            "trace_id":             trace_id,
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
        if isinstance(cached.get("metadata"), dict):
            cached["metadata"]["cache_hit"] = True
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
            conf   = float(agent_result.get("confidence", 0.5))
            conf   = max(0.0, min(conf, 1.0))
            resp = {
                "answer":               answer,
                "confidence":           conf,
                "decision":             decision,
                "source":               "agent",
                "session_id":           session_id,
                "request_id":           trace_id,
                "agent_latency":        agent_latency,
                "latency":              round(time.time() - start, 3),
                "sources":              [],
                "is_fallback":          False,
                "hallucination_warning": conf < settings.AGENT_LOW_CONFIDENCE,
                "trace_id":             trace_id,
                "metadata":             {"source": "agent"},
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
                "answer":               "No relevant documents found. Please ingest documents first.",
                "confidence":           0.0,
                "decision":             decision,
                "source":               "rag",
                "session_id":           session_id,
                "request_id":           trace_id,
                "latency":              round(time.time() - start, 3),
                "sources":              [],
                "is_fallback":          False,
                "hallucination_warning": True,
                "trace_id":             trace_id,
            }

        # RESULT FUSION — SECTION 4.8
        fused = fusion.fuse(retrieved, session_id=session_id)
        if not fused:
            return {
                "answer":               "No relevant documents found. Please ingest documents first.",
                "confidence":           0.0,
                "decision":             decision,
                "source":               "rag",
                "session_id":           session_id,
                "request_id":           trace_id,
                "latency":              round(time.time() - start, 3),
                "sources":              [],
                "is_fallback":          False,
                "hallucination_warning": True,
                "trace_id":             trace_id,
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
                "answer":               "No relevant documents found. Please ingest documents first.",
                "confidence":           0.0,
                "decision":             decision,
                "source":               "rag",
                "session_id":           session_id,
                "request_id":           trace_id,
                "latency":              round(time.time() - start, 3),
                "sources":              [],
                "is_fallback":          False,
                "hallucination_warning": True,
                "trace_id":             trace_id,
            }

        final_docs = reranked[:settings.RAG_TOP_K]

        # BUILD CANONICAL SOURCES[] WITH cite_key BEFORE REASONING
        from app.core.response import build_sources
        canonical_sources = build_sources(final_docs)

        # REASONING — SECTION 4.8
        t_reason = time.time()

        try:
            output = reasoning.generate_answer(
                query=query,
                retrieved_docs=final_docs,
                memory_context=memory_context,
                session_id=session_id,
                sources=canonical_sources,
            )
            reasoning_latency = round(time.time() - t_reason, 3)
            _record_llm_latency("gguf_mistral", reasoning_latency)
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

        # PHASE 24.8 — build standardised sources[] from reranked docs
        p248_sources = _build_sources_array(final_docs, max_items=min(3, len(final_docs)))

        # CONFIDENCE — prefer executor value when valid; else compute from source scores
        raw_conf = output.get("confidence")
        if raw_conf is not None:
            try:
                raw_conf = float(raw_conf)
                if not (0.0 <= raw_conf <= 1.0):
                    raw_conf = None
            except (TypeError, ValueError):
                raw_conf = None
        confidence = raw_conf if raw_conf is not None else _confidence_from_sources(p248_sources)
        confidence = round(max(0.0, min(confidence, 1.0)), 6)

        hallucination_warning = confidence < settings.AGENT_LOW_CONFIDENCE

        # FINAL sources[] — prefer the Phase 24.8 array built from reranked docs.
        # Also keep canonical_sources for LLM citation rendering.
        out_sources = output.get("sources")
        if not isinstance(out_sources, list):
            out_sources = canonical_sources

        total_latency = round(time.time() - start, 3)

        response = {
            "answer":               answer,
            "confidence":           confidence,
            "decision":             decision,
            "source":               "agent",
            "session_id":           session_id,
            "request_id":           trace_id,
            "latency":              total_latency,
            "sources":              p248_sources,
            "is_fallback":          False,
            "hallucination_warning": hallucination_warning,
            "sources_used":         len(p248_sources),
            "trace_id":             trace_id,
            "metadata": {
                "agent_latency":     agent_latency,
                "retrieval_latency": retrieval_latency,
                "reasoning_latency": reasoning_latency,
                "docs_used":         len(final_docs),
                "queries_expanded":  len(queries),
                "memory_injected":   bool(memory_context),
                "cache_hit":         False,
                "canonical_sources": out_sources,
            },
        }

        if (
            answer
            and "No relevant documents" not in answer
            and "Something went wrong" not in answer
            and "No answer generated" not in answer
            and output.get("confidence", 0) > 0.15
        ):
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
            "answer":               "Something went wrong. Please try again.",
            "confidence":           0.0,
            "decision":             "fallback",
            "source":               "error",
            "session_id":           session_id,
            "request_id":           trace_id,
            "latency":              round(time.time() - start, 3),
            "sources":              [],
            "is_fallback":          True,
            "hallucination_warning": True,
            "trace_id":             trace_id,
            "error":                str(e),
        }


# ASYNC WRAPPER — SECTION 4.6

async def query_pipeline_async(
    query: str,
    session_id: str = "default",
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, query_pipeline, query, session_id),
        timeout=settings.REQUEST_TIMEOUT_SEC,
    )

