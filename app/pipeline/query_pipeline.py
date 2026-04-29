from typing import Dict, Any, List
import time
import re
import json

from app.core.config import settings
from app.utils.logger import get_logger
from app.core.model_loader import model_loader

from app.agents.agent_controller import AgentController

from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.memory.summarizer import summarize_conversation

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



# REDIS RESPONSE CACHE
def _get_cache_key(session_id: str, query: str) -> str:
    q = " ".join(query.strip().lower().split())
    return f"response_cache:{session_id}:{q}"


def _get_cached_response(session_id: str, query: str):
    try:
        key = _get_cache_key(session_id, query)
        cached = _memory.client.get(key)

        if cached:
            logger.info("[Cache] HIT")
            return json.loads(cached)

    except Exception as e:
        logger.warning("[Cache] GET FAILED | %s", str(e))

    return None


def _set_cached_response(session_id: str, query: str, response: dict):
    try:
        key = _get_cache_key(session_id, query)

        _memory.client.setex(
            key,
            settings.REDIS_TTL_SECONDS,
            json.dumps(response)
        )

        logger.info("[Cache] STORED")

    except Exception as e:
        logger.warning("[Cache] SET FAILED | %s", str(e))



# SINGLETON HELPERS
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



# QUERY HELPERS
def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _enhance_query_for_modality(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["image", "photo", "diagram", "visual", "describe"]):
        query = f"{query} visual objects scene description"
    return query[:settings.MAX_PROMPT_CHARS]


def _detect_structured_query(query: str) -> bool:
    q = query.lower()
    keywords = ["table of contents", "toc", "section", "page", "begin", "starts"]
    return any(k in q for k in keywords)


def _enhance_query_for_structure(query: str) -> str:
    if _detect_structured_query(query):
        return f"{query} section number page mapping table row structure"
    return query


def _extract_numeric_answer(context: str, query: str) -> str:
    numbers = re.findall(r'\b\d+\b', context)
    if not numbers:
        return ""
    numbers = sorted(numbers, key=lambda x: int(x))
    return numbers[-1]



# MAIN PIPELINE
def query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:

    start_time = time.time()

    if not query or not query.strip():
        return {"answer": "Query cannot be empty.", "error": "invalid_query"}

    query = _normalize_query(query)

    logger.info("[QueryPipeline][START] session_id=%s", session_id)

    # CACHE CHECK
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
            response = _build_response(
                agent_result.get("response"),
                agent_result.get("sources", []),
                {"agent_only": True},
                start_time
            )
            _set_cached_response(session_id, query, response)
            return response

        # MEMORY
        history = _memory.get_history(session_id)
        filtered_history = filter_relevant_history(query, history, embedder)

        summary = ""
        if len(history) >= settings.MEMORY_SUMMARY_THRESHOLD:
            summary = summarize_conversation(llm, history)

        memory_context = build_memory_context(summary, filtered_history)
        memory_context = memory_context[:settings.MAX_PROMPT_CHARS // 2]

        # QUERY ENHANCEMENT
        enhanced_query = _enhance_query_for_modality(query)
        enhanced_query = _enhance_query_for_structure(enhanced_query)

        # DECOMPOSITION
        sub_queries = decomposer.decompose(enhanced_query)

        # RETRIEVAL
        retrieval_results = []
        top_k = settings.DEFAULT_TOP_K
        candidate_k = top_k * settings.HYBRID_CANDIDATES_MULTIPLIER
        is_structured = _detect_structured_query(query)

        for sub_q in sub_queries:
            try:
                results = hybrid.search(
                    query=sub_q,
                    session_id=session_id,
                    top_k=candidate_k
                )

                if is_structured:
                    results = sorted(
                        results,
                        key=lambda x: 1 if any(c.isdigit() for c in x.get("text", "")) else 0,
                        reverse=True
                    )

                if results:
                    retrieval_results.extend(results)

            except Exception as e:
                logger.warning("[QueryPipeline][RETRIEVAL_FAIL] %s", str(e))

        retrieval_results = retrieval_results[:settings.MAX_CHUNKS]

        if not retrieval_results:
            response = _build_response(
                "I couldn't find relevant information.",
                [],
                {"empty_retrieval": True},
                start_time
            )
            _set_cached_response(session_id, query, response)
            return response

        retrieval_results = _deduplicate(retrieval_results)
        fused_results = _fusion.fuse(retrieval_results)

        reranked = reranker.predict(
            [(query, doc["text"]) for doc in fused_results]
        )

        if not reranked:
            response = _build_response(
                "No useful information found.",
                [],
                {"empty_rerank": True},
                start_time
            )
            _set_cached_response(session_id, query, response)
            return response

        context = _build_context(memory_context, reranked)

        numeric_answer = ""
        if is_structured:
            numeric_answer = _extract_numeric_answer(context, query)

        reasoning_output = reasoning_engine.generate_answer(
            query=query,
            retrieved_docs=reranked,
            memory_context=context
        )

        answer = reasoning_output.get("answer", "").strip()

        if numeric_answer:
            answer = numeric_answer

        if not answer:
            answer = "I could not generate a proper answer."

        confidence = min(max(reasoning_output.get("confidence", 0.5), 0.0), 1.0)

        latency = round(time.time() - start_time, 2)

        response = {
            "answer": answer,
            "confidence": confidence,
            "sources": _extract_sources(reranked),
            "trace": {
                "sub_queries": sub_queries,
                "retrieved_docs": len(reranked),
                "memory_used": bool(filtered_history),
                "structured_query": is_structured
            },
            "latency": latency
        }

        # STORE CACHE
        _set_cached_response(session_id, query, response)

        # MEMORY WRITE
        _memory.add_message(session_id, "user", query)
        _memory.add_message(session_id, "assistant", answer)

        logger.info("[QueryPipeline][SUCCESS] latency=%ss", latency)

        return response

    except Exception as e:
        logger.error("[QueryPipeline][ERROR] %s", str(e))
        return {"answer": "Something went wrong.", "error": str(e)}



# HELPERS
def _deduplicate(docs: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for d in docs:
        key = (
            d.get("text", "")[:200],
            str(d.get("metadata", {}).get("doc_id")),
            str(d.get("metadata", {}).get("chunk_id"))
        )

        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique


def _build_context(memory_context: str, docs: List[Dict]) -> str:
    max_chars = settings.MAX_PROMPT_CHARS

    parts = [memory_context.strip()] if memory_context else []
    current_len = len(memory_context)

    for d in docs:
        text = d.get("text", "").strip()
        if not text:
            continue

        chunk = text[:400]

        if current_len + len(chunk) > max_chars:
            break

        parts.append(chunk)
        current_len += len(chunk)

    return "\n\n".join(parts)


def _extract_sources(docs: List[Dict]) -> List[str]:
    return list({
        d.get("metadata", {}).get("source")
        for d in docs
        if d.get("metadata", {}).get("source")
    })


def _build_response(answer, sources, trace, start_time):
    return {
        "answer": answer,
        "sources": sources,
        "trace": trace,
        "latency": round(time.time() - start_time, 2)
    }