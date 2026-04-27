from typing import Dict, Any, List
import time

from app.core.config import settings
from app.utils.logger import get_logger
from app.core.model_loader import model_loader

from app.agents.agent_controller import AgentController

from app.memory.redis_memory import RedisMemory
from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.memory.summarizer import summarize_conversation

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.vectorstore.qdrant_store import QdrantVectorStore

from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.query_decomposer import QueryDecomposer
from app.reasoning.result_fusion import ResultFusion


logger = get_logger(__name__)


# Lazy singletons
_agent = AgentController()
_bm25 = BM25Retriever()
_vector_store = QdrantVectorStore()
_reranker = Reranker()
_fusion = ResultFusion()


# MODALITY-AWARE QUERY ENHANCEMENT
def _enhance_query_for_modality(query: str) -> str:
    q = query.lower()

    vision_keywords = [
        "image", "photo", "picture", "diagram",
        "what is in", "describe", "shown", "visual"
    ]

    if any(k in q for k in vision_keywords):
        enhanced = f"{query} visual description objects scene image content"
        return enhanced[:settings.MAX_PROMPT_CHARS]

    return query


def _get_models():
    llm = model_loader.get_llm()
    embedder = model_loader.get_embedder()

    reasoning_engine = ReasoningEngine(llm)
    decomposer = QueryDecomposer(llm)

    return llm, embedder, reasoning_engine, decomposer


def query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:
    start_time = time.time()

    if not query or not query.strip():
        return {
            "answer": "Query cannot be empty.",
            "error": "invalid_query"
        }

    logger.info(f"[QueryPipeline][START] session_id={session_id}")

    try:
        llm, embedder, reasoning_engine, decomposer = _get_models()

        # Agent Layer
        agent_result = _agent.handle(query, session_id)

        if agent_result.get("source") in ["search", "direct", "memory"]:
            return _build_response(
                answer=agent_result.get("response"),
                sources=agent_result.get("sources", []),
                trace={"agent_only": True},
                start_time=start_time
            )

        # Memory Layer
        memory = RedisMemory()
        history = memory.get_history(session_id)

        filtered_history = filter_relevant_history(
            query=query,
            history=history,
            embedder=embedder
        )

        summary = ""
        if len(history) >= settings.MEMORY_SUMMARY_THRESHOLD:
            summary = summarize_conversation(llm, history)

        memory_context = build_memory_context(summary, filtered_history)

        # APPLY MODALITY ENHANCEMENT
        enhanced_query = _enhance_query_for_modality(query)

        # Query Decomposition
        sub_queries = decomposer.decompose(enhanced_query)

        # Retrieval
        hybrid = HybridRetriever(
            bm25_retriever=_bm25,
            vector_store=_vector_store,
            embedder=embedder
        )

        retrieval_results: List[Dict] = []

        top_k = settings.DEFAULT_TOP_K
        candidate_k = top_k * settings.HYBRID_CANDIDATES_MULTIPLIER

        for sub_q in sub_queries:
            results = hybrid.search(
                query=sub_q,
                session_id=session_id,
                top_k=candidate_k
            )

            if results:
                retrieval_results.extend(results)

        if not retrieval_results:
            return _build_response(
                answer="I couldn't find relevant information.",
                sources=[],
                trace={"empty_retrieval": True},
                start_time=start_time
            )

        # Deduplication before fusion
        retrieval_results = _deduplicate(retrieval_results)

        # Fusion
        fused_results = _fusion.fuse(retrieval_results)

        # Rerank
        reranked = _reranker.rerank(
            query, 
            fused_results,
            top_k=settings.RERANK_TOP_K
        )

        if not reranked:
            return _build_response(
                answer="No useful information after ranking.",
                sources=[],
                trace={"empty_rerank": True},
                start_time=start_time
            )

        # Context Build
        context = _build_context(memory_context, reranked)

        # Reasoning
        reasoning_output = reasoning_engine.generate_answer(
            query=query, 
            retrieved_docs=reranked,
            memory_context=context
        )

        answer = reasoning_output.get("answer", "")
        confidence = reasoning_output.get("confidence", 0.5)
        sources_used = reasoning_output.get("sources_used", 0)

        # Memory Write
        memory.add_message(session_id, "user", query)
        memory.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": _extract_sources(reranked),
            "trace": {
                "sub_queries": sub_queries,
                "retrieved_docs": len(reranked),
                "memory_used": bool(filtered_history),
                "sources_used": sources_used
            },
            "latency": round(time.time() - start_time, 2)
        }

    except Exception as e:
        logger.error(f"[QueryPipeline][ERROR] {str(e)}")

        return {
            "answer": "Something went wrong.",
            "error": str(e)
        }


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

        chunk = text[:500]

        if current_len + len(chunk) > max_chars:
            break

        parts.append(chunk)
        current_len += len(chunk)

    return "\n\n".join(parts)


def _extract_sources(docs: List[Dict]) -> List[str]:
    sources = set()

    for d in docs:
        src = d.get("metadata", {}).get("source")
        if src:
            sources.add(src)

    return list(sources)


def _build_response(answer, sources, trace, start_time):
    return {
        "answer": answer,
        "sources": sources,
        "trace": trace,
        "latency": round(time.time() - start_time, 2)
    }