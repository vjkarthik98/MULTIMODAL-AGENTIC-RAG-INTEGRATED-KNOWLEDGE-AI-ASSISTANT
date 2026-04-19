from typing import Dict, Any, List
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

import time
import os

# Logger
logger = get_logger(__name__)

# GLOBALS
embedder = model_loader.get_embedder()
llm = model_loader.get_llm()

reasoning_engine = ReasoningEngine(llm)
decomposer = QueryDecomposer(llm)
fusion = ResultFusion(top_k=6)
agent = AgentController()
reranker = Reranker()
bm25 = BM25Retriever()
vector_store = QdrantVectorStore()


# MAIN PIPELINE
def query_pipeline(query: str, session_id: str = "default") -> Dict[str, Any]:
    start_time = time.time()

    logger.info(f"[QueryPipeline][START] session_id={session_id}")

    try:
        # STEP 1: Agent
        agent_result = agent.handle(query, session_id)

        if agent_result.get("source") in ["search", "direct", "memory"]:
            return _build_response(
                answer=agent_result.get("response"),
                sources=agent_result.get("sources", []),
                trace={"agent_only": True},
                start_time=start_time
            )
        
        # STEP 2: Memory
        memory = RedisMemory()
        history = memory.get_history(session_id)
        
        filtered_history = filter_relevant_history(query=query, history=history, embedder=embedder)

        summary = ""
        if len(history) > 6:
            summary = summarize_conversation(llm, history)


        memory_context = build_memory_context(summary, filtered_history)

        # STEP 3: Query Decomposition
        sub_queries = decomposer.decompose(query)

        # STEP 4: Hybrid Retrieval
        hybrid = HybridRetriever(
            bm25_retriever=bm25,
            vector_store=vector_store,
            embedder=embedder
        )

        retrieval_batches: List[List[Dict]] = []

        for sub_q in sub_queries:
            results = hybrid.search(query=sub_q, session_id=session_id, top_k=10)
            
            if results:
                retrieval_batches.extend(results)
        if not retrieval_batches:
            return _build_response(
                answer="I couldn't find relevant information.",
                sources = [],
                trace={"empty_retrieval": True},
                start_time=start_time
            )

        # STEP 5: Result Fusion
        fused_results = fusion.fuse(retrieval_batches)

        # STEP 6: Rerank
        reranked = reranker.rerank(query, fused_results, top_k=5)

        if not reranked:
            return _build_response(
                answer="No useful information after ranking.",
                sources=[],
                trace={"empty_rerank": True},
                start_time=start_time
            )
        
        # STEP 7: Final Context Build
        context = _build_context(memory_context, reranked)

        # STEP 8: Reasoning
        reasoning_output = reasoning_engine.generate_answer(
            query=query,
            retrieved_docs=reranked,
            memory_context=context
        )

        # reasoning_engine returns dict
        answer = reasoning_output.get("answer", "")
        confidence = reasoning_output.get("confidence", 0.5)
        sources_used = reasoning_output.get("sources_used", 0)

        # STEP 9: Memory Write
        memory.add_message(session_id, "user", query)
        memory.add_message(session_id, "assistant", answer)

        # STEP 10: Final Response
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": _extract_sources(reranked),
            "trace": {
                "sub_queries": sub_queries,
                "retrieved_docs": len(reranked),
                "memory_used": len(filtered_history) > 0,
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
    

# HELPERS
def _build_context(memory_context: str, docs: List[Dict]) -> str:
    doc_texts = []

    for d in docs:
        text = d.get("text", "").strip()
        if text:
            doc_texts.append(text[:500])

    knowledge = "\n\n".join(doc_texts)

    return f"""
{memory_context}


[RETRIEVED KNOWLEDGE]
{knowledge}
""".strip()

def _extract_sources(docs: List[Dict]) -> List[str]:
    sources = set()

    for d in docs:
        metadata = d.get("metadata", {})
        src = metadata.get("source")
        if src:
            sources.add(src)

    return list(sources)

def _build_response(answer, sources, trace, start_time):
    return {
        "answer": answer,
        "sources": sources,
        "trace": trace,
        "latency": round(time.time() - start_time, 2)}