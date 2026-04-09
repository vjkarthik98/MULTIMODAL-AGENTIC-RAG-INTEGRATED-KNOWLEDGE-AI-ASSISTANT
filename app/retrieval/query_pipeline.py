from app.memory.redis_memory import RedisMemory
from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.memory.summarizer import summarize_conversation
from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.query_decomposer import QueryDecomposer
from app.core.model_loader import model_loader
from app.reasoning.result_fusion import ResultFusion
from app.agents.agent_controller import AgentController
from app.retrieval.hybrid_retriever import HybridRetriever
from app.ingestion.pipeline import bm25, vector_store
from moviepy.editor import VideoFileClip
from app.retrieval.reranker import Reranker
import os
import logging

# Logger
logger = logging.getLogger(__name__)


embedder = model_loader.get_embedder()
clip_embedder = model_loader.get_clip_text_embedder()
llm = model_loader.get_llm()
reasoning_engine = ReasoningEngine(llm)
decomposer = QueryDecomposer(llm)
fusion = ResultFusion(top_k=5)
agent = AgentController()
reranker = Reranker()


def detect_query_type(query: str):
    query = query.lower()

    if "image" in query or "picture" in query or "photo" in query:
        return "image"
    elif "audio" in query or "voice" in query:
        return "audio"
    elif "video" in query:
        return "video"
    else:
        return "text"


# ======================
# TEXT QUERY
# ======================
def query_text(query: str, session_id: str = "default"):
    try:
        logger.info(f"[QueryPipeline] session_id={session_id} | New query received")

        # Step 0: Agent Decision
        decision = agent.decide(query, session_id=session_id)

        logger.info(
            f"[QueryPipeline] session_id={session_id} | Agent decision={decision['action']}"
        )

        # Step 0.5 : Agent Routing
        if decision["action"] == "multimodal":
            query_type = detect_query_type(query)
            logger.info(
                f"[QueryPipeline] session_id={session_id} | Multimodal route={query_type}"
            )

            if query_type == "image":
                results = query_image(query, session_id)

                if not results:
                    return {
                        "answer": "No relevant image information found.",
                        "sources": []
                    }

                answer = reasoning_engine.generate_answer(
                    query=query,
                    retrieved_docs=results,
                    memory_context=""
                )

                return {
                    "answer": answer,
                    "sources": results
                }

            elif query_type == "audio":
                return {
                    "answer": "Audio query detected. Please use audio upload endpoint.",
                    "sources": []
                }

            elif query_type == "video":
                return {
                    "answer": "Video query detected. Please use video upload endpoint.",
                    "sources": []
                }

        # Step 1: Memory
        memory = RedisMemory()
        history = memory.get_history(session_id)

        logger.debug(
            f"[QueryPipeline] session_id={session_id} | History size={len(history)}"
        )

        # Step 2: Filter memory
        filtered_history = filter_relevant_history(
            query,
            history,
            embedder
        )

        # Step 3: Summarize
        summary = ""
        if len(history) > 6:
            summary = summarize_conversation(llm, history)

        # Step 4: Memory context
        memory_context = build_memory_context(
            summary,
            filtered_history
        )

        # Step 5: Query decomposition
        sub_queries = decomposer.decompose(query)

        logger.debug(
            f"[QueryPipeline] session_id={session_id} | Sub-queries={sub_queries}"
        )

        # Hybrid Retrieval
        hybrid = HybridRetriever(bm25, vector_store, embedder)

        all_results = []

        for sub_q in sub_queries:
            hybrid_results = hybrid.search(sub_q, top_k=10)
            all_results.extend(hybrid_results)

        # Deduplicate
        seen = set()
        unique_results = []

        for r in all_results:
            text = r.get("text", "")
            if text not in seen:
                seen.add(text)
                unique_results.append(r)

        # Reranking
        results = reranker.rerank(query, unique_results, top_k=5)

        logger.debug(
            f"[QueryPipeline] session_id={session_id} | Results after rerank={len(results)}"
        )

        # Query Type
        query_type = detect_query_type(query)

        # Modality filter
        if query_type == "image":
            filtered = [r for r in results if r["metadata"].get("modality") == "image"]
            if filtered:
                results = filtered 

        elif query_type == "audio":
            results = [r for r in results if r["metadata"].get("modality") == "audio"]

        elif query_type == "video":
            results = [r for r in results if r["metadata"].get("modality") == "video"]

        if not results:
            logger.warning(
                f"[QueryPipeline] session_id={session_id} | No results found"
            )
            return {
                "answer": "I couldn't find relevant information in the knowledge base.",
                "sources": []
            }
            
        logger.info(f"[DEBUG] Results count before LLM = {len(results)}")

        # Step 6: LLM
        logger.info(f"[QueryPipeline] session_id={session_id} | Generating answer")

        answer = reasoning_engine.generate_answer(
            query=query,
            retrieved_docs=results,
            memory_context=memory_context
        )

        # Step 7: Store memory
        memory.add_message(session_id, "user", query)
        memory.add_message(session_id, "assistant", answer)

        logger.info(f"[QueryPipeline] session_id={session_id} | Completed")

        return {
            "answer": answer,
            "sources": results
        }

    except Exception as e:
        logger.error(
            f"[QueryPipeline] session_id={session_id} | Failed | error={str(e)}"
        )
        return {"error": str(e)}


# ======================
# IMAGE QUERY
# ======================
def query_image(query: str, session_id: str):

    logger.debug(f"[QueryPipeline] session_id={session_id} | Image query")

    query_vector = embedder.embed_query(query)

    results = vector_store.search_text(query_vector, session_id="session_id")

    results = [
        r for r in results
        if r["metadata"].get("session_id") == session_id
    ]

    return results


# ======================
# AUDIO QUERY
# ======================
def query_audio(file_path: str) -> str:

    logger.info(f"[QueryPipeline] Audio query | file={file_path}")

    audio_model = model_loader.get_whisper()
    segments, _ = audio_model.transcribe(file_path)

    query_text = ""
    for segment in segments:
        query_text += segment.text + " "

    return query_text.strip()


# ======================
# VIDEO QUERY
# ======================
def query_video(file_path: str) -> str:

    logger.info(f"[QueryPipeline] Video query | file={file_path}")

    temp_audio = "temp_query_audio.wav"

    try:
        clip = VideoFileClip(file_path)
        clip.audio.write_audiofile(temp_audio)

        query_text_data = query_audio(temp_audio)

        return query_text_data.strip()

    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)