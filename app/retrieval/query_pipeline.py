from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.clip_text_embedder import ClipTextEmbedder
from app.ingestion.audio_ingest import model
from moviepy.editor import VideoFileClip
from app.memory.redis_memory import RedisMemory
from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.memory.summarizer import summarize_conversation
from app.reasoning.reasoning_engine import ReasoningEngine
from app.reasoning.query_decomposer import QueryDecomposer
from app.llm.gguf_model import GGUFModel
from app.reasoning.result_fusion import ResultFusion


import os


embedder = TextEmbedder()
vector_store = QdrantVectorStore()
clip_embedder = ClipTextEmbedder()
llm = GGUFModel()
reasoning_engine = ReasoningEngine(llm)
decomposer = QueryDecomposer(llm)
fusion = ResultFusion(top_k=5)


# Detect Query
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

# TEXT QUERY
def query_text(query: str, session_id: str = "default"):
    print("NEW PIPELINE RUNNING")
    try:
        # Step 1: Get Memory (Redis)
        memory = RedisMemory()
        history = memory.get_history(session_id)

        # Step 2: Filter relevant history
        filtered_history = filter_relevant_history(
            query,
            history,
            embedder
        )

        # Step 3: Summarize if long
        summary = ""
        if len(history) > 6:
            summary = summarize_conversation(model, history)

        # Step 4: Build memory context
        memory_context = build_memory_context(
            summary,
            filtered_history
        )

        # Query Decomposition
        
        sub_queries = decomposer.decompose(query)

        print("SUB QUERIES GENERATED:", sub_queries)

        print("SUB-QUERIES", sub_queries)

        all_results = []

        # Multi-query retrieval
        for sub_q in sub_queries:
            sub_vector = embedder.embed_query(sub_q)
            sub_results = vector_store.search_text(sub_vector)

            all_results.extend(sub_results)

        # Deduplicate results
        seen = set()
        unique_results = []

        for r in all_results:
            text = r.get("text", "")
            if text not in seen:
                seen.add(text)
                unique_results.append(r)

        # Fusion + Ranking
        results = fusion.fuse(unique_results)

        print("FUSED RESULTS:", len(results))

  
        # Detect query type
        query_type = detect_query_type(query)

        # Filter by Modality
        if query_type == "image":
            results = [
                r for r in results
                if r["metadata"].get("modality") == "image"
            ]

        elif query_type == "audio":
            results = [
                r for r in results
                if r ["metadata"].get("modality") == "audio"
            ]
        elif query_type == "video":
            results = [
                r for r in results
                if r["metadata"].get("modality") == "video"
            ]

        if not results:
            return {
                "answer": "I couldn't find relevant information in the knowledge base. Please upload the data or ask a different question.",
                "sources": []
            }

        # Step 5: Generate Answer (LLM)
        answer = reasoning_engine.generate_answer(
            query=query,
            retrieved_docs=results,
            memory_context=memory_context
        )

        # Step 6: Store memory
        memory.add_message(session_id, "user", query)
        memory.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": results
        }
    
    except Exception as e:
        return {"error": str(e)}

# IMAGE QUERY
def query_image(query: str):
    query_vector = clip_embedder.embed(query)
    print("DEBUG IMAGE QUERY VECTOR LENGTH:", len(query_vector))

    return vector_store.search_image(query_vector)

# AUDIO QUERY
def query_audio(file_path: str) -> str:
    """
    Convert audio query -> run RAG
    """

    # Step 1: Transcribe audio
    segments, _ = model.transcribe(file_path)

    query_text = ""
    for segment in segments:
        query_text += segment.text + " "

    # Step 2: Run existing RAG pipeline
    return query_text.strip()

# VIDEO QUERY
def query_video(file_path: str) -> str:
    """
    Convert video -> audio ->
    """    

    temp_audio = "temp_query_audio.wav"

    try:
        # Step 1: Extract audio from video
        clip = VideoFileClip(file_path)
        clip.audio.write_audiofile(temp_audio)

        # Step 3: Reuse audio query
        query_text_data = query_audio(temp_audio)

        return query_text_data.strip()
    
    finally:
        # Cleaning
        if os.path.exists(temp_audio):
            os.remove(temp_audio)