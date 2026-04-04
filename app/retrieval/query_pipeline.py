from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.clip_text_embedder import ClipTextEmbedder
from app.ingestion.audio_ingest import model
from moviepy.editor import VideoFileClip
from app.memory.redis_memory import RedisMemory
from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.memory.summarizer import summarize_conversation
import os


embedder = TextEmbedder()
vector_store = QdrantVectorStore()
clip_embedder = ClipTextEmbedder()


# TEXT QUERY
def query_text(query: str, session_id: str = "default"):
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

        # Step 5: Create final query
        final_query = query
        if memory_context:
            final_query = f"""
{memory_context}

User Question:
{query}
"""
        # Step 6: Embed + Search
        query_vector = embedder.embed_query(final_query)
        results = vector_store.search_text(query_vector)

        # Step 7: Store memory
        memory.add_message(session_id, "user", query)
        memory.add_message(session_id, "assistant", str(results))

        return results
    
    except Exception as e:
        return {"errro": str(e)}

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