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

        # Rerieval uses only Query
        query_vector = embedder.embed_query(query)

        # Detect query type
        query_type = detect_query_type(query)

        # Retrieve from vector DB
        results = vector_store.search_text(query_vector)

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

        # Step 5: Build cotext from retrieved docs
        retrieved_text = "\n".join(
            [doc.get("text", "") for doc in results]
        )

        # Step 6: Build Final Prompt
        final_prompt = f"""
You are an intelligent assistant.

Conversation Context:
{memory_context}

Knowledge Context:
{retrieved_text}

User Question:
{query}

Answer:
"""
        # Step 7: Generate Answer (LLM)
        answer = model.generate(final_prompt)

        # Step 8: Store memory
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