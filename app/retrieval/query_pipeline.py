from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.clip_text_embedder import ClipTextEmbedder
from app.ingestion.audio_ingest import model
from moviepy.editor import VideoFileClip
import os


embedder = TextEmbedder()
vector_store = QdrantVectorStore()
clip_embedder = ClipTextEmbedder()


# TEXT QUERY
def query_text(query: str):
    query_vector = embedder.embed_query(query)
    return vector_store.search_text(query_vector)

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