from app.embeddings.text_embedder import TextEmbedder
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.clip_text_embedder import ClipTextEmbedder


embedder = TextEmbedder()
vector_store = QdrantVectorStore()
clip_embedder = ClipTextEmbedder()

def query_text(query: str):
    query_vector = embedder.embed_text(query)
    return vector_store.search_text(query_vector)

def query_image(query: str):
    query_vector = clip_embedder.embed(query)
    print("DEBUG IMAGE QUERY VECTOR LENGTH:", len(query_vector))

    return vector_store.search_image(query_vector)
    