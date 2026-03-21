from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder

store = QdrantVectorStore()
embedder = TextEmbedder()

query = "What is Artificial Intelligence?"

query_vector = embedder.embed_text(query)

results = store.search(query_vector)

for r in results:
    print("\nResult:")
    print("Score:", r["score"])
    print("Text:", r["text"])