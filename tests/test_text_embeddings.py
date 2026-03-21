from app.vectorstore.qdrant_store import QdrantVectorStore

store = QdrantVectorStore()

store.create_collection("multimodal_rag")

docs = [
    {
        "text": "AI is transforming industries",
        "embedding": [0.1] * 384,
        "metadata": {"source": "test"}
    }
]

store.insert_documents(docs)
print("Inserted successfully")