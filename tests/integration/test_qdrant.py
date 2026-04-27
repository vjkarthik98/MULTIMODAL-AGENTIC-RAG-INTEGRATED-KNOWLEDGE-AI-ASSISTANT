from app.vectorstore.qdrant_store import QdrantVectorStore
from app.core.config import settings

def test_qdrant_connection():

    vector_store = QdrantVectorStore()

    vector_store.create_collection("multimodal_rag")

    print("Collection created successfully!")

if __name__ == "__main__":

    test_qdrant_connection()