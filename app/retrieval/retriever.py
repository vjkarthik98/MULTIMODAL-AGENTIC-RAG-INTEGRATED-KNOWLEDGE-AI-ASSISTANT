from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder
from app.core.config import settings


class Retriever:

    def __init__(self):

        self.vector_store =  QdrantVectorStore()
        self.embedder = TextEmbedder()


    def retrieval(self, query: str, top_k: int = 3):

        # Step 1 : convert query -> embedding
        query_vector = self.embedder.embed_text(query)

        # Step 2: search vector DB
        results = self.vector_store.search(query_vector, limit=top_k)

        # Step 3: format output 
        documents = []

        for result in results:
            documents.append({
                "text": result["text"],
                "source": result["metadata"].get("source", "unknown"),
                "modality": result["metadata"].get("modality", "text"),
                "score": result["score"]
                })

        return documents