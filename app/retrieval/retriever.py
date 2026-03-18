from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder
from app.core.config import settings


class Retriever:

    def __init__(self):

        self.vector_store =  QdrantVectorStore()
        self.embedder = TextEmbedder()


    def retrieval(self, query: str, top_k: int = 3):

        # convert query -> embedding
        query_vector = self.embedder.embed_text(query)

        # search vector DB
        results = self.vector_store.search_vector(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )

        documents = []

        for result in results:
            documents.append({
                "text": result.payload["text"],
                "source": result.payload.get("source", "unknown"),
                "modality": result.payload.get("modality", "text"),
                "score": result.score
                })

        return documents