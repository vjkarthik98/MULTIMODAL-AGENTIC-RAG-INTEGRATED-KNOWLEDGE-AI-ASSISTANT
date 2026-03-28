from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder
from app.core.config import settings


class Retriever:

    def __init__(self):

        self.vector_store =  QdrantVectorStore()
        self.embedder = TextEmbedder()


    def retrieval(self, query: str, top_k: int = 5):

        # Step 1 : convert query -> embedding
        query_vector = self.embedder.embed_text(query)

        # Step 2: search only text collection
        # (text + audio + video all stored here)
        results = self.vector_store.search_text(
            query_vector, 
            limit=top_k
        )

        # Step 3: Return Results     
        return results