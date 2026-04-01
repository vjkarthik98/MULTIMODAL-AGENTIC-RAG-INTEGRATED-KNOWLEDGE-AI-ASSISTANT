from sentence_transformers import SentenceTransformer

from app.core.config import settings


class TextEmbedder:

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

        print(f" Using embedding model: {settings.EMBEDDING_MODEL}")

    # Single Text
    def embed_text(self, text: str):
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    # Document Embedding
    def embed_documents(self, documents):
        texts = [doc.text for doc in documents]

        embeddings = self.model.encode(
            texts,
            batch_size=32, 
            show_progress_bar=True
        )

        for i, emb in enumerate(embeddings):
            documents[i].embedding = emb.tolist()

            print(f" Embedded {len(documents)} documents")

        return documents
    
    # Query Embedding
    def embed_query(self, query: str):
        embedding = self.model.encode(query)

        print(" Query embedding generated")

        return embedding.tolist()