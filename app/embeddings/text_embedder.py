from sentence_transformers import SentenceTransformer

from app.core.config import settings


class TextEmbedder:

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed_text(self, text: str):
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_documents(self, documents):
        texts = [doc.text for doc in documents]

        embeddings = self.model.encode(
            texts,
            batch_size=32, 
            show_progress_bar=True
        )

        for i, emb in enumerate(embeddings):
            documents[i].embedding = emb.tolist()

        return documents