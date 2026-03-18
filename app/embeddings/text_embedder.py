from sentence_transformers import SentenceTransformer

from app.core.config import settings


class TextEmbedder:

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed_text(self, text: str):
        embedding = self.model.encode(text)
        return embedding.tolist()