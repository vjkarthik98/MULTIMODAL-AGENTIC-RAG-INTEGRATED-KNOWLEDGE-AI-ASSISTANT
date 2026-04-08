from sentence_transformers import SentenceTransformer
from app.utils.logger import get_logger
import torch

logger = get_logger(__name__)


class TextEmbedder:
    def __init__(self, model_name: str):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        logger.info(
            f"[TextEmbedder] Model loaded: {self.model_name} | device={self.device}"
        )

    # Single Text
    def embed_text(self, text: str):
        embedding = self.model.encode(text)

        logger.debug("[TextEmbedder] Single text embedded")

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

        logger.debug(
            f"[TextEmbedder] Embedded documents count={len(documents)}"
        )

        return documents

    # Query Embedding
    def embed_query(self, query: str):
        embedding = self.model.encode(query)

        logger.debug("[TextEmbedder] Query embedding generated")

        return embedding.tolist()