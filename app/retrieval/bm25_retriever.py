from rank_bm25 import BM25Okapi
import numpy as np
from app.utils.logger import get_logger

logger = get_logger(__name__)

class BM25Retriever:
    def __init__(self):
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None


    # BUILD INDEX
    def build_index(self, documents):
        self.documents = documents
        corpus = [doc.text for doc in documents]

        self.tokenized_corpus = [
            text.lower().split() for text in corpus
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"[BM25] Index built | docs ={len(documents)}")

    # SEARCH
    def search(self, query, top_k = 10):

        if not self.bm25:
            logger.warning("[BM25] Index not built")
            return []
        
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            self.documents[i] for i in top_indices
        ]

        logger.info(f"[BM25] Retrieved top_k={top_k}")

        return results