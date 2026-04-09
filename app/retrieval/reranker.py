from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)

class Reranker:
    def __init__(self):
        # Load model from ModelLoader
        self.model = model_loader.get_reranker()
        logger.info("[Reranker] Loaded from ModelLoader")
        
    def rerank(self, query, documents, top_k=5):
        try:
            if not documents:
                logger.warning("[Reranker] No documents to rerank")
                return []
            
            # Prepare query-doc pairs
            pairs = []
            valid_docs = []

            for doc in documents:
                text = doc.get("text", "")
                if text:
                    pairs.append((query, text))
                    valid_docs.append(doc)

                if not pairs:
                    logger.warning("[Reranker] No valid text found")
                    return []
                
                # Predict Scores
                scores = self.model.predict(pairs)

                # Attach scores
                scored_docs = list(zip(valid_docs, scores))

                # Sort by score(descending)
                scored_docs.sort(key=lambda x:x[1], reverse=True)

                # Final Output
                reranked = [
                    {
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {}),
                        "score": float(score)
                    }
                    for doc, score in scored_docs[:top_k]
                ]
                logger.info(f"[Reranker] Top {top_k} documents selected")

                return reranked
            
        except Exception as e:
            logger.error(f"[Reranked] Failed | error = {str(e)}")
            return documents[::top_k]