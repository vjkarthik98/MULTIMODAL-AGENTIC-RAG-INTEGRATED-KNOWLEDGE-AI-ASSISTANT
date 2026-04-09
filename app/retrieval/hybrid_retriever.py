from app.utils.logger import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self, bm25_retriever, vector_store, embedder):
        self.bm25 = bm25_retriever
        self.vector_store = vector_store
        self.embedder = embedder 

    def search(self, query, top_k=10):
        try:
            # BM25 SEARCH
            bm25_results = self.bm25.search(query, top_k=top_k)

            # VECTOR SEARCH

            query_vector = self.embedder.embed_query(query)
            vector_results = self.vector_store.search_text(
                query_vector,
                session_id="default"
            )

            # MERGE RESULTS
            combined = []

            # Add BM25 results
            for doc in bm25_results:
                combined.append({
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "score": 1.0
                })
            # Add Vector results
            for doc in vector_results:
                combined.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": doc.get("score", 0.5)
                })

            # 4. REMOVE DUPLICATES
            seen = set()
            unique_results = []

            for item in combined:
                key = item["text"]

                if key not in seen:
                    seen.add(key)
                    unique_results.append(item)
            logger.info(f"[HybridRetriever] Combined results = {len(unique_results)}")

            return unique_results[:top_k]
    
        except Exception as e:
            logger.error(f"[HybridRetriever] Failed | error = {str(e)}")
            return []