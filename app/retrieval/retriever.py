from app.vectorstore.qdrant_store import QdrantVectorStore
from app.core.model_loader import model_loader
import logging

# Logger
logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self):
        self.vector_store = QdrantVectorStore()
        self.embedder = model_loader.get_embedder()
        self.reranker = model_loader.get_reranker()

    # -----------------------
    # QUERY REWRITING
    # -----------------------
    def _rewrite_query(self, query: str) -> str:
        q = query.lower()

        if "video about" in q or "what is the video about" in q:
            return "motivation speech encouragement message meaning topic"

        if "what is happening" in q:
            return "scene action activity people doing"

        if "who" in q:
            return "person people man woman speaker"

        if "describe" in q:
            return "description scene objects people environment"

        return query

    # -----------------------
    # RERANKING
    # -----------------------
    def _rerank(self, query: str, results: list, top_k: int):
        if not results:
            logger.debug("[Retriever] No results to rerank")
            return []

        pairs = [(query, r["text"]) for r in results]
        scores = self.reranker.predict(pairs)

        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])

        results = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        logger.debug(f"[Retriever] Reranking completed | top_k={top_k}")

        return results[:top_k]

    # -----------------------
    # MAIN RETRIEVAL
    # -----------------------
    def retrieval(self, query: str, top_k: int = 5, source: str = None):

        try:
            logger.info(f"[Retriever] Retrieval started | query={query}")

            # Step 1: Rewrite
            rewritten_query = self._rewrite_query(query)

            logger.debug(f"[Retriever] Rewritten query={rewritten_query}")

            query_vector = self.embedder.embed_query(rewritten_query)

            # Step 2: Retrieve
            results = self.vector_store.search_text(
                query_vector,
                limit=top_k * 3,
                source_filter=source
            )

            logger.debug(f"[Retriever] Raw results count={len(results)}")

            # Step 3: Rerank
            results = self._rerank(query, results, top_k * 2)

            # Step 4: Separate modalities
            audio_docs, frame_docs, text_docs = [], [], []

            for r in results:
                modality = r["metadata"].get("modality", "text")

                if modality in ["audio", "video_audio"]:
                    audio_docs.append(r)

                elif modality == "video_frame":
                    frame_docs.append(r)

                else:
                    text_docs.append(r)

            # Step 5: Balanced selection
            if audio_docs:
                final_results = (
                    audio_docs[:3] +
                    frame_docs[:2] +
                    text_docs[:1]
                )
            elif frame_docs:
                final_results = frame_docs[:top_k]
            else:
                final_results = text_docs[:top_k]

            logger.info(
                f"[Retriever] Retrieval completed | final_count={len(final_results)}"
            )

            return final_results

        except Exception as e:
            logger.error(f"[Retriever] Failed | error={str(e)}")
            return []