import time

from app.embeddings.clip_text_embedder import ClipTextEmbedder
from app.embeddings.text_embedder import TextEmbedder
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.utils.logger import get_logger
from app.vectorstore.qdrant_store import QdrantVectorStore


logger = get_logger(__name__)


class _NullVectorStore:
    def search_text(self, query_vector, limit=5, session_id=None):
        return []

    def search_vision(self, query_vector, limit=5, session_id=None):
        return []


class Retriever:
    def __init__(self):
        try:
            from app.ingestion.pipeline import bm25 as shared_bm25
            from app.ingestion.pipeline import vector_store as shared_vector_store

            self.bm25 = shared_bm25
            self.vector_store = shared_vector_store
        except Exception:
            self.bm25 = BM25Retriever()
            try:
                self.vector_store = QdrantVectorStore()
            except Exception:
                self.vector_store = _NullVectorStore()

        self.embedder = TextEmbedder()
        self.clip_text_embedder = ClipTextEmbedder()
        self.reranker = Reranker()
        self.hybrid = HybridRetriever(
            self.bm25,
            self.vector_store,
            self.embedder,
            self.clip_text_embedder,
        )

    def _rewrite_query(self, query: str) -> str:
        lowered = query.lower()
        hints = []

        if "video about" in lowered:
            hints.append("speech scenes explanation")
        if "what is happening" in lowered:
            hints.append("scene action activity visual description")
        if "who" in lowered:
            hints.append("person speaker individual face")
        if "describe" in lowered:
            hints.append("visual scene objects environment")
        if "what did they say" in lowered or "speech" in lowered:
            hints.append("spoken content transcript dialogue")
        if "at what time" in lowered or "when" in lowered:
            hints.append("timestamp event time moment")
        if "show frame" in lowered or "image from video" in lowered:
            hints.append("video frame snapshot")

        if not hints:
            return query
        return f"{query} {' '.join(hints)}"

    def _rerank(self, query: str, results: list, top_k: int):
        if not results:
            logger.warning("[Retriever] No results to rerank")
            return []

        try:
            reranked = self.reranker.rerank(query, results, top_k=top_k)
            return reranked[:top_k]
        except Exception as exc:
            logger.error("[Retriever] Rerank failed | error=%s", exc)
            return results[:top_k]

    def _cross_modal_fusion(self, results):
        fused_results = []
        used_indices = set()

        for index, result in enumerate(results):
            if index in used_indices:
                continue

            metadata = result.get("metadata", {})
            modality = metadata.get("modality")
            content_type = metadata.get("content_type")

            if modality == "video" and content_type == "video_speech":
                segment_index = metadata.get("segment_index")
                combined_text = result["text"]
                combined_score = result.get("score", 0.0)

                for other_index, other in enumerate(results):
                    if other_index == index:
                        continue

                    other_metadata = other.get("metadata", {})
                    if (
                        other_metadata.get("modality") == "video"
                        and other_metadata.get("content_type") == "video_frame"
                        and other_metadata.get("linked_segment_index") == segment_index
                    ):
                        combined_text += " | Visual: " + other["text"]
                        combined_score = max(combined_score, other.get("score", 0.0))
                        used_indices.add(other_index)

                fused_results.append(
                    {
                        "text": combined_text,
                        "metadata": metadata,
                        "score": combined_score,
                    }
                )
                used_indices.add(index)
            else:
                fused_results.append(result)

        return fused_results

    def retrieval(self, query: str, session_id: str = "default", top_k: int = 5):
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if not session_id:
            raise ValueError("session_id required")

        start_time = time.time()

        try:
            logger.info("[Retriever][START] session_id=%s | query=%s", session_id, query)
            rewritten_query = self._rewrite_query(query)

            results = self.hybrid.search(
                query=rewritten_query,
                session_id=session_id,
                top_k=top_k * 3,
            )
            if not results:
                logger.warning("[Retriever] No results from hybrid retrieval")
                return []

            unique_results = {}
            for result in results:
                metadata = result.get("metadata", {})
                key = (
                    result.get("text"),
                    str(metadata.get("doc_id")),
                    str(metadata.get("source")),
                    str(metadata.get("chunk_id")),
                    str(metadata.get("embedding_space")),
                )
                if key not in unique_results:
                    unique_results[key] = result

            results = list(unique_results.values())
            scores = [result.get("score", 0.0) for result in results]
            if scores:
                max_score = max(scores) + 1e-6
                for result in results:
                    result["score"] = result.get("score", 0.0) / max_score

            results = self._rerank(query, results, top_k=top_k * 2)
            results = self._cross_modal_fusion(results)

            text_docs = []
            image_docs = []
            audio_docs = []
            video_docs = []

            for result in results:
                modality = result.get("metadata", {}).get("modality", "text")
                if modality == "video":
                    video_docs.append(result)
                elif modality == "image":
                    image_docs.append(result)
                elif modality == "audio":
                    audio_docs.append(result)
                else:
                    text_docs.append(result)

            def video_priority(document):
                content_type = document.get("metadata", {}).get("content_type")
                if content_type == "video_speech":
                    return 2
                if content_type == "video_frame":
                    return 1
                return 0

            video_docs = sorted(video_docs, key=video_priority, reverse=True)

            final_results = []
            text_index = image_index = audio_index = video_index = 0

            while len(final_results) < top_k:
                appended = False

                if video_index < len(video_docs):
                    final_results.append(video_docs[video_index])
                    video_index += 1
                    appended = True
                    if len(final_results) >= top_k:
                        break

                if text_index < len(text_docs):
                    final_results.append(text_docs[text_index])
                    text_index += 1
                    appended = True
                    if len(final_results) >= top_k:
                        break

                if audio_index < len(audio_docs):
                    final_results.append(audio_docs[audio_index])
                    audio_index += 1
                    appended = True
                    if len(final_results) >= top_k:
                        break

                if image_index < len(image_docs):
                    final_results.append(image_docs[image_index])
                    image_index += 1
                    appended = True

                if not appended:
                    break

            if not final_results:
                final_results = results[:top_k]

            latency = time.time() - start_time
            logger.info(
                "[Retriever][SUCCESS] session_id=%s | results=%s | latency=%.2fs",
                session_id,
                len(final_results),
                latency,
            )
            return final_results

        except Exception as exc:
            logger.error("[Retriever][FAILED] session_id=%s | error=%s", session_id, exc)
            return []
