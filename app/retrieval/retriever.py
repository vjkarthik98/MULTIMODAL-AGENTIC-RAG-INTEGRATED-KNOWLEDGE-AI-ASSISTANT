import time
from typing import List, Dict

from app.core.config import settings
from app.core.model_loader import model_loader

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.vectorstore.qdrant_store import QdrantVectorStore

from app.ingestion.pipeline import pipeline 
from app.utils.logger import get_logger


logger = get_logger(__name__)


class _NullVectorStore:
    def search_text(self, *args, **kwargs):
        return []

    def search_vision(self, *args, **kwargs):
        return []


class Retriever:
    def __init__(self):
        try:
            self.bm25 = pipeline.bm25
            logger.info(f"[Retriever INIT] BM25 docs={len(self.bm25.documents)}")
            self.vector_store = pipeline.vector_store
        except Exception:
            self.bm25 = BM25Retriever()
            try:
                self.vector_store = QdrantVectorStore()
            except Exception:
                self.vector_store = _NullVectorStore()

        self.embedder = model_loader.get_embedder()
        self.clip_text_embedder = model_loader.get_clip_text_embedder()

        self.reranker = Reranker()

        self.hybrid = HybridRetriever(
            self.bm25,
            self.vector_store,
            self.embedder,
            self.clip_text_embedder,
        )

    # QUERY REWRITE 
    def _rewrite_query(self, query: str) -> str:
        lowered = query.lower()
        hints = []

        if "video" in lowered:
            hints.append("speech scene action visual description")
        if "who" in lowered:
            hints.append("person speaker individual")
        if "describe" in lowered:
            hints.append("visual objects environment")
        if "what did they say" in lowered or "speech" in lowered:
            hints.append("spoken content transcript")
        if "when" in lowered:
            hints.append("timestamp event time")

        if not hints:
            return query

        rewritten = f"{query} {' '.join(hints)}"
        return rewritten[:settings.MAX_PROMPT_CHARS]

    # RERANK 
    def _rerank(self, query: str, results: List[Dict], top_k: int):
        if not results:
            return []

        try:
            return self.reranker.rerank(query, results, top_k=top_k)
        except Exception as e:
            logger.warning("[Retriever] Rerank fallback | %s", str(e))
            return results[:top_k]

    # DEDUP 
    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []

        for r in results:
            metadata = r.get("metadata", {})
            key = (
                r.get("text", "")[:300], 
                str(metadata.get("doc_id")),
                str(metadata.get("chunk_id")),
            )

            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    # FINAL NORMALIZATION 
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results

        max_score = max(r.get("score", 0.0) for r in results) or 1.0

        for r in results:
            r["score"] = r.get("score", 0.0) / (max_score + 1e-6)

        return results

    # CROSS MODAL FUSION 
    def _cross_modal_fusion(self, results: List[Dict]) -> List[Dict]:
        fused = []
        used = set()

        for i, r in enumerate(results):
            if i in used:
                continue

            metadata = r.get("metadata", {})
            modality = metadata.get("modality")
            content_type = metadata.get("content_type")

            if modality == "video" and content_type == "video_speech":
                segment = metadata.get("segment_index")

                combined_text = r.get("text", "")
                combined_score = r.get("score", 0.0)

                for j, other in enumerate(results):
                    if j == i:
                        continue

                    other_meta = other.get("metadata", {})

                    if (
                        other_meta.get("modality") == "video"
                        and other_meta.get("content_type") == "video_frame"
                        and other_meta.get("linked_segment_index") == segment
                    ):
                        combined_text += " | Visual: " + other.get("text", "")
                        combined_score = max(combined_score, other.get("score", 0.0))
                        used.add(j)

                fused.append({
                    "text": combined_text,
                    "metadata": metadata,  
                    "score": combined_score
                })
                used.add(i)

            else:
                fused.append(r)

        return fused

    # MAIN 
    def retrieval(
        self,
        query: str,
        session_id: str = "default",
        top_k: int = None
    ) -> List[Dict]:
        logger.info(f"[Retriever] BM25 docs at query={len(self.bm25.documents)}")

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not session_id:
            raise ValueError("session_id required")

        top_k = top_k or settings.DEFAULT_TOP_K
        candidate_k = top_k * settings.HYBRID_CANDIDATES_MULTIPLIER

        start = time.time()

        try:
            logger.info("[Retriever][START] session_id=%s", session_id)

            rewritten_query = self._rewrite_query(query)

            # HYBRID 
            results = self.hybrid.search(
                query=rewritten_query,
                session_id=session_id,
                top_k=candidate_k
            )

            if not results:
                logger.warning("[Retriever] No hybrid results")
                return []

            logger.info("[Retriever] hybrid_results=%s", len(results))

            # DEDUP 
            results = self._deduplicate(results)

        
            # RERANK 
            results = self._rerank(query, results, top_k=top_k * 2)

            logger.info("[Retriever] after_rerank=%s", len(results))

            # FUSION 
            results = self._cross_modal_fusion(results)

            # FINAL NORMALIZATION
            results = self._normalize_scores(results)

            final_results = results[:top_k]

            latency = round(time.time() - start, 2)

            logger.info(
                "[Retriever][SUCCESS] session_id=%s | results=%s | latency=%.2fs",
                session_id,
                len(final_results),
                latency
            )

            return final_results

        except Exception as e:
            logger.error("[Retriever][FAILED] session_id=%s | error=%s", session_id, str(e))
            return []