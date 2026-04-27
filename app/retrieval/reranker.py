import time
from typing import List, Dict

import numpy as np

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger


logger = get_logger(__name__)


class Reranker:

    def __init__(self):
        self.model = model_loader.get_reranker()

        self.top_k = settings.RERANK_TOP_K
        self.max_inputs = settings.RERANK_MAX_INPUT
        self.context_chars = settings.RERANK_CONTEXT_MAX_CHARS

        self.score_weight = settings.RERANK_MODEL_WEIGHT
        self.fusion_weight = settings.RERANK_FUSION_WEIGHT

        self.modality_weights = getattr(settings, "RERANK_MODALITY_WEIGHTS", {})

        logger.info("[Reranker] initialized")

    # BUILD CONTEXT 
    def _build_context(self, document: Dict) -> str:
        metadata = document.get("metadata", {}) or {}
        structure = metadata.get("structure", {}) or {}

        modality = metadata.get("modality", "text")
        source = metadata.get("source_type") or metadata.get("source")
        page = metadata.get("page")

        content_type = metadata.get("content_type", structure.get("content_type"))
        embedding_space = metadata.get(
            "embedding_space",
            structure.get("embedding_space", "text")
        )

        header = []

        if source:
            header.append(f"[SOURCE: {str(source).upper()}]")
        if page:
            header.append(f"[PAGE: {page}]")

        if modality == "table":
            header.append("[TABLE]")
        elif modality == "image":
            header.append("[IMAGE]")
        elif modality == "audio":
            header.append("[AUDIO]")
        elif modality == "video":
            if content_type == "video_speech":
                header.append("[VIDEO SPEECH]")
            elif content_type == "video_frame":
                header.append("[VIDEO FRAME]")

        if embedding_space == "vision":
            header.append("[VISION MATCH]")

        text = str(document.get("text", ""))[:self.context_chars]

        return "\n".join(header) + "\n" + text

    # MAIN 
    def rerank(self, query: str, documents: List[Dict], top_k: int = None):

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not documents:
            return []

        start = time.time()
        top_k = top_k or self.top_k

        try:
            documents = documents[:self.max_inputs]

            pairs = []
            valid_docs = []

            for doc in documents:
                text = doc.get("text", "")
                if not text:
                    continue

                context = self._build_context(doc)

                pairs.append((
                    query[:settings.MAX_PROMPT_CHARS],
                    context[:self.context_chars]
                ))
                valid_docs.append(doc)

            if not pairs:
                return []

            # MODEL 
            scores = np.asarray(self.model.predict(pairs)).reshape(-1)

            if not np.isfinite(scores).all():
                logger.warning("[Reranker] invalid scores detected")
                scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

            if scores.size != len(valid_docs):
                scores = scores[: len(valid_docs)]
                valid_docs = valid_docs[: len(scores)]

            # ROBUST NORMALIZATION 
            min_s, max_s = scores.min(), scores.max()

            if max_s - min_s > 1e-6:
                scores = (scores - min_s) / (max_s - min_s)
            else:
                # preserve signal instead of flattening
                scores = np.clip(scores, 0.0, 1.0)

            results = []

            for doc, score in zip(valid_docs, scores):

                metadata = doc.get("metadata", {}) or {}
                structure = metadata.get("structure", {}) or {}

                modality = metadata.get("modality", "text")
                chunk_index = structure.get("chunk_index", 0)
                fusion_score = doc.get("score", 0.0)

                # SAFER POSITION BOOST 
                position_boost = 1.0 + (
                    settings.RERANK_POSITION_WEIGHT * (1.0 / (chunk_index + 2))
                )

                modality_boost = self.modality_weights.get(modality, 1.0)

                # BALANCED SCORING 
                final_score = (
                    self.score_weight * float(score) +
                    self.fusion_weight * float(fusion_score)
                ) * position_boost * modality_boost

                results.append(
                    {
                        "text": doc["text"][:settings.RAG_DOC_MAX_CHARS],
                        "metadata": metadata,
                        "score": float(final_score),
                    }
                )

            results.sort(key=lambda x: x["score"], reverse=True)

            # DEDUP 
            final = []
            seen = set()

            for r in results:
                key = (
                    r["metadata"].get("doc_id"),
                    r["metadata"].get("chunk_id"),
                    r["text"][:200],
                )

                if key in seen:
                    continue

                seen.add(key)
                final.append(r)

                if len(final) >= top_k:
                    break

            latency = round(time.time() - start, 2)

            logger.info(
                "[Reranker] success | output=%s latency=%ss",
                len(final),
                latency
            )

            return final

        except Exception as e:
            logger.error("[Reranker] failed | %s", str(e))

            return [
                {
                    "text": d.get("text", ""),
                    "metadata": d.get("metadata", {}),
                    "score": d.get("score", 0.0),
                }
                for d in documents[:top_k]
            ]