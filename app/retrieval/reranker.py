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

    #  NORMALIZE QUERY 
    def _normalize_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  CLEAN CONTEXT 
    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())

    #  BUILD CONTEXT 
    def _build_context(self, document: Dict) -> str:

        metadata = document.get("metadata", {}) or {}
        structure = metadata.get("structure", {}) or {}

        modality = metadata.get("modality", "text")
        source = metadata.get("source_type") or metadata.get("source")

        content_type = metadata.get("content_type", structure.get("content_type"))
        embedding_space = metadata.get(
            "embedding_space",
            structure.get("embedding_space", "text")
        )

        header = []

        if source:
            header.append(f"[SRC:{str(source)[:20]}]")

        if modality:
            header.append(f"[{modality.upper()}]")

        if content_type:
            header.append(f"[{content_type.upper()}]")

        if embedding_space == "vision":
            header.append("[VISION]")

        text = self._clean_text(document.get("text", ""))[:self.context_chars]

        return " ".join(header) + " " + text

    #  FILTER LOW QUALITY 
    def _filter_candidates(self, docs: List[Dict]) -> List[Dict]:
        return [d for d in docs if d.get("score", 0.0) > 0.02 and d.get("text")]

    #  MAIN 
    def rerank(self, query: str, documents: List[Dict], top_k: int = None):

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not documents:
            return []

        start = time.time()
        top_k = top_k or self.top_k

        try:
            query = self._normalize_query(query)

            documents = self._filter_candidates(documents)
            documents = documents[:self.max_inputs]

            pairs = []
            valid_docs = []

            for doc in documents:
                context = self._build_context(doc)

                if not context:
                    continue

                pairs.append((
                    query[:settings.MAX_PROMPT_CHARS],
                    context[:self.context_chars]
                ))

                valid_docs.append(doc)

            if not pairs:
                return []

            #  MODEL 
            t_model = time.time()

            scores = np.asarray(self.model.predict(pairs)).reshape(-1)

            model_latency = round(time.time() - t_model, 2)

            if not np.isfinite(scores).all():
                scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

            if scores.size != len(valid_docs):
                scores = scores[: len(valid_docs)]
                valid_docs = valid_docs[: len(scores)]

            #  NORMALIZATION 
            min_s, max_s = scores.min(), scores.max()

            if max_s - min_s > 1e-6:
                scores = (scores - min_s) / (max_s - min_s)
            else:
                scores = np.clip(scores, 0.0, 1.0)

            results = []

            for doc, score in zip(valid_docs, scores):

                metadata = doc.get("metadata", {}) or {}
                structure = metadata.get("structure", {}) or {}

                modality = metadata.get("modality", "text")
                chunk_index = structure.get("chunk_index", 0)

                fusion_score = float(doc.get("score", 0.0))

                # BALANCED FUSION (CLIPPED)
                fusion_score = min(fusion_score, 1.0)

                position_boost = 1.0 + (
                    settings.RERANK_POSITION_WEIGHT / (chunk_index + 2)
                )

                modality_boost = self.modality_weights.get(modality, 1.0)

                final_score = (
                    self.score_weight * float(score) +
                    self.fusion_weight * fusion_score
                ) * position_boost * modality_boost

                results.append(
                    {
                        "text": doc["text"][:settings.RAG_DOC_MAX_CHARS],
                        "metadata": metadata,
                        "score": float(final_score),
                    }
                )

            results.sort(key=lambda x: x["score"], reverse=True)

            #  STRONG DEDUP 
            final = []
            seen = set()

            for r in results:
                meta = r["metadata"]

                key = (
                    meta.get("doc_id"),
                    meta.get("chunk_id"),
                    meta.get("source"),
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
                "[Reranker] success | output=%s latency=%ss model=%ss",
                len(final),
                latency,
                model_latency
            )

            return final

        except Exception as e:
            logger.error("[Reranker] failed | %s", str(e))

            # SMART FALLBACK (SORT BY ORIGINAL SCORE)
            fallback = sorted(
                documents,
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )

            return fallback[:top_k]