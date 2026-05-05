import time
import hashlib
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

        self.w_model = settings.RERANK_MODEL_WEIGHT
        self.w_fusion = settings.RERANK_FUSION_WEIGHT

        self.modality_weights = getattr(settings, "RERANK_MODALITY_WEIGHTS", {})

        logger.info(event="reranker_initialized")

    #  HASH 
    def _hash(self, doc: Dict) -> str:
        meta = doc.get("metadata", {}) or {}
        base = f"{meta.get('doc_id')}|{meta.get('chunk_id')}|{doc.get('text')[:200]}"
        return hashlib.sha256(base.encode()).hexdigest()

    #  NORMALIZE 
    def _normalize_query(self, q: str) -> str:
        return " ".join(q.strip().split())

    def _clean(self, text: str) -> str:
        return " ".join(text.split())

    #  CONTEXT 
    def _context(self, doc: Dict) -> str:

        meta = doc.get("metadata", {}) or {}
        structure = meta.get("structure", {}) or {}

        modality = meta.get("modality", "text")
        source = meta.get("source") or meta.get("source_type")

        content_type = meta.get("content_type", structure.get("content_type"))
        emb_space = meta.get("embedding_space", structure.get("embedding_space", "text"))

        header = []

        if source:
            header.append(f"[SRC:{str(source)[:20]}]")

        if modality:
            header.append(f"[{modality.upper()}]")

        if content_type:
            header.append(f"[{content_type.upper()}]")

        if emb_space == "vision":
            header.append("[VISION]")

        text = self._clean(doc.get("text", ""))[:self.context_chars]

        return " ".join(header) + " " + text

    #  FILTER 
    def _filter(self, docs: List[Dict]) -> List[Dict]:
        return [
            d for d in docs
            if d.get("text") and d.get("score", 0.0) > 0.01
        ]

    #  NORMALIZE SCORES 
    def _normalize_scores(self, scores: np.ndarray):

        if scores.size == 0:
            return scores

        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        min_s, max_s = scores.min(), scores.max()

        if max_s - min_s > 1e-6:
            return (scores - min_s) / (max_s - min_s)

        return np.clip(scores, 0.0, 1.0)

    #  MAIN 
    def rerank(self, query: str, documents: List[Dict], top_k: int = None):

        if not query or not documents:
            return []

        start = time.time()
        top_k = top_k or self.top_k

        try:
            query = self._normalize_query(query)

            docs = self._filter(documents)[:self.max_inputs]

            pairs = []
            valid_docs = []

            for d in docs:
                ctx = self._context(d)
                if not ctx:
                    continue

                pairs.append((
                    query[:settings.MAX_PROMPT_CHARS],
                    ctx
                ))
                valid_docs.append(d)

            if not pairs:
                return []

            #  MODEL 
            t_model = time.time()

            scores = np.asarray(self.model.predict(pairs)).reshape(-1)
            scores = self._normalize_scores(scores)

            model_latency = round(time.time() - t_model, 2)

            if scores.size != len(valid_docs):
                min_len = min(len(scores), len(valid_docs))
                scores = scores[:min_len]
                valid_docs = valid_docs[:min_len]

            #  FUSION 
            results = []

            for d, s in zip(valid_docs, scores):

                meta = d.get("metadata", {}) or {}
                structure = meta.get("structure", {}) or {}

                modality = meta.get("modality", "text")
                chunk_idx = structure.get("chunk_index", 0)

                fusion_score = min(float(d.get("score", 0.0)), 1.0)

                position_boost = 1.0 + (
                    settings.RERANK_POSITION_WEIGHT / (chunk_idx + 2)
                )

                modality_boost = self.modality_weights.get(modality, 1.0)

                final_score = (
                    self.w_model * float(s) +
                    self.w_fusion * fusion_score
                ) * position_boost * modality_boost

                results.append({
                    "text": d["text"][:settings.RAG_DOC_MAX_CHARS],
                    "metadata": meta,
                    "score": float(final_score),
                })

            results.sort(key=lambda x: x["score"], reverse=True)

            #  STRONG DEDUP 
            final = []
            seen = set()

            for r in results:
                h = self._hash(r)
                if h in seen:
                    continue

                seen.add(h)
                final.append(r)

                if len(final) >= top_k:
                    break

            logger.info(
                event="rerank_success",
                output=len(final),
                latency=round(time.time() - start, 2),
                model_latency=model_latency
            )

            return final

        except Exception as e:
            logger.error(event="rerank_failed", error=str(e))

            # fallback
            fallback = sorted(
                documents,
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )

            return fallback[:top_k]