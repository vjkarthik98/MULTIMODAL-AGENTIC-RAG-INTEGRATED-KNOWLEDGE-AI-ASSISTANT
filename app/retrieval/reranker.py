import hashlib
import math
import time
from typing import Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:

    def __init__(self) -> None:
        self.model         = model_loader.get_reranker()
        self.top_k         = settings.RERANK_TOP_K
        self.max_inputs    = settings.RERANK_MAX_INPUT
        self.context_chars = settings.RERANK_CONTEXT_MAX_CHARS
        self.w_model       = settings.RERANK_MODEL_WEIGHT
        self.w_fusion      = settings.RERANK_FUSION_WEIGHT
        self.modality_weights = settings.RERANK_MODALITY_WEIGHTS

        logger.info(event="reranker_initialized")

    # HASH

    def _hash(self, doc: Dict) -> str:
        meta = doc.get("metadata", {}) or {}
        base = f"{meta.get('doc_id')}|{meta.get('chunk_id')}|{doc.get('text', '')[:200]}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # NORMALIZE

    def _normalize_query(self, q: str) -> str:
        return " ".join(q.strip().split())

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())

    # CONTEXT BUILDER

    def _context(self, doc: Dict) -> str:
        meta      = doc.get("metadata", {}) or {}
        structure = meta.get("structure", {}) or {}

        modality     = meta.get("modality", "text")
        source       = meta.get("source") or meta.get("source_type")
        content_type = meta.get("content_type") or structure.get("content_type")
        emb_space    = meta.get("embedding_space") or structure.get("embedding_space", "text")
        page         = meta.get("page") or structure.get("page")

        header: List[str] = []

        if source:
            header.append(f"[SRC:{str(source)[:20]}]")

        if modality:
            header.append(f"[{modality.upper()}]")

        if content_type:
            header.append(f"[{str(content_type).upper()}]")

        if emb_space == "vision":
            header.append("[VISION]")

        if page:
            header.append(f"[PG:{page}]")

        text = self._clean_text(doc.get("text", ""))[:self.context_chars]

        return (" ".join(header) + " " + text).strip()

    # FILTER

    def _filter(self, docs: List[Dict]) -> List[Dict]:
        return [
            d for d in docs
            if d.get("text") and d.get("score", 0.0) > 0.01
        ]

    # SCORE NORMALIZATION

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores

        scores   = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        min_s    = scores.min()
        max_s    = scores.max()

        if max_s - min_s > 1e-6:
            return (scores - min_s) / (max_s - min_s)

        return np.clip(scores, 0.0, 1.0)

    # FINAL SCORE VALID

    def _valid_score(self, score: float) -> bool:
        return not (math.isnan(score) or math.isinf(score)) and score > 0.0

    # DEDUP

    def _dedup(self, results: List[Dict], top_k: int) -> List[Dict]:
        seen:  set       = set()
        final: List[Dict] = []

        for r in results:
            h = self._hash(r)
            if h in seen:
                continue
            seen.add(h)
            final.append(r)

            if len(final) >= top_k:
                break

        return final

    # MAIN

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        session_id: str = "",
    ) -> List[Dict]:

        if not query or not documents:
            return []

        start   = time.time()
        top_k   = top_k or self.top_k
        query   = self._normalize_query(query)

        try:
            docs        = self._filter(documents)[:self.max_inputs]
            input_count = len(documents)
            filtered    = len(docs)

            pairs:      List         = []
            valid_docs: List[Dict]   = []

            for d in docs:
                ctx = self._context(d)
                if not ctx:
                    continue

                pairs.append((
                    query[:self.context_chars],
                    ctx,
                ))
                valid_docs.append(d)

            if not pairs:
                logger.warning(
                    event="rerank_no_pairs",
                    session_id=session_id,
                )
                return documents[:top_k]

            # CROSS-ENCODER INFERENCE
            t_model = time.time()

            scores        = np.asarray(self.model.predict(pairs)).reshape(-1)
            scores        = self._normalize_scores(scores)
            model_latency = round(time.time() - t_model, 2)

            if model_latency * 1000 > settings.LATENCY_TARGET_CROSS_MODAL_MS:
                logger.warning(
                    event="rerank_latency_exceeded",
                    latency_ms=round(model_latency * 1000, 1),
                    target_ms=settings.LATENCY_TARGET_CROSS_MODAL_MS,
                    session_id=session_id,
                )

            # ALIGN LENGTHS
            if scores.size != len(valid_docs):
                min_len    = min(len(scores), len(valid_docs))
                scores     = scores[:min_len]
                valid_docs = valid_docs[:min_len]

            # SCORE FUSION
            results: List[Dict] = []

            for d, s in zip(valid_docs, scores):
                meta      = d.get("metadata", {}) or {}
                structure = meta.get("structure", {}) or {}

                modality     = meta.get("modality", "text")
                chunk_idx    = structure.get("chunk_index", 0)
                fusion_score = min(float(d.get("score", 0.0)), 1.0)

                position_boost = 1.0 + (
                    settings.RERANK_POSITION_WEIGHT / (chunk_idx + 2)
                )
                modality_boost = self.modality_weights.get(modality, 1.0)

                final_score = (
                    self.w_model  * float(s) +
                    self.w_fusion * fusion_score
                ) * position_boost * modality_boost

                if not self._valid_score(final_score):
                    continue

                results.append({
                    "text":     d["text"][:settings.RAG_DOC_MAX_CHARS],
                    "metadata": meta,
                    "score":    round(float(final_score), 5),
                })

            results.sort(key=lambda x: x["score"], reverse=True)

            final = self._dedup(results, top_k)

            logger.info(
                event="rerank_success",
                input_count=input_count,
                filtered_count=filtered,
                output=len(final),
                model_latency=model_latency,
                latency=round(time.time() - start, 2),
                session_id=session_id,
            )

            return final

        except Exception as e:
            logger.error(
                event="rerank_failed",
                error=str(e),
                session_id=session_id,
            )

            fallback = sorted(
                documents,
                key=lambda x: x.get("score", 0.0),
                reverse=True,
            )

            return self._dedup(fallback, top_k)