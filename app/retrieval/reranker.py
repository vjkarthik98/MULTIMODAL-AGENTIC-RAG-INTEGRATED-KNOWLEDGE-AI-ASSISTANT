from __future__ import annotations

import asyncio
import hashlib
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pybreaker
    _reranker_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

    class _DummyBreaker:
        def __call__(self, fn):
            return fn

    _reranker_breaker = _DummyBreaker()  # type: ignore[assignment]


# MODALITY WEIGHTS FOR FINAL SCORE BOOSTING
_DEFAULT_MODALITY_WEIGHTS: Dict[str, float] = {
    "text":  1.0,
    "table": 1.1,
    "image": 1.05,
    "audio": 1.1,
    "video": 1.1,
}


class Reranker:

    def __init__(self) -> None:
        from app.core.model_loader import model_loader

        self.model = model_loader.get_reranker()
        self.top_k = settings.RERANK_TOP_K
        self.max_inputs = settings.RERANK_MAX_INPUT
        self.context_chars = settings.RERANK_CONTEXT_MAX_CHARS
        self.w_model = settings.RERANK_MODEL_WEIGHT
        self.w_fusion = settings.RERANK_FUSION_WEIGHT
        self.position_weight = settings.RERANK_POSITION_WEIGHT
        self.score_threshold = settings.RERANK_SCORE_THRESHOLD
        self.mmr_enabled = settings.MMR_ENABLED
        self.mmr_lambda = settings.MMR_LAMBDA

        self.modality_weights: Dict[str, float] = getattr(
            settings, "RERANK_MODALITY_WEIGHTS", _DEFAULT_MODALITY_WEIGHTS
        )

        logger.info(
            event="reranker_initialized",
            top_k=self.top_k,
            max_inputs=self.max_inputs,
            w_model=self.w_model,
            w_fusion=self.w_fusion,
            mmr_enabled=self.mmr_enabled,
        )

    # HASH

    def _hash(self, doc: Dict) -> str:
        meta = doc.get("metadata", {}) or {}
        base = (
            f"{meta.get('doc_id', '')}|"
            f"{meta.get('chunk_id', '')}|"
            f"{doc.get('text', '')[:200]}"
        )
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # QUERY NORMALIZATION

    def _normalize_query(self, q: str) -> str:
        import unicodedata
        q = unicodedata.normalize("NFC", str(q or ""))
        return " ".join(q.strip().split())[:self.context_chars]

    # TEXT CLEAN

    def _clean_text(self, text: str) -> str:
        return " ".join(str(text or "").split())

    # CONTEXT BUILDER — ENRICHES DOC WITH METADATA LABELS

    def _context(self, doc: Dict) -> str:
        meta = doc.get("metadata", {}) or {}
        structure = meta.get("structure", {}) or {}

        modality = meta.get("modality", "text")
        source = meta.get("source") or meta.get("source_type", "")
        content_type = meta.get("content_type") or structure.get("content_type", "")
        emb_space = meta.get("embedding_space") or structure.get("embedding_space", "text")
        page = meta.get("page") or structure.get("page")
        speaker = structure.get("speaker")
        language = meta.get("language") or structure.get("language")
        ts_start = structure.get("timestamp_start")

        header_parts: List[str] = []

        if source:
            header_parts.append(f"[SRC:{str(source)[:20]}]")

        if modality:
            header_parts.append(f"[{modality.upper()}]")

        if content_type:
            header_parts.append(f"[{str(content_type).upper()}]")

        if emb_space == "vision":
            header_parts.append("[VISION]")

        if page:
            header_parts.append(f"[PG:{page}]")

        if speaker:
            header_parts.append(f"[SPK:{speaker}]")

        if language and language != "en":
            header_parts.append(f"[LANG:{language}]")

        if ts_start is not None:
            header_parts.append(f"[T:{round(float(ts_start), 1)}s]")

        text = self._clean_text(doc.get("text", ""))[:self.context_chars]
        header = " ".join(header_parts)

        return (header + " " + text).strip()

    # PRE-FILTER — REMOVE EMPTY / TOO-SHORT DOCS ONLY
    # Score threshold is NOT applied here: upstream RRF scores are inherently
    # small (~1/61) and the cross-encoder re-scores everything from scratch.

    def _filter(self, docs: List[Dict]) -> List[Dict]:
        return [
            d for d in docs
            if d.get("text")
            and len(str(d.get("text", "")).strip()) >= settings.CHUNK_MIN_SIZE
        ]

    # SCORE NORMALIZATION

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores

        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        min_s = float(scores.min())
        max_s = float(scores.max())

        if max_s - min_s > 1e-6:
            return (scores - min_s) / (max_s - min_s)

        return np.clip(scores, 0.0, 1.0)

    # SCORE VALID

    def _valid_score(self, score: float) -> bool:
        return (
            isinstance(score, (int, float))
            and not math.isnan(score)
            and not math.isinf(score)
            and score > 0.0
        )

    # DEDUP

    def _dedup(self, results: List[Dict], top_k: int) -> List[Dict]:
        seen: set = set()
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

    # MMR DIVERSITY

    def _mmr(self, results: List[Dict], top_k: int) -> List[Dict]:
        if not self.mmr_enabled or not results:
            return results[:top_k]

        selected: List[Dict] = []
        candidates = list(results)

        while candidates and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")

            for i, candidate in enumerate(candidates):
                relevance = candidate.get("score", 0.0)
                max_sim = max(
                    (
                        self._text_overlap(
                            candidate.get("text", ""),
                            s.get("text", ""),
                        )
                        for s in selected
                    ),
                    default=0.0,
                )
                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * max_sim
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(candidates.pop(best_idx))

        return selected

    # TEXT OVERLAP FOR MMR

    def _text_overlap(self, left: str, right: str) -> float:
        a = set(left.lower().split())
        b = set(right.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # CROSS-ENCODER INFERENCE WITH RETRY + CIRCUIT BREAKER

    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            min=settings.RETRY_WAIT_MIN_SEC,
            max=settings.RETRY_WAIT_MAX_SEC,
        ),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
        reraise=True,
    )
    def _cross_encoder_predict(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        def _do():
            return self.model.predict(pairs)

        if _PYBREAKER_AVAILABLE:
            raw = _reranker_breaker(_do)()
        else:
            raw = _do()

        return np.asarray(raw).reshape(-1)

    # BUILD PAIRS

    def _build_pairs(
        self,
        query: str,
        docs: List[Dict],
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        pairs: List[Tuple[str, str]] = []
        valid_docs: List[Dict] = []

        for d in docs:
            ctx = self._context(d)
            if not ctx:
                continue
            pairs.append((
                query[:self.context_chars],
                ctx,
            ))
            valid_docs.append(d)

        return pairs, valid_docs

    # COMPUTE FINAL SCORES

    def _compute_final_scores(
        self,
        valid_docs: List[Dict],
        model_scores: np.ndarray,
    ) -> List[Dict]:
        results: List[Dict] = []

        for d, s in zip(valid_docs, model_scores):
            meta = d.get("metadata", {}) or {}
            structure = meta.get("structure", {}) or {}

            modality = meta.get("modality", "text")
            chunk_idx = structure.get("chunk_index", 0)
            fusion_score = min(float(d.get("score", 0.0)), 1.0)

            # POSITION BOOST: EARLIER CHUNKS GET SLIGHT BOOST
            position_boost = 1.0 + (
                self.position_weight / (int(chunk_idx) + 2)
            )

            modality_boost = self.modality_weights.get(modality, 1.0)

            final_score = (
                self.w_model * float(s) +
                self.w_fusion * fusion_score
            ) * position_boost * modality_boost

            if not self._valid_score(final_score):
                continue

            results.append({
                "text": d["text"][:settings.RAG_DOC_MAX_CHARS],
                "metadata": meta,
                "score": round(float(final_score), 5),
                "embedding": d.get("embedding"),
            })

        return results

    # HALLUCINATION GUARD — FLAG DOCS WITH LOW MODEL SCORE

    def _flag_low_confidence(self, results: List[Dict]) -> List[Dict]:
        for r in results:
            if r.get("score", 1.0) < self.score_threshold * 2:
                r.setdefault("metadata", {})["low_confidence"] = True
        return results

    # MAIN RERANK

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        session_id: str = "",
    ) -> List[Dict]:
        if not query or not documents:
            return []

        start = time.time()
        top_k = top_k or self.top_k
        query = self._normalize_query(query)

        input_count = len(documents)

        try:
            # PRE-FILTER
            docs = self._filter(documents)[:self.max_inputs]
            filtered_count = len(docs)

            if not docs:
                logger.warning(
                    event="rerank_no_valid_docs_after_filter",
                    input_count=input_count,
                    session_id=session_id,
                )
                return self._fallback(documents, top_k)

            # BUILD PAIRS
            pairs, valid_docs = self._build_pairs(query, docs)

            if not pairs:
                logger.warning(
                    event="rerank_no_pairs_built",
                    session_id=session_id,
                )
                return self._fallback(documents, top_k)

            # CROSS-ENCODER INFERENCE
            t_model = time.time()

            try:
                raw_scores = self._cross_encoder_predict(pairs)
            except Exception as exc:
                logger.error(
                    event="rerank_model_inference_failed",
                    error=str(exc),
                    session_id=session_id,
                )
                return self._fallback(documents, top_k)

            model_latency = round(time.time() - t_model, 3)

            if model_latency * 1000 > settings.LATENCY_TARGET_CROSS_MODAL_MS:
                logger.warning(
                    event="rerank_latency_exceeded",
                    latency_ms=round(model_latency * 1000, 1),
                    target_ms=settings.LATENCY_TARGET_CROSS_MODAL_MS,
                    session_id=session_id,
                )

            # NORMALIZE MODEL SCORES
            norm_scores = self._normalize_scores(raw_scores)

            # ALIGN LENGTHS
            min_len = min(len(norm_scores), len(valid_docs))
            norm_scores = norm_scores[:min_len]
            valid_docs = valid_docs[:min_len]

            # COMPUTE FINAL SCORES
            results = self._compute_final_scores(valid_docs, norm_scores)

            # SORT BY FINAL SCORE
            results.sort(key=lambda x: x["score"], reverse=True)

            # FLAG LOW CONFIDENCE
            results = self._flag_low_confidence(results)

            # MMR DIVERSITY
            results = self._mmr(results, top_k * 2)

            # DEDUP
            final = self._dedup(results, top_k)

            total_latency = round(time.time() - start, 3)

            logger.info(
                event="rerank_success",
                input_count=input_count,
                filtered_count=filtered_count,
                pairs_built=len(pairs),
                output=len(final),
                model_latency=model_latency,
                total_latency=total_latency,
                session_id=session_id,
            )

            return final

        except Exception as exc:
            logger.error(
                event="rerank_failed",
                error=str(exc),
                input_count=input_count,
                session_id=session_id,
            )
            return self._fallback(documents, top_k)

    # ASYNC WRAPPER

    async def async_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        session_id: str = "",
    ) -> List[Dict]:
        return await asyncio.to_thread(self.rerank, query, documents, top_k, session_id)

    # FALLBACK — SCORE-SORTED WITHOUT CROSS-ENCODER

    def _fallback(self, documents: List[Dict], top_k: int) -> List[Dict]:
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        return self._dedup(sorted_docs, top_k)

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model is not None,
            "top_k": self.top_k,
            "max_inputs": self.max_inputs,
            "circuit_breaker": _PYBREAKER_AVAILABLE,
            "mmr_enabled": self.mmr_enabled,
            "w_model": self.w_model,
            "w_fusion": self.w_fusion,
            "score_threshold": self.score_threshold,
        }

