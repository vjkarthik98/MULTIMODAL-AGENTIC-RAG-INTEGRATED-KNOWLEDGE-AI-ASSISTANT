from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from typing import Any

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
# Image/video OCR is noisy; penalise relative to clean text so fact-retrieval
# queries don't get dominated by ticker-display captions or cover-scan fragments.
# Table weight raised to 1.15 (plan Phase 5.1) — finance queries are table-heavy.
_DEFAULT_MODALITY_WEIGHTS: dict[str, float] = {
    "text": 1.0,
    "table": 1.15,
    "image": 0.75,
    "jpg": 0.75,
    "audio": 0.85,
    "mp3": 0.85,
    "video": 0.75,
    "mp4": 0.75,
}

# Common English stopwords excluded from exact-match boost calculation.
_STOPWORDS: set[str] = frozenset(
    {  # type: ignore[assignment]
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "in",
        "on",
        "at",
        "for",
        "of",
        "to",
        "and",
        "or",
        "not",
        "but",
        "with",
        "from",
        "by",
        "this",
        "that",
        "these",
        "those",
        "its",
        "it",
        "we",
        "they",
        "you",
        "he",
        "she",
        "their",
        "our",
        "your",
        "my",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
    }
)

# Absolute relevance floor on sigmoid-calibrated CE score.
# sigmoid(logit) < 0.04 ≈ logit < −3.18 → document is almost certainly off-topic.
_FLOOR_SIGMOID: float = 0.04

# TRANSCRIPT SECTION RE-WEIGHTING (audio/video/txt-transcript chunks only).
# A cross-encoder ranks a Q&A chunk that literally contains "cut rates by 50
# basis points" (a reporter's question) above the prepared-remarks ANNOUNCEMENT
# that actually states the cut ("...by a half percentage point") — the
# announcement never says the digits. For a FACTUAL query about what a speaker
# said/did, the authoritative prepared-remarks statement should outrank a Q&A
# fragment. Applies ONLY to chunks carrying `call_section` (audio/video and
# txt-transcript) — PDF/DOCX/XLSX/image have none, so their factor is 1.0.
# Boost/penalty are uniform within a section, so they only ever change
# prepared_remarks-vs-qa_session ORDER; within-section ordering is untouched.
_SECTION_PREPARED_BOOST: float = 1.18
_SECTION_QA_PENALTY: float = 0.74

# Meta-queries explicitly ABOUT the Q&A ("what did reporter X ask") legitimately
# want the question/answer chunks — the re-weighting is disabled for them.
_META_QUERY_RE = re.compile(
    r"\b(ask|asked|asks|asking|question(?:ed|s)?|inquir\w+|"
    r"reporter|journalist|correspondent|analyst)\b",
    re.IGNORECASE,
)


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
        self.batch_size = getattr(settings, "RERANK_BATCH_SIZE", 8)
        self.source_max_chunks = getattr(settings, "RERANK_SOURCE_MAX_CHUNKS", 3)
        self.mmr_enabled = settings.MMR_ENABLED
        self.mmr_lambda = settings.MMR_LAMBDA

        self.modality_weights: dict[str, float] = getattr(
            settings, "RERANK_MODALITY_WEIGHTS", _DEFAULT_MODALITY_WEIGHTS
        )

        logger.info(
            event="reranker_initialized",
            top_k=self.top_k,
            max_inputs=self.max_inputs,
            w_model=self.w_model,
            w_fusion=self.w_fusion,
            batch_size=self.batch_size,
            source_max_chunks=self.source_max_chunks,
            mmr_enabled=self.mmr_enabled,
        )

    # HASH

    def _hash(self, doc: dict) -> str:
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
        return " ".join(q.strip().split())[: self.context_chars]

    # TEXT CLEAN

    def _clean_text(self, text: str) -> str:
        return " ".join(str(text or "").split())

    # CONTEXT BUILDER FOR CROSS-ENCODER
    # The cross-encoder receives (query, context) pairs. Adding semantic
    # locator headers — page number, section heading, sheet name, timestamp,
    # speaker — lets the cross-encoder score chunks from the right section or
    # time window higher than identical-text chunks from an irrelevant one.
    # We omit: source UUID, generic content_type (uniform across the corpus,
    # add noise without signal), and "[TEXT]" for text modality (the default).

    def _context(self, doc: dict) -> str:
        meta = doc.get("metadata", {}) or {}
        modality = meta.get("modality", "text")
        source_type = meta.get("source_type", "")
        page = meta.get("page")
        speaker = meta.get("speaker")
        ts_start = meta.get("timestamp_start")
        section = meta.get("section_title") or ""
        sheet = meta.get("sheet_name") or ""
        sub_idx = meta.get("sub_chunk_index")
        sub_total = meta.get("total_sub_chunks")
        row_start = meta.get("row_start")
        row_end = meta.get("row_end")

        header_parts: list[str] = []

        # Modality prefix — only for non-text (text is the default)
        if modality and modality != "text":
            header_parts.append(f"[{modality.upper()}]")

        # PDF page + sub-chunk position
        if page:
            if sub_idx is not None and sub_total and int(sub_total) > 1:
                header_parts.append(f"[PG:{page} {int(sub_idx)+1}/{int(sub_total)}]")
            else:
                header_parts.append(f"[PG:{page}]")

        # DOCX section heading — critical for heading-based queries
        if source_type == "word" and section:
            header_parts.append(f"[{section[:50]}]")

        # XLSX sheet + row range — grounds the chunk in the spreadsheet
        if source_type == "excel":
            if sheet and row_start is not None and row_end is not None:
                header_parts.append(f"[{sheet[:30]} R{row_start}-{row_end}]")
            elif sheet:
                header_parts.append(f"[{sheet[:30]}]")

        # Audio/video speaker
        if speaker:
            header_parts.append(f"[SPK:{speaker}]")

        # Audio/video timestamp
        if ts_start is not None:
            header_parts.append(f"[T:{round(float(ts_start), 1)}s]")

        text = self._clean_text(doc.get("text", ""))[: self.context_chars]
        if not header_parts:
            return text
        return (" ".join(header_parts) + " " + text).strip()

    # PRE-FILTER — REMOVE EMPTY / TOO-SHORT DOCS ONLY
    # Score threshold is NOT applied here: upstream RRF scores are inherently
    # small (~1/61) and the cross-encoder re-scores everything from scratch.

    def _filter(self, docs: list[dict]) -> list[dict]:
        return [
            d
            for d in docs
            if d.get("text") and len(str(d.get("text", "")).strip()) >= settings.CHUNK_MIN_SIZE
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
            and score >= 0.0
        )

    # DEDUP

    def _dedup(self, results: list[dict], top_k: int) -> list[dict]:
        seen: set = set()
        final: list[dict] = []

        for r in results:
            h = self._hash(r)
            if h in seen:
                continue
            seen.add(h)
            final.append(r)

            if len(final) >= top_k:
                break

        return final

    # COSINE SIMILARITY BETWEEN EMBEDDING VECTORS

    def _cosine_sim(self, emb_a: Any, emb_b: Any) -> float:
        try:
            a = np.asarray(emb_a, dtype=float)
            b = np.asarray(emb_b, dtype=float)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom < 1e-9:
                return 0.0
            return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))
        except Exception:
            return 0.0

    # INTER-DOCUMENT SIMILARITY
    # Uses embedding cosine when both docs carry stored embeddings (standard for
    # Qdrant results); falls back to Jaccard bag-of-words when they do not.

    def _similarity(self, doc_a: dict, doc_b: dict) -> float:
        emb_a = doc_a.get("embedding")
        emb_b = doc_b.get("embedding")
        if emb_a is not None and emb_b is not None:
            return self._cosine_sim(emb_a, emb_b)
        return self._text_overlap(doc_a.get("text", ""), doc_b.get("text", ""))

    # TEXT OVERLAP FOR MMR (JACCARD FALLBACK)

    def _text_overlap(self, left: str, right: str) -> float:
        a = set(left.lower().split())
        b = set(right.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # EXACT-MATCH BOOST
    # Rewards chunks where non-trivial query terms appear verbatim. Compensates
    # for CE misses on exact financial figures, model numbers, or proper names.
    # Max bonus is 5% of the final score.

    def _exact_match_boost(self, query: str, text: str) -> float:
        q_lower = query.lower()
        t_lower = text.lower()
        terms = [t for t in re.findall(r"\b\w{4,}\b", q_lower) if t not in _STOPWORDS]
        if not terms:
            return 0.0
        hits = sum(1 for t in terms if re.search(r"\b" + re.escape(t) + r"\b", t_lower))
        return 0.05 * (hits / len(terms))

    # SOURCE DIVERSITY CAP
    # Hard-limit chunks per doc_id so one document cannot dominate top_k.

    def _cap_key(self, r: dict) -> tuple[str, str | None]:
        # Cap on (doc_id, sheet_name) rather than doc_id alone: for a multi-sheet
        # XLSX workbook, doc_id is identical across every sheet, so a single huge
        # sheet (e.g. 1000+ rows -> 50+ near-duplicate row-group chunks) can
        # otherwise crowd out every other sheet's candidates with zero diversity
        # enforcement (accuracy phase 2026-07). sheet_name is None for every
        # non-XLSX modality, so this is a no-op there — the key degenerates back
        # to plain doc_id and behavior is unchanged for PDF/DOCX/etc.
        m = r.get("metadata") or {}
        return (m.get("doc_id", ""), m.get("sheet_name"))

    def _apply_source_cap(self, results: list[dict]) -> list[dict]:
        # When results come from a single document (and, for XLSX, a single
        # sheet), skip the cap to avoid dropping relevant chunks — diversity
        # filtering isn't needed for a mono-doc/mono-sheet result set.
        unique_keys = {self._cap_key(r) for r in results}
        if len(unique_keys) <= 1:
            return results

        count: dict[tuple[str, str | None], int] = {}
        output: list[dict] = []
        for r in results:
            key = self._cap_key(r)
            n = count.get(key, 0)
            if n >= self.source_max_chunks:
                continue
            count[key] = n + 1
            output.append(r)
        return output

    # MMR DIVERSITY
    # Uses embedding cosine similarity (not Jaccard) when embeddings are present.

    def _mmr(self, results: list[dict], top_k: int) -> list[dict]:
        if not self.mmr_enabled or not results:
            return results[:top_k]

        selected: list[dict] = []
        candidates = list(results)

        while candidates and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")

            for i, candidate in enumerate(candidates):
                relevance = candidate.get("score", 0.0)
                max_sim = max(
                    (self._similarity(candidate, s) for s in selected),
                    default=0.0,
                )
                mmr_score = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(candidates.pop(best_idx))

        return selected

    # SINGLE-BATCH CROSS-ENCODER INFERENCE WITH RETRY + CIRCUIT BREAKER

    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            min=settings.RETRY_WAIT_MIN_SEC,
            max=settings.RETRY_WAIT_MAX_SEC,
        ),
        retry=retry_if_exception_type((RuntimeError, TimeoutError)),
        reraise=True,
    )
    def _predict_batch(self, batch: list[tuple[str, str]]) -> np.ndarray:
        def _do():
            return self.model.predict(batch)

        if _PYBREAKER_AVAILABLE:
            raw = _reranker_breaker(_do)()
        else:
            raw = _do()

        return np.asarray(raw).reshape(-1)

    # BATCHED CROSS-ENCODER INFERENCE
    # Splits pairs into self.batch_size chunks to prevent OOM on large inputs.
    # Each sub-batch retries independently via _predict_batch.

    def _cross_encoder_predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=float)

        batch_size = max(1, self.batch_size)
        batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]
        parts = [self._predict_batch(b) for b in batches]
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    # BUILD PAIRS

    def _build_pairs(
        self,
        query: str,
        docs: list[dict],
    ) -> tuple[list[tuple[str, str]], list[dict]]:
        pairs: list[tuple[str, str]] = []
        valid_docs: list[dict] = []

        for d in docs:
            ctx = self._context(d)
            if not ctx:
                continue
            pairs.append(
                (
                    query[: self.context_chars],
                    ctx,
                )
            )
            valid_docs.append(d)

        return pairs, valid_docs

    # COMPUTE FINAL SCORES

    @staticmethod
    def _section_factor(meta: dict, query_is_meta: bool) -> float:
        """prepared-remarks vs Q&A re-weighting — see module notes. Returns 1.0
        (no-op) for any chunk without a `call_section` (every non-transcript
        modality) and for meta-queries about the Q&A itself."""
        section = meta.get("call_section")
        if not section or query_is_meta:
            return 1.0
        if section == "prepared_remarks":
            return _SECTION_PREPARED_BOOST
        if section == "qa_session":
            return _SECTION_QA_PENALTY
        return 1.0

    def _compute_final_scores(
        self,
        valid_docs: list[dict],
        model_scores: np.ndarray,
        raw_sigmoid: np.ndarray,
        query: str = "",
    ) -> list[dict]:
        results: list[dict] = []

        # Computed once per query: is this query explicitly ABOUT the Q&A?
        query_is_meta = bool(_META_QUERY_RE.search(query or ""))

        for i, (d, s) in enumerate(zip(valid_docs, model_scores)):
            meta = dict(d.get("metadata", {}) or {})

            modality = meta.get("modality", "text")
            # sub_chunk_index is a top-level Qdrant/BM25 payload field (not
            # nested in a "structure" sub-dict). chunk_id is a fallback.
            chunk_idx = meta.get("sub_chunk_index") or meta.get("chunk_id") or 0
            fusion_score = min(float(d.get("score", 0.0)), 1.0)

            # POSITION BOOST: earlier sub-chunks get slight boost
            position_boost = 1.0 + (self.position_weight / (int(chunk_idx) + 2))

            modality_boost = self.modality_weights.get(modality, 1.0)

            # EXACT MATCH BOOST: reward verbatim key-term hits (up to +5%)
            em_boost = 1.0 + self._exact_match_boost(query, d.get("text", ""))

            # TRANSCRIPT SECTION FACTOR (prepared-remarks vs Q&A; transcripts only)
            section_factor = self._section_factor(meta, query_is_meta)

            final_score = (
                (self.w_model * float(s) + self.w_fusion * fusion_score)
                * position_boost
                * modality_boost
                * em_boost
                * section_factor
            )

            if not self._valid_score(final_score):
                continue

            # Stamp sigmoid-calibrated CE score for absolute relevance filtering.
            # This is sigmoid(raw_logit) ∈ [0, 1] — a true relevance probability,
            # not a relative rank. Used by the floor filter and confidence flag.
            if i < len(raw_sigmoid):
                meta["_reranker_raw"] = round(float(raw_sigmoid[i]), 5)

            results.append(
                {
                    "text": d["text"][: settings.RAG_DOC_MAX_CHARS],
                    "metadata": meta,
                    "score": round(float(final_score), 5),
                    "embedding": d.get("embedding"),
                }
            )

        return results

    # HALLUCINATION GUARD — FLAG DOCS WHERE CE SIGMOID SCORE IS LOW

    def _flag_low_confidence(self, results: list[dict]) -> list[dict]:
        for r in results:
            # Use the calibrated sigmoid score, not the composite final score,
            # so the flag is an absolute relevance signal independent of boosts.
            raw = (r.get("metadata") or {}).get("_reranker_raw", 1.0)
            if raw < self.score_threshold:
                r.setdefault("metadata", {})["low_confidence"] = True
        return results

    # MAIN RERANK

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
        session_id: str = "",
        max_inputs: int | None = None,
    ) -> list[dict]:
        if not query or not documents:
            return []

        start = time.time()
        top_k = top_k or self.top_k
        # Per-call cross-encoder budget. Defaults to self.max_inputs (unchanged
        # for every existing caller); a caller can widen it — e.g. a single-
        # meeting scoped audio query, where the specific fact chunk ranks below
        # the usual cap in dense/fusion but the cross-encoder pins it once seen.
        eff_max = max_inputs or self.max_inputs
        query = self._normalize_query(query)

        input_count = len(documents)

        try:
            # PRE-FILTER: remove empty/too-short docs
            docs = self._filter(documents)[:eff_max]
            filtered_count = len(docs)

            if not docs:
                logger.warning(
                    event="rerank_no_valid_docs_after_filter",
                    input_count=input_count,
                    session_id=session_id,
                )
                return self._fallback(documents, top_k)

            # DEDUP BEFORE INFERENCE: remove exact duplicates so cross-encoder
            # slots are not wasted on identical chunks retrieved by both BM25
            # and the dense vector retriever.
            docs = self._dedup(docs, eff_max)

            # BUILD PAIRS
            pairs, valid_docs = self._build_pairs(query, docs)

            if not pairs:
                logger.warning(
                    event="rerank_no_pairs_built",
                    session_id=session_id,
                )
                return self._fallback(documents, top_k)

            # CROSS-ENCODER INFERENCE (batched, with retry + circuit breaker)
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

            # SIGMOID CALIBRATION
            # bge-reranker-v2-m3 (and most sentence-transformers CrossEncoders)
            # return raw logits in ≈[−10, 10]. Apply sigmoid to get calibrated
            # relevance probabilities in [0, 1] for absolute threshold comparisons.
            clipped = np.clip(raw_scores, -20.0, 20.0)
            raw_sigmoid = 1.0 / (1.0 + np.exp(-clipped))

            # NORMALIZE LOGITS → [0, 1] for the model-score term in the formula
            norm_scores = self._normalize_scores(raw_scores)

            # ALIGN LENGTHS
            min_len = min(len(norm_scores), len(valid_docs), len(raw_sigmoid))
            norm_scores = norm_scores[:min_len]
            raw_sigmoid = raw_sigmoid[:min_len]
            valid_docs = valid_docs[:min_len]

            # COMPUTE FINAL SCORES (raw_sigmoid stamped as _reranker_raw)
            results = self._compute_final_scores(valid_docs, norm_scores, raw_sigmoid, query)

            # SORT BY FINAL SCORE
            results.sort(key=lambda x: x["score"], reverse=True)

            # ABSOLUTE RELEVANCE FLOOR
            # Drop chunks where sigmoid < 0.04 (≈ logit < −3.18), meaning the
            # CE is highly confident they are off-topic. Always keep index 0 so
            # the caller never receives an empty list.
            if len(results) > 1:
                results = [results[0]] + [
                    r
                    for r in results[1:]
                    if (r.get("metadata") or {}).get("_reranker_raw", 1.0) >= _FLOOR_SIGMOID
                ]

            # FLAG LOW CONFIDENCE (uses calibrated sigmoid score)
            results = self._flag_low_confidence(results)

            # SOURCE DIVERSITY CAP: limit chunks per doc_id before MMR
            results = self._apply_source_cap(results)

            # MMR DIVERSITY (embedding cosine similarity when available)
            results = self._mmr(results, top_k)

            # TRANSCRIPT AUTHORITY REORDER (deterministic; transcripts only).
            # Runs LAST — after MMR — so nothing reshuffles it. For a FACTUAL
            # (non-meta) question about what a speaker said/did, a prepared-
            # remarks ANNOUNCEMENT is the authoritative source. But a cross-
            # encoder ranks a Q&A chunk that literally repeats the query's digits
            # ("cut rates by 50 basis points", a reporter's hypothetical) above
            # the announcement ("...by a half percentage point"), which then
            # becomes the citation and drives the answer. When the #1 result is a
            # qa_session chunk and a prepared_remarks chunk sits in the top window
            # with ≥40% of the top score (genuinely relevant, not noise), promote
            # that announcement to #1. No-op for meta-queries ("what did reporter
            # X ask") and for every non-transcript modality (no call_section).
            if len(results) > 1 and not _META_QUERY_RE.search(query or ""):
                top_meta = results[0].get("metadata") or {}
                if top_meta.get("call_section") == "qa_session":
                    top_score = results[0].get("score", 0.0) or 0.0
                    for j in range(1, min(len(results), 6)):
                        m = results[j].get("metadata") or {}
                        if (
                            m.get("call_section") == "prepared_remarks"
                            and (results[j].get("score", 0.0) or 0.0) >= 0.40 * top_score
                        ):
                            results.insert(0, results.pop(j))
                            break

            total_latency = round(time.time() - start, 3)

            logger.info(
                event="rerank_success",
                input_count=input_count,
                filtered_count=filtered_count,
                pairs_built=len(pairs),
                output=len(results),
                model_latency=model_latency,
                total_latency=total_latency,
                session_id=session_id,
            )

            return results

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
        documents: list[dict],
        top_k: int | None = None,
        session_id: str = "",
    ) -> list[dict]:
        return await asyncio.to_thread(self.rerank, query, documents, top_k, session_id)

    # FALLBACK — SCORE-SORTED WITHOUT CROSS-ENCODER

    def _fallback(self, documents: list[dict], top_k: int) -> list[dict]:
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        return self._dedup(sorted_docs, top_k)

    # HEALTH CHECK

    def health_check(self) -> dict[str, Any]:
        return {
            "model_loaded": self.model is not None,
            "top_k": self.top_k,
            "max_inputs": self.max_inputs,
            "batch_size": self.batch_size,
            "source_max_chunks": self.source_max_chunks,
            "circuit_breaker": _PYBREAKER_AVAILABLE,
            "mmr_enabled": self.mmr_enabled,
            "w_model": self.w_model,
            "w_fusion": self.w_fusion,
            "score_threshold": self.score_threshold,
        }
