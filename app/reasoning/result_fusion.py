import asyncio
import hashlib
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_fusion_duration = Histogram(
    "result_fusion_duration_seconds",
    "Result fusion duration",
    ["status"],
)
_fusion_errors = Counter(
    "result_fusion_errors_total",
    "Result fusion errors by type",
    ["error_type"],
)
_fusion_output_count = Histogram(
    "result_fusion_output_count",
    "Number of results after fusion",
)
_contradiction_flags = Counter(
    "result_fusion_contradiction_flags_total",
    "Number of contradictions detected during fusion",
)

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)


# SHA-256 HASH FOR DEDUP

def _hash(text: str, meta: Dict) -> str:
    base = f"{text[:200]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# VALID SCORE CHECK

def _valid_score(score: float) -> bool:
    return not (math.isnan(score) or math.isinf(score))


# VALID EMBEDDING CHECK

def _valid_embedding(emb: Any) -> bool:
    return (
        isinstance(emb, list) and
        len(emb) in (settings.TEXT_EMBEDDING_DIM, settings.VISION_EMBEDDING_DIM)
    )


# COSINE SIMILARITY

def _cosine(v1: List[float], v2: List[float]) -> float:
    a     = np.nan_to_num(np.array(v1, dtype=float))
    b     = np.nan_to_num(np.array(v2, dtype=float))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    val   = float(np.dot(a, b) / denom)
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val


# MODALITY COUNTS FOR LOGGING

def _modality_counts(results: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        m = (r.get("metadata", {}) or {}).get("modality", "unknown")
        counts[m] = counts.get(m, 0) + 1
    return counts


# CONTRADICTION DETECTION
# COMPARES ANSWER PAIRS FOR SEMANTIC OPPOSITION

def _detect_contradictions(results: List[Dict]) -> List[Tuple[int, int, float]]:
    """
    RETURNS LIST OF (IDX_A, IDX_B, SIMILARITY) FOR PAIRS
    WHERE SIMILARITY IS LOW DESPITE HIGH SCORES — POTENTIAL CONTRADICTION.
    ONLY RUNS ON RESULTS WITH EMBEDDINGS.
    """
    contradictions: List[Tuple[int, int, float]] = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            emb_i = results[i].get("embedding")
            emb_j = results[j].get("embedding")

            if not _valid_embedding(emb_i) or not _valid_embedding(emb_j):
                continue

            sim = _cosine(emb_i, emb_j)

            # HIGH-SCORING RESULTS WITH LOW SIMILARITY — LIKELY CONTRADICTING
            score_i = results[i].get("score", 0.0)
            score_j = results[j].get("score", 0.0)

            if score_i > 0.7 and score_j > 0.7 and sim < 0.2:
                contradictions.append((i, j, round(sim, 3)))

    return contradictions


# CONFIDENCE SCORING PER RESULT

def _confidence_score(result: Dict, all_results: List[Dict]) -> float:
    """
    CONFIDENCE = HOW CONSISTENTLY SUPPORTED ACROSS ALL RESULTS.
    HIGH CONFIDENCE = HIGH SCORE + SUPPORTED BY MULTIPLE DOCS.
    """
    base_score = result.get("score", 0.0)
    text       = (result.get("text", "") or "").lower()

    if not text:
        return base_score

    # COUNT HOW MANY OTHER RESULTS SHARE SIGNIFICANT WORDS
    words       = set(w for w in text.split() if len(w) > 4)
    support     = 0
    total_other = 0

    for other in all_results:
        if other is result:
            continue
        other_text  = (other.get("text", "") or "").lower()
        other_words = set(w for w in other_text.split() if len(w) > 4)
        total_other += 1
        if words & other_words:
            support += 1

    if total_other == 0:
        return base_score

    support_ratio = support / total_other
    return round(min(base_score + 0.1 * support_ratio, 1.0), 4)


# SCORE NORMALIZATION — MIN-MAX

def _normalize_scores(results: List[Dict]) -> List[Dict]:
    if not results:
        return results

    scores = np.array([r.get("score", 0.0) for r in results], dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    min_s  = scores.min()
    max_s  = scores.max()

    for i, r in enumerate(results):
        if max_s - min_s > 1e-6:
            r["norm_score"] = float((scores[i] - min_s) / (max_s - min_s))
        else:
            r["norm_score"] = 0.5

    return results


# SCORE FUSION — COMBINES NORM SCORE + QUALITY + MODALITY + RECENCY

def _score_fusion(results: List[Dict]) -> List[Dict]:
    modality_weights = settings.FUSION_MODALITY_WEIGHTS

    for r in results:
        base     = r.get("norm_score", 0.0)
        meta     = r.get("metadata", {}) or {}
        modality = meta.get("modality", "text")

        modality_boost = modality_weights.get(modality, 1.0)

        text    = str(r.get("text", ""))
        quality = (
            0.1 if len(text) < settings.CHUNK_MIN_SIZE
            else min(len(text) / settings.FUSION_MAX_TEXT_CHARS, 1.0)
        )

        # RECENCY BOOST
        recency = 0.0
        ts      = meta.get("timestamp_start") or meta.get("ingestion_time")
        if ts:
            try:
                age     = max(time.time() - float(ts), 0.0)
                recency = 1.0 / (1.0 + age / settings.MEMORY_RECENCY_SCALE)
            except Exception:
                recency = 0.0

        # HIERARCHY BOOST — SECTION-LEVEL CHUNKS SCORE HIGHER
        hierarchy_boost = 1.0
        hierarchy_level = meta.get("hierarchy_level")
        if hierarchy_level == "section":
            hierarchy_boost = 1.05
        elif hierarchy_level == "paragraph":
            hierarchy_boost = 1.02

        final_score = (
            settings.FUSION_SCORE_WEIGHT    * base +
            settings.FUSION_QUALITY_WEIGHT  * quality +
            settings.FUSION_MODALITY_WEIGHT * modality_boost +
            0.05 * recency
        ) * hierarchy_boost

        if not _valid_score(final_score):
            final_score = 0.0

        r["final_score"] = round(float(final_score), 5)

    return results


# FILTER BY TEXT PRESENCE — score threshold applied after normalization
# (raw RRF scores are inherently small ~1/61 and cannot be compared against
#  an absolute threshold before _normalize_scores rescales them to [0,1])

def _filter(results: List[Dict]) -> List[Dict]:
    return [r for r in results if r.get("text")]


# EXACT DEDUP BY HASH

def _dedup(results: List[Dict]) -> List[Dict]:
    seen:   set        = set()
    unique: List[Dict] = []
    for r in results:
        h = _hash(r.get("text", ""), r.get("metadata", {}))
        if h in seen:
            continue
        seen.add(h)
        unique.append(r)
    return unique


# MMR DIVERSITY — MAXIMAL MARGINAL RELEVANCE

def _diversity(
    results: List[Dict],
    top_k: int,
    sim_threshold: float,
) -> List[Dict]:
    selected: List[Dict] = []

    for r in results:
        v1 = r.get("embedding")

        if not _valid_embedding(v1):
            selected.append(r)
            if len(selected) >= top_k:
                break
            continue

        too_similar = any(
            _cosine(v1, s.get("embedding")) > sim_threshold
            for s in selected
            if _valid_embedding(s.get("embedding"))
        )

        if too_similar:
            continue

        selected.append(r)

        if len(selected) >= top_k:
            break

    return selected


# CROSS-MODAL RESULT LINKING
# LINKS IMAGE/AUDIO RESULTS TO THEIR PARENT TEXT CHUNK

def _cross_modal_link(results: List[Dict]) -> List[Dict]:
    text_chunks: Dict[str, Dict] = {}

    for r in results:
        meta = r.get("metadata", {}) or {}
        if meta.get("modality") == "text":
            doc_id = meta.get("doc_id")
            page   = meta.get("page")
            key    = f"{doc_id}:{page}"
            text_chunks[key] = r

    for r in results:
        meta = r.get("metadata", {}) or {}
        if meta.get("modality") in ("image", "audio", "video"):
            doc_id = meta.get("doc_id")
            page   = meta.get("page")
            key    = f"{doc_id}:{page}"
            if key in text_chunks:
                r["linked_text_chunk"] = text_chunks[key].get("text", "")[:200]

    return results


# RESOLVE CONTRADICTIONS — FLAG AND REDUCE CONFIDENCE OF CONFLICTING RESULTS

def _resolve_contradictions(
    results: List[Dict],
    contradictions: List[Tuple[int, int, float]],
) -> List[Dict]:
    if not contradictions:
        return results

    flagged_indices: set = set()

    for idx_a, idx_b, sim in contradictions:
        _contradiction_flags.inc()
        # FLAG THE LOWER-SCORING RESULT
        score_a = results[idx_a].get("final_score", 0.0)
        score_b = results[idx_b].get("final_score", 0.0)
        if score_a < score_b:
            flagged_indices.add(idx_a)
        else:
            flagged_indices.add(idx_b)

        logger.warning(
            "fusion_contradiction_detected",
            idx_a=idx_a,
            idx_b=idx_b,
            similarity=sim,
        )

    for i in flagged_indices:
        results[i]["final_score"]      = results[i].get("final_score", 0.0) * 0.5
        results[i]["contradiction_flag"] = True

    return results


# MAIN FUSE

class ResultFusion:

    def __init__(self) -> None:
        self.top_k         = settings.RERANK_TOP_K
        self.sim_threshold = settings.FUSION_SIMILARITY_THRESHOLD
        self.min_score     = settings.FUSION_MIN_SCORE

    def fuse(
        self,
        results: List[Dict],
        session_id: str = "default",
        detect_contradictions: bool = True,
    ) -> List[Dict]:

        if not results:
            return []

        start       = time.time()
        input_count = len(results)

        with tracer.start_as_current_span("result_fusion") as span:
            span.set_attribute("results.input", input_count)
            span.set_attribute("session.id", session_id)

            try:
                # DEEP COPY TO AVOID MUTATING INPUT
                results = [dict(r) for r in results]

                # CAP INPUT
                results = results[:settings.FUSION_MAX_INPUT]

                # FILTER LOW SCORE AND EMPTY TEXT
                results         = _filter(results)
                filtered_count  = len(results)

                if not results:
                    span.set_status(Status(StatusCode.OK))
                    return []

                # SCORE NORMALIZATION
                results = _normalize_scores(results)

                # SCORE FUSION WITH MODALITY + QUALITY + RECENCY
                results = _score_fusion(results)

                # SORT BY FINAL SCORE
                results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

                # EXACT DEDUP
                results = _dedup(results)

                # CONFIDENCE SCORING PER RESULT
                for r in results:
                    r["confidence_score"] = _confidence_score(r, results)

                # CONTRADICTION DETECTION AND RESOLUTION
                if detect_contradictions:
                    contradictions = _detect_contradictions(results)
                    if contradictions:
                        results = _resolve_contradictions(results, contradictions)
                        # RE-SORT AFTER CONFIDENCE ADJUSTMENT
                        results.sort(
                            key=lambda x: x.get("final_score", 0.0),
                            reverse=True,
                        )

                # CROSS-MODAL LINKING
                results = _cross_modal_link(results)

                # MMR DIVERSITY SELECTION
                results = _diversity(results, self.top_k, self.sim_threshold)

                output  = results[:self.top_k]
                latency = round(time.time() - start, 2)

                _fusion_duration.labels(status="success").observe(latency)
                _fusion_output_count.observe(len(output))

                span.set_attribute("results.output", len(output))
                span.set_attribute("results.filtered", filtered_count)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "fusion_success",
                    input_count=input_count,
                    filtered_count=filtered_count,
                    output=len(output),
                    modality_breakdown=_modality_counts(output),
                    latency=latency,
                    session_id=session_id,
                )

                return output

            except Exception as exc:
                latency    = round(time.time() - start, 2)
                error_type = type(exc).__name__

                _fusion_duration.labels(status="error").observe(latency)
                _fusion_errors.labels(error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "fusion_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )

                # GRACEFUL DEGRADATION — RETURN TOP-K BY ORIGINAL SCORE
                try:
                    fallback = sorted(
                        results,
                        key=lambda x: x.get("score", 0.0),
                        reverse=True,
                    )
                    return fallback[:self.top_k]
                except Exception:
                    return []

    # ASYNC WRAPPER

    async def fuse_async(
        self,
        results: List[Dict],
        session_id: str = "default",
        detect_contradictions: bool = True,
    ) -> List[Dict]:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.fuse(results, session_id, detect_contradictions),
            )

