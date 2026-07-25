import asyncio
import hashlib
import heapq
import math
import threading
import time
import unicodedata
from typing import Any

import numpy as np
import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_filter_duration = Histogram(
    "memory_filter_duration_seconds",
    "Memory filter duration",
    ["status"],
)
_filter_errors = Counter(
    "memory_filter_errors_total",
    "Memory filter errors by type",
    ["error_type"],
)
_filter_results = Histogram(
    "memory_filter_results_count",
    "Number of memory items selected after filtering",
)

# SCORE WEIGHTS
_W_SIM = 0.5
_W_RECENCY = 0.2
_W_ROLE = 0.2
_W_IMPORTANCE = 0.1

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)


# NORMALIZE QUERY


def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    return " ".join(query.strip().split())


# SHA-256 HASH FOR DEDUP


def _hash(msg: dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# COSINE SIMILARITY


def _cosine(v1: list[float], v2: list[float]) -> float:
    a = np.nan_to_num(np.array(v1, dtype=float))
    b = np.nan_to_num(np.array(v2, dtype=float))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    val = float(np.dot(a, b) / denom)
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val


# EMBEDDING VALID CHECK


def _valid_embedding(vec: Any) -> bool:
    return isinstance(vec, list) and len(vec) in (
        settings.TEXT_EMBEDDING_DIM,
        settings.VISION_EMBEDDING_DIM,
    )


# RECENCY SCORE — EXPONENTIAL DECAY


def _recency(ts: Any, now: float) -> float:
    try:
        if not ts:
            return 1.0
        age = max(now - float(ts), 0.0)
        return 1.0 / (1.0 + age / settings.MEMORY_RECENCY_SCALE)
    except Exception:
        return 1.0


# ROLE WEIGHT


def _role_weight(role: str) -> float:
    return settings.MEMORY_ROLE_WEIGHTS.get(role, 1.0)


# IMPORTANCE SCORE


def _importance(msg: dict) -> float:
    try:
        return max(0.0, min(float(msg.get("importance", 1.0)), 1.0))
    except Exception:
        return 0.5


# MODALITY WEIGHT


def _modality_weight(msg: dict) -> float:
    modality = msg.get("modality", "text")
    weights = {
        "text": 1.0,
        "image": 1.05,
        "audio": 1.1,
        "video": 1.1,
        "table": 1.0,
    }
    return weights.get(modality, 1.0)


# DEDUP BY CONTENT HASH


def _dedup(history: list[dict]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for m in history:
        try:
            h = _hash(m)
            if h in seen:
                continue
            seen.add(h)
            out.append(m)
        except Exception:
            continue
    return out


# PII SCRUB — STRIP SENSITIVE CONTENT BEFORE SCORING


def _scrub_pii(text: str) -> str:
    if not settings.PII_DETECTION_ENABLED:
        return text
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        entities = getattr(
            settings,
            "PII_ENTITIES",
            [
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "CREDIT_CARD",
                "LOCATION",
                "IP_ADDRESS",
            ],
        )
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text, entities=entities, language="en")
        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pii_scrub_failed", error=str(exc))
    return text


# ADAPTIVE THRESHOLD — LOWER THRESHOLD WHEN FEW CANDIDATES


def _adaptive_threshold(base: float, scored_count: int) -> float:
    if scored_count < 5:
        return base * 0.7
    if scored_count < 10:
        return base * 0.85
    return base


# SCORE NORMALIZATION — MIN-MAX


def _normalize_scores(scored: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    if not scored:
        return scored
    raw = np.array([s for s, _ in scored], dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    max_s = max(float(raw.max()), 1e-6)
    normed = raw / max_s
    return [(float(normed[i]), m) for i, (_, m) in enumerate(scored)]


# EMBED WITH CACHE — thread-safe LRU, bounded at _MAX_CACHE entries

_embed_cache: dict[str, list[float]] = {}
_embed_cache_lock = threading.Lock()
_MAX_CACHE = 500


def _embed_text(text: str, embedder: Any, session_id: str) -> list[float] | None:
    cache_key = hashlib.sha256(text[:500].encode()).hexdigest()
    with _embed_cache_lock:
        if cache_key in _embed_cache:
            return _embed_cache[cache_key]
    try:
        vec = embedder.embed_query(text, session_id=session_id)
        if _valid_embedding(vec):
            with _embed_cache_lock:
                if cache_key not in _embed_cache:
                    if len(_embed_cache) >= _MAX_CACHE:
                        oldest = next(iter(_embed_cache))
                        del _embed_cache[oldest]
                    _embed_cache[cache_key] = vec
            return vec
    except Exception as exc:
        logger.warning("embed_text_failed", error=str(exc), session_id=session_id)
    return None


# MAIN SYNC FILTER


def filter_relevant_history(
    query: str,
    history: list[dict],
    embedder: Any,
    top_k: int | None = None,
    threshold: float | None = None,
    session_id: str = "default",
    modality: str | None = None,
) -> list[dict]:

    if not history:
        return []

    start = time.time()
    query = _normalize(query)
    top_k = top_k or settings.MEMORY_TOP_K
    threshold = threshold or settings.MEMORY_SIM_THRESHOLD

    with tracer.start_as_current_span("memory_filter") as span:
        span.set_attribute("history.size", len(history))
        span.set_attribute("session.id", session_id)

        try:
            # DEDUP AND WINDOW
            history = _dedup(history[-settings.MAX_HISTORY_MESSAGES :])

            # EMBED QUERY
            query_vec = _embed_text(query, embedder, session_id)
            if not query_vec:
                logger.warning(
                    "memory_filter_query_embed_failed",
                    session_id=session_id,
                )
                return history[:top_k]

            now = time.time()
            # Min-heap of (score, msg_id, msg) capped at top_k entries.
            # O(n log k) vs O(n log n) sort-then-slice. id(msg) breaks
            # comparison ties without comparing dicts (which would error).
            _heap: list[tuple[float, int, dict]] = []
            scored_count = 0
            filtered_count = 0
            embed_failed = 0

            for msg in history:
                try:
                    text = str(msg.get("content", "")).strip()
                    if len(text) < 3:
                        continue

                    # MODALITY FILTER
                    if modality and msg.get("modality", "text") != modality:
                        continue

                    role = msg.get("role", "user")
                    text = text[: settings.MAX_PROMPT_CHARS]

                    # PII SCRUB BEFORE EMBEDDING
                    clean_text = _scrub_pii(text)

                    # USE STORED EMBEDDING OR COMPUTE ON-THE-FLY
                    vec = msg.get("embedding")

                    if not _valid_embedding(vec):
                        vec = _embed_text(clean_text, embedder, session_id)

                    if not _valid_embedding(vec):
                        embed_failed += 1
                        continue

                    sim = _cosine(query_vec, vec)
                    scored_count += 1

                    # ADAPTIVE THRESHOLD
                    adaptive = _adaptive_threshold(threshold, scored_count)
                    if sim < adaptive:
                        filtered_count += 1
                        continue

                    modality_boost = _modality_weight(msg)

                    score = (
                        _W_SIM * sim
                        + _W_RECENCY * _recency(msg.get("timestamp"), now)
                        + _W_ROLE * _role_weight(role)
                        + _W_IMPORTANCE * _importance(msg)
                    ) * modality_boost

                    if math.isnan(score) or math.isinf(score):
                        continue

                    entry = (score, id(msg), msg)
                    if len(_heap) < top_k:
                        heapq.heappush(_heap, entry)
                    elif score > _heap[0][0]:
                        heapq.heapreplace(_heap, entry)

                except Exception as exc:
                    logger.warning(
                        "memory_filter_msg_failed",
                        error=str(exc),
                        session_id=session_id,
                    )
                    continue

            if not _heap:
                logger.debug(
                    "memory_filter_no_results",
                    scored_count=scored_count,
                    filtered_count=filtered_count,
                    embed_failed=embed_failed,
                    session_id=session_id,
                )
                span.set_status(Status(StatusCode.OK))
                return []

            # Extract heap as (score, msg) tuples, normalize, then sort k entries.
            # top_k heap always contains the global max, so _normalize_scores
            # produces identical values to normalising the full list then slicing.
            scored = [(s, m) for s, _, m in _heap]
            scored = _normalize_scores(scored)
            scored.sort(key=lambda x: x[0], reverse=True)

            result = [m for _, m in scored]
            latency = round(time.time() - start, 3)

            _filter_duration.labels(status="success").observe(latency)
            _filter_results.observe(len(result))

            span.set_attribute("results.count", len(result))
            span.set_attribute("scored.count", scored_count)
            span.set_attribute("filtered.count", filtered_count)
            span.set_status(Status(StatusCode.OK))

            logger.debug(
                "memory_filter_success",
                selected=len(result),
                scored_count=scored_count,
                filtered_count=filtered_count,
                embed_failed=embed_failed,
                latency=latency,
                session_id=session_id,
            )

            return result

        except Exception as exc:
            latency = round(time.time() - start, 3)
            error_type = type(exc).__name__

            _filter_duration.labels(status="error").observe(latency)
            _filter_errors.labels(error_type=error_type).inc()

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)

            logger.error(
                "memory_filter_failed",
                error=str(exc),
                error_type=error_type,
                session_id=session_id,
            )
            return []


# ASYNC WRAPPER


async def filter_relevant_history_async(
    query: str,
    history: list[dict],
    embedder: Any,
    top_k: int | None = None,
    threshold: float | None = None,
    session_id: str = "default",
    modality: str | None = None,
) -> list[dict]:

    async with _semaphore:
        return await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: filter_relevant_history(
                query,
                history,
                embedder,
                top_k,
                threshold,
                session_id,
                modality,
            ),
        )
