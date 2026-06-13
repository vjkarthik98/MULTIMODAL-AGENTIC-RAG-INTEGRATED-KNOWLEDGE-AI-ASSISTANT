import asyncio
import hashlib
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_decompose_duration = Histogram(
    "query_decomposer_duration_seconds",
    "Query decomposer duration",
    ["status"],
)
_decompose_errors = Counter(
    "query_decomposer_errors_total",
    "Query decomposer errors by type",
    ["error_type"],
)
_subquery_count = Histogram(
    "query_decomposer_subquery_count",
    "Number of subqueries produced per decomposition",
)

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)

# QUERY TYPE LABELS
QUERY_TYPE_FACTUAL      = "factual"
QUERY_TYPE_COMPARATIVE  = "comparative"
QUERY_TYPE_TEMPORAL     = "temporal"
QUERY_TYPE_AGGREGATION  = "aggregation"
QUERY_TYPE_MULTIHOP     = "multihop"
QUERY_TYPE_SIMPLE       = "simple"

# COMPARATIVE KEYWORDS
_COMPARATIVE_KEYWORDS = {
    "compare", "difference", "vs", "versus", "contrast",
    "better", "worse", "pros", "cons", "advantages", "disadvantages",
    "similar", "unlike", "whereas",
}

# TEMPORAL KEYWORDS
_TEMPORAL_KEYWORDS = {
    "when", "before", "after", "during", "since", "until",
    "latest", "recent", "current", "today", "yesterday",
    "history", "timeline", "trend", "evolution",
}

# AGGREGATION KEYWORDS
_AGGREGATION_KEYWORDS = {
    "how many", "count", "total", "sum", "average", "list all",
    "enumerate", "all the", "every", "each",
}

# MULTIHOP INDICATORS
_MULTIHOP_KEYWORDS = {
    "and then", "subsequently", "after which", "as a result",
    "therefore", "because of", "due to", "leading to",
}


# SHA-256 HASH FOR DEDUP

def _hash(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()


# NORMALIZE QUERY

def _normalize(q: str) -> str:
    q = unicodedata.normalize("NFC", str(q or ""))
    return " ".join(q.strip().split())


# QUERY TYPE DETECTION

def detect_query_type(query: str) -> str:
    q      = query.lower()
    tokens = set(q.split())

    if bool(tokens & _COMPARATIVE_KEYWORDS):
        return QUERY_TYPE_COMPARATIVE

    if bool(tokens & _TEMPORAL_KEYWORDS):
        return QUERY_TYPE_TEMPORAL

    if any(kw in q for kw in _AGGREGATION_KEYWORDS):
        return QUERY_TYPE_AGGREGATION

    if any(kw in q for kw in _MULTIHOP_KEYWORDS):
        return QUERY_TYPE_MULTIHOP

    if query.count("?") > 1:
        return QUERY_TYPE_MULTIHOP

    word_count = len(q.split())
    if word_count > settings.DECOMPOSITION_MIN_WORDS:
        if any(k in q for k in settings.DECOMPOSITION_KEYWORDS):
            return QUERY_TYPE_MULTIHOP

    return QUERY_TYPE_FACTUAL


# COMPLEXITY CHECK

def _is_complex(query: str) -> bool:
    ql    = query.lower()
    words = ql.split()

    if len(words) > settings.DECOMPOSITION_MIN_WORDS:
        return True

    if any(k in ql for k in settings.DECOMPOSITION_KEYWORDS):
        return True

    if query.count("?") > 1:
        return True

    query_type = detect_query_type(query)
    if query_type in (
        QUERY_TYPE_COMPARATIVE,
        QUERY_TYPE_MULTIHOP,
        QUERY_TYPE_AGGREGATION,
    ):
        return True

    return False


# SUBQUERY CONFIDENCE SCORING

def _confidence(query: str) -> float:
    words = query.split()
    n     = len(words)

    if n < 4:
        return 0.2

    if n < 6:
        return 0.5

    if query.endswith("?"):
        return min(0.6 + (n / 20.0), 1.0)

    return min(0.5 + (n / 20.0), 1.0)


# PROMPT BUILDER — QUERY-TYPE AWARE

def _build_prompt(query: str, query_type: str, max_subqueries: int) -> str:

    type_hint = {
        QUERY_TYPE_COMPARATIVE: (
            "This is a COMPARATIVE query. "
            "Create separate queries for each entity being compared."
        ),
        QUERY_TYPE_TEMPORAL: (
            "This is a TEMPORAL query. "
            "Create queries for each time period or event sequence."
        ),
        QUERY_TYPE_AGGREGATION: (
            "This is an AGGREGATION query. "
            "Create queries that retrieve the individual facts needed."
        ),
        QUERY_TYPE_MULTIHOP: (
            "This is a MULTI-HOP query. "
            "Create sequential queries where each builds on the previous."
        ),
        QUERY_TYPE_FACTUAL: (
            "This is a FACTUAL query. "
            "Create focused sub-queries targeting different aspects."
        ),
    }.get(query_type, "")

    instruction = (
        f"{type_hint}\n"
        f"Break into {max_subqueries} independent retrieval queries.\n"
        "Rules:\n"
        "- Each query must be self-contained and specific\n"
        "- No explanations — queries only\n"
        "- End each with a question mark\n\n"
    )

    format_block = "\n".join(f"{i + 1}. <query>" for i in range(max_subqueries)) + "\n\n"

    max_chars  = settings.MAX_PROMPT_CHARS
    body_limit = max_chars - len(instruction) - len(format_block) - 50
    query_trunc = query[:max(body_limit, 0)]

    return f"{instruction}{format_block}Query:\n{query_trunc}"


# PARSE LLM RESPONSE INTO SUBQUERY LIST

def _parse(text: str) -> List[str]:
    if not text:
        return []

    lines: List[str] = []

    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•]\s*",       "", line)
        line = line.strip()

        if len(line) < 5:
            continue

        if not line.endswith("?"):
            line += "?"

        lines.append(line)

    return lines


# FILTER SUBQUERIES — DEDUP + CONFIDENCE THRESHOLD

def _filter(
    queries: List[str],
    min_confidence: float,
    max_subqueries: int,
) -> List[str]:
    seen: set        = set()
    out:  List[str]  = []

    for q in queries:
        norm = q.lower().strip()

        if len(norm.split()) < 4:
            continue

        conf = _confidence(q)
        if conf < min_confidence:
            continue

        h = _hash(norm)
        if h in seen:
            continue

        seen.add(h)
        out.append(q.strip())

    return out[:max_subqueries]


# RULE-BASED FALLBACK — NO LLM NEEDED

def _rule_based_fallback(query: str, max_subqueries: int) -> List[str]:
    # SPLIT ON CONJUNCTIONS
    pattern = r"\band\b|\bthen\b|\balso\b|\bbut also\b|\bas well as\b|\bin addition to\b"
    parts   = re.split(pattern, query, flags=re.IGNORECASE)
    parts   = [p.strip() for p in parts if len(p.strip()) > 5]

    if len(parts) > 1:
        result = []
        for p in parts[:max_subqueries]:
            if not p.endswith("?"):
                p += "?"
            result.append(p)
        return result

    # COMPARATIVE SPLIT
    for kw in ("vs", "versus", "compared to", "difference between"):
        if kw in query.lower():
            idx   = query.lower().find(kw)
            left  = query[:idx].strip()
            right = query[idx + len(kw):].strip()
            subs  = []
            if left:
                subs.append(f"What is {left}?")
            if right:
                subs.append(f"What is {right}?")
            subs.append(f"What are the differences between {left} and {right}?")
            return subs[:max_subqueries]

    return [query if query.endswith("?") else query + "?"]


# FINANCE-SPECIFIC DECOMPOSITION PATTERNS (Phase 7)

_QQ_PATTERN = re.compile(
    r'Q([1-4])\s+(?:vs\.?\s*)?(?:and\s+)?Q([1-4])',
    re.IGNORECASE,
)
_CEO_AND_FINANCIALS = re.compile(
    r'(.+?)\s+(?:said|stated|commented|noted|mentioned)\s+(.+)\s+and\s+(.+)',
    re.IGNORECASE,
)


def _finance_decompose(query: str, max_subqueries: int) -> Optional[List[str]]:
    """
    Detect finance-specific patterns and produce targeted sub-queries.
    Returns a list when a pattern matched, or None to fall through to LLM.
    """
    q = query.strip()

    # "Compare Q3 vs Q4 revenue" → ["Q3 revenue?", "Q4 revenue?", "Q3 vs Q4 revenue change?"]
    m = _QQ_PATTERN.search(q)
    if m:
        q1, q2  = m.group(1), m.group(2)
        topic   = q[m.end():].strip(" ?.,") or "results"
        subs = [
            f"Q{q1} {topic}?",
            f"Q{q2} {topic}?",
            f"Q{q1} versus Q{q2} {topic} change?",
        ]
        return subs[:max_subqueries]

    # "CEO said X AND financials show Y" → ["CEO statement about X?", "Y financials?"]
    m2 = _CEO_AND_FINANCIALS.match(q)
    if m2:
        speaker_part   = m2.group(1).strip()
        statement_part = m2.group(2).strip()
        financial_part = m2.group(3).strip()
        subs = [
            f"What did {speaker_part} say about {statement_part}?",
            f"{financial_part} financial data?",
        ]
        return subs[:max_subqueries]

    # "X YoY" or "year-over-year X" → needs current + prior year data
    if re.search(r'\byoy\b|year[- ]over[- ]year', q, re.IGNORECASE):
        base = re.sub(r'\byoy\b|year[- ]over[- ]year', '', q, flags=re.IGNORECASE).strip(" ?.,")
        year_m = re.search(r'\b(20\d{2})\b', q)
        if year_m:
            yr   = int(year_m.group(1))
            subs = [f"{base} {yr}?", f"{base} {yr - 1}?", f"{base} YoY change {yr}?"]
            return subs[:max_subqueries]

    return None


# DEDUPLICATE AGAINST ORIGINAL QUERY

def _dedup_against_original(
    subqueries: List[str],
    original: str,
) -> List[str]:
    orig_hash = _hash(original)
    return [
        q for q in subqueries
        if _hash(q) != orig_hash
    ]


class QueryDecomposer:

    def __init__(self, llm: Any) -> None:
        self.llm            = llm
        self.max_subqueries = settings.MAX_SUBQUERIES
        self.min_confidence = settings.DECOMPOSITION_CONFIDENCE_THRESHOLD

    # LLM CALL WITH RETRY

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=False,
    )
    def _call_llm(self, prompt: str, session_id: str) -> Optional[str]:
        try:
            t_start  = time.time()
            response = self.llm.generate(
                prompt,
                max_tokens=settings.SUBQUERY_MAX_TOKENS,
                temperature=0.0,
                session_id=session_id,
            )
            elapsed = time.time() - t_start

            if elapsed > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    "decomposer_llm_timeout",
                    elapsed=round(elapsed, 2),
                    session_id=session_id,
                )
                return None

            return response

        except Exception as exc:
            logger.warning(
                "decomposer_llm_failed",
                error=str(exc),
                session_id=session_id,
            )
            return None

    # MAIN SYNC DECOMPOSE

    def decompose(
        self,
        query: str,
        session_id: str = "default",
    ) -> List[str]:

        if not query:
            return []

        start = time.time()

        with tracer.start_as_current_span("query_decompose") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("session.id", session_id)

            try:
                query = _normalize(query)

                if not _is_complex(query):
                    span.set_attribute("decompose.skipped", True)
                    span.set_status(Status(StatusCode.OK))
                    return [query]

                # FINANCE PATTERN FAST-PATH — no LLM needed for well-known patterns
                finance_subs = _finance_decompose(query, self.max_subqueries)
                if finance_subs:
                    logger.info(
                        "decompose_finance_pattern",
                        count=len(finance_subs),
                        session_id=session_id,
                    )
                    span.set_attribute("decompose.method", "finance_pattern")
                    span.set_status(Status(StatusCode.OK))
                    return finance_subs

                query_type = detect_query_type(query)
                span.set_attribute("query.type", query_type)

                prompt   = _build_prompt(query, query_type, self.max_subqueries)
                response = self._call_llm(prompt, session_id)

                if not response:
                    logger.warning(
                        "decomposer_llm_no_response",
                        session_id=session_id,
                    )
                    fallback = _rule_based_fallback(query, self.max_subqueries)
                    span.set_attribute("decompose.fallback", True)
                    span.set_status(Status(StatusCode.OK))
                    return fallback

                parsed   = _parse(response)
                filtered = _filter(parsed, self.min_confidence, self.max_subqueries)

                if not filtered:
                    logger.warning(
                        "decomposer_no_valid_subqueries",
                        parsed_count=len(parsed),
                        session_id=session_id,
                    )
                    fallback = _rule_based_fallback(query, self.max_subqueries)
                    span.set_attribute("decompose.fallback", True)
                    span.set_status(Status(StatusCode.OK))
                    return fallback

                # REMOVE SUBQUERIES IDENTICAL TO ORIGINAL
                filtered = _dedup_against_original(filtered, query)

                if not filtered:
                    span.set_status(Status(StatusCode.OK))
                    return [query]

                latency = round(time.time() - start, 2)

                _decompose_duration.labels(status="success").observe(latency)
                _subquery_count.observe(len(filtered))

                span.set_attribute("subqueries.count", len(filtered))
                span.set_attribute("query.type", query_type)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "decompose_success",
                    count=len(filtered),
                    query_type=query_type,
                    parsed_count=len(parsed),
                    filtered_count=len(parsed) - len(filtered),
                    latency=latency,
                    session_id=session_id,
                )

                return filtered

            except Exception as exc:
                latency    = round(time.time() - start, 2)
                error_type = type(exc).__name__

                _decompose_duration.labels(status="error").observe(latency)
                _decompose_errors.labels(error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "decompose_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )
                return [query]

    # ASYNC WRAPPER

    async def decompose_async(
        self,
        query: str,
        session_id: str = "default",
    ) -> List[str]:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.decompose(query, session_id),
            )


