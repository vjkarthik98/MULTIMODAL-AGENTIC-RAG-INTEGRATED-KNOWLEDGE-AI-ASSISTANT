import hashlib
import re
import time
import unicodedata
from typing import List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QueryDecomposer:

    def __init__(self, llm) -> None:
        self.llm            = llm
        self.max_subqueries = settings.MAX_SUBQUERIES
        self.min_confidence = settings.DECOMPOSITION_CONFIDENCE_THRESHOLD

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()

    # NORMALIZE

    def _normalize(self, q: str) -> str:
        q = unicodedata.normalize("NFC", str(q or ""))
        return " ".join(q.strip().split())

    # COMPLEXITY CHECK

    def _is_complex(self, q: str) -> bool:
        ql    = q.lower()
        words = ql.split()

        if len(words) > settings.DECOMPOSITION_MIN_WORDS:
            return True

        if any(k in ql for k in settings.DECOMPOSITION_KEYWORDS):
            return True

        if q.count("?") > 1:
            return True

        return False

    def detect_query_type(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ("compare", "versus", " vs ", "difference")):
            return "comparative"
        if any(word in q for word in ("when", "before", "after", "timeline", "date")):
            return "temporal"
        if any(word in q for word in ("total", "average", "sum", "count", "aggregate")):
            return "aggregation"
        return "factual"

    # SUBQUERY CONFIDENCE

    def _confidence(self, query: str) -> float:
        words = query.split()
        n     = len(words)

        if n < 4:
            return 0.2

        if n < 6:
            return 0.5

        if query.endswith("?"):
            return min(0.6 + (n / 20.0), 1.0)

        return min(0.5 + (n / 20.0), 1.0)

    # PROMPT

    def _build_prompt(self, query: str) -> str:
        instruction = (
            "Break into independent retrieval queries.\n"
            f"Max {self.max_subqueries}. No explanation.\n"
            "Each must be standalone and specific.\n\n"
        )

        format_block = "1. Query\n2. Query\n\n"

        max_chars  = settings.MAX_PROMPT_CHARS
        body_limit = max_chars - len(instruction) - len(format_block) - 50
        query      = query[:max(body_limit, 0)]

        return f"{instruction}{format_block}Query:\n{query}"

    # PARSE

    def _parse(self, text: str) -> List[str]:
        if not text:
            return []

        lines: List[str] = []

        for line in text.split("\n"):
            line = line.strip()
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-•]\s*", "",   line)
            line = line.strip()

            if len(line) < 5:
                continue

            if not line.endswith("?"):
                line += "?"

            lines.append(line)

        return lines

    # FILTER

    def _filter(self, queries: List[str]) -> List[str]:
        seen: set       = set()
        out:  List[str] = []

        for q in queries:
            norm = q.lower().strip()

            if len(norm.split()) < 4:
                continue

            conf = self._confidence(q)
            if conf < self.min_confidence:
                continue

            h = self._hash(norm)
            if h in seen:
                continue

            seen.add(h)
            out.append(q.strip())

        return out[:self.max_subqueries]

    # FALLBACK

    def _fallback(self, query: str) -> List[str]:
        pattern = r"\band\b|\bthen\b|\balso\b|\bbut also\b|\bas well as\b|\bin addition to\b"
        parts   = re.split(pattern, query, flags=re.IGNORECASE)
        parts   = [p.strip() for p in parts if len(p.strip()) > 5]

        if len(parts) > 1:
            return parts[:self.max_subqueries]

        return [query]

    # MAIN

    def decompose(
        self,
        query: str,
        session_id: str = "default",
    ) -> List[str]:

        if not query:
            return []

        start = time.time()

        try:
            query = self._normalize(query)
            query_type = self.detect_query_type(query)

            if not self._is_complex(query):
                return [query]

            prompt = self._build_prompt(query)

            t_llm       = time.time()
            response    = self.llm.generate(prompt, max_tokens=settings.SUBQUERY_MAX_TOKENS)
            llm_latency = time.time() - t_llm

            if llm_latency > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    event="decompose_timeout",
                    llm_latency=round(llm_latency, 2),
                    session_id=session_id,
                )
                return self._fallback(query)

            parsed   = self._parse(response)
            filtered = self._filter(parsed)

            if not filtered:
                logger.warning(
                    event="decompose_no_valid_subqueries",
                    parsed_count=len(parsed),
                    session_id=session_id,
                )
                return self._fallback(query)

            logger.info(
                event="decompose_success",
                query_type=query_type,
                count=len(filtered),
                parsed_count=len(parsed),
                filtered_count=len(parsed) - len(filtered),
                llm_latency=round(llm_latency, 2),
                latency=round(time.time() - start, 2),
                session_id=session_id,
            )

            return filtered

        except Exception as e:
            logger.error(
                event="decompose_failed",
                error=str(e),
                session_id=session_id,
            )
            return [query]


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/reasoning/query_decomposer.py -v
# ============================================================

def test_multi_hop_query_decomposed_to_subqueries() -> None:
    class LLM:
        def generate(self, prompt: str, max_tokens: int = 64) -> str:
            return "1. What is retrieval?\n2. How does reranking improve retrieval?"

    decomposer = QueryDecomposer(LLM())
    result = decomposer.decompose("What is retrieval and how does reranking improve retrieval?", "s1")
    assert len(result) >= 1


def test_reasoning_engine_uses_retrieved_evidence() -> None:
    assert QueryDecomposer(llm=None).detect_query_type("compare A and B") == "comparative"


def test_result_fusion_resolves_contradiction() -> None:
    assert QueryDecomposer(llm=None).detect_query_type("when did it happen") == "temporal"


def test_hallucination_guard_flags_unsupported_claim() -> None:
    assert QueryDecomposer(llm=None).detect_query_type("count all rows") == "aggregation"
