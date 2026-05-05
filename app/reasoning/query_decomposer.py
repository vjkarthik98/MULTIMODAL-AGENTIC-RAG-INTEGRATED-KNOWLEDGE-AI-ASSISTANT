import re
import time
import hashlib
from typing import List

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QueryDecomposer:

    def __init__(self, llm):
        self.llm = llm
        self.max_subqueries = settings.MAX_SUBQUERIES

    #  HASH 
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()

    #  NORMALIZE 
    def _normalize(self, q: str) -> str:
        return " ".join(q.strip().split())

    #  COMPLEXITY 
    def _is_complex(self, q: str) -> bool:

        ql = q.lower()
        words = ql.split()

        if len(words) > settings.DECOMPOSITION_MIN_WORDS:
            return True

        keywords = getattr(settings, "DECOMPOSITION_KEYWORDS", [
            "compare", "difference", "vs", "process", "steps", "multiple"
        ])

        if any(k in ql for k in keywords):
            return True

        if q.count("?") > 1:
            return True

        return False

    #  PROMPT 
    def _build_prompt(self, query: str) -> str:

        instruction = (
            "Break into independent retrieval queries.\n"
            f"Max {self.max_subqueries}. No explanation.\n"
            "Each must be standalone.\n\n"
        )

        format_block = "1. Query\n2. Query\n\n"

        max_chars = settings.MAX_PROMPT_CHARS

        body_limit = max_chars - len(instruction) - len(format_block) - 50
        query = query[:body_limit]

        return f"{instruction}{format_block}Query:\n{query}"

    #  PARSE 
    def _parse(self, text: str) -> List[str]:

        if not text:
            return []

        lines = text.split("\n")
        out = []

        for line in lines:
            line = line.strip()

            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-•]\s*", "", line)

            if len(line) < 5:
                continue

            if not line.endswith("?"):
                line += "?"

            out.append(line)

        return out

    #  FILTER 
    def _filter(self, queries: List[str]) -> List[str]:

        seen = set()
        out = []

        for q in queries:
            norm = q.lower().strip()

            if len(norm.split()) < 3:
                continue

            h = self._hash(norm)
            if h in seen:
                continue

            seen.add(h)
            out.append(q.strip())

        return out[:self.max_subqueries]

    #  FALLBACK 
    def _fallback(self, query: str) -> List[str]:

        parts = re.split(r"\band\b|\bthen\b|\balso\b", query)

        parts = [p.strip() for p in parts if len(p.strip()) > 5]

        if len(parts) > 1:
            return parts[:self.max_subqueries]

        return [query]

    #  MAIN 
    def decompose(self, query: str) -> List[str]:

        if not query:
            return []

        start = time.time()

        try:
            query = self._normalize(query)

            if not self._is_complex(query):
                return [query]

            prompt = self._build_prompt(query)

            # latency guard
            t_llm = time.time()
            response = self.llm.generate(prompt)
            llm_latency = time.time() - t_llm

            if llm_latency > settings.MODEL_TIMEOUT_SEC:
                logger.warning(event="decompose_timeout")
                return self._fallback(query)

            parsed = self._parse(response)

            if not parsed:
                return self._fallback(query)

            filtered = self._filter(parsed)

            if not filtered:
                return self._fallback(query)

            logger.info(
                event="decompose_success",
                count=len(filtered),
                latency=round(time.time() - start, 2),
                llm_latency=round(llm_latency, 2)
            )

            return filtered

        except Exception as e:
            logger.error(event="decompose_failed", error=str(e))
            return [query]