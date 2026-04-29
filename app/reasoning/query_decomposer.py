import re
import time
from typing import List

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class QueryDecomposer:

    def __init__(self, llm):
        self.llm = llm
        self.max_subqueries = settings.MAX_SUBQUERIES

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  COMPLEXITY CHECK (IMPROVED) 
    def _is_complex(self, query: str) -> bool:

        q = query.lower()

        if len(q.split()) > settings.DECOMPOSITION_MIN_WORDS:
            return True

        keywords = getattr(settings, "DECOMPOSITION_KEYWORDS", [
            "and", "compare", "difference", "vs",
            "multiple", "steps", "process", "how"
        ])

        if any(k in q for k in keywords):
            return True

        # MULTI-INTENT DETECTION
        if "?" in q and q.count("?") > 1:
            return True

        return False

    #  SAFE PROMPT 
    def _build_prompt(self, query: str) -> str:

        instruction = (
            "You are an expert query planner.\n\n"
            "Break the query into independent retrieval questions.\n\n"
            "Rules:\n"
            f"- Max {self.max_subqueries} questions\n"
            "- No overlap\n"
            "- Preserve intent\n"
            "- Each query must be standalone\n"
            "- No explanation\n\n"
        )

        format_block = (
            "Format:\n"
            "1. Question\n"
            "2. Question\n\n"
        )

        body = f"Query:\n{query}"

        max_chars = settings.MAX_PROMPT_CHARS

        available = max_chars - len(instruction) - len(format_block) - 50
        body = body[:available]

        return instruction + format_block + body

    #  ROBUST PARSER 
    def _parse_response(self, text: str) -> List[str]:

        if not text:
            return []

        try:
            lines = text.split("\n")

            queries = []

            for line in lines:
                line = line.strip()

                # HANDLE MULTIPLE FORMATS
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                line = re.sub(r"^[-•]\s*", "", line)

                if len(line) < 5:
                    continue

                # ENSURE QUESTION-LIKE
                if not line.endswith("?"):
                    line = line + "?"

                queries.append(line)

            return queries

        except Exception:
            return []

    #  FILTER 
    def _filter_queries(self, queries: List[str]) -> List[str]:

        filtered = []
        seen = set()

        for q in queries:

            q_norm = q.lower().strip()

            if len(q_norm.split()) < 3:
                continue

            if q_norm in seen:
                continue

            seen.add(q_norm)
            filtered.append(q.strip())

        return filtered[:self.max_subqueries]

    #  FALLBACK 
    def _fallback(self, query: str) -> List[str]:

        # SIMPLE SPLIT HEURISTIC
        parts = re.split(r"\band\b|\bthen\b|\balso\b", query)

        parts = [p.strip() for p in parts if len(p.strip()) > 5]

        if len(parts) > 1:
            return parts[:self.max_subqueries]

        return [query]

    #  MAIN 
    def decompose(self, query: str) -> List[str]:

        if not query or not query.strip():
            return []

        start = time.time()

        try:
            logger.info("[QueryDecomposer][START]")

            query = self._normalize(query)

            if not self._is_complex(query):
                return [query]

            prompt = self._build_prompt(query)

            #  LLM 
            t_llm = time.time()
            response = self.llm.generate(prompt)
            llm_latency = round(time.time() - t_llm, 2)

            parsed = self._parse_response(response)

            if not parsed:
                logger.warning("[QueryDecomposer] fallback triggered")
                return self._fallback(query)

            sub_queries = self._filter_queries(parsed)

            if not sub_queries:
                return self._fallback(query)

            latency = round(time.time() - start, 2)

            logger.info(
                "[QueryDecomposer][SUCCESS] count=%s latency=%ss llm=%ss",
                len(sub_queries),
                latency,
                llm_latency
            )

            return sub_queries

        except Exception as e:
            logger.error("[QueryDecomposer][FAILED] %s", str(e))
            return [query]