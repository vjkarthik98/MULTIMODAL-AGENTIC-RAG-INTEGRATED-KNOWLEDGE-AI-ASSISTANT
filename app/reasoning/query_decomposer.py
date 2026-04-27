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

    # MAIN 
    def decompose(self, query: str) -> List[str]:

        if not query or not query.strip():
            return []

        start = time.time()

        try:
            logger.info("[QueryDecomposer][START]")

            query = self._sanitize(query)

            # Step 1: Complexity check
            if not self._is_complex(query):
                return [query]

            # Step 2: Prompt
            prompt = self._build_prompt(query)

            if len(prompt) > settings.MAX_PROMPT_CHARS:
                prompt = prompt[-settings.MAX_PROMPT_CHARS:]

            # Step 3: LLM call
            response = self.llm.generate(prompt)

            # Step 4: Parse
            sub_queries = self._parse_response(response)

            if not sub_queries:
                logger.warning("[QueryDecomposer] fallback to original")
                return [query]

            # Step 5: Post-process
            sub_queries = self._post_process(sub_queries)

            latency = round(time.time() - start, 2)

            logger.info(
                "[QueryDecomposer][SUCCESS] count=%s latency=%ss",
                len(sub_queries),
                latency
            )

            return sub_queries

        except Exception as e:
            logger.error("[QueryDecomposer][FAILED] %s", str(e))
            return [query]

    # SANITIZE 
    def _sanitize(self, query: str) -> str:
        query = query.strip()

        if len(query) > settings.MAX_PROMPT_CHARS:
            query = query[:settings.MAX_PROMPT_CHARS]

        return query

    # COMPLEXITY CHECK 
    def _is_complex(self, query: str) -> bool:
        q = query.lower()

        if len(query.split()) > settings.DECOMPOSITION_MIN_WORDS:
            return True

        keywords = getattr(settings, "DECOMPOSITION_KEYWORDS", [
            "and", "compare", "difference", "vs", "multiple"
        ])

        return any(word in q for word in keywords)

    # PROMPT 
    def _build_prompt(self, query: str) -> str:
        return (
            "You are an expert query planner.\n\n"
            "Break the query into independent retrieval questions.\n\n"
            "Rules:\n"
            "- Max {n} questions\n"
            "- No overlap\n"
            "- Preserve intent\n"
            "- No explanation\n\n"
            "Format:\n"
            "1. Question?\n"
            "2. Question?\n\n"
            f"Query:\n{query}"
        ).format(n=self.max_subqueries)

    # PARSER 
    def _parse_response(self, text: str) -> List[str]:
        if not text:
            return []

        try:
            matches = re.findall(r"\d+\.\s*(.+?\?)", text)
            return [m.strip() for m in matches if len(m.strip()) > 5]
        except Exception:
            return []

    # POST PROCESS 
    def _post_process(self, queries: List[str]) -> List[str]:
        seen = set()
        unique = []

        for q in queries:
            q_norm = q.lower().strip()

            if not q_norm:
                continue

            if q_norm not in seen:
                seen.add(q_norm)
                unique.append(q.strip())

        return unique[:self.max_subqueries]