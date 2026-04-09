"""
query_decomposer.py

Breaks complex queries into smaller sub-queries
for better retrieval and reasoning.
"""

from typing import List
import re
import logging

# Logger
logger = logging.getLogger(__name__)


class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        """

        try:
            logger.debug("[QueryDecomposer] Decomposition started")

            prompt = f"""
You are an expert AI assistant.

Break the following query into clear, independent questions.

STRICT RULES:
- Return ONLY questions
- Do NOT write explanations
- Do Not write "..."
- Each question must be meaningful and complete
- Maximum 4 questions

FORMAT (STRICT)
1. Question one?
2. Question two?
3. Question three?

Query:
{query}
"""

            response = self.llm.generate(prompt)

            logger.debug("[QueryDecomposer] LLM response received")

            # Extract numbered questions
            matches = re.findall(r"\d+\.\s*(.+?\?)", response)

            sub_queries = [m.strip() for m in matches if len(m.strip()) > 5]

            # Fallback if decomposition fails
            if not sub_queries:
                logger.warning("[QueryDecomposer] No decomposition found, using original query")
                return [query]

            logger.debug(f"[QueryDecomposer] Sub-queries generated count={len(sub_queries)}")

            return sub_queries[:4]

        except Exception as e:
            logger.error(f"[QueryDecomposer] Failed | error={str(e)}")
            return [query]