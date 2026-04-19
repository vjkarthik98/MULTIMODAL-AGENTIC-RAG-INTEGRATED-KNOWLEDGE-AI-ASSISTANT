from typing import List
import re
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm

    # Main API
    def decompose(self, query: str) -> List[str]:
        try:
            logger.info("[QueryDecomposer] Started")

            # Step 1: Check if decomposition needed
            if not self._is_complex(query):
                logger.debug("[QueryDecomposer] Query not complex -> skipping")
                return [query]
            
            # Step 2: LLM Decomposition
            prompt = self._build_prompt(query)

            response = self.llm.generate(prompt)

            # Step 3: Parse
            sub_queries = self._parse_response(response)

            if not sub_queries:
                logger.warning("[QueryDecomposer] Fallback to original query")
                return [query]
            
            # Step 4: Dedup + Clean
            sub_queries = self._post_process(sub_queries)

            logger.info(f"[QueryDecomposer] Generated {len(sub_queries)} sub-queries")

            return sub_queries
        
        except Exception as e:
            logger.error(f"[QueryDecomposer] Failed | {str(e)}")
            return [query]
        
    # Complexity Check
    def _is_complex(self, query: str) -> bool:

        q = query.lower()

        # Heuristics
        if len(query.split()) > 12:
            return True
        
        if any(word in q for word in ["and", "compare", "difference", "vs", "mulitiple"]):
            return True
        
        return False
    
    # Prompt
    def _build_prompt(self, query: str) -> str:

        return f"""
You are an expert query planner.

Break the query into independent, retrieval-optimized questions.

STRICT RULES:
- Max 4 questions
- Each question must retrieve different information
- Avoid overlap
- Preserve original intent
- No explanation

FORMAT:
1. Question?
2. Question?
3. Quesition?

Query:
{query}
"""
    # Parser
    def _parse_response(self, text: str) -> List[str]:

        matches = re.findall(r"\d+\.\s*(.+?\?)", text)

        return [m.strip() for m in matches if len(m.strip()) > 5]
    
    # Post Processing
    def _post_process(self, queries: List[str]) -> List[str]:

        seen = set()
        unique = []

        for q in queries:
            q_norm = q.lower().strip()

            if q_norm not in seen:
                seen.add(q_norm)
                unique.append(q.strip())

        return unique[:4]




