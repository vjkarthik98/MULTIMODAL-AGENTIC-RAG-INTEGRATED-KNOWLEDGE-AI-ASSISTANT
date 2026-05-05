import time

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:

    def __init__(self):
        self.max_chars = settings.MAX_PROMPT_CHARS

    #  CLEAN 
    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    #  TRUNCATE 
    def _truncate(self, text: str, limit: int) -> str:
        if not text:
            return ""
        return text[:limit]

    #  DEDUP 
    def _deduplicate(self, memory: str, context: str):

        if not memory or not context:
            return memory, context

        key = memory[:200]
        if key and key in context:
            context = context.replace(key, "")

        return memory, context

    #  MODE 
    def _is_structured(self, query: str) -> bool:

        q = query.lower()

        return any(k in q for k in [
            "table", "row", "column",
            "page number", "which page",
            "toc", "section"
        ])

    #  SYSTEM 
    def _system(self, structured: bool) -> str:

        if structured:
            return (
                "You are a strict extraction system.\n"
                "- Use ONLY context\n"
                "- Return exact value\n"
                "- No explanation\n"
                "- If missing → I don't know\n\n"
            )

        return (
            "You are a grounded assistant.\n"
            "- Use ONLY context\n"
            "- No hallucination\n"
            "- If unsure → I don't know\n"
            "- Be precise\n\n"
        )

    #  MAIN 
    def build_prompt(
        self,
        query: str,
        context: str,
        memory: str = "",
        session_id: str = "default"
    ) -> str:

        start = time.time()

        try:
            query = self._clean(query)
            context = self._clean(context)
            memory = self._clean(memory)

            if not query:
                raise ValueError("EMPTY_QUERY")

            structured = self._is_structured(query)

            # dedup
            memory, context = self._deduplicate(memory, context)

            # budgets
            mem_budget = int(self.max_chars * 0.25)
            ctx_budget = int(self.max_chars * 0.5)

            memory = self._truncate(memory, mem_budget)
            context = self._truncate(context, ctx_budget)

            # blocks
            system = self._system(structured)

            mem_block = f"MEMORY:\n{memory}\n\n" if memory else ""
            ctx_block = f"CONTEXT:\n{context}\n\n" if context else ""

            if structured:
                query_block = f"TASK:\n{query}\n\n"
                output_block = "OUTPUT:\n<exact answer>"
            else:
                query_block = f"QUERY:\n{query}\n\n"
                output_block = (
                    "FORMAT:\n"
                    "Answer:\n<text>\n"
                    "Confidence:\n<0-1>"
                )

            prompt = system + mem_block + ctx_block + query_block + output_block

            # final guard
            if len(prompt) > self.max_chars:

                fixed = system + query_block + output_block
                allowed = self.max_chars - len(fixed) - 20

                middle = (mem_block + ctx_block)[:allowed]
                prompt = system + middle + query_block + output_block

                logger.warning(event="prompt_truncated")

            logger.debug(
                event="prompt_built",
                size=len(prompt),
                latency=round(time.time() - start, 3)
            )

            return prompt

        except Exception as e:
            logger.error(event="prompt_failed", error=str(e))
            raise