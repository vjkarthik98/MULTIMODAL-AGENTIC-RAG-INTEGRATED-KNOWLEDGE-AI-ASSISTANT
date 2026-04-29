import time

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class PromptBuilder:

    def __init__(self):
        self.max_chars = settings.MAX_PROMPT_CHARS

    # CLEAN
    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    # SAFE TRUNCATION
    def _truncate(self, text: str, limit: int) -> str:
        if not text:
            return ""
        return text[:limit].rstrip() + ("..." if len(text) > limit else "")

    # DEDUP
    def _deduplicate(self, memory: str, context: str):

        if not memory or not context:
            return memory, context

        if memory[:200] in context:
            context = context.replace(memory[:200], "")

        return memory, context

    # DETECT STRUCTURED QUERY
    def _is_structured_query(self, query: str) -> bool:

        q = query.lower()

        keywords = [
            "table of contents",
            "toc",
            "which page",
            "page number",
            "how many",
            "which section",
            "table",
            "row",
            "column"
        ]

        return any(k in q for k in keywords)

    # BUILD SYSTEM PROMPT
    def _build_system_block(self, structured: bool) -> str:

        if structured:
            return (
                "<s>[INST]\n\n"
                "You are a precise information extraction system.\n\n"
                "STRICT RULES:\n"
                "- Extract ONLY from provided context\n"
                "- Do NOT generate or infer beyond context\n"
                "- Identify correct row and column alignment\n"
                "- Return ONLY the exact value\n"
                "- Do NOT explain\n"
                "- Do NOT include extra text\n"
                "- If not found, return: I don't know\n\n"
            )

        return (
            "<s>[INST]\n\n"
            "You are a highly reliable AI assistant.\n\n"
            "STRICT RULES:\n"
            "- Answer ONLY from provided context\n"
            "- NEVER hallucinate\n"
            "- If information is missing say: I don't know\n"
            "- Do NOT assume facts\n"
            "- Be precise and factual\n"
            "- Avoid repetition\n"
            "- Prefer grounded evidence\n\n"
        )

    # MAIN
    def build_prompt(
        self,
        query: str,
        context: str,
        memory: str = "",
        session_id: str = "default"
    ) -> str:

        start = time.time()

        try:
            # NORMALIZE
            query = self._clean(query)
            context = self._clean(context)
            memory = self._clean(memory)

            if not query:
                raise ValueError("query cannot be empty")

            # DETECT MODE
            structured = self._is_structured_query(query)

            # DEDUP
            memory, context = self._deduplicate(memory, context)

            # SMART BUDGET
            memory_budget = int(self.max_chars * 0.25)
            context_budget = int(self.max_chars * 0.5)

            memory = self._truncate(memory, memory_budget)
            context = self._truncate(context, context_budget)

            # SYSTEM BLOCK
            system_block = self._build_system_block(structured)

            # MEMORY
            memory_block = ""
            if memory:
                memory_block = (
                    "=====\n"
                    "MEMORY\n"
                    "=====\n"
                    f"{memory}\n\n"
                )

            # CONTEXT
            context_block = ""
            if context:
                context_block = (
                    "=====\n"
                    "CONTEXT\n"
                    "=====\n"
                    f"{context}\n\n"
                )

            # QUERY
            if structured:
                query_block = (
                    "=====\n"
                    "TASK\n"
                    "=====\n"
                    "Find the exact value from the structured data.\n"
                    f"Query: {query}\n\n"
                )
            else:
                query_block = (
                    "=====\n"
                    "USER QUERY\n"
                    "=====\n"
                    f"{query}\n\n"
                )

            # OUTPUT FORMAT
            if structured:
                output_block = (
                    "=====\n"
                    "OUTPUT\n"
                    "=====\n"
                    "<exact answer only>\n\n"
                    "[/INST]"
                )
            else:
                output_block = (
                    "=====\n"
                    "OUTPUT FORMAT\n"
                    "=====\n"
                    "Answer:\n"
                    "<final answer>\n\n"
                    "Confidence:\n"
                    "<0 to 1>\n\n"
                    "[/INST]"
                )

            # COMPOSE
            prompt = (
                system_block +
                memory_block +
                context_block +
                query_block +
                output_block
            )

            # SAFE GLOBAL TRUNCATION
            if len(prompt) > self.max_chars:

                logger.warning("[PromptBuilder] safe truncation applied")

                fixed_parts = system_block + query_block + output_block
                allowed_middle = self.max_chars - len(fixed_parts) - 20

                middle = (memory_block + context_block)[:allowed_middle]

                prompt = system_block + middle + query_block + output_block

            latency = round(time.time() - start, 2)

            logger.debug(
                "[PromptBuilder][SUCCESS] session_id=%s | structured=%s | size=%s | latency=%ss",
                session_id,
                structured,
                len(prompt),
                latency
            )

            return prompt

        except Exception as e:
            logger.error(
                "[PromptBuilder][FAILED] session_id=%s | %s",
                session_id,
                str(e)
            )
            raise