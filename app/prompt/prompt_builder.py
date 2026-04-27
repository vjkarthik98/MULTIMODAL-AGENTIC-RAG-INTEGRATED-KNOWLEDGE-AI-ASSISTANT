import time

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class PromptBuilder:
    def __init__(self):
        self.max_chars = settings.MAX_PROMPT_CHARS

    def _sanitize(self, text: str) -> str:
        if not text:
            return ""

        text = str(text).strip()

        if len(text) > self.max_chars:
            return text[:self.max_chars]

        return text

    def _truncate_block(self, text: str, limit: int) -> str:
        if not text:
            return ""

        if len(text) <= limit:
            return text

        return text[:limit] + "..."

    def build_prompt(
        self,
        query: str,
        context: str,
        memory: str = "",
        session_id: str = "default"
    ) -> str:

        start = time.time()

        try:
            # SANITIZE 
            query = self._sanitize(query)
            context = self._sanitize(context)
            memory = self._sanitize(memory)

            if not query:
                raise ValueError("query cannot be empty")

            # CONTROL BLOCK SIZES 
            context = self._truncate_block(
                context,
                settings.CONTEXT_MAX_CHARS
            )

            memory = self._truncate_block(
                memory,
                settings.MEMORY_MAX_CONTEXT_CHARS
            )

            # BUILD PROMPT 
            prompt = (
                "<s>[INST]\n\n"
                "You are a highly reliable AI assistant.\n\n"

                "====================\n"
                "MEMORY\n"
                "====================\n"
                f"{memory}\n\n"

                "====================\n"
                "CONTEXT\n"
                "====================\n"
                f"{context}\n\n"

                "====================\n"
                "USER QUERY\n"
                "====================\n"
                f"{query}\n\n"

                "====================\n"
                "INSTRUCTIONS\n"
                "====================\n"
                "- Answer ONLY using provided context\n"
                "- Do NOT hallucinate\n"
                '- If missing -> say "I don\'t know"\n'
                "- Be concise and structured\n"
                "- Combine multiple sources logically\n"
                "- Respect modality differences:\n"
                "  * Audio -> spoken content\n"
                "  * Image/Video -> visual description\n"
                "  * Text -> factual information\n\n"

                "====================\n"
                "OUTPUT FORMAT\n"
                "====================\n"
                "Answer:\n"
                "<final answer>\n\n"
                "Confidence:\n"
                "<0 to 1>\n\n"

                "[/INST]"
            )

            # FINAL SAFETY 
            if len(prompt) > self.max_chars:
                logger.warning("[PromptBuilder] truncating prompt")
                prompt = prompt[-self.max_chars:]

            latency = round(time.time() - start, 2)

            logger.debug(
                "[PromptBuilder][SUCCESS] session_id=%s | size=%s | latency=%ss",
                session_id,
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