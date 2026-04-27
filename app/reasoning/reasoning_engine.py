import time
from typing import List, Dict, Any

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ReasoningEngine:

    def __init__(self, llm):
        self.llm = llm
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS

    # MAIN 
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = "",
        session_id: str = "default"
    ) -> Dict[str, Any]:

        start = time.time()

        try:
            logger.info("[ReasoningEngine][START]")

            query = self._sanitize(query)

            knowledge = self._prepare_knowledge(retrieved_docs)
            memory = self._prepare_memory(memory_context)

            prompt = self._build_prompt(query, knowledge, memory)

            # Global safety
            if len(prompt) > self.max_prompt_chars:
                logger.warning("[ReasoningEngine] truncating prompt")
                prompt = prompt[-self.max_prompt_chars:]

            response = self.llm.generate(prompt)

            parsed = self._parse_response(response)

            latency = round(time.time() - start, 2)

            logger.info(
                "[ReasoningEngine][SUCCESS] latency=%ss",
                latency
            )

            return parsed

        except Exception as e:
            logger.error("[ReasoningEngine][FAILED] %s", str(e))

            return {
                "answer": "I couldn't generate a reliable answer.",
                "confidence": 0.2,
                "sources_used": 0
            }

    # SANITIZE 
    def _sanitize(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        if len(text) > self.max_prompt_chars:
            text = text[:self.max_prompt_chars]

        return text

    # KNOWLEDGE 
    def _prepare_knowledge(self, docs: List[Dict]) -> str:
        if not docs:
            return ""

        max_docs = settings.RAG_TOP_K
        max_chars_per_doc = settings.RAG_DOC_MAX_CHARS

        selected = []

        for d in docs[:max_docs]:
            try:
                text = str(d.get("text", "")).strip()
                if not text:
                    continue

                source = d.get("source", "unknown")

                text = text[:max_chars_per_doc]

                selected.append(f"[Source: {source}] {text}")

            except Exception:
                continue

        return "\n\n".join(selected)

    # MEMORY 
    def _prepare_memory(self, memory: str) -> str:
        if not memory:
            return ""

        return memory[:settings.MEMORY_MAX_CONTEXT_CHARS]

    # PROMPT 
    def _build_prompt(self, query: str, knowledge: str, memory: str) -> str:
        return (
            "You are a highly reliable AI assistant.\n\n"
            "Rules:\n"
            "- Answer ONLY from context\n"
            "- Do NOT hallucinate\n"
            '- If missing -> say "I don\'t know"\n\n'

            "MEMORY:\n"
            f"{memory}\n\n"

            "KNOWLEDGE:\n"
            f"{knowledge}\n\n"

            "QUERY:\n"
            f"{query}\n\n"

            "OUTPUT:\n"
            "Answer:\n"
            "<final answer>\n\n"
            "Confidence:\n"
            "<0-1>\n\n"
            "Sources Used:\n"
            "<number>"
        )

    # PARSER 
    def _parse_response(self, text: str) -> Dict[str, Any]:

        if not text:
            return {
                "answer": "",
                "confidence": 0.5,
                "sources_used": 0
            }

        try:
            answer = []
            confidence = 0.7
            sources = 0

            lines = text.split("\n")
            capture_answer = False

            for line in lines:
                line_clean = line.strip()

                if line_clean.lower().startswith("answer"):
                    capture_answer = True
                    continue

                if line_clean.lower().startswith("confidence"):
                    capture_answer = False
                    try:
                        confidence = float(line_clean.split(":")[1].strip())
                    except Exception:
                        confidence = 0.5
                    continue

                if line_clean.lower().startswith("sources"):
                    try:
                        sources = int(line_clean.split(":")[1].strip())
                    except Exception:
                        sources = 0
                    continue

                if capture_answer and line_clean:
                    answer.append(line_clean)

            final_answer = " ".join(answer).strip()

            if not final_answer:
                final_answer = text.strip()

            return {
                "answer": final_answer,
                "confidence": confidence,
                "sources_used": sources
            }

        except Exception as e:
            logger.error("[ReasoningEngine][PARSE_FAIL] %s", str(e))

            return {
                "answer": text.strip(),
                "confidence": 0.5,
                "sources_used": 0
            }