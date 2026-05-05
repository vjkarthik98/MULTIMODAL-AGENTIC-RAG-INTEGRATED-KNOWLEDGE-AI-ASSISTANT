import time
import hashlib
from typing import List, Dict, Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningEngine:

    def __init__(self, llm):
        self.llm = llm
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS

    #  HASH 
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    #  NORMALIZE 
    def _normalize(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    #  MAIN 
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = "",
        session_id: str = "default"
    ) -> Dict[str, Any]:

        if not query:
            return self._fallback()

        start = time.time()

        try:
            query = self._normalize(query)

            knowledge = self._prepare_knowledge(retrieved_docs)
            memory = self._prepare_memory(memory_context)

            prompt = self._build_prompt(query, knowledge, memory)

            if len(prompt) > self.max_prompt_chars:
                prompt = self._truncate(prompt)

            #  LLM 
            t_llm = time.time()
            response = self.llm.generate(prompt)
            llm_latency = round(time.time() - t_llm, 2)

            parsed = self._parse(response, retrieved_docs)

            logger.info(
                event="reasoning_success",
                latency=round(time.time() - start, 2),
                llm_latency=llm_latency
            )

            return parsed

        except Exception as e:
            logger.error(event="reasoning_failed", error=str(e))
            return self._fallback()

    #  TRUNCATION 
    def _truncate(self, prompt: str) -> str:

        parts = prompt.split("KNOWLEDGE:")

        if len(parts) < 2:
            return prompt[:self.max_prompt_chars]

        header = parts[0]
        body = "KNOWLEDGE:" + parts[1]

        allowed = self.max_prompt_chars - len(header) - 20

        return header + body[:allowed]

    #  KNOWLEDGE 
    def _prepare_knowledge(self, docs: List[Dict]) -> str:

        if not docs:
            return ""

        max_docs = settings.RAG_TOP_K
        max_chars = settings.RAG_DOC_MAX_CHARS

        seen = set()
        parts = []

        for d in docs[:max_docs]:

            text = self._normalize(d.get("text", ""))
            if not text:
                continue

            h = self._hash(text[:200])
            if h in seen:
                continue
            seen.add(h)

            meta = d.get("metadata", {}) or {}

            source = meta.get("source", "unknown")
            modality = meta.get("modality", "text")

            parts.append(
                f"[{modality.upper()} | {source}] {text[:max_chars]}"
            )

        return "\n\n".join(parts)

    #  MEMORY 
    def _prepare_memory(self, memory: str) -> str:

        if not memory:
            return ""

        memory = self._normalize(memory)

        return memory[:settings.MEMORY_MAX_CONTEXT_CHARS]

    #  PROMPT 
    def _build_prompt(self, query: str, knowledge: str, memory: str) -> str:

        instruction = (
            "You are a grounded AI system.\n"
            "Rules:\n"
            "- Use ONLY provided knowledge\n"
            "- If missing → say 'I don't know'\n"
            "- No hallucination\n"
            "- Be concise and factual\n\n"
        )

        memory_block = f"MEMORY:\n{memory}\n\n" if memory else ""
        knowledge_block = f"KNOWLEDGE:\n{knowledge}\n\n"
        query_block = f"QUERY:\n{query}\n\n"

        output_format = (
            "FORMAT:\n"
            "Answer: <text>\n"
            "Confidence: <0-1>\n"
            "Sources Used: <int>\n"
        )

        return instruction + memory_block + knowledge_block + query_block + output_format

    #  PARSE 
    def _parse(self, text: str, docs: List[Dict]) -> Dict[str, Any]:

        if not text:
            return self._fallback()

        try:
            answer = ""
            confidence = 0.5
            sources = 0

            for line in text.split("\n"):
                l = line.lower().strip()

                if l.startswith("answer"):
                    answer = line.split(":", 1)[-1].strip()

                elif l.startswith("confidence"):
                    try:
                        confidence = float(line.split(":", 1)[-1].strip())
                    except:
                        confidence = 0.5

                elif l.startswith("sources"):
                    try:
                        sources = int(line.split(":", 1)[-1].strip())
                    except:
                        sources = 0

            if not answer:
                answer = text.strip()

            confidence = max(0.0, min(confidence, 1.0))
            sources = min(sources, len(docs))

            return {
                "answer": answer,
                "confidence": confidence,
                "sources_used": sources
            }

        except Exception:
            return self._fallback(text)

    #  FALLBACK 
    def _fallback(self, text: str = "") -> Dict[str, Any]:
        return {
            "answer": text.strip() if text else "I couldn't generate a reliable answer.",
            "confidence": 0.3,
            "sources_used": 0
        }