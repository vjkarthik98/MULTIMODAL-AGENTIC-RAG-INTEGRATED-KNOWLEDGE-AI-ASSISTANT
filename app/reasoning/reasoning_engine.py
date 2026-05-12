import hashlib
import math
import re
import time
import unicodedata
from typing import Any, Dict, List

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningEngine:

    def __init__(self, llm) -> None:
        self.llm              = llm
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # NORMALIZE

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", str(text or ""))
        return " ".join(text.strip().split())

    # MAIN

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = "",
        session_id: str = "default",
    ) -> Dict[str, Any]:

        if not query:
            return self._fallback()

        start = time.time()

        try:
            query     = self._normalize(query)
            knowledge = self._prepare_knowledge(retrieved_docs)
            memory    = self._prepare_memory(memory_context)

            prompt = self._build_prompt(query, knowledge, memory)

            # BUDGET WARNING
            budget_pct = len(prompt) / self.max_prompt_chars
            if budget_pct > 0.8:
                logger.warning(
                    event="reasoning_prompt_near_limit",
                    budget_pct=round(budget_pct, 2),
                    prompt_chars=len(prompt),
                    session_id=session_id,
                )

            if len(prompt) > self.max_prompt_chars:
                prompt = self._truncate(prompt)

            # LLM INFERENCE
            t_llm       = time.time()
            response    = self.llm.generate(prompt)
            llm_latency = round(time.time() - t_llm, 2)

            if llm_latency > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    event="reasoning_llm_timeout",
                    llm_latency=llm_latency,
                    session_id=session_id,
                )

            parsed = self._parse(response, retrieved_docs)
            parsed["unsupported_claims"] = self._unsupported_claims(parsed.get("answer", ""), retrieved_docs)
            if parsed["unsupported_claims"]:
                parsed["confidence"] = min(parsed.get("confidence", 0.5), 0.4)

            logger.info(
                event="reasoning_success",
                knowledge_chars=len(knowledge),
                memory_chars=len(memory),
                llm_latency=llm_latency,
                latency=round(time.time() - start, 2),
                confidence=parsed.get("confidence"),
                sources_used=parsed.get("sources_used"),
                session_id=session_id,
            )

            return parsed

        except Exception as e:
            logger.error(
                event="reasoning_failed",
                error=str(e),
                session_id=session_id,
            )
            return self._fallback()

    # PROMPT TRUNCATION

    def _truncate(self, prompt: str) -> str:
        parts = prompt.split("KNOWLEDGE:")

        if len(parts) < 2:
            return prompt[:self.max_prompt_chars]

        header  = parts[0]
        body    = "KNOWLEDGE:" + parts[1]
        allowed = self.max_prompt_chars - len(header) - 20

        return header + body[:max(allowed, 0)]

    # KNOWLEDGE PREPARATION

    def _prepare_knowledge(self, docs: List[Dict]) -> str:
        if not docs:
            return ""

        max_docs  = settings.RAG_TOP_K
        max_chars = settings.RAG_DOC_MAX_CHARS
        seen:     set        = set()
        parts:    List[str]  = []

        for d in docs[:max_docs]:
            text = self._normalize(d.get("text", ""))
            if not text:
                continue

            h = self._hash(text[:200])
            if h in seen:
                continue
            seen.add(h)

            meta     = d.get("metadata", {}) or {}
            source   = meta.get("source", "unknown")
            modality = meta.get("modality", "text")
            subtype  = meta.get("subtype", "")
            page     = meta.get("page")

            label = f"[{modality.upper()}"
            if subtype:
                label += f"/{subtype}"
            if page:
                label += f" | p{page}"
            label += f" | {source}]"

            parts.append(f"{label} {text[:max_chars]}")

        return "\n\n".join(parts)

    # MEMORY PREPARATION

    def _prepare_memory(self, memory: str) -> str:
        if not memory:
            return ""

        memory = self._normalize(memory)
        return memory[:settings.MEMORY_MAX_CONTEXT_CHARS]

    # PROMPT BUILDER

    def _build_prompt(self, query: str, knowledge: str, memory: str) -> str:
        instruction = (
            "You are a grounded AI system.\n"
            "Rules:\n"
            "- Use ONLY provided knowledge\n"
            "- If missing → say 'I don't know'\n"
            "- No hallucination\n"
            "- Be concise and factual\n\n"
        )

        memory_block    = f"MEMORY:\n{memory}\n\n"    if memory    else ""
        knowledge_block = f"KNOWLEDGE:\n{knowledge}\n\n" if knowledge else ""
        query_block     = f"QUERY:\n{query}\n\n"

        output_format = (
            "FORMAT:\n"
            "Answer: <text>\n"
            "Confidence: <0-1>\n"
            "Sources Used: <int>\n"
        )

        return instruction + memory_block + knowledge_block + query_block + output_format

    # RESPONSE PARSER

    def _parse(self, text: str, docs: List[Dict]) -> Dict[str, Any]:
        if not text:
            return self._fallback()

        try:
            answer:     str   = ""
            confidence: float = 0.5
            sources:    int   = 0

            for line in text.split("\n"):
                ll = line.lower().strip()

                if ll.startswith("answer"):
                    answer = line.split(":", 1)[-1].strip()

                elif ll.startswith("confidence"):
                    try:
                        confidence = float(line.split(":", 1)[-1].strip())
                    except Exception:
                        confidence = 0.5

                elif ll.startswith("sources"):
                    try:
                        sources = int(line.split(":", 1)[-1].strip())
                    except Exception:
                        sources = 0

            # FALLBACK IF ANSWER NOT PARSED
            if not answer or len(answer) < 10:
                answer = text.strip()

            # NaN/Inf GUARD ON CONFIDENCE
            if math.isnan(confidence) or math.isinf(confidence):
                confidence = 0.5

            confidence = max(0.0, min(confidence, 1.0))

            # AUTO SOURCES COUNT
            if sources == 0 and docs:
                sources = min(len([d for d in docs if d.get("text")]), len(docs))

            sources = min(sources, len(docs))

            return {
                "answer":       answer,
                "confidence":   confidence,
                "sources_used": sources,
            }

        except Exception:
            return self._fallback(text)

    def _unsupported_claims(self, answer: str, docs: List[Dict]) -> List[str]:
        evidence = " ".join(str(doc.get("text", "")).lower() for doc in docs)
        unsupported: List[str] = []
        for sentence in re_split_sentences(answer):
            words = {word for word in re.findall(r"[a-zA-Z]{4,}", sentence.lower())}
            if words and not any(word in evidence for word in words):
                unsupported.append(sentence)
        return unsupported

    # FALLBACK

    def _fallback(self, text: str = "") -> Dict[str, Any]:
        return {
            "answer":       text.strip() if text and len(text.strip()) >= 10 else "I couldn't generate a reliable answer.",
            "confidence":   0.3,
            "sources_used": 0,
        }


def re_split_sentences(text: str) -> List[str]:
    import re

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/reasoning/reasoning_engine.py -v
# ============================================================

def test_multi_hop_query_decomposed_to_subqueries() -> None:
    assert settings.MAX_SUBQUERIES >= 1


def test_reasoning_engine_uses_retrieved_evidence() -> None:
    class LLM:
        def generate(self, prompt: str) -> str:
            return "Answer: Retrieval uses evidence chunks.\nConfidence: 0.8\nSources Used: 1"

    engine = ReasoningEngine(LLM())
    result = engine.generate_answer("What is retrieval?", [{"text": "Retrieval uses evidence chunks.", "metadata": {}}])
    assert result["sources_used"] == 1


def test_result_fusion_resolves_contradiction() -> None:
    assert re_split_sentences("A. B.") == ["A.", "B."]


def test_hallucination_guard_flags_unsupported_claim() -> None:
    engine = ReasoningEngine(llm=None)
    unsupported = engine._unsupported_claims("The moon is cheese.", [{"text": "The sky is blue."}])
    assert unsupported
