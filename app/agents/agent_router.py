from typing import Dict
import json
import re
import time

from app.core.config import settings
from app.core.model_loader import model_loader
from app.agents.agent_schema import AgentDecision
from app.utils.logger import get_logger


logger = get_logger(__name__)


class AgentRouter:

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  MAIN 
    def route(self, query: str, session_id: str) -> AgentDecision:

        if not query or not query.strip():
            return self._decision("direct", "empty_query", 0.0)

        start = time.time()
        query = self._normalize(query)

        logger.info("[AgentRouter][START] session_id=%s", session_id)

        signals = self._analyze_query(query)

        #  HARD RULES 
        if signals["is_recent"]:
            return self._decision("search", "recent_query", 0.95)

        if signals["is_memory"]:
            return self._decision("memory", "memory_reference", 0.9)

        #  LLM ROUTING 
        decision = self._llm_route(query, signals, session_id)

        validated = self._validate_decision(decision, signals)

        latency = round(time.time() - start, 2)

        logger.info(
            "[AgentRouter][FINAL] action=%s confidence=%.2f latency=%ss",
            validated.action,
            validated.confidence,
            latency
        )

        return validated

    #  SIGNALS 
    def _analyze_query(self, query: str) -> Dict:

        q = query.lower()

        return {
            "is_recent": any(w in q for w in [
                "latest", "today", "news", "recent", "current", "update"
            ]),
            "is_memory": any(w in q for w in [
                "earlier", "previous", "last time", "we discussed"
            ]),
            "is_complex": (
                len(q.split()) > settings.DECOMPOSITION_MIN_WORDS or
                any(k in q for k in ["compare", "difference", "process", "steps"])
            ),
            "is_reasoning": any(k in q for k in [
                "why", "how", "explain", "reason"
            ]),
            "has_multimodal_hint": any(w in q for w in [
                "image", "photo", "video", "diagram", "chart", "audio"
            ])
        }

    #  SAFE PROMPT 
    def _build_prompt(self, query: str, signals: dict) -> str:

        instruction = (
            "You are an AI routing system.\n\n"
            "Choose best action:\n"
            "- rag\n- search\n- direct\n- memory\n- hybrid\n\n"
            "Rules:\n"
            "- search → recent\n"
            "- memory → past conversation\n"
            "- rag → documents\n"
            "- hybrid → rag + memory\n"
            "- direct → general knowledge\n\n"
        )

        body = f"Signals:\n{signals}\n\nQuery:\n{query}\n\n"

        format_block = '{"action":"...", "reason":"..."}'

        max_chars = settings.MAX_PROMPT_CHARS
        available = max_chars - len(instruction) - len(format_block) - 50

        body = body[:available]

        return instruction + body + format_block

    #  LLM ROUTING 
    def _llm_route(self, query: str, signals: dict, session_id: str):

        try:
            llm = model_loader.get_llm()
        except Exception:
            return self._decision("rag", "llm_unavailable", 0.5)

        prompt = self._build_prompt(query, signals)

        try:
            t = time.time()

            response = llm.generate(
                prompt,
                max_tokens=80,
                temperature=0.0,
                top_p=1.0
            )

            logger.debug("[AgentRouter] llm_latency=%.2fs", time.time() - t)

            return self._parse_response(response, signals)

        except Exception:
            return self._decision("rag", "llm_failure", 0.5)

    #  PARSER 
    def _parse_response(self, text: str, signals: dict) -> AgentDecision:

        try:
            cleaned = self._clean_json(text)

            data = json.loads(cleaned)

            action = data.get("action", "").strip().lower()
            reason = data.get("reason", "")

            confidence = self._compute_confidence(action, signals)

            return AgentDecision(
                action=action,
                reason=reason,
                confidence=confidence,
                signals=signals
            )

        except Exception:
            return self._decision("rag", "parse_failure", 0.6)

    #  CONFIDENCE 
    def _compute_confidence(self, action: str, signals: dict) -> float:

        base = 0.6

        if signals.get("is_recent") and action == "search":
            base += 0.3

        if signals.get("is_memory") and action == "memory":
            base += 0.25

        if signals.get("is_complex") and action in {"rag", "hybrid"}:
            base += 0.2

        return min(base, 0.95)

    #  CLEAN JSON 
    def _clean_json(self, text: str) -> str:

        text = text.strip()

        if "```" in text:
            text = text.split("```")[1]

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("Invalid JSON")

    #  VALIDATE 
    def _validate_decision(self, decision: AgentDecision, signals: dict) -> AgentDecision:

        allowed = {"rag", "search", "direct", "memory", "hybrid"}

        if decision.action not in allowed:
            return self._decision("rag", "invalid_action", 0.5)

        if signals["is_recent"]:
            return self._decision("search", "override_recent", 0.95)

        return decision

    #  DECISION 
    def _decision(self, action: str, reason: str, confidence: float) -> AgentDecision:
        return AgentDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            signals={}
        )