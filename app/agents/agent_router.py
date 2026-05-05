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

        if not query:
            return self._decision("direct", "empty_query", 0.0)

        start = time.time()
        query = self._normalize(query)

        signals = self._analyze(query)

        #  HARD RULES 
        if signals["is_recent"]:
            return self._decision("search", "recent_query", 0.95)

        if signals["is_memory"]:
            return self._decision("memory", "memory_reference", 0.9)

        #  LLM ROUTING 
        decision = self._llm_route(query, signals)

        validated = self._validate(decision, signals)

        logger.info(
            event="router_final",
            action=validated.action,
            confidence=validated.confidence,
            latency=round(time.time() - start, 3)
        )

        return validated

    #  SIGNALS 
    def _analyze(self, query: str) -> Dict:

        q = query.lower()

        return {
            "is_recent": any(w in q for w in ["latest", "today", "news", "recent", "update"]),
            "is_memory": any(w in q for w in ["earlier", "previous", "last time", "we discussed"]),
            "is_complex": (
                len(q.split()) > settings.DECOMPOSITION_MIN_WORDS or
                any(k in q for k in ["compare", "difference", "process", "steps"])
            ),
            "is_reasoning": any(k in q for k in ["why", "how", "explain"]),
            "has_multimodal_hint": any(w in q for w in ["image", "video", "diagram", "chart", "audio"]),
        }

    #  PROMPT 
    def _build_prompt(self, query: str, signals: dict) -> str:

        instruction = (
            "Route query to one:\n"
            "rag | search | direct | memory | hybrid\n"
            "Return JSON only.\n\n"
        )

        body = f"Signals:{signals}\nQuery:{query}\n"

        format_block = '{"action":"...", "reason":"..."}'

        max_chars = settings.MAX_PROMPT_CHARS
        available = max_chars - len(instruction) - len(format_block) - 50

        return instruction + body[:available] + format_block

    #  LLM 
    def _llm_route(self, query: str, signals: dict) -> AgentDecision:

        try:
            llm = model_loader.get_llm()
        except Exception:
            return self._decision("rag", "llm_unavailable", 0.5)

        prompt = self._build_prompt(query, signals)

        try:
            t = time.time()

            response = llm.generate(
                prompt,
                max_tokens=60,
                temperature=0.0,
                top_p=1.0
            )

            if time.time() - t > settings.MODEL_TIMEOUT_SEC:
                return self._decision("rag", "timeout", 0.5)

            return self._parse(response, signals)

        except Exception:
            return self._decision("rag", "llm_failure", 0.5)

    #  PARSE 
    def _parse(self, text: str, signals: dict) -> AgentDecision:

        try:
            cleaned = self._extract_json(text)
            data = json.loads(cleaned)

            action = data.get("action", "").lower().strip()
            reason = data.get("reason", "")

            confidence = self._confidence(action, signals)

            return AgentDecision(
                action=action,
                reason=reason,
                confidence=confidence,
                signals=signals
            )

        except Exception:
            return self._decision("rag", "parse_failure", 0.6)

    #  CONFIDENCE 
    def _confidence(self, action: str, signals: dict) -> float:

        score = 0.6

        if signals["is_recent"] and action == "search":
            score += 0.3

        if signals["is_memory"] and action == "memory":
            score += 0.25

        if signals["is_complex"] and action in {"rag", "hybrid"}:
            score += 0.2

        if signals["has_multimodal_hint"] and action in {"rag", "hybrid"}:
            score += 0.1

        return min(score, 0.95)

    #  JSON CLEAN 
    def _extract_json(self, text: str) -> str:

        text = text.strip()

        if "```" in text:
            parts = text.split("```")
            text = next((p for p in parts if "{" in p), text)

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            return match.group(0)

        raise ValueError("NO_JSON")

    #  VALIDATE 
    def _validate(self, decision: AgentDecision, signals: dict) -> AgentDecision:

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