import json
import re
import time
import unicodedata
from typing import Dict

from app.agents.agent_schema import AgentDecision
from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)


# HARD RULE KEYWORDS

_RECENT_WORDS     = {"latest", "today", "news", "recent", "update", "current", "now", "live"}
_MEMORY_WORDS     = {"earlier", "previous", "last time", "we discussed", "you said", "before"}
_COMPLEX_KEYWORDS = {"compare", "difference", "process", "steps", "vs", "versus", "explain"}
_REASONING_WORDS  = {"why", "how", "explain", "reason", "cause", "because"}
_MULTIMODAL_WORDS = {"image", "video", "diagram", "chart", "audio", "photo", "picture", "figure"}
_CODE_WORDS       = {"code", "function", "implement", "script", "syntax", "class", "debug"}
_GREETING_WORDS   = {"hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"}


class AgentRouter:

    # NORMALIZE

    def _normalize(self, query: str) -> str:
        query = unicodedata.normalize("NFC", str(query or ""))
        return " ".join(query.strip().split())

    # MAIN

    def route(self, query: str, session_id: str) -> AgentDecision:

        if not query:
            return self._decision("direct", "empty_query", 0.0, session_id)

        start = time.time()
        query = self._normalize(query)

        signals = self._analyze(query)

        # HARD RULE: GREETING / CHITCHAT
        if signals["is_greeting"]:
            return self._decision("direct", "greeting_detected", 0.95, session_id)

        # HARD RULE: CODE QUERY
        if signals["is_code"]:
            return self._decision("direct", "code_query", 0.9, session_id)

        # HARD RULE: RECENT/WEB SEARCH
        if signals["is_recent"]:
            return self._decision("search", "recent_query", 0.95, session_id)

        # HARD RULE: MEMORY REFERENCE
        if signals["is_memory"]:
            return self._decision("memory", "memory_reference", 0.9, session_id)

        # LLM ROUTING
        decision  = self._llm_route(query, signals, session_id)
        validated = self._validate(decision, signals, session_id)

        validated.set_latency(start)

        logger.info(
            event="router_final",
            action=validated.action,
            confidence=validated.confidence,
            signal_summary={k: v for k, v in signals.items() if v},
            latency_ms=validated.latency_ms,
            session_id=session_id,
        )

        return validated

    # SIGNAL ANALYSIS

    def _analyze(self, query: str) -> Dict:
        q      = query.lower()
        tokens = set(q.split())

        return {
            "is_recent":          bool(tokens & _RECENT_WORDS),
            "is_memory":          any(w in q for w in _MEMORY_WORDS),
            "is_complex":         (
                len(tokens) > settings.DECOMPOSITION_MIN_WORDS or
                bool(tokens & _COMPLEX_KEYWORDS)
            ),
            "is_reasoning":       bool(tokens & _REASONING_WORDS),
            "has_multimodal_hint": bool(tokens & _MULTIMODAL_WORDS),
            "is_code":            bool(tokens & _CODE_WORDS),
            "is_greeting":        bool(tokens & _GREETING_WORDS) and len(tokens) <= 4,
        }

    # PROMPT

    def _build_prompt(self, query: str, signals: dict) -> str:
        instruction = (
            "Route query to one:\n"
            "rag | search | direct | memory | hybrid\n"
            "Return JSON only.\n\n"
        )

        signal_str   = str({k: v for k, v in signals.items() if v})
        body         = f"Signals:{signal_str}\nQuery:{query}\n"
        format_block = '{"action":"...", "reason":"..."}'

        max_chars = settings.MAX_PROMPT_CHARS
        available = max_chars - len(instruction) - len(format_block) - 50

        return instruction + body[:max(available, 0)] + format_block

    # LLM ROUTING

    def _llm_route(self, query: str, signals: dict, session_id: str) -> AgentDecision:
        try:
            llm = model_loader.get_llm()
        except Exception:
            return self._decision("rag", "llm_unavailable", 0.5, session_id)

        prompt = self._build_prompt(query, signals)

        try:
            t_start  = time.time()

            response = llm.generate(
                prompt,
                max_tokens=80,
                temperature=0.0,
                top_p=1.0,
                session_id=session_id,
            )

            elapsed = time.time() - t_start

            if elapsed > settings.AGENT_STEP_TIMEOUT_SEC:
                logger.warning(
                    event="router_llm_timeout",
                    elapsed=round(elapsed, 2),
                    session_id=session_id,
                )
                return self._decision("rag", "llm_timeout", 0.5, session_id)

            return self._parse(response, signals, session_id)

        except Exception as e:
            logger.warning(
                event="router_llm_failed",
                error=str(e),
                session_id=session_id,
            )
            return self._decision("rag", "llm_failure", 0.5, session_id)

    # PARSE

    def _parse(self, text: str, signals: dict, session_id: str) -> AgentDecision:
        try:
            cleaned = self._extract_json(text)
            data    = json.loads(cleaned)

            action = str(data.get("action", "")).lower().strip()
            reason = str(data.get("reason", ""))

            confidence = self._score_confidence(action, signals)

            return AgentDecision(
                action=action,
                reason=reason,
                confidence=confidence,
                signals=signals,
                session_id=session_id,
            )

        except Exception:
            return self._decision("rag", "parse_failure", 0.6, session_id)

    # CONFIDENCE SCORING

    def _score_confidence(self, action: str, signals: dict) -> float:
        score = 0.6

        if signals.get("is_recent") and action == "search":
            score += 0.30

        if signals.get("is_memory") and action == "memory":
            score += 0.25

        if signals.get("is_complex") and action in {"rag", "hybrid"}:
            score += 0.20

        if signals.get("has_multimodal_hint") and action in {"rag", "hybrid"}:
            score += 0.10

        if signals.get("is_reasoning") and action in {"rag", "direct"}:
            score += 0.05

        return min(round(score, 3), 0.95)

    # JSON EXTRACTION

    def _extract_json(self, text: str) -> str:
        text = text.strip()

        if "```" in text:
            parts = text.split("```")
            text  = next((p for p in parts if "{" in p), text)

        match = re.search(r"\{.*?\}", text, re.DOTALL)

        if match:
            return match.group(0)

        raise ValueError("NO_JSON_FOUND")

    # VALIDATE

    def _validate(
        self,
        decision: AgentDecision,
        signals: dict,
        session_id: str,
    ) -> AgentDecision:

        if decision.action not in {"rag", "search", "direct", "memory", "hybrid"}:
            return self._decision("rag", "invalid_action_fallback", 0.5, session_id)

        # OVERRIDE: recent signal always forces search
        if signals.get("is_recent"):
            return self._decision("search", "override_recent_signal", 0.95, session_id)

        return decision

    # DECISION FACTORY

    def _decision(
        self,
        action: str,
        reason: str,
        confidence: float,
        session_id: str = "default",
    ) -> AgentDecision:
        return AgentDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            signals={},
            session_id=session_id,
        )