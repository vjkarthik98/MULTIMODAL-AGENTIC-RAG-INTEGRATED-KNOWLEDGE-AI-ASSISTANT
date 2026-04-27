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

    def route(self, query: str, session_id: str) -> AgentDecision:
        if not query or not query.strip():
            return self._decision("direct", "empty_query")

        start = time.time()

        logger.info("[AgentRouter][START] session_id=%s", session_id)

        signals = self._analyze_query(query)

        # Only strict override for RECENT queries
        if signals["is_recent"]:
            return self._decision("search", "Detected time-sensitive query")

        if signals["is_memory"]:
            return self._decision("memory", "User referring to past conversation")

        decision = self._llm_route(query, signals, session_id)

        validated = self._validate_decision(decision, signals)

        logger.info(
            "[AgentRouter][FINAL] session_id=%s | action=%s | latency=%.2fs",
            session_id,
            validated.action,
            time.time() - start
        )

        return validated

    def _analyze_query(self, query: str) -> Dict:
        query_lower = query.lower()

        return {
            "is_recent": any(w in query_lower for w in [
                "latest", "today", "news", "recent", "current", "update"
            ]),
            "is_memory": any(w in query_lower for w in [
                "earlier", "previous", "last time", "we discussed"
            ]),
            "is_complex": len(query.split()) > 15,
            "has_multimodal_hint": any(w in query_lower for w in [
                "image", "photo", "picture", "audio", "video", "diagram", "chart"
            ])
        }

    def _llm_route(self, query: str, signals: dict, session_id: str):
        try:
            llm = model_loader.get_llm()
        except Exception as e:
            logger.warning(
                "[AgentRouter] LLM unavailable → fallback to RAG | %s", str(e)
            )
            return self._decision("rag", "llm_unavailable")

        prompt = f"""
You are an AI routing system.

Choose the best action for answering a query.

Actions:
- rag
- search
- direct
- memory
- hybrid

Rules:
- search → recent info ONLY
- memory → past conversation
- rag → document-based (default)
- hybrid → rag + memory
- direct → general knowledge

Signals:
{signals}

Return STRICT JSON:
{{"action": "...", "reason": "..."}}

Query:
{query}
"""

        prompt = prompt[:settings.MAX_PROMPT_CHARS]

        try:
            response = llm.generate(
                prompt,
                max_tokens=100,
                temperature=0.0,
                top_p=1.0
            )

            return self._parse_response(response, session_id, signals)

        except Exception as e:
            logger.error(
                "[AgentRouter][LLM_FAIL] session_id=%s | error=%s",
                session_id,
                str(e)
            )
            return self._decision("rag", "llm_failure_fallback")

    def _parse_response(self, text: str, session_id: str, signals: dict) -> AgentDecision:
        try:
            cleaned = self._clean_json(text)
            data = json.loads(cleaned)

            action = data.get("action", "").strip().lower()
            reason = data.get("reason", "")

            return AgentDecision(
                action=action,
                reason=reason,
                confidence=0.8,
                signals=signals
            )

        except Exception as e:
            logger.error(
                "[AgentRouter][PARSE_FAIL] session_id=%s | error=%s",
                session_id,
                str(e)
            )

            # fallback to RAG instead of SEARCH
            return self._decision("rag", "parse_failure")

    def _clean_json(self, text: str) -> str:
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("No JSON found")

    def _validate_decision(self, decision: AgentDecision, signals: dict) -> AgentDecision:
        allowed = {"rag", "search", "direct", "memory", "hybrid"}

        if decision.action not in allowed:
            logger.warning("[AgentRouter] Invalid action → fallback to RAG")
            return self._decision("rag", "invalid_action")

        if signals["is_recent"]:
            return self._decision("search", "override_recent")

        return decision

    def _decision(self, action: str, reason: str) -> AgentDecision:
        return AgentDecision(
            action=action,
            reason=reason,
            confidence=0.9,
            signals={}
        )