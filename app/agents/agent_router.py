from app.core.model_loader import model_loader
from app.agents.agent_schema import AgentDecision
import json
import re
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


class AgentRouter:

    # PUBLIC API
    def route(self, query: str, session_id: str) -> AgentDecision:

        logger.info(f"[AgentRouter][START] session_id={session_id}")

        # STEP 1: PRE-ANALYSIS 
        signals = self._analyze_query(query)

        # Hard Overrides
        if signals["is_recent"]:
            return self._decision("search", "Detected time-sensitive query")
        
        if signals["is_memory"]:
            return self._decision("memory", "User referring to past conversation")
        
        # STEP 2: LLM DECISION(CONTROLLED)
        decision = self._llm_route(query, signals, session_id)

        # STEP 3: POST VALIDATION
        validated = self._validate_decision(decision, signals)

        logger.info(
            f"[AgentRouter][FINAL] session_id={session_id} | "
            f"action= {validated.action} | reason={validated.reason}"
        )

        return validated
    
    #  QUERY ANALYSIS
    def _analyze_query(self, query: str) -> dict:
        query_lower = query.lower()

        return {
            "is_recent": any(word in query_lower for word in [
                "latest", "today", "news", "recent", "current", "update"
            ]), 
            "is_memory": any(word in query_lower for word in [
                "earlier", "previous", "last time", "we discussed"
            ]),
            "is_complex": len(query.split()) > 15,
            "has_multimodal_hint": any(word in query_lower for word in [
                "text", "document","image", "audio", "video", "diagram", "chart"
            ])
        }

    # LLM ROUTING
    def _llm_route(self, query: str, signals: dict, session_id: str) -> AgentDecision:

        prompt = f"""
You are an advanced AI agent router.

Your job is to decide the BEST strategy to answer a query.

SYSTEM CAPABILITIES:
- RAG -> internal documents (multimodal: text, document, image, audio, video)
- SEARCH -> external real-time info
- MEMORY -> past conversation context
- DIRECT -> general LLM Knowledge

AVAILABLE ACTIONS:
- "rag"
- "search"
- "direct"
- "memory"
- "hybrid" (rag + memory)

DECISION RULES:
- Use "search" for recent/time-sensitive queries
- Use "memory" if user refers to past conversation
- Use "rag" for document-based queries
- Use "hybrid" if both memory + knowledge needed
- Use "direct" only if no external context needed

SIGNALS:
{signals}

Return STRICT JSON:
{{
    "action": "...",
    "reason": "..."
}}

Query:
{query}
"""

        response = model_loader.generate(prompt)

        return self._parse_response(response, session_id, signals)
    
    # RESPONSE PARSER
    def _parse_response(self, text: str, session_id: str, signals: dict) -> AgentDecision:
        
        try:
            cleaned = self._clean_json(text)
            data = json.loads(cleaned)

            return AgentDecision(
                action=data.get("action"),
                reason=data.get("reason"),
                confidence=data.get("confidence", 0.8),
                signals=signals 
            )
        
        except Exception as e:
            logger.error(
                f"[AgentRouter][PARSE_FAIL] session_id={session_id} | {str(e)}"
            )

            return self._decision("search", "Fallback due to parsing failure")
        
    # CLEAN JSON
    def _clean_json(self, text: str) -> str:
        if "```" in text:

            text = text.split("```")[1]

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            return match.group(0)
        
        raise ValueError("No JSON found")
    

    # VALIDATION
    def _validate_decision(self, decision: AgentDecision, signals: dict) -> AgentDecision:

        allowed = {"rag", "search", "direct", "memory", "hybrid"}

        if decision.action not in allowed:
            logger.warning("[AgentRouter] Invalid action, forcing safe fallback")
            return self._decision("search", "Invalid action fallback")
        
        # Safety overrides
        if signals["is_recent"]:
            return self._decision("search", "override: recent query")
        
        return decision
    
    # HELPER
    def _decision(self, action: str, reason:str) -> AgentDecision:
        return AgentDecision(action=action, reason=reason)

        