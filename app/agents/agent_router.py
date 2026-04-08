from app.core.model_loader import model_loader
from app.agents.agent_schema import AgentDecision
import json
import logging

# Logger
logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Decides how to handle a user query:
    - RAG (internal knowledge)
    - SEARCH (external/fresh info)
    - DIRECT (LLM answer)
    """

    def route(self, query: str, session_id: str) -> AgentDecision:

        logger.info(f"[AgentRouter] session_id={session_id} | Routing started")

        prompt = f"""
        You are an intelligent AI routing system.

        Your job is to decide how to answer a user query.

        You MUST choose ONE action:

        1. "rag"
            -> Use internal knowledge base if:
                - user refers to uploaded files
                - question depends on stored documents
                - long or detailed queries needing retrieval

        2. "search"
            -> Use external search if:
                - query asks for latest information
                - news, updates, recent events
                - anything time-sensitive
        
        3. "direct"
            -> Use LLM directly if:
                - general knowledge
                - definitions, explanations
                - no external or stored data needed
        Rules:
        - Prefer "search" over "direct" for anything recent
        - Prefer "rag" when documents are involved
        - Do NOT guess - choose the most reliable source

        Return ONLY JSON:

        {{
            "action": "rag" | "search" | "direct",
            "reason": "short explanation
        }}

        User Query:
        {query}
        """

        response_text = model_loader.generate(prompt)

        try:
            cleaned = response_text.strip()

            if "```" in cleaned:
                parts = cleaned.split("```")
                cleaned = parts[1] if len(parts) > 1 else parts[0]

            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1

            if start == -1 and end == -1:
                raise ValueError("No JSON found")

            cleaned = cleaned[start:end]

            decision_dict = json.loads(cleaned)

            # Validation
            if decision_dict["action"] not in ["rag", "search", "direct"]:
                raise ValueError("Invalid action")

            logger.info(
                f"[AgentRouter] session_id={session_id} | Decision: {decision_dict['action']} | Reason: {decision_dict['reason']}"
            )

            return AgentDecision(**decision_dict)

        except Exception as e:
            logger.error(
                f"[AgentRouter] session_id={session_id} | ROUTER ERROR: {str(e)}"
            )
            logger.error(
                f"[AgentRouter] session_id={session_id} | RAW OUTPUT: {response_text}"
            )

            return AgentDecision(
                action="search",
                reason="Fallback: forcing search"
            )