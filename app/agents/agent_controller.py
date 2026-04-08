"""
agent_controller.py

Handles decision-making for incoming queries.
Determines which pipeline/path to use.
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AgentController:
    """
    Agent layer that decides how to process a user query.
    """
    def decide(self, query: str, session_id: str) -> Dict:
        """
        Decide which pipeline to use based on query.

        Args:
            query (str): user input
            session_id (str): unique session identifier
        
        Returns:
            dict with:
            - action: "rag" | "memory" | "multimodal" | "direct"
        """

        logger.info(
            f"[AgentController] session_id={session_id} | Recieved query: {query}"
        )

        query_lower = query.lower()

        # Simple heuristics 
        if any(word in query_lower for word in ["image", "photo", "diagram"]):
            return {
                "action": "multimodal",
                "reason": "Image-related query detected"
            }
        
        elif any(word in query_lower for word in ["previous", "earlier", "before"]):
            return {
                "action": "memory",
                "reason": "Follow-up question detected"
            }
        
        elif len(query.split()) > 12:
            return {
                "action": "rag",
                "reason": "Complex query -> use retrieval"
            }
        
        else:
            decision =  {
                "action": "rag",
                "reason": "Default to RAG"
            }
        
        logger.info(
            f"[AgentController] session_id={session_id} | Decision: {decision['action']} | Reason: {decision['reason']}"
        )

        return decision 