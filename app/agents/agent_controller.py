from typing import Dict, Any
from app.utils.logger import get_logger
from app.agents.agent_executor import AgentExecutor
from app.core.model_loader import model_loader

# Logger
logger = get_logger(__name__)


class AgentController:
    def __init__(self):
        self.executor = AgentExecutor()

    # MAIN ENTRY
    def handle(self, query: str, session_id: str) -> Dict[str, Any]:

        logger.info(
            f"[AgentController][START] session_id={session_id} | query={query}"
        )

        try:
            # STEP 1: Execute Agent
            result = self.executor.run(query, session_id)

            # STEP 2: Post Processing
            response = self._format_response(result)

            logger.info(
                f"[AgentController][SUCCESS] session_id={session_id}"
            )

            return response
        
        except Exception as e:
            logger.error(
                f"[AgentController][ERROR] session_id={session_id} | {str(e)}"
            )

            # Safe Fallback
            return self._fallback_response(query)
    
    # RESPONSE FORMATTER
    def _format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "response": result.get("response"),
            "source": result.get("source"),
            "decision": result.get("decision"),
            "reason": result.get("reason"),
            "metadata": result.get("metadata", {})
        }
    
    # FALLBACK
    def _fallback_response(self, query: str) -> Dict[str, Any]:

        logger.warning("[AgentController] Using fallback LLM response")

        response = model_loader.generate(query)

        return {
            "response": response,
            "source": "fallback",
            "decision": "direct",
            "reason": "Controller fallback due to error",
            "latency": None,
            "metadata": {}
        }



    
