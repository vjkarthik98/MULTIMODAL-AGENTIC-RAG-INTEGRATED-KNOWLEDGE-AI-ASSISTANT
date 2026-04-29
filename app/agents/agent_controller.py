from typing import Dict, Any
import time

from app.core.config import settings
from app.utils.logger import get_logger
from app.agents.agent_executor import AgentExecutor
from app.core.model_loader import model_loader


logger = get_logger(__name__)


class AgentController:

    def __init__(self):
        self.executor = AgentExecutor()
        self.timeout = getattr(settings, "AGENT_TIMEOUT", 10)

    #  NORMALIZE 
    def _normalize(self, query: str) -> str:
        return " ".join(query.strip().split())

    #  MAIN 
    def handle(self, query: str, session_id: str) -> Dict[str, Any]:

        if not query or not query.strip():
            return {
                "response": "Query cannot be empty.",
                "source": "validation",
                "decision": "reject",
                "reason": "empty_query",
                "confidence": 0.0,
                "metadata": {}
            }

        start = time.time()

        query = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        logger.info("[AgentController][START] session_id=%s", session_id)

        try:
            #  SAFE EXECUTION 
            t_agent = time.time()

            result = self._safe_execute(query, session_id)

            agent_latency = round(time.time() - t_agent, 2)

            #  VALIDATION 
            validated = self._validate_result(result)

            if not validated:
                raise ValueError("Invalid agent output")

            response = self._format_response(result)

            response["latency"] = round(time.time() - start, 2)
            response["agent_latency"] = agent_latency
            response["confidence"] = self._compute_confidence(result)

            logger.info("[AgentController][SUCCESS] session_id=%s", session_id)

            return response

        except Exception as e:
            logger.error(
                "[AgentController][ERROR] session_id=%s | error=%s",
                session_id,
                str(e)
            )

            return self._fallback_response(query, start)

    #  SAFE EXECUTION 
    def _safe_execute(self, query: str, session_id: str):

        try:
            return self.executor.run(query, session_id)
        except Exception as e:
            logger.warning("[AgentController][EXEC_FAIL] %s", str(e))
            raise

    #  VALIDATION 
    def _validate_result(self, result: Dict[str, Any]) -> bool:

        if not isinstance(result, dict):
            return False

        required_fields = ["response", "source", "decision"]

        for f in required_fields:
            if f not in result:
                return False

        if not result.get("response"):
            return False

        return True

    #  CONFIDENCE 
    def _compute_confidence(self, result: Dict[str, Any]) -> float:

        decision = result.get("decision")

        if decision == "direct":
            return 0.7
        if decision == "tool":
            return 0.8
        if decision == "memory":
            return 0.75

        return 0.5

    #  FORMAT 
    def _format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": result.get("response", ""),
            "source": result.get("source", "unknown"),
            "decision": result.get("decision", "unknown"),
            "reason": result.get("reason", ""),
            "metadata": result.get("metadata", {})
        }

    #  FALLBACK 
    def _fallback_response(self, query: str, start_time: float) -> Dict[str, Any]:

        logger.warning("[AgentController] fallback triggered")

        try:
            llm = model_loader.get_llm()

            safe_prompt = (
                "Answer the following question clearly and concisely:\n\n"
                f"{query}"
            )

            response = llm.generate(
                safe_prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.3,  # safer
                top_p=settings.LLM_TOP_P,
            )

        except Exception as e:
            logger.error("[AgentController][FALLBACK_FAIL] %s", str(e))
            response = "I'm unable to process your request right now."

        return {
            "response": response,
            "source": "fallback",
            "decision": "direct",
            "reason": "controller_error_fallback",
            "confidence": 0.3,
            "latency": round(time.time() - start_time, 2),
            "metadata": {}
        }