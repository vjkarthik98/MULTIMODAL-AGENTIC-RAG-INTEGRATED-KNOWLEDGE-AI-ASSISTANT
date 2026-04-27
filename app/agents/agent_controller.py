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

    # Main entry
    def handle(self, query: str, session_id: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "response": "Query cannot be empty.",
                "source": "validation",
                "decision": "reject",
                "reason": "empty_query",
                "metadata": {}
            }

        start = time.time()

        logger.info(
            "[AgentController][START] session_id=%s",
            session_id
        )

        try:
            result = self.executor.run(query, session_id)

            if not isinstance(result, dict):
                raise ValueError("Invalid agent response format")

            response = self._format_response(result)

            response["latency"] = round(time.time() - start, 2)

            logger.info(
                "[AgentController][SUCCESS] session_id=%s",
                session_id
            )

            return response

        except Exception as e:
            logger.error(
                "[AgentController][ERROR] session_id=%s | error=%s",
                session_id,
                str(e)
            )

            return self._fallback_response(query, start)

    # Response formatter
    def _format_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": result.get("response", ""),
            "source": result.get("source", "unknown"),
            "decision": result.get("decision", "unknown"),
            "reason": result.get("reason", ""),
            "metadata": result.get("metadata", {})
        }

    # Fallback (safe LLM call)
    def _fallback_response(self, query: str, start_time: float) -> Dict[str, Any]:
        logger.warning("[AgentController] Fallback triggered")

        try:
            llm = model_loader.get_llm()

            response = llm.generate(
                query[:settings.MAX_PROMPT_CHARS],
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
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
            "latency": round(time.time() - start_time, 2),
            "metadata": {}
        }