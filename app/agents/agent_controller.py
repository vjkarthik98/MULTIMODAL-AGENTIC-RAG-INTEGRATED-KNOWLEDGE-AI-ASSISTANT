import time
from typing import Dict, Any

from app.core.config import settings
from app.utils.logger import get_logger
from app.agents.agent_executor import AgentExecutor
from app.core.model_loader import model_loader

logger = get_logger(__name__)


class AgentController:

    def __init__(self):
        self.executor = AgentExecutor()
        self.timeout = getattr(settings, "AGENT_TIMEOUT_SEC", 10)

    #  NORMALIZE 
    def _normalize(self, q: str) -> str:
        return " ".join(q.strip().split())

    #  MAIN 
    def handle(self, query: str, session_id: str) -> Dict[str, Any]:

        if not query:
            return self._reject("empty_query")

        start = time.time()
        query = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        try:
            logger.info(event="agent_start", session_id=session_id)

            t_agent = time.time()
            result = self._execute_with_timeout(query, session_id)
            agent_latency = round(time.time() - t_agent, 3)

            if not self._validate(result):
                raise ValueError("INVALID_AGENT_OUTPUT")

            response = self._format(result)

            response.update({
                "latency": round(time.time() - start, 3),
                "agent_latency": agent_latency,
                "confidence": self._confidence(result),
            })

            logger.info(event="agent_success", session_id=session_id)

            return response

        except Exception as e:
            logger.error(event="agent_failed", error=str(e))
            return self._fallback(query, start)

    #  TIMEOUT EXEC 
    def _execute_with_timeout(self, query: str, session_id: str):

        start = time.time()

        result = self.executor.run(query, session_id)

        if time.time() - start > self.timeout:
            raise TimeoutError("AGENT_TIMEOUT")

        return result

    #  VALIDATION 
    def _validate(self, result: Dict[str, Any]) -> bool:

        if not isinstance(result, dict):
            return False

        required = {"response", "source", "decision"}

        if not required.issubset(result.keys()):
            return False

        if not result.get("response"):
            return False

        return True

    #  CONFIDENCE 
    def _confidence(self, result: Dict[str, Any]) -> float:

        decision = result.get("decision")

        mapping = {
            "direct": 0.7,
            "tool": 0.85,
            "memory": 0.75,
        }

        return mapping.get(decision, 0.5)

    #  FORMAT 
    def _format(self, result: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "response": result.get("response", ""),
            "source": result.get("source", "unknown"),
            "decision": result.get("decision", "unknown"),
            "reason": result.get("reason", ""),
            "metadata": result.get("metadata", {}),
        }

    #  REJECT 
    def _reject(self, reason: str) -> Dict[str, Any]:
        return {
            "response": "Invalid query.",
            "source": "validation",
            "decision": "reject",
            "reason": reason,
            "confidence": 0.0,
            "metadata": {},
        }

    #  FALLBACK 
    def _fallback(self, query: str, start_time: float) -> Dict[str, Any]:

        try:
            llm = model_loader.get_llm()

            prompt = f"Answer clearly:\n{query}"

            response = llm.generate(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.2,
                top_p=settings.LLM_TOP_P,
            )

        except Exception as e:
            logger.error(event="fallback_failed", error=str(e))
            response = "Unable to process request."

        return {
            "response": response,
            "source": "fallback",
            "decision": "direct",
            "reason": "controller_failure",
            "confidence": 0.3,
            "latency": round(time.time() - start_time, 3),
            "metadata": {},
        }