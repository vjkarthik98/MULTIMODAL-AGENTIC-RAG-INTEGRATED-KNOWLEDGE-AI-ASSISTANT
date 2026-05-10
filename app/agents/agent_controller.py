import time
import unicodedata
import uuid
from typing import Any, Dict

from app.agents.agent_executor import AgentExecutor
from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)


# CONFIDENCE MAP BY DECISION TYPE

_CONFIDENCE_MAP: Dict[str, float] = {
    "rag":     0.80,
    "search":  0.85,
    "direct":  0.70,
    "memory":  0.75,
    "hybrid":  0.85,
    "fallback": 0.30,
    "reject":  0.00,
}


class AgentController:

    def __init__(self) -> None:
        self.executor   = AgentExecutor()
        self.timeout    = max(settings.AGENT_TIMEOUT_SEC, 1)

        if settings.AGENT_TIMEOUT_SEC <= 0:
            logger.warning(
                event="agent_timeout_invalid",
                value=settings.AGENT_TIMEOUT_SEC,
                using=self.timeout,
            )

    # NORMALIZE

    def _normalize(self, q: str) -> str:
        q = unicodedata.normalize("NFC", str(q or ""))
        return " ".join(q.strip().split())

    # MAIN

    def handle(self, query: str, session_id: str) -> Dict[str, Any]:

        if not query or not query.strip():
            return self._reject("empty_query")

        start      = time.time()
        request_id = str(uuid.uuid4())
        query      = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        try:
            logger.info(
                event="agent_start",
                request_id=request_id,
                query_len=len(query),
                session_id=session_id,
            )

            t_agent = time.time()
            result  = self._execute_with_timeout(query, session_id)
            agent_latency = round(time.time() - t_agent, 3)

            if not self._validate(result):
                raise ValueError("INVALID_AGENT_OUTPUT")

            response = self._format(result)

            confidence = self._confidence(result)

            response.update({
                "request_id":    request_id,
                "latency":       round(time.time() - start, 3),
                "agent_latency": agent_latency,
                "agent_latency_ms": round(agent_latency * 1000, 1),
                "confidence":    confidence,
            })

            logger.info(
                event="agent_success",
                request_id=request_id,
                decision=response.get("decision"),
                confidence=confidence,
                latency=response["latency"],
                session_id=session_id,
            )

            return response

        except Exception as e:
            logger.error(
                event="agent_failed",
                request_id=request_id,
                error=str(e),
                session_id=session_id,
            )
            return self._fallback(query, start, session_id, request_id)

    # TIMEOUT EXECUTION

    def _execute_with_timeout(self, query: str, session_id: str) -> Dict[str, Any]:
        start  = time.time()
        result = self.executor.run(query, session_id)
        elapsed = time.time() - start

        if elapsed > self.timeout:
            raise TimeoutError(f"AGENT_TIMEOUT_{elapsed:.1f}s > {self.timeout}s")

        return result

    # VALIDATION

    def _validate(self, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False

        required = {"response", "source", "decision"}
        if not required.issubset(result.keys()):
            return False

        response = result.get("response", "")
        if not response or len(str(response).strip()) < 3:
            return False

        return True

    # CONFIDENCE SCORING

    def _confidence(self, result: Dict[str, Any]) -> float:
        # PREFER EXECUTOR-PROVIDED CONFIDENCE FROM METADATA
        metadata   = result.get("metadata", {}) or {}
        meta_conf  = metadata.get("confidence")

        if meta_conf is not None:
            try:
                val = float(meta_conf)
                if 0.0 <= val <= 1.0:
                    return round(val, 3)
            except (TypeError, ValueError):
                pass

        # FALLBACK TO DECISION-BASED MAP
        decision = result.get("decision", "unknown")
        return _CONFIDENCE_MAP.get(decision, 0.5)

    # FORMAT

    def _format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "response": result.get("response", ""),
            "source":   result.get("source", "unknown"),
            "decision": result.get("decision", "unknown"),
            "reason":   result.get("reason", ""),
            "metadata": result.get("metadata", {}),
        }

    # REJECT

    def _reject(self, reason: str) -> Dict[str, Any]:
        return {
            "response":   "Invalid query.",
            "source":     "validation",
            "decision":   "reject",
            "reason":     reason,
            "confidence": 0.0,
            "latency":    0.0,
            "metadata":   {},
        }

    # FALLBACK

    def _fallback(
        self,
        query: str,
        start_time: float,
        session_id: str = "default",
        request_id: str = "",
    ) -> Dict[str, Any]:

        try:
            llm      = model_loader.get_llm()
            prompt   = f"Answer clearly:\n{query}"
            response = llm.generate(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.2,
                top_p=settings.LLM_TOP_P,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(
                event="controller_fallback_failed",
                request_id=request_id,
                error=str(e),
                session_id=session_id,
            )
            response = "Unable to process your request at this time."

        return {
            "response":    response,
            "source":      "fallback",
            "decision":    "direct",
            "reason":      "controller_failure",
            "confidence":  0.3,
            "request_id":  request_id,
            "latency":     round(time.time() - start_time, 3),
            "metadata":    {},
        }