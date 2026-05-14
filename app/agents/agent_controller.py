import asyncio
import time
import unicodedata
import uuid
from typing import Any, Dict, Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, Gauge
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agents.agent_executor import AgentExecutor
from app.core.config import settings
from app.core.model_loader import model_loader

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_controller_duration = Histogram(
    "agent_controller_duration_seconds",
    "Agent controller total duration",
    ["decision", "status"],
)
_controller_errors = Counter(
    "agent_controller_errors_total",
    "Agent controller errors by type",
    ["error_type"],
)
_fallback_invocations = Counter(
    "agent_controller_fallback_total",
    "Number of fallback invocations",
)
_active_requests = Gauge(
    "agent_controller_active_requests",
    "Currently active agent controller requests",
)

# CONFIDENCE MAP BY DECISION TYPE
_CONFIDENCE_MAP: Dict[str, float] = {
    "rag":      0.80,
    "search":   0.85,
    "direct":   0.70,
    "memory":   0.75,
    "hybrid":   0.85,
    "fallback": 0.30,
    "reject":   0.00,
}

# SEMAPHORE — CAP CONCURRENT CONTROLLER REQUESTS
_semaphore = asyncio.Semaphore(settings.MAX_PARALLEL_REQUESTS)


# NORMALIZE QUERY

def _normalize(q: str) -> str:
    q = unicodedata.normalize("NFC", str(q or ""))
    return " ".join(q.strip().split())


# INJECT SANITIZATION — STRIP PROMPT INJECTION PATTERNS

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard the above",
    "forget everything",
    "you are now",
    "act as",
    "jailbreak",
    "override instructions",
    "new instructions",
    "system prompt",
    "developer mode",
    "dan mode",
    "admin override",
    "bypass",
]


def _sanitize(query: str) -> str:
    lower = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            idx   = query.lower().find(pattern)
            query = query[:idx].strip()
            logger.warning(
                "controller_injection_detected",
                pattern=pattern,
                query_prefix=query[:80],
            )
            break
    return query


class AgentController:

    def __init__(self) -> None:
        self.executor = AgentExecutor()
        self.timeout  = max(settings.AGENT_TIMEOUT_SEC, 1)

        if settings.AGENT_TIMEOUT_SEC <= 0:
            logger.warning(
                "agent_timeout_invalid",
                value=settings.AGENT_TIMEOUT_SEC,
                using=self.timeout,
            )

    # NORMALIZE QUERY

    def _normalize(self, q: str) -> str:
        return _normalize(q)

    # MAIN SYNC HANDLE

    def handle(self, query: str, session_id: str) -> Dict[str, Any]:

        if not query or not query.strip():
            return self._reject("empty_query")

        start      = time.time()
        request_id = str(uuid.uuid4())
        query      = self._normalize(query)[:settings.MAX_PROMPT_CHARS]

        # INJECTION SANITIZATION
        query = _sanitize(query)
        if not query.strip():
            return self._reject("injection_detected")

        _active_requests.inc()

        with tracer.start_as_current_span("agent_controller_handle") as span:
            span.set_attribute("request.id", request_id)
            span.set_attribute("session.id", session_id)
            span.set_attribute("query.length", len(query))

            try:
                logger.info(
                    "agent_start",
                    request_id=request_id,
                    query_len=len(query),
                    session_id=session_id,
                )

                t_agent       = time.time()
                result        = self._execute_with_timeout(query, session_id)
                agent_latency = round(time.time() - t_agent, 3)

                if not self._validate(result):
                    raise ValueError("INVALID_AGENT_OUTPUT")

                response   = self._format(result)
                confidence = self._confidence(result)
                decision   = response.get("decision", "unknown")

                response.update({
                    "request_id":       request_id,
                    "latency":          round(time.time() - start, 3),
                    "agent_latency":    agent_latency,
                    "agent_latency_ms": round(agent_latency * 1000, 1),
                    "confidence":       confidence,
                })

                _controller_duration.labels(
                    decision=decision,
                    status="success",
                ).observe(response["latency"])

                span.set_attribute("decision", decision)
                span.set_attribute("confidence", confidence)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "agent_success",
                    request_id=request_id,
                    decision=decision,
                    confidence=confidence,
                    latency=response["latency"],
                    session_id=session_id,
                )

                return response

            except Exception as exc:
                latency    = round(time.time() - start, 3)
                error_type = type(exc).__name__

                _controller_errors.labels(error_type=error_type).inc()
                _controller_duration.labels(
                    decision="fallback",
                    status="error",
                ).observe(latency)

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "agent_failed",
                    request_id=request_id,
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )

                return self._fallback(query, start, session_id, request_id)

            finally:
                _active_requests.dec()

    # TIMEOUT EXECUTION

    def _execute_with_timeout(
        self,
        query: str,
        session_id: str,
    ) -> Dict[str, Any]:
        start   = time.time()
        result  = self.executor.run(query, session_id)
        elapsed = time.time() - start

        if elapsed > self.timeout:
            raise TimeoutError(
                f"AGENT_TIMEOUT_{elapsed:.1f}s > {self.timeout}s"
            )

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
        metadata  = result.get("metadata", {}) or {}
        meta_conf = metadata.get("confidence")

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

    # FORMAT RESULT

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

    # FALLBACK — LLM DIRECT GENERATION

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=False,
    )
    def _fallback(
        self,
        query: str,
        start_time: float,
        session_id: str = "default",
        request_id: str = "",
    ) -> Dict[str, Any]:

        _fallback_invocations.inc()

        try:
            llm      = model_loader.get_llm()
            prompt   = f"Answer clearly and concisely:\n{query}"
            response = llm.generate(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.2,
                top_p=settings.LLM_TOP_P,
                session_id=session_id,
            )

        except Exception as exc:
            logger.error(
                "controller_fallback_failed",
                request_id=request_id,
                error=str(exc),
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

    # ASYNC HANDLE

    async def handle_async(
        self,
        query: str,
        session_id: str,
    ) -> Dict[str, Any]:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.handle(query, session_id),
            )

