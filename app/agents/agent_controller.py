import asyncio
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, Gauge
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.model_loader import model_loader

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent_ctrl")

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

# SEMAPHORE — created lazily inside async context to avoid missing event loop
_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_lock = threading.Lock()


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        with _semaphore_lock:
            if _semaphore is None:
                _semaphore = asyncio.Semaphore(settings.MAX_PARALLEL_REQUESTS)
    return _semaphore


# NORMALIZE QUERY

def _normalize(q: str) -> str:
    q = unicodedata.normalize("NFC", str(q or ""))
    return " ".join(q.strip().split())


# INPUT GUARD — blocking check (raises GuardrailBlocked on injection/jailbreak/SSRF)

def _guard_input(query: str, session_id: str = "", correlation_id: str = "") -> str:
    """Full blocking input check. Returns safe query text or raises GuardrailBlocked."""
    from app.guardrails.input_guard import check as _input_check
    from app.guardrails.exceptions import GuardrailBlocked
    guarded = _input_check(query, surface="agent_controller", session_id=session_id, correlation_id=correlation_id)
    return guarded.text


# OUTPUT GUARD — scrubs template artifacts, PII, fabricated citations before return

def _guard_output(response: str, session_id: str = "", correlation_id: str = "") -> str:
    """Non-blocking output scrub. Returns cleaned response text."""
    try:
        from app.guardrails.output_guard import check as _output_check
        og = _output_check(
            response,
            context_chunks=[],
            sources=[],
            session_id=session_id,
            correlation_id=correlation_id,
            surface="agent_controller_output",
        )
        return og.text
    except Exception:
        return response


class AgentExecutor:
    """Delegates routing to AgentRouter. Only handles routes that bypass RAG
    (direct/search/memory). RAG/hybrid routes return a thin signal so the
    pipeline performs retrieval — this executor must NEVER fabricate an answer
    to a factual document question."""

    # Whole-word greeting tokens. Substring matching previously caused "hi"
    # to match "which", "this", "while" — routing factual questions to direct.
    _CONVERSATIONAL_TOKENS = frozenset({
        "hello", "hi", "hey", "thanks", "bye", "goodbye",
    })
    _CONVERSATIONAL_PHRASES = frozenset({
        "thank you", "how are you", "good morning",
        "good afternoon", "good evening",
    })

    def __init__(self) -> None:
        from app.agents.agent_router import AgentRouter
        self._router = AgentRouter()

    def _is_conversational(self, query: str) -> bool:
        lower = query.lower().strip().rstrip("?!.,")
        tokens = set(lower.split())
        if tokens & self._CONVERSATIONAL_TOKENS and len(tokens) <= 4:
            return True
        if any(p == lower or lower.startswith(p + " ") for p in self._CONVERSATIONAL_PHRASES):
            return True
        return False

    def run(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        # Cheap fast-path for genuine greetings (whole-word, short query only)
        if self._is_conversational(query):
            return self._direct(query, session_id, "greeting")

        # Delegate to the router for all other queries. The router uses
        # signal analysis + LLM and returns one of rag/search/direct/memory/hybrid.
        try:
            decision = self._router.route(query, session_id)
            action = decision.action
            reason = decision.reason
            confidence = float(decision.confidence)
        except Exception as exc:
            logger.warning("executor_router_failed", error=str(exc), session_id=session_id)
            action, reason, confidence = "rag", "router_exception", 0.5

        # RAG/hybrid → return a signal; the pipeline performs retrieval.
        if action in {"rag", "hybrid"}:
            return {
                "response": "Routing to knowledge base.",
                "source":   "rag",
                "decision": action,
                "reason":   reason or "knowledge_query",
                "metadata": {"confidence": max(confidence, 0.6)},
            }

        # search/memory → also requires the pipeline to fulfil; pass through.
        if action in {"search", "memory"}:
            return {
                "response": f"Routing to {action}.",
                "source":   action,
                "decision": action,
                "reason":   reason or f"{action}_route",
                "metadata": {"confidence": confidence},
            }

        # direct → only safe when router is confident this is NOT a doc question.
        return self._direct(query, session_id, reason or "router_direct")

    def _direct(self, query: str, session_id: str, reason: str) -> Dict[str, Any]:
        try:
            llm      = model_loader.get_llm()
            response = llm.generate(
                f"Answer briefly and helpfully:\n{query}",
                max_tokens=256,
                temperature=0.3,
                session_id=session_id,
            )
            if not response or len(response.strip()) < 3:
                raise ValueError("empty_llm_response")
        except Exception:
            response = "I can help with that. Please ask a more specific question."

        response = _guard_output(response, session_id=session_id)
        return {
            "response": response,
            "source":   "llm",
            "decision": "direct",
            "reason":   reason,
            "metadata": {"confidence": 0.70},
        }


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

        # RATE LIMIT CHECK — reject banned sessions before any processing
        try:
            from app.guardrails.exceptions import GuardrailBlocked
            from app.guardrails.output_guard import safe_refusal
            from app.guardrails.rate_limiter import enforce as _rl_enforce, check_and_record as _rl_record
            _rl_enforce(session_id=session_id, surface="agent_controller", correlation_id=request_id)
        except GuardrailBlocked as gb:
            return {
                "response":   "Too many blocked requests. Please wait before retrying.",
                "source":     "guardrail",
                "decision":   "reject",
                "reason":     gb.reason,
                "confidence": 0.0,
                "request_id": request_id,
                "latency":    round(time.time() - start, 3),
                "metadata":   {"guard_type": gb.guard_type},
            }

        # INPUT GUARD — blocking: raises GuardrailBlocked on injection/jailbreak/SSRF
        try:
            query = _guard_input(query, session_id=session_id, correlation_id=request_id)
        except GuardrailBlocked as gb:
            _rl_record(session_id=session_id, surface="agent_controller", correlation_id=request_id)
            return {
                "response":   safe_refusal(gb.reason),
                "source":     "guardrail",
                "decision":   "reject",
                "reason":     gb.reason,
                "confidence": 0.0,
                "request_id": request_id,
                "latency":    round(time.time() - start, 3),
                "metadata":   {"guard_type": gb.guard_type},
            }
        if not query.strip():
            return self._reject("empty_after_normalization")

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

                try:
                    return self._fallback(query, start, session_id, request_id)
                except Exception:
                    return {
                        "response":   "Unable to process your request at this time.",
                        "source":     "fallback",
                        "decision":   "direct",
                        "reason":     "all_retries_exhausted",
                        "confidence": 0.1,
                        "request_id": request_id,
                        "latency":    round(time.time() - start, 3),
                        "metadata":   {},
                    }

            finally:
                _active_requests.dec()

    # TIMEOUT EXECUTION — uses a real future so the timeout actually preempts

    def _execute_with_timeout(
        self,
        query: str,
        session_id: str,
    ) -> Dict[str, Any]:
        future = _executor.submit(self.executor.run, query, session_id)
        try:
            return future.result(timeout=self.timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"AGENT_TIMEOUT: executor did not finish within {self.timeout}s"
            )

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
        reraise=True,
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

        response = _guard_output(response, session_id=session_id, correlation_id=request_id)
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

        async with _get_semaphore():
            return await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.handle(query, session_id),
            )

