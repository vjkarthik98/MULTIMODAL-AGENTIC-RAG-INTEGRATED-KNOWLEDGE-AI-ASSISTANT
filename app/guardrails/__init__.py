"""Phase 26 — Consolidated Guardrails.

Single source of truth for all input/output safety. Import from here:

    from app.guardrails import InputGuard, OutputGuard, GuardrailBlocked

Public API:
    InputGuard.check(query, surface, session_id, correlation_id)
        → GuardedInput  (or raises GuardrailBlocked)

    InputGuard.sanitize(text, surface, session_id, correlation_id)
        → str  (non-blocking, for embedder/captioner paths)

    OutputGuard.check(answer, context_chunks, sources, session_id, correlation_id)
        → GuardedOutput  (or raises GuardrailBlocked on toxicity)

    GuardrailBlocked(reason, surface, guard_type, correlation_id)
        — exception with full attribution

    warm_up()  — call from server startup to pre-init Presidio + jailbreak corpus
"""
from app.guardrails.exceptions import GuardrailBlocked
from app.guardrails.input_guard import GuardedInput, check as _input_check, sanitize as _input_sanitize
from app.guardrails.output_guard import GuardedOutput, check as _output_check, safe_refusal


class InputGuard:
    """Namespace wrapper for input guard functions."""

    @staticmethod
    def check(
        query: str,
        surface: str = "api",
        session_id: str = "",
        correlation_id: str = "",
    ) -> GuardedInput:
        return _input_check(query, surface=surface, session_id=session_id, correlation_id=correlation_id)

    @staticmethod
    def sanitize(
        text: str,
        surface: str = "embedder",
        session_id: str = "",
        correlation_id: str = "",
    ) -> str:
        return _input_sanitize(text, surface=surface, session_id=session_id, correlation_id=correlation_id)


class OutputGuard:
    """Namespace wrapper for output guard functions."""

    @staticmethod
    def check(
        answer: str,
        context_chunks=None,
        sources=None,
        session_id: str = "",
        correlation_id: str = "",
        surface: str = "output",
    ) -> GuardedOutput:
        return _output_check(
            answer,
            context_chunks=context_chunks,
            sources=sources,
            session_id=session_id,
            correlation_id=correlation_id,
            surface=surface,
        )

    @staticmethod
    def safe_refusal(reason: str) -> str:
        return safe_refusal(reason)


def warm_up() -> None:
    """Pre-initialize all guardrail subsystems at server startup.

    Presidio (PII), jailbreak corpus embeddings, policy cache.
    Call from app/core/startup_optimizer.py or app/main.py lifespan.
    """
    try:
        from app.guardrails.pii import warm_up as pii_warmup
        pii_warmup()
    except Exception:
        pass

    try:
        from app.guardrails.jailbreak import initialize as jb_init
        jb_init()
    except Exception:
        pass

    try:
        from app.guardrails._policy_loader import get_policy
        get_policy()
    except Exception:
        pass


__all__ = [
    "GuardrailBlocked",
    "GuardedInput",
    "GuardedOutput",
    "InputGuard",
    "OutputGuard",
    "warm_up",
    "safe_refusal",
]
