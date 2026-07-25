"""Guardrail exception hierarchy.

GuardrailBlocked is raised by InputGuard or OutputGuard whenever a
request or response violates policy. The caller (API layer or pipeline)
catches it and converts it to the appropriate HTTP / response shape.
"""

from __future__ import annotations


class GuardrailBlocked(Exception):
    """Raised when a guardrail blocks a request or response.

    Attributes:
        reason      Short machine-readable slug (e.g. "injection_detected").
        surface     "input" or "output".
        guard_type  Category: "injection", "jailbreak", "pii", "ssrf",
                    "toxicity", "groundedness", "length", "encoding",
                    "citation", "template_artifact", "scope_violation".
        correlation_id  Request trace ID for audit cross-reference.
        detail      Optional human-readable detail (never exposed to end users
                    in production; used only in audit logs).
    """

    def __init__(
        self,
        reason: str,
        surface: str = "input",
        guard_type: str = "injection",
        correlation_id: str = "",
        detail: str = "",
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.surface = surface
        self.guard_type = guard_type
        self.correlation_id = correlation_id
        self.detail = detail

    def __repr__(self) -> str:
        return (
            f"GuardrailBlocked(reason={self.reason!r}, surface={self.surface!r}, "
            f"guard_type={self.guard_type!r}, correlation_id={self.correlation_id!r})"
        )
