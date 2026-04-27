from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.core.config import settings


_ALLOWED_ACTIONS = {"rag", "search", "direct", "memory", "hybrid"}


class AgentDecision(BaseModel):
    action: str
    reason: str
    confidence: float = 0.8
    signals: Optional[Dict[str, Any]] = None
    suggested_tools: Optional[List[str]] = None
    trace: Optional[Dict[str, Any]] = None

    def normalize(self):
        if self.action:
            self.action = self.action.strip().lower()

        if self.reason:
            self.reason = self.reason.strip()

        if self.signals is None:
            self.signals = {}

        if self.suggested_tools is None:
            self.suggested_tools = []

        if self.trace is None:
            self.trace = {}

        return self

    def validate(self):
        if not self.action or self.action not in _ALLOWED_ACTIONS:
            self.action = "search"
            self.reason = self.reason or "invalid_action_fallback"

        if not self.reason:
            self.reason = "no_reason_provided"

        if not isinstance(self.confidence, (int, float)):
            self.confidence = 0.5

        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

        return self

    def finalize(self):
        self.normalize()
        self.validate()
        return self

    def is_high_confidence(self) -> bool:
        threshold = getattr(settings, "AGENT_HIGH_CONFIDENCE", 0.7)
        return self.confidence >= threshold

    def requires_fallback(self) -> bool:
        threshold = getattr(settings, "AGENT_LOW_CONFIDENCE", 0.4)
        return self.confidence < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "signals": self.signals or {},
            "suggested_tools": self.suggested_tools or [],
            "trace": self.trace or {}
        }