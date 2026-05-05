from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Literal

from app.core.config import settings


ActionType = Literal["rag", "search", "direct", "memory", "hybrid"]


class AgentDecision(BaseModel):

    action: ActionType
    reason: str

    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    signals: Dict[str, Any] = Field(default_factory=dict)
    suggested_tools: List[str] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)

    #  NORMALIZE 
    def normalize(self):

        self.action = str(self.action).strip().lower()
        self.reason = (self.reason or "").strip()

        if not self.reason:
            self.reason = "no_reason_provided"

        return self

    #  VALIDATION 
    def validate_strict(self):

        allowed = {"rag", "search", "direct", "memory", "hybrid"}

        if self.action not in allowed:
            raise ValueError(f"INVALID_ACTION_{self.action}")

        self._normalize_confidence()

        return self

    def validate_safe(self):

        allowed = {"rag", "search", "direct", "memory", "hybrid"}

        if self.action not in allowed:
            self.action = "rag"
            self.reason = self.reason or "invalid_action_fallback"

        self._normalize_confidence()

        return self

    #  CONFIDENCE 
    def _normalize_confidence(self):

        if not isinstance(self.confidence, (int, float)):
            self.confidence = 0.5

        self.confidence = float(self.confidence)
        self.confidence = max(0.0, min(self.confidence, 1.0))

    #  FINALIZE 
    def finalize(self, strict: bool = False):

        self.normalize()

        if strict:
            return self.validate_strict()

        return self.validate_safe()

    #  DECISION LOGIC 
    def is_high_confidence(self) -> bool:
        return self.confidence >= getattr(settings, "AGENT_HIGH_CONFIDENCE", 0.7)

    def requires_fallback(self) -> bool:
        return self.confidence < getattr(settings, "AGENT_LOW_CONFIDENCE", 0.4)

    def is_retrieval(self) -> bool:
        return self.action in {"rag", "search", "hybrid"}

    def is_memory_based(self) -> bool:
        return self.action == "memory"

    def is_direct(self) -> bool:
        return self.action == "direct"

    #  TRACE 
    def add_trace(self, key: str, value: Any):
        try:
            self.trace[key] = value
        except Exception:
            pass
        return self

    #  SERIALIZATION 
    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:

        if minimal:
            return {
                "action": self.action,
                "confidence": self.confidence
            }

        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "signals": self.signals,
            "suggested_tools": self.suggested_tools,
            "trace": self.trace
        }