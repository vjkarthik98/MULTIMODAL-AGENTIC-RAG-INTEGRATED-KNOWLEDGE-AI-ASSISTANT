from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal

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

        self.action = self.action.strip().lower()
        self.reason = self.reason.strip()

        return self

    #  STRICT VALIDATION 
    def validate_strict(self):

        if self.action not in {"rag", "search", "direct", "memory", "hybrid"}:
            raise ValueError(f"Invalid action: {self.action}")

        if not self.reason:
            self.reason = "no_reason_provided"

        if not isinstance(self.confidence, (int, float)):
            self.confidence = 0.5

        self.confidence = max(0.0, min(self.confidence, 1.0))

        return self

    #  SAFE VALIDATION
    def validate_safe(self):

        if self.action not in {"rag", "search", "direct", "memory", "hybrid"}:
            self.action = "rag"
            self.reason = self.reason or "invalid_action_fallback"

        if not self.reason:
            self.reason = "no_reason_provided"

        if not isinstance(self.confidence, (int, float)):
            self.confidence = 0.5

        self.confidence = max(0.0, min(self.confidence, 1.0))

        return self

    #  FINALIZE 
    def finalize(self, strict: bool = False):

        self.normalize()

        if strict:
            return self.validate_strict()

        return self.validate_safe()

    #  DECISION INTELLIGENCE 
    def is_high_confidence(self) -> bool:
        threshold = getattr(settings, "AGENT_HIGH_CONFIDENCE", 0.7)
        return self.confidence >= threshold

    def requires_fallback(self) -> bool:
        threshold = getattr(settings, "AGENT_LOW_CONFIDENCE", 0.4)
        return self.confidence < threshold

    def is_retrieval(self) -> bool:
        return self.action in {"rag", "search", "hybrid"}

    def is_memory_based(self) -> bool:
        return self.action == "memory"

    def is_direct(self) -> bool:
        return self.action == "direct"

    #  TRACE ENRICHMENT 
    def add_trace(self, key: str, value: Any):
        self.trace[key] = value
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