import math
import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings


# ACTION TYPE

ActionType = Literal["rag", "search", "direct", "memory", "hybrid"]

ALLOWED_ACTIONS = {"rag", "search", "direct", "memory", "hybrid"}


class AgentDecision(BaseModel):

    model_config = {"validate_assignment": True}

    # CORE
    action:     ActionType
    reason:     str

    # CONFIDENCE
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # CONTEXT
    session_id: str = Field(default="default")
    signals:    Dict[str, Any]  = Field(default_factory=dict)
    suggested_tools: List[str]  = Field(default_factory=list)
    action_history:  List[str]  = Field(default_factory=list)
    trace:      Dict[str, Any]  = Field(default_factory=dict)

    # TIMING
    created_at: float  = Field(default_factory=time.time)
    latency_ms: Optional[float] = Field(default=None)

    # NORMALIZE

    def normalize(self) -> "AgentDecision":
        self.action = str(self.action).strip().lower()
        self.reason = (self.reason or "").strip()
        self.reason = self.reason or "no_reason_provided"
        return self

    # VALIDATE STRICT

    def validate_strict(self) -> "AgentDecision":
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"INVALID_ACTION_{self.action}")
        self._normalize_confidence()
        return self

    # VALIDATE SAFE

    def validate_safe(self) -> "AgentDecision":
        if self.action not in ALLOWED_ACTIONS:
            self.action_history.append(self.action)
            self.action = "rag"
            self.reason = self.reason or "invalid_action_fallback"
        self._normalize_confidence()
        return self

    # CONFIDENCE NORMALIZATION

    def _normalize_confidence(self) -> None:
        if not isinstance(self.confidence, (int, float)):
            self.confidence = 0.5
            return

        val = float(self.confidence)

        if math.isnan(val) or math.isinf(val):
            self.confidence = 0.5
            return

        self.confidence = max(0.0, min(val, 1.0))

    # FINALIZE

    def finalize(self, strict: bool = False) -> "AgentDecision":
        self.normalize()
        if strict:
            return self.validate_strict()
        return self.validate_safe()

    # DECISION HELPERS

    def is_high_confidence(self) -> bool:
        return self.confidence >= settings.AGENT_HIGH_CONFIDENCE

    def requires_fallback(self) -> bool:
        return self.confidence < settings.AGENT_LOW_CONFIDENCE

    def is_retrieval(self) -> bool:
        return self.action in {"rag", "search", "hybrid"}

    def is_memory_based(self) -> bool:
        return self.action == "memory"

    def is_direct(self) -> bool:
        return self.action == "direct"

    def is_multimodal(self) -> bool:
        return bool(self.signals.get("has_multimodal_hint", False))

    # TRACE

    def add_trace(self, key: str, value: Any) -> "AgentDecision":
        try:
            self.trace[key] = value
        except Exception:
            pass
        return self

    def record_action(self, action: str) -> "AgentDecision":
        self.action_history.append(action)
        return self

    def set_latency(self, start_time: float) -> "AgentDecision":
        self.latency_ms = round((time.time() - start_time) * 1000, 2)
        return self

    # SERIALIZATION

    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:
        if minimal:
            return {
                "action":     self.action,
                "confidence": self.confidence,
            }

        return {
            "action":          self.action,
            "reason":          self.reason,
            "confidence":      self.confidence,
            "session_id":      self.session_id,
            "signals":         self.signals,
            "suggested_tools": self.suggested_tools,
            "action_history":  self.action_history,
            "trace":           self.trace,
            "created_at":      self.created_at,
            "latency_ms":      self.latency_ms,
        }

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "action":     self.action,
            "reason":     self.reason,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
        }