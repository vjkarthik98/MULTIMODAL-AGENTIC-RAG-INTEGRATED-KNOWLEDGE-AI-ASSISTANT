import math
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

# ACTION TYPE

ActionType = Literal["rag", "search", "direct", "memory", "hybrid"]

ALLOWED_ACTIONS = {"rag", "search", "direct", "memory", "hybrid"}

# STEP COST LEVELS
COST_LOW = "low"
COST_MEDIUM = "medium"
COST_HIGH = "high"


# TOOL CALL SCHEMA — SINGLE TOOL INVOCATION CONTRACT


class ToolCall(BaseModel):

    model_config = {"validate_assignment": True}

    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any | None = Field(default=None)
    status: str = Field(default="pending")
    latency_ms: float | None = Field(default=None)
    error: str | None = Field(default=None)
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# EXECUTION STEP SCHEMA — SINGLE PLAN STEP


class ExecutionStep(BaseModel):

    model_config = {"validate_assignment": True}

    tool: str
    description: str = Field(default="")
    optional: bool = Field(default=False)
    cost: str = Field(default=COST_MEDIUM)
    depends_on: list[str] = Field(default_factory=list)
    parallel: bool = Field(default=False)

    @field_validator("cost")
    @classmethod
    def validate_cost(cls, v: str) -> str:
        allowed = {COST_LOW, COST_MEDIUM, COST_HIGH}
        if v not in allowed:
            return COST_MEDIUM
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "description": self.description,
            "optional": self.optional,
            "cost": self.cost,
            "depends_on": self.depends_on,
            "parallel": self.parallel,
        }


# EXECUTION PLAN SCHEMA


class ExecutionPlan(BaseModel):

    model_config = {"validate_assignment": True}

    steps: list[ExecutionStep] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_cost(self) -> str:
        cost_rank = {COST_LOW: 0, COST_MEDIUM: 1, COST_HIGH: 2}
        if not self.steps:
            return COST_LOW
        max_cost = max(cost_rank.get(s.cost, 1) for s in self.steps)
        return [COST_LOW, COST_MEDIUM, COST_HIGH][max_cost]

    def tool_sequence(self) -> list[str]:
        return [s.tool for s in self.steps]

    def parallel_groups(self) -> list[list[ExecutionStep]]:
        """
        GROUP STEPS INTO SEQUENTIAL BATCHES WHERE PARALLEL STEPS
        CAN EXECUTE CONCURRENTLY AND NON-PARALLEL STEPS BLOCK.
        """
        groups: list[list[ExecutionStep]] = []
        current_group: list[ExecutionStep] = []

        for step in self.steps:
            if step.parallel:
                current_group.append(step)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([step])

        if current_group:
            groups.append(current_group)

        return groups

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.steps]


# AGENT SIGNAL SCHEMA — STRUCTURED SIGNALS FROM ROUTER ANALYSIS


class AgentSignals(BaseModel):

    model_config = {"validate_assignment": True}

    is_recent: bool = False
    is_memory: bool = False
    is_web: bool = False
    is_complex: bool = False
    is_reasoning: bool = False
    has_multimodal_hint: bool = False
    is_code: bool = False
    is_greeting: bool = False
    is_math: bool = False
    is_security: bool = False
    multi_question: bool = False
    has_question_mark: bool = False
    token_count: int = 0
    # Finance domain signals (Phase 7)
    is_earnings_call_query: bool = False
    is_regulatory_query: bool = False
    is_financial_model_query: bool = False
    is_market_data_query: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def active_signals(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if isinstance(v, bool) and v]


# AGENT DECISION SCHEMA — CORE ROUTING DECISION


class AgentDecision(BaseModel):

    model_config = {"validate_assignment": True}

    # CORE FIELDS
    action: ActionType
    reason: str

    # CONFIDENCE
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # CONTEXT
    session_id: str = Field(default="default")
    signals: dict[str, Any] = Field(default_factory=dict)
    suggested_tools: list[str] = Field(default_factory=list)
    action_history: list[str] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    query_type: str | None = Field(default=None)
    # Finance retrieval filters (Phase 7)
    modality_hint: str | None = Field(default=None)
    source_type_filter: list[str] = Field(default_factory=list)
    subtype_filter: list[str] = Field(default_factory=list)
    call_section_filter: str | None = Field(default=None)

    # TIMING
    created_at: float = Field(default_factory=time.time)
    latency_ms: float | None = Field(default=None)

    # VALIDATORS

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v: Any) -> str:
        v = str(v).strip().lower()
        if v not in ALLOWED_ACTIONS:
            return "rag"
        return v

    @field_validator("reason", mode="before")
    @classmethod
    def validate_reason(cls, v: Any) -> str:
        v = str(v or "").strip()
        return v or "no_reason_provided"

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                return 0.5
            return max(0.0, min(val, 1.0))
        except (TypeError, ValueError):
            return 0.5

    # NORMALIZE

    def normalize(self) -> "AgentDecision":
        self.action = str(self.action).strip().lower()
        self.reason = (self.reason or "").strip() or "no_reason_provided"
        return self

    # VALIDATE STRICT — RAISES ON INVALID ACTION

    def validate_strict(self) -> "AgentDecision":
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"INVALID_ACTION_{self.action}")
        self._normalize_confidence()
        return self

    # VALIDATE SAFE — CORRECTS INVALID ACTION TO RAG

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

    def is_complex(self) -> bool:
        return bool(self.signals.get("is_complex", False))

    # TRACE METHODS

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

    def add_tool_call(self, tool_call: ToolCall) -> "AgentDecision":
        self.tool_calls.append(tool_call)
        return self

    # SERIALIZATION

    def to_dict(self, minimal: bool = False) -> dict[str, Any]:
        if minimal:
            return {
                "action": self.action,
                "confidence": self.confidence,
            }

        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "signals": self.signals,
            "suggested_tools": self.suggested_tools,
            "action_history": self.action_history,
            "trace": self.trace,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "query_type": self.query_type,
            "modality_hint": self.modality_hint,
            "source_type_filter": self.source_type_filter,
            "subtype_filter": self.subtype_filter,
            "call_section_filter": self.call_section_filter,
            "created_at": self.created_at,
            "latency_ms": self.latency_ms,
        }

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
            "query_type": self.query_type,
        }


# AGENT RESPONSE SCHEMA — FINAL OUTPUT FROM AGENT


class AgentResponse(BaseModel):

    model_config = {"validate_assignment": True}

    response: str
    source: str = "agent"
    decision: str = "unknown"
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    request_id: str = Field(default="")
    session_id: str = Field(default="default")
    latency: float = Field(default=0.0, ge=0.0)
    agent_latency: float = Field(default=0.0, ge=0.0)
    agent_latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    hallucination_warning: bool = Field(default=False)
    is_fallback: bool = Field(default=False)
    sources: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                return 0.5
            return max(0.0, min(val, 1.0))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, v: Any) -> list[dict[str, Any]]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "source": self.source,
            "decision": self.decision,
            "reason": self.reason,
            "confidence": self.confidence,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "latency": self.latency,
            "agent_latency": self.agent_latency,
            "agent_latency_ms": self.agent_latency_ms,
            "metadata": self.metadata,
            "hallucination_warning": self.hallucination_warning,
            "is_fallback": self.is_fallback,
            "sources": self.sources,
        }
