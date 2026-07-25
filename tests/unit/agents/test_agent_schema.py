"""Unit tests for app/agents/agent_schema.py — Phase 24.9."""
from __future__ import annotations

import math
import time

import pytest

from app.agents.agent_schema import (
    ALLOWED_ACTIONS,
    AgentDecision,
    AgentResponse,
    AgentSignals,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
)
from app.core.config import settings


# ── AgentDecision ────────────────────────────────────────────────────────────

class TestAgentDecision:

    def test_valid_action_accepted(self):
        for action in ALLOWED_ACTIONS:
            d = AgentDecision(action=action, reason="ok")
            assert d.action == action

    def test_invalid_action_corrected_to_rag_by_validate_safe(self):
        d = AgentDecision(action="INVALID_ACTION", reason="test")
        d.validate_safe()
        assert d.action == "rag"

    def test_invalid_action_preserved_before_validate(self):
        # Pydantic validator converts to rag at construction
        d = AgentDecision(action="bogus", reason="test")
        assert d.action == "rag"

    def test_confidence_clamped_to_zero_to_one(self):
        d = AgentDecision(action="rag", reason="test", confidence=2.5)
        assert d.confidence <= 1.0
        d2 = AgentDecision(action="rag", reason="test", confidence=-0.5)
        assert d2.confidence >= 0.0

    def test_nan_confidence_replaced_with_0_5(self):
        d = AgentDecision(action="rag", reason="test", confidence=float("nan"))
        assert d.confidence == 0.5

    def test_inf_confidence_replaced_with_0_5(self):
        d = AgentDecision(action="rag", reason="test", confidence=float("inf"))
        assert d.confidence == 0.5

    def test_finalize_calls_validate_safe(self):
        d = AgentDecision(action="rag", reason="test")
        result = d.finalize()
        assert result.action in ALLOWED_ACTIONS

    def test_finalize_strict_raises_on_invalid(self):
        d = AgentDecision(action="rag", reason="test")
        # Force an invalid action after construction (bypass Pydantic)
        object.__setattr__(d, "action", "bad_action")
        with pytest.raises(ValueError, match="INVALID_ACTION"):
            d.validate_strict()

    def test_is_retrieval_for_rag_hybrid_search(self):
        for action in ("rag", "search", "hybrid"):
            d = AgentDecision(action=action, reason="test")
            assert d.is_retrieval() is True

    def test_is_retrieval_false_for_direct_memory(self):
        for action in ("direct", "memory"):
            d = AgentDecision(action=action, reason="test")
            assert d.is_retrieval() is False

    def test_is_high_confidence(self):
        d = AgentDecision(action="rag", reason="test", confidence=settings.AGENT_HIGH_CONFIDENCE)
        assert d.is_high_confidence() is True

    def test_requires_fallback(self):
        d = AgentDecision(action="rag", reason="test", confidence=0.0)
        assert d.requires_fallback() is True

    def test_to_dict_contains_required_keys(self):
        d = AgentDecision(action="rag", reason="test", session_id="s1")
        result = d.to_dict()
        for key in ("action", "reason", "confidence", "session_id", "signals", "trace"):
            assert key in result

    def test_add_trace_stores_value(self):
        d = AgentDecision(action="rag", reason="test")
        d.add_trace("my_key", "my_val")
        assert d.trace["my_key"] == "my_val"

    def test_set_latency(self):
        d = AgentDecision(action="rag", reason="test")
        start = time.time() - 0.1
        d.set_latency(start)
        assert d.latency_ms is not None
        assert d.latency_ms > 0

    def test_record_action(self):
        d = AgentDecision(action="rag", reason="test")
        d.record_action("rag")
        assert "rag" in d.action_history

    def test_reason_defaults_when_empty(self):
        d = AgentDecision(action="rag", reason="")
        assert d.reason == "no_reason_provided"


# ── AgentResponse ────────────────────────────────────────────────────────────

class TestAgentResponse:

    def test_confidence_clamped(self):
        r = AgentResponse(response="answer", confidence=5.0)
        assert r.confidence <= 1.0

    def test_nan_confidence_replaced(self):
        r = AgentResponse(response="answer", confidence=float("nan"))
        assert r.confidence == 0.5

    def test_sources_defaults_to_empty_list(self):
        r = AgentResponse(response="answer")
        assert r.sources == []

    def test_sources_none_coerced_to_empty(self):
        r = AgentResponse(response="answer", sources=None)
        assert r.sources == []

    def test_to_dict_has_all_fields(self):
        r = AgentResponse(response="answer", session_id="s1", decision="rag")
        d = r.to_dict()
        for key in ("response", "confidence", "sources", "decision", "session_id",
                    "hallucination_warning", "is_fallback"):
            assert key in d

    def test_hallucination_warning_default_false(self):
        r = AgentResponse(response="answer")
        assert r.hallucination_warning is False


# ── ExecutionStep ────────────────────────────────────────────────────────────

class TestExecutionStep:

    def test_valid_cost_accepted(self):
        for cost in ("low", "medium", "high"):
            s = ExecutionStep(tool="rag", cost=cost)
            assert s.cost == cost

    def test_invalid_cost_defaults_to_medium(self):
        s = ExecutionStep(tool="rag", cost="super_expensive")
        assert s.cost == "medium"

    def test_to_dict_has_required_keys(self):
        s = ExecutionStep(tool="rag", description="retrieve docs")
        d = s.to_dict()
        for key in ("tool", "description", "optional", "cost"):
            assert key in d


# ── ExecutionPlan ────────────────────────────────────────────────────────────

class TestExecutionPlan:

    def test_empty_plan_has_low_cost(self):
        p = ExecutionPlan()
        assert p.total_cost == "low"

    def test_tool_sequence_returns_list(self):
        p = ExecutionPlan(steps=[
            ExecutionStep(tool="retrieve"),
            ExecutionStep(tool="reason"),
        ])
        assert p.tool_sequence() == ["retrieve", "reason"]

    def test_to_list_serializes_steps(self):
        p = ExecutionPlan(steps=[ExecutionStep(tool="rag")])
        lst = p.to_list()
        assert len(lst) == 1
        assert lst[0]["tool"] == "rag"


# ── AgentSignals ─────────────────────────────────────────────────────────────

class TestAgentSignals:

    def test_defaults_all_false(self):
        s = AgentSignals()
        assert s.is_recent is False
        assert s.is_memory is False
        assert s.is_greeting is False

    def test_active_signals_returns_true_keys(self):
        s = AgentSignals(is_greeting=True, is_code=True)
        active = s.active_signals()
        assert "is_greeting" in active
        assert "is_code" in active
        assert "is_recent" not in active


# ── ToolCall ─────────────────────────────────────────────────────────────────

class TestToolCall:

    def test_default_status_pending(self):
        tc = ToolCall(name="rag_retrieve", input={})
        assert tc.status == "pending"

    def test_to_dict_has_required_keys(self):
        tc = ToolCall(name="rag_retrieve", input={"query": "test"})
        d = tc.to_dict()
        for key in ("name", "input", "output", "status", "latency_ms"):
            assert key in d
