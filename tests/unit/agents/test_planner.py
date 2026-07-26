import pytest

from app.agents.agent_schema import AgentDecision, ExecutionPlan, ExecutionStep
from app.agents.planner import Planner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decision(action: str, signals: dict = None) -> AgentDecision:
    return AgentDecision(
        action=action,
        reason="test",
        confidence=0.8,
        session_id="s1",
        signals=signals or {},
    )


def _make_planner() -> Planner:
    return Planner()


# ---------------------------------------------------------------------------
# ExecutionPlan helpers
# ---------------------------------------------------------------------------

class TestExecutionPlan:

    def test_tool_sequence_returns_tool_names(self):
        plan = ExecutionPlan(
            steps=[
                ExecutionStep(tool="rag", description="retrieve"),
                ExecutionStep(tool="reason", description="answer"),
            ]
        )
        assert plan.tool_sequence() == ["rag", "reason"]

    def test_total_cost_is_highest(self):
        plan = ExecutionPlan(
            steps=[
                ExecutionStep(tool="memory", cost="low"),
                ExecutionStep(tool="rag",    cost="medium"),
                ExecutionStep(tool="reason", cost="high"),
            ]
        )
        assert plan.total_cost == "high"

    def test_total_cost_empty_plan(self):
        plan = ExecutionPlan(steps=[])
        assert plan.total_cost == "low"

    def test_to_list_returns_dicts(self):
        plan = ExecutionPlan(
            steps=[ExecutionStep(tool="rag", description="retrieve")]
        )
        lst = plan.to_list()
        assert isinstance(lst, list)
        assert isinstance(lst[0], dict)
        assert lst[0]["tool"] == "rag"

    def test_parallel_groups_non_parallel_steps_each_alone(self):
        plan = ExecutionPlan(
            steps=[
                ExecutionStep(tool="memory", parallel=False),
                ExecutionStep(tool="rag",    parallel=False),
                ExecutionStep(tool="reason", parallel=False),
            ]
        )
        groups = plan.parallel_groups()
        assert len(groups) == 3
        for g in groups:
            assert len(g) == 1

    def test_parallel_groups_parallel_steps_grouped(self):
        plan = ExecutionPlan(
            steps=[
                ExecutionStep(tool="rag",    parallel=True),
                ExecutionStep(tool="search", parallel=True),
                ExecutionStep(tool="reason", parallel=False),
            ]
        )
        groups = plan.parallel_groups()
        assert len(groups) == 2
        assert len(groups[0]) == 2  # rag + search together
        assert len(groups[1]) == 1  # reason alone


# ---------------------------------------------------------------------------
# ExecutionStep
# ---------------------------------------------------------------------------

class TestExecutionStep:

    def test_invalid_cost_defaults_to_medium(self):
        step = ExecutionStep(tool="rag", cost="invalid_cost")
        assert step.cost == "medium"

    def test_valid_costs_accepted(self):
        for cost in ("low", "medium", "high"):
            step = ExecutionStep(tool="rag", cost=cost)
            assert step.cost == cost

    def test_to_dict_has_required_keys(self):
        step = ExecutionStep(tool="rag", description="test", cost="medium")
        d = step.to_dict()
        for key in ("tool", "description", "optional", "cost", "depends_on", "parallel"):
            assert key in d


# ---------------------------------------------------------------------------
# Planner._direct
# ---------------------------------------------------------------------------

class TestPlannerDirect:

    def setup_method(self):
        self.planner = _make_planner()

    def test_direct_plan_has_one_step(self):
        plan = self.planner.create_plan(_decision("direct"), "say hello", "s1")
        assert len(plan.steps) == 1

    def test_direct_plan_step_is_reason(self):
        plan = self.planner.create_plan(_decision("direct"), "say hello", "s1")
        assert plan.steps[0].tool == "reason"

    def test_direct_plan_cost_is_high(self):
        plan = self.planner.create_plan(_decision("direct"), "say hello", "s1")
        assert plan.steps[0].cost == "high"

    def test_direct_code_description(self):
        plan = self.planner.create_plan(
            _decision("direct", {"is_code": True}), "write code", "s1"
        )
        assert "Code" in plan.steps[0].description

    def test_direct_math_description(self):
        plan = self.planner.create_plan(
            _decision("direct", {"is_math": True}), "solve integral", "s1"
        )
        assert "Math" in plan.steps[0].description


# ---------------------------------------------------------------------------
# Planner._search
# ---------------------------------------------------------------------------

class TestPlannerSearch:

    def setup_method(self):
        self.planner = _make_planner()

    def test_search_plan_has_search_step(self):
        plan = self.planner.create_plan(_decision("search"), "latest news", "s1")
        tools = plan.tool_sequence()
        assert "search" in tools

    def test_search_plan_ends_with_reason(self):
        plan = self.planner.create_plan(_decision("search"), "latest news", "s1")
        assert plan.steps[-1].tool == "reason"

    def test_search_simple_has_no_memory_step(self):
        plan = self.planner.create_plan(
            _decision("search", {"is_complex": False, "is_reasoning": False}),
            "news today",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "memory" not in tools

    def test_search_complex_adds_memory(self):
        plan = self.planner.create_plan(
            _decision("search", {"is_complex": True, "is_reasoning": False}),
            "complex news query",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "memory" in tools

    def test_search_reasoning_adds_memory(self):
        plan = self.planner.create_plan(
            _decision("search", {"is_reasoning": True, "is_complex": False}),
            "why does this happen?",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "memory" in tools


# ---------------------------------------------------------------------------
# Planner._memory
# ---------------------------------------------------------------------------

class TestPlannerMemory:

    def setup_method(self):
        self.planner = _make_planner()

    def test_memory_plan_starts_with_memory(self):
        plan = self.planner.create_plan(_decision("memory"), "what did we discuss?", "s1")
        assert plan.steps[0].tool == "memory"

    def test_memory_plan_ends_with_reason(self):
        plan = self.planner.create_plan(_decision("memory"), "what did we discuss?", "s1")
        assert plan.steps[-1].tool == "reason"

    def test_memory_simple_no_rag(self):
        plan = self.planner.create_plan(
            _decision("memory", {"is_complex": False}),
            "what did we say?",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "rag" not in tools

    def test_memory_complex_adds_rag(self):
        plan = self.planner.create_plan(
            _decision("memory", {"is_complex": True}),
            "earlier complex discussion",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "rag" in tools

    def test_memory_rag_is_optional(self):
        plan = self.planner.create_plan(
            _decision("memory", {"is_complex": True}),
            "earlier context",
            "s1",
        )
        for step in plan.steps:
            if step.tool == "rag":
                assert step.optional is True


# ---------------------------------------------------------------------------
# Planner._rag
# ---------------------------------------------------------------------------

class TestPlannerRag:

    def setup_method(self):
        self.planner = _make_planner()

    def test_rag_simple_has_rag_and_reason(self):
        plan = self.planner.create_plan(
            _decision("rag", {}),
            "what is machine learning?",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "rag" in tools
        assert "reason" in tools

    def test_rag_simple_no_decompose(self):
        plan = self.planner.create_plan(
            _decision("rag", {"is_complex": False, "is_reasoning": False, "multi_question": False}),
            "simple question",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "decompose" not in tools

    def test_rag_complex_adds_decompose(self):
        plan = self.planner.create_plan(
            _decision("rag", {"is_complex": True}),
            "complex multi-part question",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "decompose" in tools

    def test_rag_complex_adds_fusion(self):
        plan = self.planner.create_plan(
            _decision("rag", {"is_complex": True}),
            "complex question with fusion",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "fusion" in tools

    def test_rag_multimodal_adds_fusion(self):
        plan = self.planner.create_plan(
            _decision("rag", {"has_multimodal_hint": True}),
            "describe the image",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "fusion" in tools

    def test_rag_reason_not_optional(self):
        plan = self.planner.create_plan(_decision("rag", {}), "question", "s1")
        reason_step = next(s for s in plan.steps if s.tool == "reason")
        assert reason_step.optional is False

    def test_rag_rag_step_not_optional(self):
        plan = self.planner.create_plan(_decision("rag", {}), "question", "s1")
        rag_step = next(s for s in plan.steps if s.tool == "rag")
        assert rag_step.optional is False

    def test_rag_multi_question_adds_decompose(self):
        plan = self.planner.create_plan(
            _decision("rag", {"multi_question": True}),
            "what is X? what is Y?",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "decompose" in tools


# ---------------------------------------------------------------------------
# Planner._hybrid
# ---------------------------------------------------------------------------

class TestPlannerHybrid:

    def setup_method(self):
        self.planner = _make_planner()

    def test_hybrid_always_has_memory_rag_fusion_reason(self):
        plan = self.planner.create_plan(_decision("hybrid", {}), "hybrid query", "s1")
        tools = plan.tool_sequence()
        for required in ("memory", "rag", "fusion", "reason"):
            assert required in tools

    def test_hybrid_recent_adds_search(self):
        plan = self.planner.create_plan(
            _decision("hybrid", {"is_recent": True}),
            "latest and archived info",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "search" in tools

    def test_hybrid_non_recent_no_search(self):
        plan = self.planner.create_plan(
            _decision("hybrid", {"is_recent": False}),
            "general complex question",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "search" not in tools

    def test_hybrid_complex_adds_decompose(self):
        plan = self.planner.create_plan(
            _decision("hybrid", {"is_complex": True}),
            "complex multi-part question",
            "s1",
        )
        tools = plan.tool_sequence()
        assert "decompose" in tools

    def test_hybrid_reason_not_optional(self):
        plan = self.planner.create_plan(_decision("hybrid", {}), "hybrid question", "s1")
        reason_step = next(s for s in plan.steps if s.tool == "reason")
        assert reason_step.optional is False

    def test_hybrid_ends_with_reason(self):
        plan = self.planner.create_plan(_decision("hybrid", {}), "hybrid query", "s1")
        assert plan.steps[-1].tool == "reason"


# ---------------------------------------------------------------------------
# Planner.create_plan edge cases
# ---------------------------------------------------------------------------

class TestPlannerEdgeCases:

    def setup_method(self):
        self.planner = _make_planner()

    def test_none_decision_returns_fallback(self):
        plan = self.planner.create_plan(None, "query", "s1")
        assert isinstance(plan, ExecutionPlan)
        assert plan.steps[0].tool == "reason"

    def test_returns_execution_plan_instance(self):
        plan = self.planner.create_plan(_decision("rag"), "question", "s1")
        assert isinstance(plan, ExecutionPlan)

    def test_unknown_action_returns_fallback(self):
        d = AgentDecision(action="rag", reason="test", confidence=0.5, session_id="s1")
        d.action = "completely_unknown"
        plan = self.planner.create_plan(d, "query", "s1")
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) >= 1

    def test_plan_has_at_most_max_steps(self):
        from app.core.config import settings
        plan = self.planner.create_plan(
            _decision("hybrid", {"is_complex": True, "is_recent": True, "multi_question": True}),
            "very complex multi-part recent query",
            "s1",
        )
        assert len(plan.steps) <= settings.AGENT_MAX_STEPS

    def test_no_duplicate_tools_in_rag_plan(self):
        plan = self.planner.create_plan(
            _decision("rag", {"is_complex": True}),
            "complex question",
            "s1",
        )
        tools = plan.tool_sequence()
        assert len(tools) == len(set(tools))

    def test_no_duplicate_tools_in_hybrid_plan(self):
        plan = self.planner.create_plan(
            _decision("hybrid", {"is_complex": True, "is_recent": True}),
            "complex recent question",
            "s1",
        )
        tools = plan.tool_sequence()
        assert len(tools) == len(set(tools))
