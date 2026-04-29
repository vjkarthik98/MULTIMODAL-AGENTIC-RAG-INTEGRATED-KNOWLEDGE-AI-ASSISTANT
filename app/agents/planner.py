from typing import List, Dict, Any

from app.core.config import settings
from app.utils.logger import get_logger
from app.agents.agent_schema import AgentDecision


logger = get_logger(__name__)


class ExecutionStep:
    def __init__(
        self,
        tool: str,
        description: str = "",
        optional: bool = False
    ):
        self.tool = tool
        self.description = description
        self.optional = optional

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "description": self.description,
            "optional": self.optional
        }


class ExecutionPlan:
    def __init__(self, steps: List[ExecutionStep], trace: Dict = None):
        self.steps = steps or []
        self.trace = trace or {}

    def to_list(self) -> List[Dict[str, Any]]:
        return [step.to_dict() for step in self.steps]


class Planner:

    def create_plan(
        self,
        decision: AgentDecision,
        query: str
    ) -> ExecutionPlan:

        if not decision or not getattr(decision, "action", None):
            logger.warning("[Planner] Invalid decision → fallback")
            return self._fallback_plan("invalid_decision")

        action = str(decision.action).strip().lower()
        signals = decision.signals or {}

        logger.info("[Planner] action=%s", action)

        try:
            if action == "direct":
                return self._direct_plan(query)

            if action == "search":
                return self._search_plan(query)

            if action == "memory":
                return self._memory_plan(query)

            if action == "rag":
                return self._rag_plan(query, signals)

            if action == "hybrid":
                return self._hybrid_plan(query, signals)

            return self._fallback_plan("unknown_action")

        except Exception as e:
            logger.error("[Planner][ERROR] %s", str(e))
            return self._fallback_plan("planner_exception")

    #  PLANS 

    def _direct_plan(self, query: str) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[
                ExecutionStep("reason", "Direct reasoning")
            ],
            trace={"type": "direct"}
        )

    def _search_plan(self, query: str) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[
                ExecutionStep("search", "External retrieval"),
                ExecutionStep("reason", "Answer generation")
            ],
            trace={"type": "search"}
        )

    def _memory_plan(self, query: str) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[
                ExecutionStep("memory", "Fetch memory"),
                ExecutionStep("reason", "Answer with memory")
            ],
            trace={"type": "memory"}
        )

    def _rag_plan(self, query: str, signals: Dict) -> ExecutionPlan:

        steps: List[ExecutionStep] = []

        is_complex = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)

        if is_complex or is_reasoning:
            steps.append(
                ExecutionStep("decompose", "Break query", optional=True)
            )

        steps.append(
            ExecutionStep("rag", "Retrieve knowledge")
        )

        if is_complex:
            steps.append(
                ExecutionStep("fusion", "Merge results", optional=True)
            )

        steps.append(
            ExecutionStep("reason", "Final reasoning")
        )

        return ExecutionPlan(
            steps=self._optimize_steps(steps),
            trace={"type": "rag", "complex": is_complex}
        )

    def _hybrid_plan(self, query: str, signals: Dict) -> ExecutionPlan:

        steps: List[ExecutionStep] = []

        is_complex = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)

        steps.append(
            ExecutionStep("memory", "Fetch memory")
        )

        if is_complex or is_reasoning:
            steps.append(
                ExecutionStep("decompose", "Split query", optional=True)
            )

        steps.append(
            ExecutionStep("rag", "Retrieve knowledge")
        )

        steps.append(
            ExecutionStep("fusion", "Combine results", optional=True)
        )

        steps.append(
            ExecutionStep("reason", "Final reasoning")
        )

        return ExecutionPlan(
            steps=self._optimize_steps(steps),
            trace={"type": "hybrid", "complex": is_complex}
        )

    #  OPTIMIZATION 

    def _optimize_steps(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:

        seen = set()
        optimized = []

        for step in steps:
            if step.tool not in seen:
                seen.add(step.tool)
                optimized.append(step)

        return self._limit_steps(optimized)

    #  LIMIT 

    def _limit_steps(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:

        max_steps = getattr(settings, "AGENT_MAX_STEPS", 10)

        if len(steps) > max_steps:
            logger.warning("[Planner] truncated to max_steps=%s", max_steps)
            return steps[:max_steps]

        return steps

    #  FALLBACK 

    def _fallback_plan(self, reason: str) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[ExecutionStep("reason", "Fallback reasoning")],
            trace={"fallback": reason}
        )