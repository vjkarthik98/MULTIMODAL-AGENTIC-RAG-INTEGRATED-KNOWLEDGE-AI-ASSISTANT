from typing import List, Dict, Any

from app.core.config import settings
from app.utils.logger import get_logger
from app.agents.agent_schema import AgentDecision


logger = get_logger(__name__)


class ExecutionStep:
    def __init__(self, tool: str, description: str = ""):
        self.tool = tool
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "description": self.description
        }


class ExecutionPlan:
    def __init__(self, steps: List[ExecutionStep]):
        self.steps = steps or []

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
            return self._fallback_plan()

        action = str(decision.action).strip().lower()
        signals = decision.signals or {}

        logger.info("[Planner] Creating plan | action=%s", action)

        try:
            if action == "direct":
                return self._direct_plan()

            if action == "search":
                return self._search_plan()

            if action == "memory":
                return self._memory_plan()

            if action == "rag":
                return self._rag_plan(signals)

            if action == "hybrid":
                return self._hybrid_plan(signals)

            logger.warning("[Planner] Unknown action=%s → fallback", action)
            return self._fallback_plan()

        except Exception as e:
            logger.error("[Planner][ERROR] %s", str(e))
            return self._fallback_plan()

    # PLAN TYPES

    def _direct_plan(self) -> ExecutionPlan:
        return ExecutionPlan([
            ExecutionStep("reason", "Direct reasoning without retrieval")
        ])

    def _search_plan(self) -> ExecutionPlan:
        return ExecutionPlan([
            ExecutionStep("search", "Fetch external data"),
            ExecutionStep("reason", "Summarize and answer")
        ])

    def _memory_plan(self) -> ExecutionPlan:
        return ExecutionPlan([
            ExecutionStep("memory", "Fetch past context"),
            ExecutionStep("reason", "Answer using memory")
        ])

    def _rag_plan(self, signals: Dict[str, Any]) -> ExecutionPlan:
        steps: List[ExecutionStep] = []

        is_complex = bool(signals.get("is_complex"))

        if is_complex:
            steps.append(
                ExecutionStep("decompose", "Break query into sub-queries")
            )

        steps.append(
            ExecutionStep("rag", "Retrieve knowledge")
        )

        if is_complex:
            steps.append(
                ExecutionStep("fusion", "Merge retrieval results")
            )

        steps.append(
            ExecutionStep("reason", "Generate final answer")
        )

        return ExecutionPlan(self._limit_steps(steps))

    def _hybrid_plan(self, signals: Dict[str, Any]) -> ExecutionPlan:
        steps: List[ExecutionStep] = []

        is_complex = bool(signals.get("is_complex"))

        steps.append(
            ExecutionStep("memory", "Fetch conversation context")
        )

        if is_complex:
            steps.append(
                ExecutionStep("decompose", "Split query")
            )

        steps.append(
            ExecutionStep("rag", "Retrieve knowledge")
        )

        steps.append(
            ExecutionStep("fusion", "Combine results")
        )

        steps.append(
            ExecutionStep("reason", "Final reasoning with memory + knowledge")
        )

        return ExecutionPlan(self._limit_steps(steps))

    # SAFETY

    def _limit_steps(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        max_steps = getattr(settings, "AGENT_MAX_STEPS", 10)

        if len(steps) > max_steps:
            logger.warning("[Planner] Plan truncated to max_steps=%s", max_steps)
            return steps[:max_steps]

        return steps

    def _fallback_plan(self) -> ExecutionPlan:
        return ExecutionPlan([
            ExecutionStep("reason", "Fallback reasoning")
        ])