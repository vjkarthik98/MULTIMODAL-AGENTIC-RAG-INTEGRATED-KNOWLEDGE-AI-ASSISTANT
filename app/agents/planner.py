from typing import List, Dict, Any

from app.core.config import settings
from app.utils.logger import get_logger
from app.agents.agent_schema import AgentDecision

logger = get_logger(__name__)


class ExecutionStep:
    def __init__(self, tool: str, description: str = "", optional: bool = False):
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
        return [s.to_dict() for s in self.steps]


class Planner:

    #  MAIN 
    def create_plan(self, decision: AgentDecision, query: str) -> ExecutionPlan:

        if not decision or not getattr(decision, "action", None):
            return self._fallback("invalid_decision")

        action = decision.action.strip().lower()
        signals = decision.signals or {}

        try:
            if action == "direct":
                return self._direct()

            if action == "search":
                return self._search()

            if action == "memory":
                return self._memory()

            if action == "rag":
                return self._rag(signals)

            if action == "hybrid":
                return self._hybrid(signals)

            return self._fallback("unknown_action")

        except Exception as e:
            logger.error(event="planner_failed", error=str(e))
            return self._fallback("planner_exception")

    #  PLANS 
    def _direct(self):
        return ExecutionPlan(
            steps=[ExecutionStep("reason", "Direct reasoning")],
            trace={"type": "direct"}
        )

    def _search(self):
        return ExecutionPlan(
            steps=[
                ExecutionStep("search", "External retrieval"),
                ExecutionStep("reason", "Answer generation")
            ],
            trace={"type": "search"}
        )

    def _memory(self):
        return ExecutionPlan(
            steps=[
                ExecutionStep("memory", "Fetch memory"),
                ExecutionStep("reason", "Answer with memory")
            ],
            trace={"type": "memory"}
        )

    def _rag(self, signals: Dict):

        steps = []

        is_complex = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)

        if is_complex or is_reasoning:
            steps.append(ExecutionStep("decompose", "Break query", optional=True))

        steps.append(ExecutionStep("rag", "Retrieve knowledge"))

        if is_complex:
            steps.append(ExecutionStep("fusion", "Merge results", optional=True))

        steps.append(ExecutionStep("reason", "Final reasoning"))

        return ExecutionPlan(
            steps=self._optimize(steps),
            trace={
                "type": "rag",
                "complex": is_complex,
                "reasoning": is_reasoning
            }
        )

    def _hybrid(self, signals: Dict):

        steps = []

        is_complex = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)

        steps.append(ExecutionStep("memory", "Fetch memory"))

        if is_complex or is_reasoning:
            steps.append(ExecutionStep("decompose", "Split query", optional=True))

        steps.append(ExecutionStep("rag", "Retrieve knowledge"))
        steps.append(ExecutionStep("fusion", "Combine results", optional=True))
        steps.append(ExecutionStep("reason", "Final reasoning"))

        return ExecutionPlan(
            steps=self._optimize(steps),
            trace={
                "type": "hybrid",
                "complex": is_complex,
                "reasoning": is_reasoning
            }
        )

    #  OPTIMIZE 
    def _optimize(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:

        seen = set()
        ordered = []

        for s in steps:
            if s.tool not in seen:
                seen.add(s.tool)
                ordered.append(s)

        return self._limit(ordered)

    #  LIMIT 
    def _limit(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:

        max_steps = getattr(settings, "AGENT_MAX_STEPS", 10)

        if len(steps) > max_steps:
            logger.warning(event="planner_truncated", max_steps=max_steps)
            return steps[:max_steps]

        return steps

    #  FALLBACK 
    def _fallback(self, reason: str):

        return ExecutionPlan(
            steps=[ExecutionStep("reason", "Fallback reasoning")],
            trace={"fallback": reason}
        )