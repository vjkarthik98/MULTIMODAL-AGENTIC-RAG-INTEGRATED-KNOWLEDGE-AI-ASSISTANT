from typing import Any, Dict, List, Optional

from app.agents.agent_schema import AgentDecision
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# STEP COSTS

_STEP_COSTS: Dict[str, str] = {
    "reason":    "high",
    "rag":       "medium",
    "search":    "high",
    "memory":    "low",
    "decompose": "medium",
    "fusion":    "medium",
}


class ExecutionStep:

    def __init__(
        self,
        tool: str,
        description: str = "",
        optional: bool = False,
        cost: str = "medium",
    ) -> None:
        self.tool        = tool
        self.description = description
        self.optional    = optional
        self.cost        = cost or _STEP_COSTS.get(tool, "medium")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool":        self.tool,
            "description": self.description,
            "optional":    self.optional,
            "cost":        self.cost,
        }


class ExecutionPlan:

    def __init__(
        self,
        steps: List[ExecutionStep],
        trace: Optional[Dict] = None,
    ) -> None:
        self.steps = steps or []
        self.trace = trace or {}

    @property
    def total_cost(self) -> str:
        cost_rank = {"low": 0, "medium": 1, "high": 2}
        if not self.steps:
            return "low"
        max_cost = max(cost_rank.get(s.cost, 1) for s in self.steps)
        return ["low", "medium", "high"][max_cost]

    def tool_sequence(self) -> List[str]:
        return [s.tool for s in self.steps]

    def to_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]


class Planner:

    # MAIN

    def create_plan(
        self,
        decision: AgentDecision,
        query: str,
        session_id: str = "default",
    ) -> ExecutionPlan:

        if not decision or not getattr(decision, "action", None):
            return self._fallback("invalid_decision", session_id)

        action  = decision.action.strip().lower()
        signals = decision.signals or {}

        try:
            if action == "direct":
                plan = self._direct(signals)

            elif action == "search":
                plan = self._search()

            elif action == "memory":
                plan = self._memory()

            elif action == "rag":
                plan = self._rag(signals)

            elif action == "hybrid":
                plan = self._hybrid(signals)

            else:
                plan = self._fallback("unknown_action", session_id)

            logger.info(
                event="plan_created",
                action=action,
                steps=plan.tool_sequence(),
                total_cost=plan.total_cost,
                session_id=session_id,
            )

            return plan

        except Exception as e:
            logger.error(
                event="planner_failed",
                error=str(e),
                session_id=session_id,
            )
            return self._fallback("planner_exception", session_id)

    # DIRECT PLAN

    def _direct(self, signals: Dict) -> ExecutionPlan:
        is_code = signals.get("is_code", False)

        return ExecutionPlan(
            steps=[
                ExecutionStep(
                    "reason",
                    "Code reasoning" if is_code else "Direct reasoning",
                    cost="high",
                )
            ],
            trace={"type": "direct", "is_code": is_code},
        )

    # SEARCH PLAN

    def _search(self) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[
                ExecutionStep("search", "External web retrieval", cost="high"),
                ExecutionStep("reason", "Answer generation",      cost="high"),
            ],
            trace={"type": "search"},
        )

    # MEMORY PLAN

    def _memory(self) -> ExecutionPlan:
        return ExecutionPlan(
            steps=[
                ExecutionStep("memory", "Fetch session memory", cost="low"),
                ExecutionStep("reason", "Answer with memory",   cost="high"),
            ],
            trace={"type": "memory"},
        )

    # RAG PLAN

    def _rag(self, signals: Dict) -> ExecutionPlan:
        steps: List[ExecutionStep] = []

        is_complex      = signals.get("is_complex",         False)
        is_reasoning    = signals.get("is_reasoning",       False)
        is_multimodal   = signals.get("has_multimodal_hint", False)

        if is_complex or is_reasoning:
            steps.append(ExecutionStep("decompose", "Break query into subqueries", optional=True, cost="medium"))

        steps.append(ExecutionStep("rag", "Retrieve knowledge from vector store", cost="medium"))

        if is_complex:
            steps.append(ExecutionStep("fusion", "Merge and rank results", optional=True, cost="medium"))

        steps.append(ExecutionStep("reason", "Final answer generation", cost="high"))

        return ExecutionPlan(
            steps=self._optimize(steps),
            trace={
                "type":       "rag",
                "complex":    is_complex,
                "reasoning":  is_reasoning,
                "multimodal": is_multimodal,
            },
        )

    # HYBRID PLAN

    def _hybrid(self, signals: Dict) -> ExecutionPlan:
        steps: List[ExecutionStep] = []

        is_complex    = signals.get("is_complex",         False)
        is_reasoning  = signals.get("is_reasoning",       False)
        is_multimodal = signals.get("has_multimodal_hint", False)

        steps.append(ExecutionStep("memory", "Fetch session memory", cost="low"))

        if is_complex or is_reasoning:
            steps.append(ExecutionStep("decompose", "Split query", optional=True, cost="medium"))

        steps.append(ExecutionStep("rag",    "Retrieve from knowledge base",      cost="medium"))
        steps.append(ExecutionStep("fusion", "Combine all retrieved results",     optional=True, cost="medium"))
        steps.append(ExecutionStep("reason", "Final answer with full context",    cost="high"))

        return ExecutionPlan(
            steps=self._optimize(steps),
            trace={
                "type":       "hybrid",
                "complex":    is_complex,
                "reasoning":  is_reasoning,
                "multimodal": is_multimodal,
            },
        )

    # OPTIMIZE

    def _optimize(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        seen:    set                   = set()
        ordered: List[ExecutionStep]   = []

        for s in steps:
            if s.tool not in seen:
                seen.add(s.tool)
                ordered.append(s)

        return self._limit(ordered)

    # LIMIT

    def _limit(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        max_steps = settings.AGENT_MAX_STEPS

        if len(steps) > max_steps:
            logger.warning(
                event="planner_steps_truncated",
                original=len(steps),
                max_steps=max_steps,
            )
            return steps[:max_steps]

        return steps

    # FALLBACK

    def _fallback(self, reason: str, session_id: str = "default") -> ExecutionPlan:
        logger.warning(
            event="planner_fallback",
            reason=reason,
            session_id=session_id,
        )
        return ExecutionPlan(
            steps=[ExecutionStep("reason", "Fallback direct reasoning", cost="high")],
            trace={"fallback": reason},
        )