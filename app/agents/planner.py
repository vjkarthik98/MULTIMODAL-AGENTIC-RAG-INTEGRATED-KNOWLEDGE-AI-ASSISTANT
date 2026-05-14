import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.agents.agent_schema import (
    AgentDecision,
    ExecutionPlan,
    ExecutionStep,
    COST_LOW,
    COST_MEDIUM,
    COST_HIGH,
)
from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_plan_duration = Histogram(
    "planner_duration_seconds",
    "Planner duration by action",
    ["action", "status"],
)
_plan_errors = Counter(
    "planner_errors_total",
    "Planner errors by type",
    ["error_type"],
)
_plan_steps_count = Histogram(
    "planner_steps_count",
    "Number of steps per plan",
    ["action"],
)

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)

# STEP COST MAP
_STEP_COSTS: Dict[str, str] = {
    "reason":    COST_HIGH,
    "rag":       COST_MEDIUM,
    "search":    COST_HIGH,
    "memory":    COST_LOW,
    "decompose": COST_MEDIUM,
    "fusion":    COST_MEDIUM,
}

# DEPENDENCY MAP — WHICH STEPS REQUIRE PRIOR STEPS
_STEP_DEPENDENCIES: Dict[str, List[str]] = {
    "reason":   ["rag", "search", "memory", "fusion"],
    "fusion":   ["rag", "decompose"],
    "decompose": [],
    "rag":      [],
    "search":   [],
    "memory":   [],
}


class Planner:

    # MAIN PLAN CREATION

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

        start = time.time()

        with tracer.start_as_current_span("planner_create_plan") as span:
            span.set_attribute("action", action)
            span.set_attribute("session.id", session_id)

            try:
                if action == "direct":
                    plan = self._direct(signals)

                elif action == "search":
                    plan = self._search(signals)

                elif action == "memory":
                    plan = self._memory(signals)

                elif action == "rag":
                    plan = self._rag(signals)

                elif action == "hybrid":
                    plan = self._hybrid(signals)

                else:
                    plan = self._fallback("unknown_action", session_id)

                latency = round(time.time() - start, 3)

                _plan_duration.labels(action=action, status="success").observe(latency)
                _plan_steps_count.labels(action=action).observe(len(plan.steps))

                span.set_attribute("plan.steps", len(plan.steps))
                span.set_attribute("plan.total_cost", plan.total_cost)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "plan_created",
                    action=action,
                    steps=plan.tool_sequence(),
                    total_cost=plan.total_cost,
                    latency=latency,
                    session_id=session_id,
                )

                return plan

            except Exception as exc:
                latency    = round(time.time() - start, 3)
                error_type = type(exc).__name__

                _plan_duration.labels(action=action, status="error").observe(latency)
                _plan_errors.labels(error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "planner_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )

                return self._fallback("planner_exception", session_id)

    # DIRECT PLAN — SIMPLE LLM ANSWER WITHOUT RETRIEVAL

    def _direct(self, signals: Dict) -> ExecutionPlan:
        is_code       = signals.get("is_code", False)
        is_math       = signals.get("is_math", False)
        is_multilingual = signals.get("is_multilingual", False)

        description = (
            "Code reasoning"       if is_code  else
            "Math reasoning"       if is_math  else
            "Multilingual reasoning" if is_multilingual else
            "Direct reasoning"
        )

        return ExecutionPlan(
            steps=[
                ExecutionStep(
                    tool="reason",
                    description=description,
                    optional=False,
                    cost=COST_HIGH,
                )
            ],
            trace={
                "type":          "direct",
                "is_code":       is_code,
                "is_math":       is_math,
                "is_multilingual": is_multilingual,
            },
        )

    # SEARCH PLAN — WEB SEARCH THEN REASON

    def _search(self, signals: Dict) -> ExecutionPlan:
        is_complex  = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)

        steps: List[ExecutionStep] = [
            ExecutionStep(
                tool="search",
                description="External web retrieval",
                optional=False,
                cost=COST_HIGH,
            )
        ]

        # FUSE SEARCH WITH MEMORY FOR BETTER CONTEXT IF COMPLEX
        if is_complex or is_reasoning:
            steps.insert(
                0,
                ExecutionStep(
                    tool="memory",
                    description="Fetch session memory for context",
                    optional=True,
                    cost=COST_LOW,
                ),
            )

        steps.append(
            ExecutionStep(
                tool="reason",
                description="Answer generation from search results",
                optional=False,
                cost=COST_HIGH,
            )
        )

        return ExecutionPlan(
            steps=steps,
            trace={
                "type":       "search",
                "is_complex": is_complex,
                "is_reasoning": is_reasoning,
            },
        )

    # MEMORY PLAN — RECALL PREVIOUS CONVERSATION THEN REASON

    def _memory(self, signals: Dict) -> ExecutionPlan:
        is_complex  = signals.get("is_complex", False)

        steps: List[ExecutionStep] = [
            ExecutionStep(
                tool="memory",
                description="Fetch relevant session memory",
                optional=False,
                cost=COST_LOW,
            )
        ]

        # AUGMENT WITH RAG FOR COMPLEX MEMORY QUERIES
        if is_complex:
            steps.append(
                ExecutionStep(
                    tool="rag",
                    description="Augment memory with knowledge base",
                    optional=True,
                    cost=COST_MEDIUM,
                )
            )

        steps.append(
            ExecutionStep(
                tool="reason",
                description="Answer using session memory",
                optional=False,
                cost=COST_HIGH,
            )
        )

        return ExecutionPlan(
            steps=steps,
            trace={
                "type":       "memory",
                "is_complex": is_complex,
            },
        )

    # RAG PLAN — RETRIEVE THEN REASON

    def _rag(self, signals: Dict) -> ExecutionPlan:
        steps: List[ExecutionStep] = []

        is_complex      = signals.get("is_complex",          False)
        is_reasoning    = signals.get("is_reasoning",        False)
        is_multimodal   = signals.get("has_multimodal_hint", False)
        is_multilingual = signals.get("is_multilingual",     False)
        multi_question  = signals.get("multi_question",      False)

        # DECOMPOSE FOR COMPLEX OR MULTI-HOP QUERIES
        if is_complex or is_reasoning or multi_question:
            steps.append(
                ExecutionStep(
                    tool="decompose",
                    description="Break query into sub-queries",
                    optional=True,
                    cost=COST_MEDIUM,
                )
            )

        steps.append(
            ExecutionStep(
                tool="rag",
                description="Retrieve knowledge from vector store",
                optional=False,
                cost=COST_MEDIUM,
            )
        )

        # FUSION FOR COMPLEX OR MULTI-MODAL QUERIES
        if is_complex or is_multimodal:
            steps.append(
                ExecutionStep(
                    tool="fusion",
                    description="Merge and rank retrieved results",
                    optional=True,
                    cost=COST_MEDIUM,
                )
            )

        steps.append(
            ExecutionStep(
                tool="reason",
                description="Final answer generation",
                optional=False,
                cost=COST_HIGH,
            )
        )

        return ExecutionPlan(
            steps=self._optimize(steps),
            trace={
                "type":         "rag",
                "complex":      is_complex,
                "reasoning":    is_reasoning,
                "multimodal":   is_multimodal,
                "multilingual": is_multilingual,
                "multi_question": multi_question,
            },
        )

    # HYBRID PLAN — MEMORY + RAG + SEARCH COMBINED

    def _hybrid(self, signals: Dict) -> ExecutionPlan:
        steps: List[ExecutionStep] = []

        is_complex      = signals.get("is_complex",          False)
        is_reasoning    = signals.get("is_reasoning",        False)
        is_multimodal   = signals.get("has_multimodal_hint", False)
        is_multilingual = signals.get("is_multilingual",     False)
        is_recent       = signals.get("is_recent",           False)
        multi_question  = signals.get("multi_question",      False)

        # ALWAYS FETCH MEMORY FIRST IN HYBRID
        steps.append(
            ExecutionStep(
                tool="memory",
                description="Fetch session memory",
                optional=True,
                cost=COST_LOW,
            )
        )

        # DECOMPOSE COMPLEX OR MULTI-HOP QUERIES
        if is_complex or is_reasoning or multi_question:
            steps.append(
                ExecutionStep(
                    tool="decompose",
                    description="Split query into sub-queries",
                    optional=True,
                    cost=COST_MEDIUM,
                )
            )

        # RECENT QUERIES GET SEARCH IN HYBRID
        if is_recent:
            steps.append(
                ExecutionStep(
                    tool="search",
                    description="Retrieve recent information from web",
                    optional=True,
                    cost=COST_HIGH,
                )
            )

        steps.append(
            ExecutionStep(
                tool="rag",
                description="Retrieve from knowledge base",
                optional=False,
                cost=COST_MEDIUM,
            )
        )

        steps.append(
            ExecutionStep(
                tool="fusion",
                description="Combine all retrieved results",
                optional=True,
                cost=COST_MEDIUM,
            )
        )

        steps.append(
            ExecutionStep(
                tool="reason",
                description="Final answer with full context",
                optional=False,
                cost=COST_HIGH,
            )
        )

        return ExecutionPlan(
            steps=self._optimize(steps),
            trace={
                "type":         "hybrid",
                "complex":      is_complex,
                "reasoning":    is_reasoning,
                "multimodal":   is_multimodal,
                "multilingual": is_multilingual,
                "is_recent":    is_recent,
                "multi_question": multi_question,
            },
        )

    # OPTIMIZE — DEDUP STEPS AND APPLY DEPENDENCY ORDERING

    def _optimize(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        seen:    set                   = set()
        ordered: List[ExecutionStep]   = []

        for s in steps:
            if s.tool not in seen:
                seen.add(s.tool)
                ordered.append(s)

        # ENFORCE DEPENDENCY ORDERING
        ordered = self._order_by_dependencies(ordered)

        return self._limit(ordered)

    # DEPENDENCY ORDERING — ENSURE REQUIRED STEPS COME BEFORE DEPENDENTS

    def _order_by_dependencies(
        self,
        steps: List[ExecutionStep],
    ) -> List[ExecutionStep]:
        """
        TOPOLOGICAL SORT — STEPS THAT DEPEND ON OTHERS
        ARE PLACED AFTER THEIR DEPENDENCIES.
        """
        tool_names = {s.tool for s in steps}
        ordered:   List[ExecutionStep] = []
        placed:    set                 = set()

        def _place(step: ExecutionStep) -> None:
            if step.tool in placed:
                return
            deps = _STEP_DEPENDENCIES.get(step.tool, [])
            for dep in deps:
                dep_step = next((s for s in steps if s.tool == dep), None)
                if dep_step and dep_step.tool not in placed:
                    _place(dep_step)
            ordered.append(step)
            placed.add(step.tool)

        for step in steps:
            _place(step)

        return ordered

    # LIMIT — CAP STEPS AT MAX_STEPS

    def _limit(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        max_steps = settings.AGENT_MAX_STEPS

        if len(steps) > max_steps:
            logger.warning(
                "planner_steps_truncated",
                original=len(steps),
                max_steps=max_steps,
            )
            return steps[:max_steps]

        return steps

    # FALLBACK — MINIMAL DIRECT REASONING PLAN

    def _fallback(
        self,
        reason: str,
        session_id: str = "default",
    ) -> ExecutionPlan:
        logger.warning(
            "planner_fallback",
            reason=reason,
            session_id=session_id,
        )
        return ExecutionPlan(
            steps=[
                ExecutionStep(
                    tool="reason",
                    description="Fallback direct reasoning",
                    optional=False,
                    cost=COST_HIGH,
                )
            ],
            trace={"fallback": reason},
        )

    # ASYNC WRAPPER

    async def create_plan_async(
        self,
        decision: AgentDecision,
        query: str,
        session_id: str = "default",
    ) -> ExecutionPlan:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.create_plan(decision, query, session_id),
            )


