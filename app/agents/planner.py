# NOT CURRENTLY WIRED IN — this module (Planner, ExecutionPlan multi-step
# building, and the _limit()/AGENT_MAX_STEPS cap below) has no callers
# anywhere in the live request path. AgentExecutor.run() in
# app/agents/agent_controller.py performs a single classify+dispatch via
# AgentRouter, not a multi-step ExecutionPlan loop, so AGENT_MAX_STEPS is
# currently NOT enforced by anything at runtime — only AGENT_TIMEOUT_SEC and
# the (now pre-capped, see agent_controller.py) AGENT_TOKEN_BUDGET are real.
# Flagged during the 2026-07 security audit: CLAUDE.md documents an
# agent_router -> planner -> tool_registry multi-step architecture that does
# not match the live code. Wiring this in for genuine multi-step tool
# chaining is a separate feature-scoped change; until then, do not treat
# max_steps as an enforced bound anywhere else in the codebase or docs.
import asyncio
import time

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.agents.agent_schema import (
    COST_HIGH,
    COST_LOW,
    COST_MEDIUM,
    AgentDecision,
    ExecutionPlan,
    ExecutionStep,
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
_STEP_COSTS: dict[str, str] = {
    "reason": COST_HIGH,
    "rag": COST_MEDIUM,
    "search": COST_HIGH,
    "memory": COST_LOW,
    "decompose": COST_MEDIUM,
    "fusion": COST_MEDIUM,
}

# DEPENDENCY MAP — WHICH STEPS REQUIRE PRIOR STEPS
_STEP_DEPENDENCIES: dict[str, list[str]] = {
    "reason": ["rag", "search", "memory", "fusion"],
    "fusion": ["rag", "decompose"],
    "decompose": [],
    "rag": [],
    "search": [],
    "memory": [],
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

        action = decision.action.strip().lower()
        signals = decision.signals or {}

        start = time.time()

        with tracer.start_as_current_span("planner_create_plan") as span:
            span.set_attribute("action", action)
            span.set_attribute("session.id", session_id)

            try:
                if action == "direct":
                    plan = self._direct(signals)

                elif action == "search":
                    # Finance market data → specialist plan
                    if signals.get("is_market_data_query"):
                        plan = self._finance_market_data(signals)
                    else:
                        plan = self._search(signals)

                elif action == "memory":
                    plan = self._memory(signals)

                elif action == "rag":
                    # Finance domain archetypes (Phase 7)
                    if signals.get("is_earnings_call_query"):
                        plan = self._finance_earnings_analysis(signals)
                    elif signals.get("is_regulatory_query"):
                        plan = self._finance_regulatory(signals)
                    elif signals.get("is_financial_model_query"):
                        plan = self._finance_model(signals)
                    else:
                        plan = self._rag(signals)

                elif action == "hybrid":
                    if signals.get("is_earnings_call_query"):
                        plan = self._finance_earnings_analysis(signals)
                    else:
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
                latency = round(time.time() - start, 3)
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

    def _direct(self, signals: dict) -> ExecutionPlan:
        is_code = signals.get("is_code", False)
        is_math = signals.get("is_math", False)

        description = (
            "Code reasoning" if is_code else "Math reasoning" if is_math else "Direct reasoning"
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
                "type": "direct",
                "is_code": is_code,
                "is_math": is_math,
            },
        )

    # SEARCH PLAN — WEB SEARCH THEN REASON

    def _search(self, signals: dict) -> ExecutionPlan:
        is_complex = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)

        steps: list[ExecutionStep] = [
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
                "type": "search",
                "is_complex": is_complex,
                "is_reasoning": is_reasoning,
            },
        )

    # MEMORY PLAN — RECALL PREVIOUS CONVERSATION THEN REASON

    def _memory(self, signals: dict) -> ExecutionPlan:
        is_complex = signals.get("is_complex", False)

        steps: list[ExecutionStep] = [
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
                "type": "memory",
                "is_complex": is_complex,
            },
        )

    # RAG PLAN — RETRIEVE THEN REASON

    def _rag(self, signals: dict) -> ExecutionPlan:
        steps: list[ExecutionStep] = []

        is_complex = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)
        is_multimodal = signals.get("has_multimodal_hint", False)
        multi_question = signals.get("multi_question", False)

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
                "type": "rag",
                "complex": is_complex,
                "reasoning": is_reasoning,
                "multimodal": is_multimodal,
                "multi_question": multi_question,
            },
        )

    # HYBRID PLAN — MEMORY + RAG + SEARCH COMBINED

    def _hybrid(self, signals: dict) -> ExecutionPlan:
        steps: list[ExecutionStep] = []

        is_complex = signals.get("is_complex", False)
        is_reasoning = signals.get("is_reasoning", False)
        is_multimodal = signals.get("has_multimodal_hint", False)
        is_recent = signals.get("is_recent", False)
        multi_question = signals.get("multi_question", False)

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
                "type": "hybrid",
                "complex": is_complex,
                "reasoning": is_reasoning,
                "multimodal": is_multimodal,
                "is_recent": is_recent,
                "multi_question": multi_question,
            },
        )

    # OPTIMIZE — DEDUP STEPS AND APPLY DEPENDENCY ORDERING

    def _optimize(self, steps: list[ExecutionStep]) -> list[ExecutionStep]:
        seen: set = set()
        ordered: list[ExecutionStep] = []

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
        steps: list[ExecutionStep],
    ) -> list[ExecutionStep]:
        """
        TOPOLOGICAL SORT — STEPS THAT DEPEND ON OTHERS
        ARE PLACED AFTER THEIR DEPENDENCIES.
        """
        {s.tool for s in steps}
        ordered: list[ExecutionStep] = []
        placed: set = set()

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

    def _limit(self, steps: list[ExecutionStep]) -> list[ExecutionStep]:
        max_steps = settings.AGENT_MAX_STEPS

        if len(steps) > max_steps:
            logger.warning(
                "planner_steps_truncated",
                original=len(steps),
                max_steps=max_steps,
            )
            return steps[:max_steps]

        return steps

    # FINANCE PLAN ARCHETYPES (Phase 7)

    def _finance_earnings_analysis(self, signals: dict) -> ExecutionPlan:
        """Archetype: earnings_analysis — audio transcript + PDF + reason."""
        return ExecutionPlan(
            steps=self._optimize(
                [
                    ExecutionStep(
                        tool="rag",
                        description="Retrieve audio/transcript chunks (earnings call)",
                        cost=COST_MEDIUM,
                    ),
                    ExecutionStep(
                        tool="rag",
                        description="Retrieve PDF supplemental (earnings release/10-Q)",
                        cost=COST_MEDIUM,
                    ),
                    ExecutionStep(
                        tool="fusion",
                        description="Merge audio and document results",
                        optional=True,
                        cost=COST_MEDIUM,
                    ),
                    ExecutionStep(
                        tool="reason",
                        description="Speaker-attributed earnings analysis",
                        cost=COST_HIGH,
                    ),
                ]
            ),
            trace={"type": "finance_earnings_analysis"},
        )

    def _finance_regulatory(self, signals: dict) -> ExecutionPlan:
        """Archetype: regulatory_filing — PDF/DOCX KB + reason."""
        return ExecutionPlan(
            steps=self._optimize(
                [
                    ExecutionStep(
                        tool="rag",
                        description="Retrieve regulatory filing from KB (PDF/DOCX)",
                        cost=COST_MEDIUM,
                    ),
                    ExecutionStep(
                        tool="reason", description="Regulatory filing analysis", cost=COST_HIGH
                    ),
                ]
            ),
            trace={"type": "finance_regulatory"},
        )

    def _finance_model(self, signals: dict) -> ExecutionPlan:
        """Archetype: financial_model — table/assumptions chunks + reason."""
        return ExecutionPlan(
            steps=self._optimize(
                [
                    ExecutionStep(
                        tool="rag",
                        description="Retrieve table and assumption chunks",
                        cost=COST_MEDIUM,
                    ),
                    ExecutionStep(
                        tool="reason", description="Financial model analysis", cost=COST_HIGH
                    ),
                ]
            ),
            trace={"type": "finance_financial_model"},
        )

    def _finance_market_data(self, signals: dict) -> ExecutionPlan:
        """Archetype: market_data — live web search + reason."""
        return ExecutionPlan(
            steps=self._optimize(
                [
                    ExecutionStep(
                        tool="search",
                        description="Retrieve live market data (finance topic)",
                        cost=COST_HIGH,
                    ),
                    ExecutionStep(
                        tool="reason", description="Market data synthesis", cost=COST_HIGH
                    ),
                ]
            ),
            trace={"type": "finance_market_data"},
        )

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
