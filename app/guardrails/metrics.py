"""Prometheus metrics for the guardrails layer.

Registered once at import time — consistent with the pattern used in
agent_router.py and query_pipeline.py. Labels mirror the GuardrailBlocked
fields so dashboards can slice by surface and guard_type.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Total block/allow/scrub decisions
guardrail_decisions_total = Counter(
    "guardrail_decisions_total",
    "Total guardrail decisions by action, guard_type, and surface",
    ["action", "guard_type", "surface"],
)

# Blocks only (convenience alias for alert rules)
guardrail_blocks_total = Counter(
    "guardrail_blocks_total",
    "Total guardrail blocks by guard_type and surface",
    ["guard_type", "surface"],
)

# Latency per guard stage
guardrail_latency_seconds = Histogram(
    "guardrail_latency_seconds",
    "Guardrail check latency in seconds by guard_type",
    ["guard_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


def record_block(guard_type: str, surface: str) -> None:
    guardrail_blocks_total.labels(guard_type=guard_type, surface=surface).inc()
    guardrail_decisions_total.labels(action="block", guard_type=guard_type, surface=surface).inc()


def record_allow(guard_type: str, surface: str) -> None:
    guardrail_decisions_total.labels(action="allow", guard_type=guard_type, surface=surface).inc()


def record_scrub(guard_type: str, surface: str) -> None:
    guardrail_decisions_total.labels(action="scrub", guard_type=guard_type, surface=surface).inc()
