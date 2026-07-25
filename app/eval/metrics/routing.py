"""Routing accuracy metrics.

Measures whether AgentController.handle() returns the expected decision.action.
Also tracks the P1-4 probe: when decision=hybrid, web sources MUST be present.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.eval.metrics.base import MetricResult

VALID_ROUTES = {"rag", "search", "memory", "direct", "hybrid"}


def route_accuracy(results: list[dict[str, Any]]) -> MetricResult:
    """Fraction of queries where decision.action == expected_route.

    results: [{"expected_route": str, "actual_route": str, "query": str}]
    """
    if not results:
        return MetricResult.empty("route_accuracy", "no routing results")

    correct = sum(1 for r in results if r.get("actual_route") == r.get("expected_route"))
    value = correct / len(results)
    wrong = [
        f"  '{r.get('query', '')[:50]}' expected={r.get('expected_route')} got={r.get('actual_route')}"
        for r in results
        if r.get("actual_route") != r.get("expected_route")
    ]
    notes = f"correct={correct}/{len(results)}"
    if wrong:
        notes += " | misroutes: " + "; ".join(wrong[:3])
    return MetricResult(name="route_accuracy", value=value, n=len(results), notes=notes)


def confusion_matrix(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Returns {expected_route: {actual_route: count}} dict."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        exp = r.get("expected_route", "unknown")
        act = r.get("actual_route", "unknown")
        matrix[exp][act] += 1
    return {k: dict(v) for k, v in matrix.items()}


def hybrid_with_web_rate(results: list[dict[str, Any]]) -> MetricResult:
    """Fraction of hybrid-route results that actually contain web sources.

    Catches P1-4: system labels decision=hybrid but returns zero web results.
    results: [{"actual_route": str, "web_source_count": int, "query": str}]
    """
    hybrid_results = [r for r in results if r.get("actual_route") == "hybrid"]
    if not hybrid_results:
        return MetricResult.empty("hybrid_with_web_rate", "no hybrid-route queries in set")

    with_web = sum(1 for r in hybrid_results if r.get("web_source_count", 0) > 0)
    value = with_web / len(hybrid_results)
    notes = f"hybrid_with_web={with_web}/{len(hybrid_results)}"
    if with_web < len(hybrid_results):
        missing = [
            r.get("query", "")[:50] for r in hybrid_results if r.get("web_source_count", 0) == 0
        ]
        notes += f" | P1-4 misses: {missing[:3]}"
    return MetricResult(
        name="hybrid_with_web_rate", value=value, n=len(hybrid_results), notes=notes
    )
