"""Routing suite runner.

Calls app/agents/agent_controller.py:AgentController.handle() directly.
Measures route_accuracy, confusion matrix, and the P1-4 hybrid probe
(hybrid route must show actual web sources, not be labelled-only).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.metrics.routing import confusion_matrix, hybrid_with_web_rate, route_accuracy


def run_routing_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the routing benchmark against real AgentController.handle()."""
    t0 = time.time()
    result = SuiteResult(suite="routing")

    try:
        from app.agents.agent_controller import AgentController
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    try:
        controller = AgentController()
    except Exception as e:
        result.breached["controller_init"] = str(e)
        return result

    gold_rows = load_gold("routing", gold_dir=cfg.gold_dir, include_todos=False)
    if not gold_rows:
        result.add(MetricResult.empty("route_accuracy", "no routing gold rows found"))
        result.duration_sec = time.time() - t0
        return result

    routing_results: List[Dict[str, Any]] = []

    for row in gold_rows:
        query = row["query"]
        expected_route = row.get("expected_route", "")
        session_id = f"{cfg.session_prefix}_routing_{row['id']}"

        try:
            response = controller.handle(query=query, session_id=session_id)
        except Exception as exc:
            result.breached[f"handle_error_{row['id']}"] = str(exc)
            continue

        # Extract action from AgentDecision nested in response
        decision = response.get("decision") or {}
        if hasattr(decision, "action"):
            action = decision.action
        elif isinstance(decision, dict):
            action = decision.get("action", "")
        else:
            # Fallback: check top-level response
            action = response.get("action") or response.get("route") or ""

        sources = response.get("sources") or []
        web_sources = [s for s in sources if isinstance(s, dict) and s.get("type") == "web"]

        routing_results.append({
            "row_id": row["id"],
            "query": query,
            "actual_route": action,
            "expected_route": expected_route,
            "sources": sources,
            "web_sources": web_sources,
            "web_source_count": len(web_sources),
            "tags": row.get("tags", []),
        })

    if not routing_results:
        result.add(MetricResult.empty("route_accuracy", "no successful routing evaluations"))
        result.duration_sec = time.time() - t0
        return result

    # Core routing metrics
    result.add(route_accuracy(routing_results))

    # P1-4 probe: hybrid route must include web sources
    hybrid_rate = hybrid_with_web_rate(routing_results)
    result.add(hybrid_rate)

    # Confusion matrix logged as sub-metrics (not threshold-checked, for diagnostics)
    cm = confusion_matrix(routing_results)
    for label, counts in cm.items():
        for pred, cnt in counts.items():
            result.add(MetricResult(
                name=f"route_cm_{label}_as_{pred}",
                value=float(cnt),
                n=len(routing_results),
                notes="confusion matrix cell",
            ))

    result.duration_sec = time.time() - t0
    return result
