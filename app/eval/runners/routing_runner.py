"""Routing suite runner.

Calls query_pipeline directly (same path as the API) so that hybrid
queries actually execute web search and return web sources.
Route decision is extracted from the pipeline response's 'decision' field.
Measures route_accuracy, confusion matrix, and the P1-4 hybrid probe
(hybrid route must show actual web sources, not be labelled-only).
"""

from __future__ import annotations

import time
from typing import Any

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.metrics.routing import confusion_matrix, hybrid_with_web_rate, route_accuracy


def run_routing_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the routing benchmark against the full query_pipeline."""
    t0 = time.time()
    result = SuiteResult(suite="routing")

    try:
        from app.pipeline.query_pipeline import query_pipeline
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    try:
        # warm up pipeline
        _ = query_pipeline  # noqa
    except Exception as e:
        result.breached["controller_init"] = str(e)
        return result

    gold_rows = load_gold("routing", gold_dir=cfg.gold_dir, include_todos=False)
    if not gold_rows:
        result.add(MetricResult.empty("route_accuracy", "no routing gold rows found"))
        result.duration_sec = time.time() - t0
        return result

    routing_results: list[dict[str, Any]] = []

    for row in gold_rows:
        query = row["query"]
        expected_route = row.get("expected_route", "")
        session_id = f"{cfg.session_prefix}_routing_{row['id']}"

        try:
            response = query_pipeline(query, session_id, None, cfg.user_id)
        except Exception as exc:
            result.breached[f"handle_error_{row['id']}"] = str(exc)
            continue

        # Extract action — query_pipeline returns {"decision": "rag"|"search"|..., ...}
        decision = response.get("decision") or ""
        if isinstance(decision, str):
            action = decision
        elif hasattr(decision, "action"):
            action = decision.action
        elif isinstance(decision, dict):
            action = decision.get("action", "")
        else:
            action = response.get("action") or response.get("route") or ""

        sources = response.get("sources") or []
        # Pipeline sets modality="web" on web sources; also check type="web" for compatibility
        web_sources = [
            s
            for s in sources
            if isinstance(s, dict) and (s.get("modality") == "web" or s.get("type") == "web")
        ]

        routing_results.append(
            {
                "row_id": row["id"],
                "query": query,
                "actual_route": action,
                "expected_route": expected_route,
                "sources": sources,
                "web_sources": web_sources,
                "web_source_count": len(web_sources),
                "tags": row.get("tags", []),
            }
        )

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
            result.add(
                MetricResult(
                    name=f"route_cm_{label}_as_{pred}",
                    value=float(cnt),
                    n=len(routing_results),
                    notes="confusion matrix cell",
                )
            )

    result.duration_sec = time.time() - t0
    return result
