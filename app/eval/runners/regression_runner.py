"""Regression runner: diffs current eval metrics against a committed baseline.

Flags any metric that drops more than REGRESSION_TOLERANCE (default 5%) from baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.eval.config import EvalConfig
from app.eval.metrics.base import MetricResult, SuiteResult

REGRESSION_TOLERANCE = 0.05  # 5% drop triggers a regression breach


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _flat_metrics(baseline: dict) -> dict[str, float]:
    """Flatten {suite: {metrics: {name: {value: float}}}} to {name: float}."""
    flat: dict[str, float] = {}
    suites = baseline.get("suites") or {}
    for suite_name, suite_data in suites.items():
        metrics = suite_data.get("metrics") or {}
        for metric_name, metric_data in metrics.items():
            val = metric_data.get("value")
            if val is not None and not math.isnan(float(val)):
                flat[f"{suite_name}.{metric_name}"] = float(val)
    return flat


def run_regression_suite(cfg: EvalConfig) -> SuiteResult:
    """Compare a fresh retrieval+generation run against the committed baseline."""
    result = SuiteResult(suite="regression")

    # Find baseline
    baseline_path: Path | None = getattr(cfg, "_baseline_path", None)
    if baseline_path is None:
        baseline_path = cfg.baselines_dir / "rag_report_v1.json"

    if not baseline_path.exists():
        result.breached["no_baseline"] = (
            f"Baseline file not found: {baseline_path}. "
            "Run --suite full once, commit the output, then set it as baseline."
        )
        return result

    baseline = _load_baseline(baseline_path)
    baseline_metrics = _flat_metrics(baseline)

    if not baseline_metrics:
        result.breached["empty_baseline"] = f"Baseline at {baseline_path} has no metric data."
        return result

    # Run a fresh retrieval suite to compare
    from app.eval.runners.retrieval_runner import run_retrieval_suite

    current_retrieval = run_retrieval_suite(cfg)

    for name, m in current_retrieval.metrics.items():
        qualified = f"retrieval.{name}"
        result.add(
            MetricResult(
                name=f"current.{name}",
                value=m.value,
                n=m.n,
                notes=m.notes,
            )
        )

        if math.isnan(m.value):
            continue
        baseline_val = baseline_metrics.get(qualified)
        if baseline_val is None:
            continue  # new metric, no baseline to compare

        # For "max" metrics (lower is better), flag if current > baseline * (1 + tol)
        # For "min" metrics (higher is better), flag if current < baseline * (1 - tol)
        is_max_metric = any(w in name for w in ("rate_max", "_max", "p95", "p99", "p50"))
        if is_max_metric:
            threshold = baseline_val * (1 + REGRESSION_TOLERANCE)
            if m.value > threshold:
                msg = f"{name}: {m.value:.4f} > baseline {baseline_val:.4f} * 1.{int(REGRESSION_TOLERANCE*100):02d}"
                result.breached[f"regression.{name}"] = msg
        else:
            threshold = baseline_val * (1 - REGRESSION_TOLERANCE)
            if m.value < threshold:
                msg = f"{name}: {m.value:.4f} < baseline {baseline_val:.4f} * 0.{int((1-REGRESSION_TOLERANCE)*100):02d}"
                result.breached[f"regression.{name}"] = msg

    # Provenance, NOT a metric. This used to be emitted as
    # MetricResult(name="baseline_path", value=0.0), which made a file path
    # masquerade as a measurement: it printed in the report as
    # `regression.baseline_path = 0.0000 (n=0) /app/.../rag_report_v1.json`,
    # and — because tier2-eval.yml's Pushgateway step forwards every numeric
    # metric that is neither None nor NaN — it was also published to Prometheus
    # as a real time series pinned at 0.0. `dataset_version` is the field that
    # already exists for "which reference data did this suite run against", is
    # serialized by SuiteResult.to_dict(), and is not treated as a number.
    result.dataset_version = str(baseline_path)

    return result
