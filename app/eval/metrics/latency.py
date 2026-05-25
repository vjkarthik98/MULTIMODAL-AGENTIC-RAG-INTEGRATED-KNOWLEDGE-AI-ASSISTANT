"""Latency metrics: p50, p95, p99 from a list of observed latency samples."""
from __future__ import annotations

import math
from typing import List

from app.eval.metrics.base import MetricResult


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def latency_stats(samples_sec: List[float], prefix: str = "") -> dict:
    """Return p50/p95/p99 MetricResult dict from a list of latency samples (seconds)."""
    tag = f"{prefix}_" if prefix else ""
    if not samples_sec:
        return {
            f"{tag}p50_sec": MetricResult.empty(f"{tag}p50_sec", "no samples"),
            f"{tag}p95_sec": MetricResult.empty(f"{tag}p95_sec", "no samples"),
            f"{tag}p99_sec": MetricResult.empty(f"{tag}p99_sec", "no samples"),
        }
    n = len(samples_sec)
    # Use bare names for threshold lookup (prefix only when multiple suites need disambiguation)
    p50_name = f"{tag}p50_sec" if tag else "p50_sec"
    p95_name = f"{tag}p95_sec" if tag else "p95_sec"
    p99_name = f"{tag}p99_sec" if tag else "p99_sec"
    return {
        p50_name: MetricResult(
            name=p50_name,
            value=_percentile(samples_sec, 50),
            n=n,
            notes=f"min={min(samples_sec):.2f}s max={max(samples_sec):.2f}s",
        ),
        p95_name: MetricResult(
            name=p95_name,
            value=_percentile(samples_sec, 95),
            n=n,
        ),
        p99_name: MetricResult(
            name=p99_name,
            value=_percentile(samples_sec, 99),
            n=n,
        ),
    }
