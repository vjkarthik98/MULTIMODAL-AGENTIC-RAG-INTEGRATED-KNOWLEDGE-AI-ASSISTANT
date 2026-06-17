"""Latency metrics: p50, p95, p99 from a list of observed latency samples."""
from __future__ import annotations

import math
from typing import List

from app.eval.metrics.base import MetricResult


def _pct(sorted_v: List[float], p: float) -> float:
    """Interpolated p-th percentile from a pre-sorted list. O(1) — no sort."""
    if not sorted_v:
        return float("nan")
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
    # Sort once; pass pre-sorted array to O(1) _pct() for all three percentiles.
    # Before: three separate sorted() calls = O(3n log n). After: O(n log n + 3).
    sorted_v = sorted(samples_sec)
    n = len(samples_sec)
    # Use bare names for threshold lookup (prefix only when multiple suites need disambiguation)
    p50_name = f"{tag}p50_sec" if tag else "p50_sec"
    p95_name = f"{tag}p95_sec" if tag else "p95_sec"
    p99_name = f"{tag}p99_sec" if tag else "p99_sec"
    return {
        p50_name: MetricResult(
            name=p50_name,
            value=_pct(sorted_v, 50),
            n=n,
            notes=f"min={sorted_v[0]:.2f}s max={sorted_v[-1]:.2f}s",
        ),
        p95_name: MetricResult(
            name=p95_name,
            value=_pct(sorted_v, 95),
            n=n,
        ),
        p99_name: MetricResult(
            name=p99_name,
            value=_pct(sorted_v, 99),
            n=n,
        ),
    }
