"""Phase 32 verification-loop eval metrics: grounding_success_rate,
citation_accuracy_v2, retry_success_rate, avg_retry_count,
verification_latency_p50/p95.

Distinct from the existing lexical `citation_accuracy` in generation.py
(regex-based [filename] presence check) — these are computed from real
VerificationReport objects (app/verification/verification_schema.py),
produced by actually running VerificationLoop.run() or the lighter
app.verification.verify() dispatch over the eval rows, so they reflect the
grounding/citation/completeness checks this phase adds, not just citation
presence.
"""

from __future__ import annotations

from typing import Any

from app.eval.metrics.base import MetricResult


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(int(round(pct / 100.0 * (len(s) - 1))), len(s) - 1)
    return s[idx]


def compute_verification_metrics(reports: list[dict[str, Any]]) -> dict[str, MetricResult]:
    """reports: [VerificationReport.to_dict()] collected over the eval run —
    one per query, from actually invoking the verification loop/dispatch.
    """
    if not reports:
        return {
            "grounding_success_rate": MetricResult.empty("grounding_success_rate", "no reports"),
            "citation_accuracy_v2": MetricResult.empty("citation_accuracy_v2", "no reports"),
            "retry_success_rate": MetricResult.empty("retry_success_rate", "no reports"),
            "avg_retry_count": MetricResult.empty("avg_retry_count", "no reports"),
            "verification_latency_p50": MetricResult.empty(
                "verification_latency_p50", "no reports"
            ),
            "verification_latency_p95": MetricResult.empty(
                "verification_latency_p95", "no reports"
            ),
        }

    n = len(reports)
    grounded = sum(1 for r in reports if not r.get("unsupported_claims"))
    cited_ok = sum(1 for r in reports if not r.get("bad_citations"))

    retried = [r for r in reports if len(r.get("attempts") or []) > 1]
    retry_passed = sum(1 for r in retried if r.get("verified"))
    retry_counts = [max(len(r.get("attempts") or []) - 1, 0) for r in reports]
    latencies_sec = [(r.get("total_duration_ms") or 0.0) / 1000.0 for r in reports]

    return {
        "grounding_success_rate": MetricResult(
            name="grounding_success_rate",
            value=grounded / n,
            n=n,
            notes=f"{grounded}/{n} answers had zero unsupported claims/numbers",
        ),
        "citation_accuracy_v2": MetricResult(
            name="citation_accuracy_v2",
            value=cited_ok / n,
            n=n,
            notes=f"{cited_ok}/{n} answers had zero bad citations (CitationVerifier)",
        ),
        "retry_success_rate": MetricResult(
            name="retry_success_rate",
            value=(retry_passed / len(retried)) if retried else float("nan"),
            n=len(retried),
            notes=(
                f"{retry_passed}/{len(retried)} retried queries eventually PASSed"
                if retried
                else "no queries needed a retry"
            ),
        ),
        "avg_retry_count": MetricResult(
            name="avg_retry_count",
            value=sum(retry_counts) / n,
            n=n,
            notes=f"mean retries per query, max={max(retry_counts) if retry_counts else 0}",
        ),
        "verification_latency_p50": MetricResult(
            name="verification_latency_p50",
            value=_percentile(latencies_sec, 50),
            n=n,
        ),
        "verification_latency_p95": MetricResult(
            name="verification_latency_p95",
            value=_percentile(latencies_sec, 95),
            n=n,
        ),
    }
