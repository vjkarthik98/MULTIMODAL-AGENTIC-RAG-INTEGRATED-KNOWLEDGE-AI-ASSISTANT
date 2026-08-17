"""Unit tests for app/eval/runners/generation_runner.py::_verification_metrics
— the first-ever baseline aggregation for thresholds.yaml's `verification.*`
section (hallucination-reduction initiative, Phase 1, 2026-08-13).

Pure aggregation logic over VerificationReport.to_dict() shapes (see
app/verification/verification_schema.py) — no server, no GPU.
"""

from __future__ import annotations

from app.eval.runners.generation_runner import _verification_metrics


def _report(verified=True, unsupported_claims=None, bad_citations=None, attempts=1, ms=1000.0):
    return {
        "verified": verified,
        "scores": {"retrieval": 90.0, "grounding": 95.0, "citation": 100.0, "overall": 95.0},
        "unsupported_claims": unsupported_claims or [],
        "bad_citations": bad_citations or [],
        "missing_aspects": [],
        "attempts": [{"attempt_number": i + 1, "strategy": "baseline"} for i in range(attempts)],
        "total_duration_ms": ms,
        "degraded": False,
        "limitation_notice": None,
        "cited_sources": [],
    }


class TestVerificationMetrics:
    def test_no_reports_returns_empty_metrics(self):
        metrics = _verification_metrics([{"verification": None}, {}])
        names = {m.name for m in metrics}
        assert names == {
            "grounding_success_rate",
            "citation_accuracy_v2",
            "retry_success_rate",
            "avg_retry_count",
            "verification_latency_p50",
            "verification_latency_p95",
        }
        assert all(m.n == 0 for m in metrics)

    def test_grounding_and_citation_success_rate(self):
        rows = [
            {"verification": _report()},
            {"verification": _report(unsupported_claims=["bad claim"])},
            {"verification": _report(bad_citations=["[fake.pdf]"])},
        ]
        metrics = {m.name: m for m in _verification_metrics(rows)}
        assert metrics["grounding_success_rate"].value == 2 / 3
        assert metrics["citation_accuracy_v2"].value == 2 / 3
        assert metrics["grounding_success_rate"].n == 3

    def test_retry_success_rate_only_counts_retried_rows(self):
        rows = [
            {"verification": _report(attempts=1)},  # not retried — excluded from denominator
            {"verification": _report(attempts=2, verified=True)},  # retried, passed
            {"verification": _report(attempts=3, verified=False)},  # retried, still failed
        ]
        metrics = {m.name: m for m in _verification_metrics(rows)}
        assert metrics["retry_success_rate"].n == 2  # only the two retried rows
        assert metrics["retry_success_rate"].value == 1 / 2

    def test_avg_retry_count(self):
        rows = [
            {"verification": _report(attempts=1)},
            {"verification": _report(attempts=4)},
        ]
        metrics = {m.name: m for m in _verification_metrics(rows)}
        # retries = attempts - 1 -> (0 + 3) / 2 = 1.5
        assert metrics["avg_retry_count"].value == 1.5

    def test_latency_percentiles_from_duration_ms(self):
        rows = [
            {"verification": _report(ms=1000.0)},
            {"verification": _report(ms=2000.0)},
            {"verification": _report(ms=3000.0)},
        ]
        metrics = {m.name: m for m in _verification_metrics(rows)}
        assert metrics["verification_latency_p50"].value == 2.0
        assert metrics["verification_latency_p95"].value > 2.0

    def test_rows_without_verification_are_excluded_not_failures(self):
        rows = [
            {"verification": _report()},
            {"verification": None},  # e.g. hybrid_web path
            {},  # missing key entirely
        ]
        metrics = {m.name: m for m in _verification_metrics(rows)}
        assert metrics["grounding_success_rate"].n == 1
