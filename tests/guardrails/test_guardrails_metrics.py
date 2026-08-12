"""Tests for app/guardrails/metrics.py's graceful-degradation fix
(monitoring Phase 7 audit): record_block/record_allow/record_scrub are
called unconditionally, with no surrounding try/except at the call site, at
every input_guard.sanitize()/output_guard.check()/rate_limiter call — i.e.
on literally every request through this system. Confirmed by testing (not
inspection) that a Prometheus failure previously propagated straight up
through these three functions; this pins that fix in place.
"""

from __future__ import annotations

from unittest.mock import patch

from app.guardrails.metrics import record_allow, record_block, record_scrub


class TestGuardrailsMetricsGracefulDegradation:
    def test_record_allow_survives_prometheus_failure(self):
        import app.guardrails.metrics as gm

        with patch.object(
            gm.guardrail_decisions_total, "labels", side_effect=RuntimeError("prometheus broke")
        ):
            record_allow("input", "query_pipeline")  # must not raise

    def test_record_block_survives_prometheus_failure_on_blocks_counter(self):
        import app.guardrails.metrics as gm

        with patch.object(
            gm.guardrail_blocks_total, "labels", side_effect=RuntimeError("prometheus broke")
        ):
            record_block("injection", "query_pipeline")  # must not raise

    def test_record_block_survives_prometheus_failure_on_decisions_counter(self):
        import app.guardrails.metrics as gm

        with patch.object(
            gm.guardrail_decisions_total, "labels", side_effect=RuntimeError("prometheus broke")
        ):
            record_block("injection", "query_pipeline")  # must not raise

    def test_record_scrub_survives_prometheus_failure(self):
        import app.guardrails.metrics as gm

        with patch.object(
            gm.guardrail_decisions_total, "labels", side_effect=RuntimeError("prometheus broke")
        ):
            record_scrub("template_artifact", "query_pipeline")  # must not raise

    def test_normal_operation_still_increments(self):
        """The fix must not turn these into no-ops on the happy path."""
        import app.guardrails.metrics as gm

        before = gm.guardrail_decisions_total.labels(
            action="allow", guard_type="input", surface="test_surface"
        )._value.get()
        record_allow("input", "test_surface")
        after = gm.guardrail_decisions_total.labels(
            action="allow", guard_type="input", surface="test_surface"
        )._value.get()
        assert after == before + 1
