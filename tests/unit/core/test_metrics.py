"""Tests for app/core/metrics.py — Prometheus circuit breaker metrics."""


class TestMetricsImport:

    def test_circuit_breaker_state_importable(self):
        from app.core.metrics import circuit_breaker_state
        assert circuit_breaker_state is not None

    def test_circuit_breaker_failures_importable(self):
        from app.core.metrics import circuit_breaker_failures
        assert circuit_breaker_failures is not None

    def test_state_has_labels_method(self):
        from app.core.metrics import circuit_breaker_state
        assert hasattr(circuit_breaker_state, "labels")

    def test_failures_has_labels_method(self):
        from app.core.metrics import circuit_breaker_failures
        assert hasattr(circuit_breaker_failures, "labels")

    def test_labels_call_does_not_raise(self):
        from app.core.metrics import circuit_breaker_state, circuit_breaker_failures
        # Whether Prometheus is enabled or not, calling labels() should not raise
        s = circuit_breaker_state.labels(service="qdrant")
        assert s is not None
        f = circuit_breaker_failures.labels(service="qdrant")
        assert f is not None

    def test_set_on_noop_does_not_raise(self):
        from app.core.metrics import circuit_breaker_state
        # set() must be callable whether it's a real Gauge or the Noop stub
        try:
            circuit_breaker_state.labels(service="test").set(1)
        except Exception as exc:
            # Only Prometheus registration errors are acceptable — not AttributeErrors
            assert "already" in str(exc).lower() or isinstance(exc, ValueError)

    def test_reranker_latency_importable(self):
        from app.core.metrics import reranker_latency
        assert reranker_latency is not None

    def test_reranker_latency_observe_does_not_raise(self):
        from app.core.metrics import reranker_latency
        # No labels() call needed — it's a single, unlabeled histogram
        # (exactly one reranker in this deployment).
        reranker_latency.observe(0.123)
