from __future__ import annotations

from app.core.config import settings

if settings.PROMETHEUS_ENABLED:
    try:
        from prometheus_client import Gauge
        circuit_breaker_state = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state per service (0=closed, 1=open)",
            ["service"],
        )
        circuit_breaker_failures = Gauge(
            "circuit_breaker_failures_total",
            "Circuit breaker failure count per service",
            ["service"],
        )
    except Exception:
        class _Noop:
            def labels(self, **_): return self
            def set(self, *_): pass
        circuit_breaker_state    = _Noop()
        circuit_breaker_failures = _Noop()
else:
    class _Noop:
        def labels(self, **_): return self
        def set(self, *_): pass
    circuit_breaker_state    = _Noop()
    circuit_breaker_failures = _Noop()
