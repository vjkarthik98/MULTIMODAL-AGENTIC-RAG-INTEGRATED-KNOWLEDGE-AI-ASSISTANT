from __future__ import annotations

from app.core.config import settings


class _Noop:
    def labels(self, **_):
        return self

    def set(self, *_):
        pass

    def inc(self, *_):
        pass

    def observe(self, *_):
        pass


def _make_gauges():
    if not settings.PROMETHEUS_ENABLED:
        return None
    try:
        from prometheus_client import Gauge

        return {
            "circuit_breaker_state": Gauge(
                "circuit_breaker_state",
                "Circuit breaker state per service (0=closed, 1=half-open, 2=open)",
                ["service"],
            ),
            "circuit_breaker_failures": Gauge(
                "circuit_breaker_failures_total",
                "Circuit breaker failure count per service",
                ["service"],
            ),
            # ONLINE EVAL (Phase 31) — populated by app/eval/jobs/online_eval.py from
            # reference-free scoring of sampled live traffic (app/eval/jobs/shadow_sampler.py).
            # These are NOT the CI gold-set retrieval metrics (recall@k etc, see
            # thresholds.yaml) — those need ground truth and only run in eval-gate.yml.
            "eval_online_faithfulness": Gauge(
                "magik_eval_online_faithfulness",
                "Reference-free lexical faithfulness on sampled live traffic (rolling window)",
            ),
            "eval_online_answer_relevancy": Gauge(
                "magik_eval_online_answer_relevancy",
                "Reference-free lexical answer relevancy on sampled live traffic (rolling window)",
            ),
            "eval_online_hallucination_rate": Gauge(
                "magik_eval_online_hallucination_rate",
                "Fraction of sampled live answers flagged by hallucination_flag_single (rolling window)",
            ),
            "eval_online_latency_p50_ms": Gauge(
                "magik_eval_online_latency_p50_ms",
                "Median end-to-end stream latency on sampled live traffic (rolling window)",
            ),
            "eval_online_latency_p95_ms": Gauge(
                "magik_eval_online_latency_p95_ms",
                "P95 end-to-end stream latency on sampled live traffic (rolling window)",
            ),
            "eval_online_sample_count": Gauge(
                "magik_eval_online_sample_count",
                "Number of sampled live traces scored in the most recent online_eval run",
            ),
            "eval_online_route_share": Gauge(
                "magik_eval_online_route_share",
                "Share of sampled live traffic per routing decision (rolling window)",
                ["route"],
            ),
        }
    except Exception:
        return None


_gauges = _make_gauges()

circuit_breaker_state = (_gauges or {}).get("circuit_breaker_state", _Noop())
circuit_breaker_failures = (_gauges or {}).get("circuit_breaker_failures", _Noop())
eval_online_faithfulness = (_gauges or {}).get("eval_online_faithfulness", _Noop())
eval_online_answer_relevancy = (_gauges or {}).get("eval_online_answer_relevancy", _Noop())
eval_online_hallucination_rate = (_gauges or {}).get("eval_online_hallucination_rate", _Noop())
eval_online_latency_p50_ms = (_gauges or {}).get("eval_online_latency_p50_ms", _Noop())
eval_online_latency_p95_ms = (_gauges or {}).get("eval_online_latency_p95_ms", _Noop())
eval_online_sample_count = (_gauges or {}).get("eval_online_sample_count", _Noop())
eval_online_route_share = (_gauges or {}).get("eval_online_route_share", _Noop())
