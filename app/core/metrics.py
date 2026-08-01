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


# Cross-module shared Histograms — moved here 2026-08-01 after a live Tier-2
# run showed these same three metric names independently re-defined in
# gguf_model.py / rag_pipeline.py / query_pipeline.py / reasoning_engine.py.
# Three of those four wrapped their own copy safely (function + try/except +
# module-level cache), so losing the registration race there just meant
# silently-empty metrics for that module. reasoning_engine.py's copy was
# unconditional, unguarded top-level code — and since it's only ever
# imported lazily, inside a function (query_pipeline.py's per-query path,
# not at module top-level), a losing collision there doesn't just no-op:
# Python evicts the failed module from sys.modules, so the NEXT query
# retries the same import, re-executes the same module body, and hits the
# same ValueError again — permanently, for the rest of the process, on
# every single query that needs ReasoningEngine. Confirmed live: one race
# lost at the first generation query took down 100% of the "full" Tier-2
# suite's generation phase, instantly and irrecoverably, for the rest of
# that run. Defining each metric exactly once, here, removes the race
# entirely instead of just making the loser's failure mode softer.
#
# llm_call_latency_seconds keeps gguf_model.py's richer ["model", "mode"]
# label shape (not the 1-label version the other three callers used) — a
# shared Histogram must have ONE label schema, and observing with a
# different label set than what a metric was registered with raises at
# observe() time, not just at registration. Callers that don't have a
# meaningful "mode" pass mode="pipeline" (see call sites) rather than being
# silently dropped.
def _make_shared_histograms():
    if not settings.PROMETHEUS_ENABLED:
        return None
    try:
        from prometheus_client import Histogram

        return {
            "llm_call_latency": Histogram(
                "llm_call_latency_seconds",
                "LLM call latency by model and mode",
                ["model", "mode"],
            ),
            "retrieval_latency": Histogram(
                "retrieval_latency_seconds",
                "Retrieval latency by retriever type",
                ["retriever_type"],
            ),
            "reasoning_engine_duration": Histogram(
                "reasoning_engine_duration_seconds",
                "Reasoning engine duration",
                ["status"],
            ),
            # Found in the same audit (2026-08-01, live "fomc_dec2024.txt:
            # INGESTION_FAILED" error surfaced in the UI) — router.py had its
            # own unguarded top-level copy of these three, colliding with
            # ingestion_pipeline.py's safely-guarded one, and router.py is
            # itself only ever imported lazily (ingestion_pipeline.py:916,
            # inside a function) — the exact same crash-and-never-recover
            # shape as reasoning_engine_duration_seconds above, just on the
            # file-upload path instead of the generation path.
            "file_ingestion_duration": Histogram(
                "file_ingestion_duration_seconds",
                "Ingestion duration by modality",
                ["modality"],
            ),
            "chunk_count_per_file": Histogram(
                "chunk_count_per_file",
                "Chunks produced per file",
                ["modality"],
            ),
            # embedding_latency_seconds: model_loader.py's own copy was
            # dead code (defined, never once observed) — deleted there
            # rather than unified, so this singleton has exactly one real
            # caller (ingestion_pipeline.py).
            "embedding_latency": Histogram(
                "embedding_latency_seconds",
                "Embedding latency by model",
                ["model"],
            ),
        }
    except Exception:
        return None


def _make_shared_counters():
    if not settings.PROMETHEUS_ENABLED:
        return None
    try:
        from prometheus_client import Counter

        return {
            "file_ingestion_errors": Counter(
                "file_ingestion_errors_total",
                "Ingestion errors by modality and error type",
                ["modality", "error_type"],
            ),
        }
    except Exception:
        return None


_shared_histograms = _make_shared_histograms()
_shared_counters = _make_shared_counters()

llm_call_latency = (_shared_histograms or {}).get("llm_call_latency", _Noop())
retrieval_latency = (_shared_histograms or {}).get("retrieval_latency", _Noop())
reasoning_engine_duration = (_shared_histograms or {}).get("reasoning_engine_duration", _Noop())
file_ingestion_duration = (_shared_histograms or {}).get("file_ingestion_duration", _Noop())
chunk_count_per_file = (_shared_histograms or {}).get("chunk_count_per_file", _Noop())
embedding_latency = (_shared_histograms or {}).get("embedding_latency", _Noop())
file_ingestion_errors = (_shared_counters or {}).get("file_ingestion_errors", _Noop())
