"""Tests for the Phase 0 monitoring work: OTel span wrappers added around
Reranker.rerank, HybridRetriever.search, query_pipeline(), RAGPipeline.run,
and RAGPipeline.stream.

These wrappers exist so the ALREADY-INSTRUMENTED spans deeper in the stack
(agent_controller.py's "agent_controller_handle", reasoning_engine.py's
"reasoning_generate_answer", qdrant_store.py's "qdrant_search",
prompt_builder.py's "prompt_builder") get a real parent span to nest under
instead of each becoming its own disconnected root trace — see the module
docstrings on query_pipeline()/RAGPipeline.run()/RAGPipeline.stream() for the
full rationale.

Verifies three things per wrapper:
1. It delegates to the renamed `_*_impl` and returns that result unchanged.
2. An exception from the impl still propagates AND marks the span as ERROR
   (monitoring must observe failures, never swallow them).
3. Nesting actually works: a span opened while a wrapper's span is current
   reports the wrapper's span as its parent (the structural fix this phase
   exists for — proven with a real SDK span exporter, not asserted by
   inspection).

`trace.get_tracer(__name__)` is called at import time in each production
module (before any SDK is configured, mirroring real startup order — app/
main.py's _setup_otel() runs during FastAPI lifespan, after every pipeline
module is already imported). The OTel API's tracer proxy resolves the
*current* global provider at span-creation time, not at get_tracer()-call
time, so setting the SDK provider here — after those modules already
imported — matches production and lets InMemorySpanExporter observe real
spans.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session")
def _sdk_exporter():
    """Install a real SDK TracerProvider + InMemorySpanExporter exactly once
    for the whole test session.

    OTel's global TracerProvider can only be set once per process (a second
    `set_tracer_provider()` call is refused with a logged warning, not an
    exception) — this mirrors production, where app/main.py's _setup_otel()
    sets it exactly once at startup. Every production module in this repo
    already calls `trace.get_tracer(__name__)` at import time (before any
    SDK is configured); the OTel API's tracer proxy resolves whatever the
    *current* global provider is at span-creation time, so installing the
    real SDK provider here — after those modules already imported, same as
    production's startup order — is what lets InMemorySpanExporter observe
    real spans instead of the no-op default.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def span_exporter(_sdk_exporter):
    """Per-test view: same session-wide exporter, cleared before each test
    so span counts/names asserted in one test can't leak into the next."""
    _sdk_exporter.clear()
    yield _sdk_exporter
    _sdk_exporter.clear()


class TestRerankerSpanWrapper:
    def test_delegates_and_returns_impl_result(self, span_exporter):
        from app.retrieval.reranker import Reranker

        r = Reranker.__new__(Reranker)  # bypass __init__ (loads a real model)
        sentinel = [{"text": "doc1", "score": 0.9}]
        r._rerank_impl = lambda *a, **k: sentinel  # type: ignore[method-assign]

        result = r.rerank("query", [{"text": "doc1"}], top_k=5, session_id="s1")

        assert result is sentinel
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "reranker_rerank"
        assert spans[0].attributes["input_count"] == 1
        assert spans[0].attributes["output_count"] == 1
        assert spans[0].status.is_ok
        # Phase 4 (Arize Phoenix) OpenInference enrichment on the same span.
        assert spans[0].attributes["openinference.span.kind"] == "RERANKER"
        assert spans[0].attributes["input.value"] == "query"
        assert spans[0].attributes["retrieval.documents.0.document.content"] == "doc1"
        assert spans[0].attributes["retrieval.documents.0.document.score"] == 0.9

    def test_exception_propagates_and_marks_span_error(self, span_exporter):
        from app.retrieval.reranker import Reranker

        r = Reranker.__new__(Reranker)

        def _boom(*a, **k):
            raise RuntimeError("cross-encoder blew up")

        r._rerank_impl = _boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="cross-encoder blew up"):
            r.rerank("query", [{"text": "doc1"}], top_k=5, session_id="s1")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert not spans[0].status.is_ok
        assert spans[0].events[0].name == "exception"


class TestHybridRetrieverSpanWrapper:
    def test_delegates_and_returns_impl_result(self, span_exporter):
        from app.retrieval.hybrid_retriever import HybridRetriever

        hr = HybridRetriever.__new__(HybridRetriever)
        sentinel = [{"text": "doc1", "score": 0.75}]
        hr._search_impl = lambda *a, **k: sentinel  # type: ignore[method-assign]

        result = hr.search("query", session_id="s1", top_k=10)

        assert result is sentinel
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "hybrid_retriever_search"
        assert spans[0].attributes["results.count"] == 1
        assert spans[0].attributes["results.top_score"] == 0.75
        assert spans[0].status.is_ok
        # Phase 4 (Arize Phoenix) OpenInference enrichment on the same span.
        assert spans[0].attributes["openinference.span.kind"] == "RETRIEVER"
        assert spans[0].attributes["input.value"] == "query"
        assert spans[0].attributes["retrieval.documents.0.document.content"] == "doc1"

    def test_exception_propagates_and_marks_span_error(self, span_exporter):
        from app.retrieval.hybrid_retriever import HybridRetriever

        hr = HybridRetriever.__new__(HybridRetriever)

        def _boom(*a, **k):
            raise ValueError("qdrant unreachable")

        hr._search_impl = _boom  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="qdrant unreachable"):
            hr.search("query", session_id="s1")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert not spans[0].status.is_ok


class TestQueryPipelineSpanWrapper:
    def test_delegates_and_sets_response_attributes(self, span_exporter, monkeypatch):
        import app.pipeline.query_pipeline as qp

        sentinel = {
            "decision": "rag",
            "confidence": 0.83,
            "cache_hit": False,
            "request_id": "req-123",
            "answer": "the answer",
        }
        monkeypatch.setattr(qp, "_query_pipeline_impl", lambda *a, **k: sentinel)

        result = qp.query_pipeline("what is the revenue", session_id="s1")

        assert result is sentinel
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "query_pipeline"
        assert span.attributes["decision"] == "rag"
        assert span.attributes["confidence"] == pytest.approx(0.83)
        assert span.attributes["request.id"] == "req-123"
        # Version attributes exist for deploy root-cause correlation.
        assert "app.version" in span.attributes
        assert "git.sha" in span.attributes
        assert "prompt.version" in span.attributes
        assert span.status.is_ok
        # Phase 4 (Arize Phoenix) OpenInference enrichment on the same span.
        assert span.attributes["openinference.span.kind"] == "CHAIN"
        assert span.attributes["input.value"] == "what is the revenue"
        assert span.attributes["output.value"] == "the answer"


class TestSpanNesting:
    """The actual structural fix: a span opened while a wrapper span is
    current must report the wrapper span as its parent, not become its own
    disconnected root trace — this is the exact bug Phase 0 closes."""

    def test_child_span_nests_under_reranker_wrapper(self, span_exporter):
        from app.retrieval.reranker import Reranker
        from app.retrieval.reranker import tracer as reranker_tracer

        r = Reranker.__new__(Reranker)

        def _impl_with_child_span(*a, **k):
            # Simulates what a real deeper call (e.g. a cross-encoder helper
            # emitting its own span) would do while the wrapper's span is
            # current.
            with reranker_tracer.start_as_current_span("inner_work"):
                pass
            return []

        r._rerank_impl = _impl_with_child_span  # type: ignore[method-assign]
        r.rerank("query", [{"text": "d"}], top_k=5, session_id="s1")

        spans = {s.name: s for s in span_exporter.get_finished_spans()}
        assert "reranker_rerank" in spans
        assert "inner_work" in spans
        parent_span = spans["reranker_rerank"]
        child_span = spans["inner_work"]
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span.context.span_id
        assert child_span.parent.trace_id == parent_span.context.trace_id

    def test_query_pipeline_root_span_is_ancestor_of_impl_span(self, span_exporter, monkeypatch):
        """Regression guard for the exact bug this phase fixes: before the
        query_pipeline() wrapper existed, a span opened by code called from
        inside _query_pipeline_impl (e.g. agent_controller's own span) had no
        active parent and became a brand new root trace. Now it must share
        query_pipeline's trace_id."""
        import app.pipeline.query_pipeline as qp
        from app.pipeline.query_pipeline import tracer as qp_tracer

        def _impl_opens_nested_span(*a, **k):
            with qp_tracer.start_as_current_span("simulated_agent_controller_handle"):
                pass
            return {"decision": "rag", "confidence": 0.5, "cache_hit": False}

        monkeypatch.setattr(qp, "_query_pipeline_impl", _impl_opens_nested_span)
        qp.query_pipeline("a query", session_id="s1")

        spans = {s.name: s for s in span_exporter.get_finished_spans()}
        root = spans["query_pipeline"]
        nested = spans["simulated_agent_controller_handle"]
        assert nested.context.trace_id == root.context.trace_id, (
            "nested span must share query_pipeline's trace_id — this is the "
            "exact fragmentation bug Phase 0 fixes"
        )


class TestRagPipelineRunSpanWrapper:
    def test_delegates_and_sets_response_attributes(self, span_exporter, monkeypatch):
        from app.pipeline.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        sentinel = {"decision": "rag", "trace_id": "req-456", "answer": "the answer"}
        monkeypatch.setattr(rag, "_run_impl", lambda *a, **k: sentinel)

        result = rag.run("what is the revenue", session_id="s1")

        assert result is sentinel
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "rag_pipeline_run"
        assert spans[0].attributes["decision"] == "rag"
        assert spans[0].attributes["request.id"] == "req-456"
        assert spans[0].status.is_ok
        # Phase 4 (Arize Phoenix) OpenInference enrichment on the same span.
        assert spans[0].attributes["openinference.span.kind"] == "CHAIN"
        assert spans[0].attributes["input.value"] == "what is the revenue"
        assert spans[0].attributes["output.value"] == "the answer"

    def test_exception_propagates_and_marks_span_error(self, span_exporter, monkeypatch):
        from app.pipeline.rag_pipeline import RAGPipeline

        rag = RAGPipeline()

        def _boom(*a, **k):
            raise RuntimeError("llm unavailable")

        monkeypatch.setattr(rag, "_run_impl", _boom)

        with pytest.raises(RuntimeError, match="llm unavailable"):
            rag.run("query", session_id="s1")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert not spans[0].status.is_ok


class TestRagPipelineStreamSpanWrapper:
    """`stream()` isn't itself a generator — it returns one lazily (see the
    module docstring on RAGPipeline.stream) — so the span only opens once the
    caller actually iterates the returned generator, and must stay open for
    the whole iteration, closing on either natural exhaustion or an
    exception raised mid-stream."""

    def test_span_closes_on_error_inside_generator(self, span_exporter, monkeypatch):
        from app.pipeline.rag_pipeline import RAGPipeline

        rag = RAGPipeline()

        def _raise_retriever(self):
            raise RuntimeError("retriever init failed")

        monkeypatch.setattr(RAGPipeline, "_get_retriever", _raise_retriever)

        gen = rag.stream("what is the revenue", session_id="s1")
        # stream()'s inner _generator() catches this internally and yields a
        # user-facing message rather than raising (see "Streaming failed."
        # in app/pipeline/rag_pipeline.py) — so the span must close OK, and
        # the generator must still be fully drainable.
        tokens = list(gen)
        assert tokens  # something was yielded, not silently swallowed

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "rag_pipeline_stream"
        assert spans[0].attributes["session.id"] == "s1"
        # generator's own try/except handles the failure and yields a
        # message, so from the span's perspective this is a clean exit.
        assert spans[0].status.is_ok
        # Phase 4 (Arize Phoenix): OUTPUT_VALUE is accumulated by
        # _traced_generator itself from the yielded tokens (not read from
        # _generator()'s internals), so it must equal what was actually
        # streamed back to the caller.
        assert spans[0].attributes["openinference.span.kind"] == "CHAIN"
        assert spans[0].attributes["output.value"] == "".join(tokens)
