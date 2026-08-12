"""Tests for app/utils/otel_attrs.py — the OpenInference span-enrichment
helpers added for monitoring Phase 4 (Arize Phoenix).

Covers the attribute shape/truncation contract directly (no real span
needed — a tiny fake with the same set_attribute(key, value) interface a
real OTel span has), and the hard safety requirement stated in the module's
own docstring: none of these functions may ever raise, even when given
malformed input, because a telemetry bug must never taint the real
operation's span status.
"""

from __future__ import annotations

from app.utils.otel_attrs import set_input_output, set_retrieval_documents, set_span_kind


class _FakeSpan:
    def __init__(self):
        self.attributes: dict = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestSetSpanKind:
    def test_sets_openinference_span_kind(self):
        span = _FakeSpan()
        set_span_kind(span, "RETRIEVER")
        assert span.attributes["openinference.span.kind"] == "RETRIEVER"

    def test_never_raises_on_broken_span(self):
        class _BrokenSpan:
            def set_attribute(self, k, v):
                raise RuntimeError("boom")

        set_span_kind(_BrokenSpan(), "RETRIEVER")  # must not raise


class TestSetInputOutput:
    def test_sets_input_value(self):
        span = _FakeSpan()
        set_input_output(span, input_value="what is revenue")
        assert span.attributes["input.value"] == "what is revenue"
        assert "output.value" not in span.attributes

    def test_sets_output_value(self):
        span = _FakeSpan()
        set_input_output(span, output_value="revenue was $1B")
        assert span.attributes["output.value"] == "revenue was $1B"
        assert "input.value" not in span.attributes

    def test_truncates_to_200_chars(self):
        span = _FakeSpan()
        long_text = "x" * 500
        set_input_output(span, input_value=long_text)
        assert len(span.attributes["input.value"]) == 200

    def test_empty_values_are_not_set(self):
        span = _FakeSpan()
        set_input_output(span, input_value="", output_value=None)
        assert "input.value" not in span.attributes
        assert "output.value" not in span.attributes

    def test_never_raises_on_broken_span(self):
        class _BrokenSpan:
            def set_attribute(self, k, v):
                raise RuntimeError("boom")

        set_input_output(_BrokenSpan(), input_value="q", output_value="a")  # must not raise


class TestSetRetrievalDocuments:
    def test_flattens_documents_with_metadata_source(self):
        span = _FakeSpan()
        docs = [{"text": "chunk one", "score": 0.9, "metadata": {"source": "10k.pdf"}}]
        set_retrieval_documents(span, docs)
        assert span.attributes["retrieval.documents.0.document.content"] == "chunk one"
        assert span.attributes["retrieval.documents.0.document.score"] == 0.9
        assert span.attributes["retrieval.documents.0.document.id"] == "10k.pdf"

    def test_flattens_documents_with_flat_source(self):
        """The final built sources array (query_pipeline.py's
        _build_sources_array) uses a flat `source` key instead of
        `metadata.source` — both shapes must work."""
        span = _FakeSpan()
        docs = [{"text": "chunk", "score": 0.5, "source": "report.docx"}]
        set_retrieval_documents(span, docs)
        assert span.attributes["retrieval.documents.0.document.id"] == "report.docx"

    def test_falls_back_to_final_score(self):
        span = _FakeSpan()
        docs = [{"text": "chunk", "final_score": 0.7}]
        set_retrieval_documents(span, docs)
        assert span.attributes["retrieval.documents.0.document.score"] == 0.7

    def test_caps_at_max_documents(self):
        span = _FakeSpan()
        docs = [{"text": f"chunk {i}", "score": 0.5} for i in range(50)]
        set_retrieval_documents(span, docs)
        indices = {
            int(k.split(".")[2]) for k in span.attributes if k.startswith("retrieval.documents.")
        }
        assert max(indices) == 9  # _MAX_DOCUMENTS = 10, so indices 0..9

    def test_truncates_content_to_200_chars(self):
        span = _FakeSpan()
        docs = [{"text": "x" * 500, "score": 0.5}]
        set_retrieval_documents(span, docs)
        assert len(span.attributes["retrieval.documents.0.document.content"]) == 200

    def test_none_documents_does_not_raise(self):
        span = _FakeSpan()
        set_retrieval_documents(span, None)
        assert span.attributes == {}

    def test_non_dict_documents_are_skipped(self):
        span = _FakeSpan()
        set_retrieval_documents(span, ["not-a-dict", 123, None])
        assert span.attributes == {}

    def test_never_raises_on_broken_span(self):
        class _BrokenSpan:
            def set_attribute(self, k, v):
                raise RuntimeError("boom")

        docs = [{"text": "chunk", "score": 0.5}]
        set_retrieval_documents(_BrokenSpan(), docs)  # must not raise
