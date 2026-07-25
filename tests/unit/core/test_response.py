"""Unit tests for app/core/response.py — Phase 24.9."""
from __future__ import annotations

import pytest

from app.core.response import (
    ErrorCode,
    Modality,
    ProcessingResult,
    ProcessingStatus,
    QueryResponse,
    Severity,
    StreamChunk,
    UniversalErrorResponse,
    ValidationResult,
    build_retrieved_source,
    build_sources,
    err,
    ok,
    query_ok,
    validation_err,
    validation_ok,
)


# ── ProcessingResult ──────────────────────────────────────────────────────────

class TestProcessingResult:

    def test_to_dict_has_all_fields(self):
        r = ProcessingResult(
            modality="pdf", session_id="s1", latency=0.5,
            chunks=10, stored=10, source="doc.pdf",
        )
        d = r.to_dict()
        for key in ("success", "modality", "session_id", "latency",
                    "chunks", "stored", "source", "warnings", "file_hash", "status"):
            assert key in d

    def test_success_is_true(self):
        r = ProcessingResult(modality="text", session_id="s1", latency=0.1)
        assert r.success is True

    def test_warnings_defaults_to_empty_list(self):
        r = ProcessingResult(modality="text", session_id="s1", latency=0.1)
        assert r.warnings == []

    def test_status_defaults_to_success(self):
        r = ProcessingResult(modality="text", session_id="s1", latency=0.1)
        assert r.status == ProcessingStatus.SUCCESS

    def test_file_hash_stored(self):
        r = ProcessingResult(modality="text", session_id="s1", latency=0.1,
                             file_hash="abc123")
        assert r.file_hash == "abc123"


# ── UniversalErrorResponse ────────────────────────────────────────────────────

class TestUniversalErrorResponse:

    def test_success_is_false(self):
        r = UniversalErrorResponse(
            code=ErrorCode.EMPTY_FILE, message="File is empty"
        )
        assert r.success is False

    def test_error_code_stored(self):
        r = UniversalErrorResponse(
            code=ErrorCode.CORRUPTED_FILE, message="Corrupt"
        )
        assert r.error.code == ErrorCode.CORRUPTED_FILE

    def test_to_dict_has_error_key(self):
        r = UniversalErrorResponse(
            code=ErrorCode.EMPTY_FILE, message="empty"
        )
        d = r.to_dict()
        assert "error" in d
        assert d["error"]["code"] == ErrorCode.EMPTY_FILE

    def test_severity_stored(self):
        r = UniversalErrorResponse(
            code=ErrorCode.EMPTY_FILE, message="empty",
            severity=Severity.HIGH,
        )
        assert r.error.severity == Severity.HIGH


# ── QueryResponse ─────────────────────────────────────────────────────────────

class TestQueryResponse:

    def test_confidence_clamped_below_zero(self):
        r = QueryResponse(answer="a", session_id="s1", latency=0.1, confidence=-0.5)
        assert r.confidence == 0.0

    def test_confidence_clamped_above_one(self):
        r = QueryResponse(answer="a", session_id="s1", latency=0.1, confidence=1.5)
        assert r.confidence == 1.0

    def test_sources_stored(self):
        src = [{"text": "chunk", "score": 0.9}]
        r = QueryResponse(answer="a", session_id="s1", latency=0.1, sources=src)
        assert len(r.sources) == 1

    def test_sources_used_inferred_from_sources(self):
        src = [{"text": "c1"}, {"text": "c2"}]
        r = QueryResponse(answer="a", session_id="s1", latency=0.1, sources=src)
        assert r.sources_used == 2

    def test_to_dict_has_required_keys(self):
        r = QueryResponse(answer="a", session_id="s1", latency=0.1)
        d = r.to_dict()
        for key in ("answer", "confidence", "sources", "decision", "cache_hit"):
            assert key in d


# ── ValidationResult ──────────────────────────────────────────────────────────

class TestValidationResult:

    def test_add_error_sets_valid_false(self):
        v = ValidationResult(valid=True, modality="text")
        v.add_error(ErrorCode.EMPTY_FILE, "empty")
        assert v.valid is False
        assert v.has_errors is True

    def test_add_warning_does_not_invalidate(self):
        v = ValidationResult(valid=True)
        v.add_warning("minor issue")
        assert v.valid is True
        assert v.has_warnings is True

    def test_to_dict_has_all_fields(self):
        v = ValidationResult(valid=True, modality="pdf", file_size=1024, mime_type="application/pdf")
        d = v.to_dict()
        for key in ("valid", "modality", "file_size", "mime_type", "errors", "warnings"):
            assert key in d

    def test_multiple_errors_accumulated(self):
        v = ValidationResult(valid=True)
        v.add_error(ErrorCode.EMPTY_FILE, "e1")
        v.add_error(ErrorCode.CORRUPTED_FILE, "e2")
        assert len(v.errors) == 2


# ── StreamChunk ───────────────────────────────────────────────────────────────

class TestStreamChunk:

    def test_done_produces_done_marker(self):
        s = StreamChunk(token="", done=True)
        assert s.to_sse() == "data: [DONE]\n\n"

    def test_token_produces_sse_line(self):
        s = StreamChunk(token="hello")
        assert s.to_sse() == "data: hello\n\n"

    def test_to_dict_has_all_fields(self):
        s = StreamChunk(token="tok", session_id="s1")
        d = s.to_dict()
        for key in ("token", "session_id", "done", "timestamp"):
            assert key in d


# ── Factory helpers ───────────────────────────────────────────────────────────

class TestFactoryHelpers:

    def test_ok_returns_processing_result(self):
        r = ok(modality="text", session_id="s1", latency=0.1, chunks=5, stored=5)
        assert isinstance(r, ProcessingResult)
        assert r.success is True

    def test_err_returns_error_response(self):
        r = err(code=ErrorCode.EMPTY_FILE, message="empty")
        assert isinstance(r, UniversalErrorResponse)
        assert r.success is False

    def test_validation_ok_returns_valid_result(self):
        r = validation_ok(modality="pdf", file_size=1024, mime_type="application/pdf")
        assert r.valid is True

    def test_validation_err_returns_invalid_result(self):
        r = validation_err(code=ErrorCode.CORRUPTED_FILE, message="corrupt")
        assert r.valid is False
        assert r.has_errors is True

    def test_query_ok_returns_query_response(self):
        r = query_ok(answer="The answer is 42", session_id="s1", latency=0.3,
                     confidence=0.85, decision="rag")
        assert isinstance(r, QueryResponse)
        assert r.answer == "The answer is 42"
        assert r.confidence == 0.85


# ── build_retrieved_source ────────────────────────────────────────────────────

class TestBuildRetrievedSource:

    def test_pdf_source_has_page(self):
        doc = {
            "text": "Some content from a PDF page.",
            "score": 0.9,
            "metadata": {"source": "report.pdf", "modality": "pdf", "page": 3},
        }
        s = build_retrieved_source(doc)
        assert s["page"] == 3
        assert s["source"] == "report.pdf"
        assert "p.3" in s["cite_key"]

    def test_audio_source_has_timestamps(self):
        doc = {
            "text": "Transcript text.",
            "score": 0.8,
            "metadata": {
                "source": "talk.mp3", "modality": "audio",
                "timestamp_start": 10.0, "timestamp_end": 25.5,
            },
        }
        s = build_retrieved_source(doc)
        assert s["timestamp_start"] == 10.0
        assert s["timestamp_end"] == 25.5

    def test_snippet_truncated_to_240_chars(self):
        long_text = "word " * 200
        doc = {"text": long_text, "score": 0.5, "metadata": {}}
        s = build_retrieved_source(doc, snippet_chars=240)
        assert len(s["snippet"]) <= 240

    def test_unknown_source_fallback(self):
        doc = {"text": "text", "score": 0.5, "metadata": {}}
        s = build_retrieved_source(doc)
        assert s["source"] == "unknown"

    def test_cite_key_present(self):
        doc = {"text": "text", "score": 0.5, "metadata": {"source": "doc.pdf", "page": 1}}
        s = build_retrieved_source(doc)
        assert s["cite_key"].startswith("[")


class TestBuildSources:

    def test_deduplicates_by_cite_key(self):
        doc = {"text": "text", "score": 0.9, "metadata": {"source": "a.pdf", "page": 1, "modality": "pdf"}}
        sources = build_sources([doc, doc])
        assert len(sources) == 1

    def test_empty_input_returns_empty(self):
        assert build_sources([]) == []

    def test_multiple_unique_docs_all_returned(self):
        docs = [
            {"text": "t1", "score": 0.9, "metadata": {"source": "a.pdf", "page": 1}},
            {"text": "t2", "score": 0.8, "metadata": {"source": "b.pdf", "page": 2}},
        ]
        sources = build_sources(docs)
        assert len(sources) == 2
