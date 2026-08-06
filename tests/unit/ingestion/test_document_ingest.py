"""Unit tests for app/ingestion/document_ingest.py — Phase 24.10."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.schema import IngestedDocument


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_doc(text: str = "Sample text content.", modality: str = "text") -> IngestedDocument:
    doc = IngestedDocument(
        text=text,
        modality=modality,
        structure={"session_id": "test", "doc_id": "abc123",
                   "content_type": "text", "embedding_space": "text"},
    )
    return doc


# ── _table_to_text ────────────────────────────────────────────────────────────

class TestTableToText:

    def test_empty_rows_returns_empty(self):
        from app.ingestion.pdf_ingest import _table_to_text
        assert _table_to_text([]) == ""

    def test_single_row_returns_joined(self):
        from app.ingestion.pdf_ingest import _table_to_text
        result = _table_to_text([["col1", "col2", "col3"]])
        assert "col1" in result
        assert "col2" in result

    def test_empty_row_skipped(self):
        # _table_to_text also drops rows with no digit/$/% (non-financial
        # prose fragments pdfplumber sometimes misreads as table rows) —
        # use a row with financial content so this test isolates the
        # empty-row-skipping behavior it's meant to cover.
        from app.ingestion.pdf_ingest import _table_to_text
        result = _table_to_text([["", "", ""], ["Revenue", "$100", "5%"]])
        assert "Revenue" in result

    def test_multirow_table_has_newlines(self):
        from app.ingestion.pdf_ingest import _table_to_text
        rows = [["h1", "h2"], ["r1", "r2"], ["r3", "r4"]]
        result = _table_to_text(rows)
        assert "\n" in result


class TestTableToMarkdown:

    def test_empty_returns_empty(self):
        from app.ingestion.pdf_ingest import _table_to_markdown
        assert _table_to_markdown([]) == ""

    def test_header_and_separator_present(self):
        from app.ingestion.pdf_ingest import _table_to_markdown
        rows = [["Name", "Age"], ["Alice", "30"]]
        result = _table_to_markdown(rows)
        assert "Name" in result
        assert "---" in result

    def test_body_rows_included(self):
        from app.ingestion.pdf_ingest import _table_to_markdown
        rows = [["Col1", "Col2"], ["val1", "val2"]]
        result = _table_to_markdown(rows)
        assert "val1" in result
        assert "val2" in result


# ── _file_hash ────────────────────────────────────────────────────────────────

class TestFileHash:

    def test_returns_64_char_hex(self, tmp_path):
        from app.ingestion.pdf_ingest import _file_hash
        p = tmp_path / "f.txt"
        p.write_text("hello world")
        result = _file_hash(str(p))
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_content_same_hash(self, tmp_path):
        from app.ingestion.pdf_ingest import _file_hash
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("same content")
        p2.write_text("same content")
        assert _file_hash(str(p1)) == _file_hash(str(p2))

    def test_different_content_different_hash(self, tmp_path):
        from app.ingestion.pdf_ingest import _file_hash
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("content one")
        p2.write_text("content two")
        assert _file_hash(str(p1)) != _file_hash(str(p2))


# ── _content_hash ─────────────────────────────────────────────────────────────

class TestContentHash:

    def test_returns_64_char_hex(self):
        from app.ingestion.pdf_ingest import _content_hash
        result = _content_hash("hello world")
        assert len(result) == 64

    def test_identical_strings_same_hash(self):
        from app.ingestion.pdf_ingest import _content_hash
        assert _content_hash("test") == _content_hash("test")

    def test_different_strings_different_hash(self):
        from app.ingestion.pdf_ingest import _content_hash
        assert _content_hash("abc") != _content_hash("xyz")


# ── Async ingest with mocked IO ───────────────────────────────────────────────

class TestDocumentIngestAsync:

    def test_missing_file_raises(self, tmp_path):
        from app.ingestion.pdf_ingest import ingest
        path = str(tmp_path / "nonexistent.pdf")

        async def _run():
            with pytest.raises((FileNotFoundError, ValueError, Exception)):
                await ingest(path, session_id="s1")

        asyncio.run(_run())

    def test_empty_session_id_raises(self, tmp_path):
        from app.ingestion.pdf_ingest import ingest
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.4 empty")

        async def _run():
            with pytest.raises(ValueError):
                await ingest(str(p), session_id="")

        asyncio.run(_run())


# ── IngestedDocument schema ───────────────────────────────────────────────────

class TestIngestedDocumentSchema:

    def test_valid_document_finalizes(self):
        doc = IngestedDocument(
            text="Valid text content for testing.",
            modality="text",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "text", "embedding_space": "text"},
        )
        result = doc.finalize()
        assert result.text

    def test_table_modality_valid(self):
        doc = IngestedDocument(
            text="Table data row1 col1 | col2 value here",
            modality="table",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "table", "embedding_space": "text"},
        )
        result = doc.finalize()
        assert result.modality == "table"

    def test_image_modality_valid(self):
        doc = IngestedDocument(
            text="Image caption text describing the photograph.",
            modality="image",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "image", "embedding_space": "text"},
        )
        result = doc.finalize()
        assert result.modality == "image"

    def test_short_text_raises(self):
        doc = IngestedDocument(
            text="ab",
            modality="text",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "text", "embedding_space": "text"},
        )
        with pytest.raises(ValueError, match="TEXT_TOO_SHORT"):
            doc.finalize()

    def test_invalid_modality_raises(self):
        doc = IngestedDocument(
            text="Valid text content.",
            modality="foobar",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "text", "embedding_space": "text"},
        )
        with pytest.raises(ValueError, match="INVALID_MODALITY"):
            doc.finalize()

    def test_content_hash_returns_64_chars(self):
        doc = _make_doc("hello world content test")
        assert len(doc.content_hash()) == 64

    def test_is_embeddable_false_without_embedding(self):
        doc = _make_doc("hello world content test")
        assert doc.is_embeddable() is False

    def test_is_embeddable_true_with_valid_embedding(self):
        from app.core.config import settings
        doc = _make_doc("hello world content test")
        doc.embedding = [0.1] * settings.TEXT_EMBEDDING_DIM
        assert doc.is_embeddable() is True

    def test_summary_has_expected_keys(self):
        doc = IngestedDocument(
            text="Summary test content text here.",
            modality="text",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "text", "embedding_space": "text"},
        )
        doc.finalize()
        s = doc.summary()
        for key in ("modality", "doc_id", "session_id", "text_length", "content_hash"):
            assert key in s

    def test_to_dict_has_all_fields(self):
        doc = _make_doc("Dict test content here.")
        d = doc.to_dict()
        for key in ("text", "modality", "structure", "extra_metadata"):
            assert key in d

    def test_is_high_quality_true_by_default(self):
        doc = IngestedDocument(
            text="Quality test content text here.",
            modality="text",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "text", "embedding_space": "text"},
        )
        doc.finalize()
        assert doc.is_high_quality() is True

    def test_clone_creates_new_doc_id(self):
        doc = IngestedDocument(
            text="Clone test content text here.",
            modality="text",
            structure={"session_id": "s1", "doc_id": "original_id",
                       "content_type": "text", "embedding_space": "text"},
        )
        doc.finalize()
        clone = doc.clone()
        assert clone.structure["doc_id"] != doc.structure["doc_id"]

    def test_normalize_strips_null_bytes(self):
        doc = IngestedDocument(
            text="He\x00llo world content.",
            modality="text",
            structure={"session_id": "s1", "doc_id": "id1",
                       "content_type": "text", "embedding_space": "text"},
        )
        doc.normalize()
        assert "\x00" not in doc.text
