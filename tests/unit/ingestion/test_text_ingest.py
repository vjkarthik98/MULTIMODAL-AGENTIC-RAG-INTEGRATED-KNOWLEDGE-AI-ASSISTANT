"""Unit tests for app/ingestion/text_ingest.py — Phase 24.10."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.ingestion.schema import IngestedDocument


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _write_bytes(tmp_path: Path, name: str, content: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


# ── Sync ingest ───────────────────────────────────────────────────────────────

class TestTextIngestSync:

    def test_valid_file_returns_list(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Hello world. " * 20)
        result = ingest_sync(path, session_id="sess_1")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_returns_ingested_document_instances(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "The quick brown fox. " * 20)
        result = ingest_sync(path, session_id="sess_1")
        for doc in result:
            assert isinstance(doc, IngestedDocument)

    def test_modality_is_text(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Sample text content. " * 20)
        result = ingest_sync(path, session_id="s1")
        for doc in result:
            assert doc.modality == "text"

    def test_session_id_required_raises(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Some content here.")
        with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
            ingest_sync(path, session_id="")

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        with pytest.raises(FileNotFoundError):
            ingest_sync(str(tmp_path / "nonexistent.txt"), session_id="s1")

    def test_empty_file_raises(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "empty.txt", "")
        with pytest.raises(ValueError):
            ingest_sync(path, session_id="s1")

    def test_binary_file_raises(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        binary_data = bytes(range(256)) * 10
        path = _write_bytes(tmp_path, "binary.txt", binary_data)
        with pytest.raises((ValueError, UnicodeDecodeError)):
            ingest_sync(path, session_id="s1")

    def test_documents_have_nonempty_text(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Retrieval augmented generation. " * 20)
        result = ingest_sync(path, session_id="s1")
        for doc in result:
            assert isinstance(doc.text, str)
            assert len(doc.text) > 0

    def test_structure_is_dict(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Structure test content. " * 20)
        result = ingest_sync(path, session_id="s1")
        for doc in result:
            assert isinstance(doc.structure, dict)

    def test_structure_has_session_id(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Session id test. " * 20)
        result = ingest_sync(path, session_id="my_session")
        for doc in result:
            assert doc.structure.get("session_id") == "my_session"

    def test_structure_has_doc_id(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Doc id test. " * 20)
        result = ingest_sync(path, session_id="s1")
        for doc in result:
            assert "doc_id" in doc.structure
            assert doc.structure["doc_id"]

    def test_bom_stripped(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        content = "﻿BOM stripped content. " * 20
        path = _write(tmp_path, "bom.txt", content)
        result = ingest_sync(path, session_id="s1")
        assert len(result) >= 1
        assert "﻿" not in result[0].text

    def test_null_bytes_stripped_from_valid_text(self, tmp_path):
        from app.ingestion.txt_ingest import _strip_null_bytes
        # The null stripping helper itself removes nulls from clean text
        text_with_nulls = "Content with nu\x00ll bytes here."
        result = _strip_null_bytes(text_with_nulls)
        assert "\x00" not in result

    def test_chunk_id_assigned(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        path = _write(tmp_path, "doc.txt", "Chunk id test content. " * 30)
        result = ingest_sync(path, session_id="s1")
        for doc in result:
            assert doc.chunk_id is not None

    def test_near_duplicate_chunks_reduced(self, tmp_path):
        from app.ingestion.txt_ingest import ingest_sync
        paragraph = "Identical paragraph content for dedup testing. " * 10
        content = paragraph + "\n\n" + paragraph
        path = _write(tmp_path, "dedup.txt", content)
        result = ingest_sync(path, session_id="s1")
        assert len(result) <= 2


class TestTextIngestAsync:

    def test_async_ingest_returns_list(self, tmp_path):
        from app.ingestion.txt_ingest import ingest
        path = _write(tmp_path, "doc.txt", "Async ingest test content. " * 20)

        async def _run():
            return await ingest(path, session_id="s1")

        result = asyncio.run(_run())
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_async_empty_session_id_raises(self, tmp_path):
        from app.ingestion.txt_ingest import ingest
        path = _write(tmp_path, "doc.txt", "Some content here.")

        async def _run():
            with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
                await ingest(path, session_id="")

        asyncio.run(_run())


# ── Internal helper functions ─────────────────────────────────────────────────

class TestTextIngestHelpers:

    def test_is_binary_detects_null_byte_file(self, tmp_path):
        from pathlib import Path
        from app.ingestion.txt_ingest import _is_binary
        p = tmp_path / "binary.bin"
        p.write_bytes(b"\x00\x01\x02\x03hello world")
        assert _is_binary(Path(p)) is True

    def test_is_binary_accepts_plain_text_file(self, tmp_path):
        from pathlib import Path
        from app.ingestion.txt_ingest import _is_binary
        p = tmp_path / "plain.txt"
        p.write_bytes(b"Hello world this is plain text.")
        assert _is_binary(Path(p)) is False

    def test_strip_bom_removes_bom(self):
        from app.ingestion.txt_ingest import _strip_bom
        result = _strip_bom("﻿Hello world")
        assert result == "Hello world"

    def test_strip_bom_no_change_without_bom(self):
        from app.ingestion.txt_ingest import _strip_bom
        result = _strip_bom("Hello world")
        assert result == "Hello world"

    def test_strip_null_bytes_removes_nulls(self):
        from app.ingestion.txt_ingest import _strip_null_bytes
        result = _strip_null_bytes("He\x00llo\x00 world")
        assert "\x00" not in result

    def test_simhash_returns_int(self):
        from app.ingestion.txt_ingest import _simhash
        result = _simhash("Hello world this is a test document")
        assert isinstance(result, int)

    def test_simhash_identical_texts_zero_distance(self):
        from app.ingestion.txt_ingest import _simhash, _simhash_distance
        text = "The quick brown fox jumps over the lazy dog"
        h1 = _simhash(text)
        h2 = _simhash(text)
        assert _simhash_distance(h1, h2) == 0

    def test_simhash_different_texts_nonzero_distance(self):
        from app.ingestion.txt_ingest import _simhash, _simhash_distance
        h1 = _simhash("The quick brown fox")
        h2 = _simhash("Completely different about mountains")
        assert _simhash_distance(h1, h2) > 0

    def test_quality_score_returns_float_in_range(self):
        from app.ingestion.txt_ingest import _quality_score
        score = _quality_score("The quick brown fox jumps over the lazy dog.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_quality_score_low_for_symbols_only(self):
        from app.ingestion.txt_ingest import _quality_score
        score = _quality_score("!!!###$$$%%%^^^&&&***((()))___+++")
        assert score < 0.5

    def test_detect_subtype_returns_string(self):
        from app.ingestion.txt_ingest import _detect_subtype
        result = _detect_subtype("# Section Heading\n\nSome paragraph text content.")
        assert isinstance(result, str)
