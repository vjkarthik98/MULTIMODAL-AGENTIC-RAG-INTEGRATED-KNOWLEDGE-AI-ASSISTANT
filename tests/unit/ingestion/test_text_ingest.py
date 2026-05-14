import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTextIngest:

    @pytest.mark.asyncio
    async def test_happy_path_returns_chunks_and_metadata(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("This is a valid document with sufficient content for ingestion testing purposes.")
        with patch("app.ingestion.text_ingest._is_binary", return_value=False), \
             patch("app.ingestion.text_ingest._detect_language", return_value="en"), \
             patch("app.ingestion.text_ingest._redact_pii", return_value=("same text", {})):
            docs = await ingest(str(f), "session-1")
        assert len(docs) >= 1
        assert all(d.modality == "text" for d in docs)
        assert all(d.structure.get("language") == "en" for d in docs)

    @pytest.mark.asyncio
    async def test_empty_file_raises_empty_content_error(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="EMPTY_FILE"):
            await ingest(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_all_whitespace_raises(self, tmp_path):
        f = tmp_path / "blank.txt"
        f.write_text("   \n\n\t  \n  ")
        with patch("app.ingestion.text_ingest._is_binary", return_value=False):
            with pytest.raises(ValueError, match="EMPTY_CONTENT_AFTER_NORMALIZE"):
                await ingest(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_binary_file_raises_mime_error(self, tmp_path):
        f = tmp_path / "fake.txt"
        f.write_bytes(b"\x89PNG\r\n\x1a\nbinary content here that looks like image")
        with pytest.raises(ValueError, match="BINARY_FILE_DISGUISED_AS_TEXT"):
            await ingest(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_pii_redacted_from_chunks(self, tmp_path):
        f = tmp_path / "pii.txt"
        f.write_text("Contact John Doe at john@example.com or call 555-123-4567 for details.")
        with patch("app.ingestion.text_ingest._is_binary", return_value=False), \
             patch("app.ingestion.text_ingest._detect_language", return_value="en"), \
             patch("app.ingestion.text_ingest._redact_pii", return_value=("Contact [REDACTED] at [REDACTED] or call [REDATED] for additional ingestion testing details and metadata validation ", {"EMAIL_ADDRESS": 1, "PERSON": 1})) as mock_pii:
            docs = await ingest(str(f), "session-1")
        mock_pii.assert_called()
        assert any(d.structure.get("pii_redacted") for d in docs)

    @pytest.mark.asyncio
    async def test_duplicate_chunk_skipped_via_simhash(self, tmp_path):
        repeated = ("The quick brown fox jumps over the lazy dog. " * 10 + "\n") * 20
        f = tmp_path / "dup.txt"
        f.write_text(repeated)
        with patch("app.ingestion.text_ingest._is_binary", return_value=False), \
             patch("app.ingestion.text_ingest._detect_language", return_value="en"), \
             patch("app.ingestion.text_ingest._redact_pii", return_value=(repeated, {})):
            docs = await ingest(str(f), "session-1")
        assert len(docs) < 20

    @pytest.mark.asyncio
    async def test_happy_path_returns_chunks_and_metadata(self, tmp_path):
        f = tmp_path / "test.txt"

        content = (
            "This is a valid document with sufficient content "
            "for ingestion testing purposes and metadata validation."
        )

        f.write_text(content, encoding="utf-8")

        with patch(
            "app.ingestion.text_ingest._is_binary",
            return_value=False
        ), patch(
            "app.ingestion.text_ingest._detect_language",
            return_value="en"
        ), patch(
            "app.ingestion.text_ingest._redact_pii",
            return_value=(content, {})
        ), patch(
            "app.ingestion.text_ingest.settings.CHUNK_MIN_SIZE",
            1
        ):

            docs = await ingest(str(f), "session-1")

        assert len(docs) >= 1
        assert all(d.modality == "text" for d in docs)
        assert all(d.structure.get("language") == "en" for d in docs)

    @pytest.mark.asyncio
    async def test_bom_stripped_correctly(self, tmp_path):
        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbfThis text has a UTF-8 BOM at the start and is valid content.")
        with patch("app.ingestion.text_ingest._is_binary", return_value=False), \
             patch("app.ingestion.text_ingest._detect_language", return_value="en"), \
             patch("app.ingestion.text_ingest._redact_pii", return_value=("This text has a UTF-8 BOM at the start and is valid content.", {})):
            docs = await ingest(str(f), "session-1")
        assert all(not d.text.startswith("\ufeff") for d in docs)

    @pytest.mark.asyncio
    async def test_no_session_id_raises(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("some content here")
        with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
            await ingest(str(f), "")

    @pytest.mark.asyncio
    async def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            await ingest("/nonexistent/path/file.txt", "session-1")

    def test_simhash_identical_texts_zero_distance(self):
        h1 = _simhash("the quick brown fox")
        h2 = _simhash("the quick brown fox")
        assert _simhash_distance(h1, h2) == 0

    def test_simhash_different_texts_nonzero_distance(self):
        h1 = _simhash("the quick brown fox jumps")
        h2 = _simhash("completely different content here now")
        assert _simhash_distance(h1, h2) > 3

    def test_strip_bom_removes_utf8_bom(self):
        text = "\ufeffHello world"
        assert _strip_bom(text) == "Hello world"

    def test_strip_null_bytes_returns_count(self):
        text       = "hello\x00world\x00"
        cleaned, n = _strip_null_bytes(text)
        assert n == 2
        assert "\x00" not in cleaned

    def test_is_binary_detects_png(self, tmp_path):
        f = tmp_path / "fake.txt"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert _is_binary(f) is True

    def test_is_binary_false_for_plain_text(self, tmp_path):
        f = tmp_path / "real.txt"
        f.write_text("This is plain text content without binary markers.")
        assert _is_binary(f) is False

    def test_quality_score_short_text_low(self):
        assert _quality_score("hi") < 0.5

    def test_quality_score_long_text_high(self):
        text = "word " * 100
        assert _quality_score(text) >= 0.6

    def test_detect_subtype_heading(self):
        assert _detect_subtype("## Introduction") == "heading"

    def test_detect_subtype_paragraph(self):
        text = "This is a normal paragraph with multiple sentences and enough words."
        assert _detect_subtype(text) == "paragraph"

    def test_metadata_schema_populated(self, tmp_path):
        f = tmp_path / "meta.txt"
        f.write_text("Metadata test content with sufficient words for a valid chunk result.")

        async def _run():
            with patch("app.ingestion.text_ingest._is_binary", return_value=False), \
                 patch("app.ingestion.text_ingest._detect_language", return_value="en"), \
                 patch("app.ingestion.text_ingest._redact_pii", return_value=("Metadata test content with sufficient words for a valid chunk result.", {})):
                return await ingest(str(f), "session-meta")

        import asyncio as _asyncio
        docs = _asyncio.run(_run())
        assert len(docs) >= 1
        s = docs[0].structure
        assert "doc_id" in s
        assert "language" in s
        assert "ingestion_time" in s
        assert "tags" in s
        assert "readability_score" in s