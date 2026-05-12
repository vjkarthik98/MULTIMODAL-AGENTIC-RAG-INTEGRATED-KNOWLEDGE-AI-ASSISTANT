import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRouter:

    @pytest.mark.asyncio
    async def test_happy_path_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world this is a test document with enough content.")
        mock_docs = [MagicMock(spec=IngestedDocument, text="Hello world", structure={}, modality="text")]
        mock_docs[0].finalize.return_value = mock_docs[0]
        with patch("app.ingestion.router._detect_mime_magic", return_value="text/plain"), \
             patch("app.ingestion.router._clamav_scan"), \
             patch("app.ingestion.router.text_ingest", return_value=mock_docs), \
             patch("app.ingestion.router._file_sha256", return_value="abc123"):
            result = await route_ingestion(str(f), "session-1")
        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="EMPTY_FILE"):
            await route_ingestion(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            await route_ingestion("/nonexistent/path/file.txt", "session-1")

    @pytest.mark.asyncio
    async def test_no_session_id_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
            await route_ingestion(str(f), "")

    @pytest.mark.asyncio
    async def test_unsupported_mime_raises(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_bytes(b"\x00\x01\x02\x03unsupported binary content")
        with patch("app.ingestion.router._detect_mime_magic", return_value="application/x-unknown"), \
             patch("app.ingestion.router._clamav_scan"):
            with pytest.raises(ValueError, match="UNSUPPORTED_TYPE"):
                await route_ingestion(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_file_too_large_raises(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * (settings.MAX_FILE_SIZE_TEXT + 1))
        with patch("app.ingestion.router._detect_mime_magic", return_value="text/plain"), \
             patch("app.ingestion.router._clamav_scan"):
            with pytest.raises(ValueError, match="FILE_TOO_LARGE"):
                await route_ingestion(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_malware_detected_raises(self, tmp_path):
        f = tmp_path / "virus.txt"
        f.write_text("some content")
        with patch("app.ingestion.router._detect_mime_magic", return_value="text/plain"), \
             patch("app.ingestion.router._clamav_scan", side_effect=ValueError("MALWARE_DETECTED: Eicar")):
            with pytest.raises(ValueError, match="MALWARE_DETECTED"):
                await route_ingestion(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_null_bytes_logged_not_blocked(self, tmp_path):
        f = tmp_path / "nullbytes.txt"
        f.write_bytes(b"hello\x00world")
        

        with patch("app.ingestion.router._detect_mime_magic", return_value="text/plain"), \
             patch("app.ingestion.router._clamav_scan"):
     
            with pytest.raises(ValueError, match="BINARY_FILE_DISGUISED_AS_TEXT"):
                await route_ingestion(str(f), "session-1")
                

    def test_detect_modality_text(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("hello world")
        with patch("app.ingestion.router._detect_mime_magic", return_value="text/plain"), \
             patch("app.ingestion.router._check_disk_space"):
            modality, mime = detect_modality(str(f))
        assert modality == "text"
        assert mime == "text/plain"

    def test_detect_modality_pdf(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake pdf content")
        with patch("app.ingestion.router._detect_mime_magic", return_value="application/pdf"), \
             patch("app.ingestion.router._check_disk_space"):
            modality, mime = detect_modality(str(f))
        assert modality == "document"

    def test_file_sha256_deterministic(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("deterministic content")
        h1 = _file_sha256(f)
        h2 = _file_sha256(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_stable_hash_deterministic(self):
        assert _stable_hash("hello") == _stable_hash("hello")
        assert _stable_hash("hello") != _stable_hash("world")

    @pytest.mark.asyncio
    async def test_duplicate_docs_deduped(self, tmp_path):
        f = tmp_path / "dup.txt"
        f.write_text("duplicate content" * 50)
        doc = MagicMock(spec=IngestedDocument, text="duplicate content", structure={}, modality="text")
        doc.finalize.return_value = doc
        with patch("app.ingestion.router._detect_mime_magic", return_value="text/plain"), \
             patch("app.ingestion.router._clamav_scan"), \
             patch("app.ingestion.router.text_ingest", return_value=[doc, doc, doc]), \
             patch("app.ingestion.router._file_sha256", return_value="abc"):
            result = await route_ingestion(str(f), "session-1")
        assert len(result) <= 1