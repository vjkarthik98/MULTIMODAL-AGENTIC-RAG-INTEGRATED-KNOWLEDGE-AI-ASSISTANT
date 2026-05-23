"""Unit tests for app/ingestion/image_ingest.py — Phase 24.10."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_jpeg(tmp_path: Path, name: str = "img.jpg", size: tuple = (100, 100)) -> str:
    """Write a minimal valid JPEG file."""
    p = tmp_path / name
    img = Image.new("RGB", size, color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    p.write_bytes(buf.getvalue())
    return str(p)


def _write_png(tmp_path: Path, name: str = "img.png", size: tuple = (100, 100)) -> str:
    """Write a minimal valid PNG file."""
    p = tmp_path / name
    img = Image.new("RGB", size, color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return str(p)


# ── _detect_mime_from_bytes ───────────────────────────────────────────────────

class TestDetectMimeFromBytes:

    def test_jpeg_header_detected(self):
        from app.ingestion.image_ingest import _detect_mime_from_bytes
        header = b"\xff\xd8\xff\xe0" + b"\x00" * 12
        assert _detect_mime_from_bytes(header, ".jpg") == "image/jpeg"

    def test_png_header_detected(self):
        from app.ingestion.image_ingest import _detect_mime_from_bytes
        header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        assert _detect_mime_from_bytes(header, ".png") == "image/png"

    def test_gif_header_detected(self):
        from app.ingestion.image_ingest import _detect_mime_from_bytes
        header = b"GIF89a" + b"\x00" * 10
        assert _detect_mime_from_bytes(header, ".gif") == "image/gif"

    def test_bmp_header_detected(self):
        from app.ingestion.image_ingest import _detect_mime_from_bytes
        header = b"BM" + b"\x00" * 14
        assert _detect_mime_from_bytes(header, ".bmp") == "image/bmp"

    def test_unknown_returns_octet_stream(self):
        from app.ingestion.image_ingest import _detect_mime_from_bytes
        header = b"\x00" * 16
        result = _detect_mime_from_bytes(header, ".xyz")
        assert result == "application/octet-stream"


# ── _validate_dimensions ──────────────────────────────────────────────────────

class TestValidateDimensions:

    def test_valid_image_passes(self):
        from app.ingestion.image_ingest import _validate_dimensions
        img = Image.new("RGB", (100, 100))
        _validate_dimensions(img)  # no raise

    def test_zero_width_raises(self):
        from app.ingestion.image_ingest import _validate_dimensions
        img = MagicMock()
        img.size = (0, 100)
        with pytest.raises(ValueError, match="INVALID_IMAGE_DIMENSIONS"):
            _validate_dimensions(img)

    def test_too_small_raises(self):
        from app.ingestion.image_ingest import _validate_dimensions
        img = Image.new("RGB", (10, 10))
        with pytest.raises(ValueError, match="IMAGE_TOO_SMALL"):
            _validate_dimensions(img)


# ── _check_aspect_ratio ───────────────────────────────────────────────────────

class TestCheckAspectRatio:

    def test_normal_ratio_returns_none(self):
        from app.ingestion.image_ingest import _check_aspect_ratio
        assert _check_aspect_ratio(100, 100) is None

    def test_extreme_wide_returns_warning(self):
        from app.ingestion.image_ingest import _check_aspect_ratio
        result = _check_aspect_ratio(2000, 10)
        assert result is not None
        assert "EXTREME" in result

    def test_zero_height_returns_none(self):
        from app.ingestion.image_ingest import _check_aspect_ratio
        # Should not crash
        result = _check_aspect_ratio(100, 0)
        assert result is None


# ── _resize_if_needed ─────────────────────────────────────────────────────────

class TestResizeIfNeeded:

    def test_small_image_unchanged(self):
        from app.ingestion.image_ingest import _resize_if_needed
        img = Image.new("RGB", (200, 200))
        result = _resize_if_needed(img)
        assert result.size == (200, 200)

    def test_large_image_resized(self):
        from app.ingestion.image_ingest import _resize_if_needed
        from app.core.config import settings
        # Very large image — should be downsized
        max_dim = settings.MAX_IMAGE_DIM + 100
        img = Image.new("RGB", (max_dim, max_dim))
        result = _resize_if_needed(img)
        assert max(result.size) <= settings.MAX_IMAGE_DIM


# ── _validate_image_file ──────────────────────────────────────────────────────

class TestValidateImageFile:

    def test_valid_jpeg_passes(self, tmp_path):
        from app.ingestion.image_ingest import _validate_image_file
        path = Path(_write_jpeg(tmp_path))
        _validate_image_file(path)  # no raise

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from app.ingestion.image_ingest import _validate_image_file
        with pytest.raises(FileNotFoundError):
            _validate_image_file(tmp_path / "missing.jpg")

    def test_empty_file_raises(self, tmp_path):
        from app.ingestion.image_ingest import _validate_image_file
        from app.ingestion.schema import EmptyFileError
        p = tmp_path / "empty.jpg"
        p.write_bytes(b"")
        with pytest.raises(EmptyFileError):
            _validate_image_file(p)

    def test_unsupported_extension_raises(self, tmp_path):
        from app.ingestion.image_ingest import _validate_image_file
        from app.ingestion.schema import UnsupportedMimeError
        p = tmp_path / "doc.xyz"
        p.write_bytes(b"fake content here")
        with pytest.raises(UnsupportedMimeError):
            _validate_image_file(p)


# ── ingest function ───────────────────────────────────────────────────────────

class TestImageIngest:

    def test_session_id_required_raises(self, tmp_path):
        from app.ingestion.image_ingest import ingest
        path = _write_jpeg(tmp_path)
        with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
            ingest(path, session_id="")

    def test_missing_file_raises(self, tmp_path):
        from app.ingestion.image_ingest import ingest
        with pytest.raises((FileNotFoundError, ValueError)):
            ingest(str(tmp_path / "missing.jpg"), session_id="s1")

    def _ingest_with_mocks(self, path_str: str, session_id: str, caption: str) -> list:
        """Helper to run ingest with chroot guard and model calls mocked."""
        from pathlib import Path
        from app.ingestion.image_ingest import ingest
        real_path = Path(path_str)

        with patch("app.ingestion.image_ingest._safe_resolve", return_value=real_path), \
             patch("app.ingestion.image_ingest._generate_caption", return_value=caption), \
             patch("app.ingestion.image_ingest._extract_ocr", return_value=""):
            return ingest(path_str, session_id=session_id)

    def test_valid_jpeg_with_mocked_models(self, tmp_path):
        path = _write_jpeg(tmp_path)
        result = self._ingest_with_mocks(
            path, "s1", "A test image with mountains and trees in the background."
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_result_modality_is_image(self, tmp_path):
        path = _write_jpeg(tmp_path)
        result = self._ingest_with_mocks(
            path, "s1", "A descriptive caption for the test image photograph."
        )
        for doc in result:
            assert doc.modality == "image"

    def test_result_has_caption_text(self, tmp_path):
        path = _write_jpeg(tmp_path)
        caption = "A beautiful mountain landscape photograph with trees and sky."
        result = self._ingest_with_mocks(path, "s1", caption)
        all_text = " ".join(doc.text for doc in result)
        assert "mountain" in all_text.lower() or "landscape" in all_text.lower()

    def test_result_has_structure_with_session_id(self, tmp_path):
        path = _write_jpeg(tmp_path)
        result = self._ingest_with_mocks(
            path, "my_sess", "Caption text here for session id testing purposes."
        )
        for doc in result:
            assert doc.structure.get("session_id") == "my_sess"

    def test_png_file_with_mocked_models(self, tmp_path):
        path = _write_png(tmp_path)
        result = self._ingest_with_mocks(
            path, "s1", "PNG image caption showing a red sunset sky view."
        )
        assert isinstance(result, list)
