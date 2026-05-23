"""
Unit tests for pure helper functions in scripts/download_model.py.
download_model() itself requires network/HuggingFace and is not tested here.
"""
import os
import pytest

from scripts.download_model import (
    _resolve_hf_token,
    _validate_download,
    _verify_checksum,
)


# ---------------------------------------------------------------------------
# _validate_download
# ---------------------------------------------------------------------------

class TestValidateDownload:

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="DOWNLOAD_MISSING"):
            _validate_download(str(tmp_path / "nonexistent.gguf"))

    def test_too_small_file_raises(self, tmp_path):
        f = tmp_path / "tiny.gguf"
        f.write_bytes(b"x" * 100)  # way below MIN_VALID_SIZE_BYTES
        with pytest.raises(RuntimeError, match="DOWNLOAD_CORRUPT"):
            _validate_download(str(f))

    def test_large_enough_file_passes(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * (1024 * 1024 * 200))  # 200 MB
        _validate_download(str(f))  # should not raise


# ---------------------------------------------------------------------------
# _verify_checksum
# ---------------------------------------------------------------------------

class TestVerifyChecksum:

    def test_no_checksum_skips_verification(self, tmp_path):
        f = tmp_path / "model.gguf"
        f.write_bytes(b"data")
        # Should not raise when expected_sha256 is None
        _verify_checksum(str(f), expected_sha256=None)

    def test_correct_checksum_passes(self, tmp_path):
        import hashlib
        data = b"test model data"
        f = tmp_path / "model.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        _verify_checksum(str(f), expected_sha256=expected)  # should not raise

    def test_wrong_checksum_raises(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"some data")
        with pytest.raises(RuntimeError, match="CHECKSUM_MISMATCH"):
            _verify_checksum(str(f), expected_sha256="0" * 64)


# ---------------------------------------------------------------------------
# _resolve_hf_token
# ---------------------------------------------------------------------------

class TestResolveHfToken:

    def test_returns_none_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        result = _resolve_hf_token()
        assert result is None

    def test_returns_hf_token_when_set(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test_token_abc")
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        result = _resolve_hf_token()
        assert result == "test_token_abc"

    def test_returns_huggingface_hub_token_when_set(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub_token_xyz")
        result = _resolve_hf_token()
        assert result == "hub_token_xyz"

    def test_hf_token_takes_priority(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "primary")
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "secondary")
        result = _resolve_hf_token()
        assert result == "primary"
