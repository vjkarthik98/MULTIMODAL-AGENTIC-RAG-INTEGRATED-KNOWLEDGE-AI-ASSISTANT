"""Unit tests for app/ingestion/audio_ingest.py — Phase 24.10."""
from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_wav(tmp_path: Path, name: str = "audio.wav", duration_s: float = 2.0,
               sample_rate: int = 16000) -> str:
    """Write a minimal valid WAV file (sine-like silence)."""
    p = tmp_path / name
    n_frames = int(sample_rate * duration_s)
    with wave.open(str(p), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return str(p)


def _make_mock_audio(duration: float = 2.0, channels: int = 1,
                     frame_rate: int = 16000) -> MagicMock:
    m = MagicMock()
    m.duration_seconds = duration
    m.channels = channels
    m.frame_rate = frame_rate
    m.dBFS = -20.0
    m.set_channels = MagicMock(return_value=m)
    m.set_frame_rate = MagicMock(return_value=m)
    return m


def _make_mock_whisper_result(text: str = "Hello world transcript.") -> dict:
    return {
        "text": text,
        "segments": [
            {
                "text": text,
                "start": 0.0,
                "end": 2.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.05,
            }
        ],
    }


# ── _detect_mime ──────────────────────────────────────────────────────────────

class TestDetectMime:

    def test_wav_magic_detected(self, tmp_path):
        from app.ingestion.audio_ingest import _detect_mime
        p = tmp_path / "test.wav"
        # RIFF....WAVE header
        p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 4)
        result = _detect_mime(p)
        assert "audio" in result

    def test_mp3_magic_detected(self, tmp_path):
        from app.ingestion.audio_ingest import _detect_mime
        p = tmp_path / "test.mp3"
        p.write_bytes(b"ID3" + b"\x00" * 13)
        result = _detect_mime(p)
        assert result == "audio/mpeg"

    def test_flac_magic_detected(self, tmp_path):
        from app.ingestion.audio_ingest import _detect_mime
        p = tmp_path / "test.flac"
        p.write_bytes(b"fLaC" + b"\x00" * 12)
        result = _detect_mime(p)
        assert result == "audio/flac"

    def test_unknown_falls_back_to_extension(self, tmp_path):
        from app.ingestion.audio_ingest import _detect_mime
        p = tmp_path / "test.ogg"
        p.write_bytes(b"\x00" * 16)
        result = _detect_mime(p)
        assert isinstance(result, str)


# ── _compute_confidence ───────────────────────────────────────────────────────

class TestComputeConfidence:

    def test_none_logprob_returns_one(self):
        from app.ingestion.audio_ingest import _compute_confidence
        assert _compute_confidence(None) == 1.0

    def test_zero_logprob_returns_one(self):
        from app.ingestion.audio_ingest import _compute_confidence
        result = _compute_confidence(0.0)
        assert result == 1.0

    def test_negative_logprob_reduces_confidence(self):
        from app.ingestion.audio_ingest import _compute_confidence
        result = _compute_confidence(-0.5)
        assert 0.0 < result < 1.0

    def test_very_negative_logprob_clamps_to_zero(self):
        from app.ingestion.audio_ingest import _compute_confidence
        result = _compute_confidence(-10.0)
        assert result == 0.0


# ── _hallucination_risk ───────────────────────────────────────────────────────

class TestHallucinationRisk:

    def test_high_confidence_low_risk(self):
        from app.ingestion.audio_ingest import _hallucination_risk
        result = _hallucination_risk(0.9, 0.05)
        assert result == "low"

    def test_low_confidence_high_risk(self):
        from app.ingestion.audio_ingest import _hallucination_risk
        result = _hallucination_risk(0.3, 0.1)
        assert result == "high"

    def test_high_no_speech_prob_high_risk(self):
        from app.ingestion.audio_ingest import _hallucination_risk
        result = _hallucination_risk(0.9, 0.9)
        assert result == "high"

    def test_medium_confidence_medium_risk(self):
        from app.ingestion.audio_ingest import _hallucination_risk
        result = _hallucination_risk(0.55, 0.3)
        assert result == "medium"


# ── _speed_flag ───────────────────────────────────────────────────────────────

class TestSpeedFlag:

    def test_normal_speech_rate_no_flag(self):
        from app.ingestion.audio_ingest import _speed_flag
        text = "word " * 100  # 100 words
        assert _speed_flag(text, 60.0) is False  # 100wpm — normal

    def test_very_fast_speech_flagged(self):
        from app.ingestion.audio_ingest import _speed_flag
        text = "word " * 300  # 300 words
        assert _speed_flag(text, 60.0) is True  # 300wpm — too fast

    def test_zero_duration_no_flag(self):
        from app.ingestion.audio_ingest import _speed_flag
        assert _speed_flag("word word word", 0.0) is False


# ── _strip_fillers ────────────────────────────────────────────────────────────

class TestStripFillers:

    def test_returns_string(self):
        from app.ingestion.audio_ingest import _strip_fillers
        result = _strip_fillers("um hello uh world")
        assert isinstance(result, str)


# ── _estimate_snr ─────────────────────────────────────────────────────────────

class TestEstimateSnr:

    def test_returns_float(self):
        from app.ingestion.audio_ingest import _estimate_snr
        mock_audio = _make_mock_audio()
        result = _estimate_snr(mock_audio)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_silence_returns_zero(self):
        from app.ingestion.audio_ingest import _estimate_snr
        mock_audio = MagicMock()
        mock_audio.dBFS = float("-inf")
        result = _estimate_snr(mock_audio)
        assert result == 0.0


# ── _validate_audio_file ──────────────────────────────────────────────────────

class TestValidateAudioFile:

    def test_valid_wav_passes(self, tmp_path):
        from app.ingestion.audio_ingest import _validate_audio_file
        path = Path(_write_wav(tmp_path))
        _validate_audio_file(path)  # no raise

    def test_missing_file_raises(self, tmp_path):
        from app.ingestion.audio_ingest import _validate_audio_file
        with pytest.raises(FileNotFoundError):
            _validate_audio_file(tmp_path / "missing.mp3")

    def test_empty_file_raises(self, tmp_path):
        from app.ingestion.audio_ingest import _validate_audio_file
        from app.ingestion.schema import EmptyFileError
        p = tmp_path / "empty.wav"
        p.write_bytes(b"")
        with pytest.raises(EmptyFileError):
            _validate_audio_file(p)

    def test_unsupported_format_raises(self, tmp_path):
        from app.ingestion.audio_ingest import _validate_audio_file
        from app.ingestion.schema import UnsupportedMimeError
        p = tmp_path / "audio.xyz"
        p.write_bytes(b"fake audio content")
        with pytest.raises(UnsupportedMimeError):
            _validate_audio_file(p)


# ── ingest function ───────────────────────────────────────────────────────────

class TestAudioIngest:

    def test_session_id_required_raises(self, tmp_path):
        from app.ingestion.audio_ingest import ingest
        path = _write_wav(tmp_path)
        with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
            ingest(path, session_id="")

    def test_missing_file_raises(self, tmp_path):
        from app.ingestion.audio_ingest import ingest
        with pytest.raises((FileNotFoundError, ValueError)):
            ingest(str(tmp_path / "missing.wav"), session_id="s1")

    def _ingest_with_mocks(self, path_str: str, session_id: str,
                            transcript: str = "Test transcript content.") -> list:
        """Run audio ingest with chroot and model calls mocked."""
        from pathlib import Path
        from app.ingestion.audio_ingest import ingest
        real_path = Path(path_str)
        mock_audio = _make_mock_audio(duration=3.0)

        # Build a fake segments_iter from the transcript
        seg = MagicMock()
        seg.text = transcript
        seg.start = 0.0
        seg.end = 3.0
        seg.avg_logprob = -0.2
        seg.no_speech_prob = 0.05
        seg.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 3.0

        with patch("app.ingestion.audio_ingest._safe_resolve", return_value=real_path), \
             patch("app.ingestion.audio_ingest._load_audio", return_value=mock_audio), \
             patch("app.ingestion.audio_ingest._to_mono", return_value=mock_audio), \
             patch("app.ingestion.audio_ingest._resample", return_value=mock_audio), \
             patch("app.ingestion.audio_ingest._transcribe_file",
                   return_value=([seg], mock_info, 0.1)):
            return ingest(path_str, session_id=session_id)

    def test_valid_wav_with_mocked_whisper(self, tmp_path):
        path = _write_wav(tmp_path, duration_s=3.0)
        result = self._ingest_with_mocks(path, "s1", "This is a test transcript.")
        assert isinstance(result, list)

    def test_result_modality_is_audio(self, tmp_path):
        path = _write_wav(tmp_path, duration_s=3.0)
        result = self._ingest_with_mocks(path, "s1", "Audio transcript text content here.")
        for doc in result:
            assert doc.modality == "audio"

    def test_result_contains_transcript_text(self, tmp_path):
        path = _write_wav(tmp_path, duration_s=3.0)
        transcript = "This is the expected transcript content."
        result = self._ingest_with_mocks(path, "s1", transcript)
        all_text = " ".join(doc.text for doc in result)
        assert "transcript" in all_text.lower() or "expected" in all_text.lower()

    def test_result_has_session_id_in_structure(self, tmp_path):
        path = _write_wav(tmp_path, duration_s=3.0)
        result = self._ingest_with_mocks(path, "audio_sess",
                                         "Structure session id test content.")
        for doc in result:
            assert doc.structure.get("session_id") == "audio_sess"
