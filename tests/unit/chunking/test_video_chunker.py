"""Unit tests for app/chunking/video_chunker.py — error_markers wiring for
low-confidence transcription (hallucination-reduction initiative, Phase 5,
2026-08-13). Mirrors audio_chunker.py's identical wiring; both share
app.chunking.av_shared._assemble_chunks for the confidence aggregation
itself (see test_av_shared.py for that logic's own tests).
"""

from __future__ import annotations

from unittest.mock import patch

from app.chunking.video_chunker import VideoChunker
from app.ingestion.schema import RawExtract, UniversalMetadata


def _video_raw_extract(file_path: str) -> RawExtract:
    return RawExtract(
        text="",
        extract_type="video_raw",
        raw_source_ref="video:call.mp4",
        extra={"file_path": file_path, "duration_seconds": 2.0},
    )


def _meta() -> UniversalMetadata:
    return UniversalMetadata(source_path="/tmp/call.mp4", modality="video")


def _words(avg_logprob: float) -> list[dict]:
    return [
        {"word": "Revenue", "start": 0.0, "end": 0.5,
         "avg_logprob": avg_logprob, "no_speech_prob": 0.01},
        {"word": "grew", "start": 0.5, "end": 1.0,
         "avg_logprob": avg_logprob, "no_speech_prob": 0.01},
        {"word": "ten", "start": 1.0, "end": 1.5,
         "avg_logprob": avg_logprob, "no_speech_prob": 0.01},
        {"word": "percent.", "start": 1.5, "end": 2.0,
         "avg_logprob": avg_logprob, "no_speech_prob": 0.01},
    ]


class TestVideoChunkerErrorMarkers:
    def test_low_confidence_transcription_flagged(self, tmp_path):
        video_file = tmp_path / "call.mp4"
        video_file.write_bytes(b"\x00" * 16)

        with patch("app.chunking.video_chunker._extract_audio", return_value=True), \
             patch(
                 "app.chunking.video_chunker._transcribe_video_audio",
                 return_value=_words(avg_logprob=-0.9),
             ), \
             patch("app.chunking.audio_chunker.diarize", return_value=[]), \
             patch(
                 "app.chunking.video_chunker._measure_snr",
                 return_value={"snr": None, "snr_degraded": False, "clipping_detected": False},
             ), \
             patch("app.ingestion.video_ingest.extract_frames", return_value=[]):
            docs = VideoChunker().chunk([_video_raw_extract(str(video_file))], _meta())

        assert len(docs) == 1
        assert docs[0].structure["error_markers"] == ["low_transcription_confidence"]
        assert docs[0].structure["hallucination_risk"] == "high"

    def test_high_confidence_transcription_not_flagged(self, tmp_path):
        video_file = tmp_path / "call.mp4"
        video_file.write_bytes(b"\x00" * 16)

        with patch("app.chunking.video_chunker._extract_audio", return_value=True), \
             patch(
                 "app.chunking.video_chunker._transcribe_video_audio",
                 return_value=_words(avg_logprob=-0.02),
             ), \
             patch("app.chunking.audio_chunker.diarize", return_value=[]), \
             patch(
                 "app.chunking.video_chunker._measure_snr",
                 return_value={"snr": None, "snr_degraded": False, "clipping_detected": False},
             ), \
             patch("app.ingestion.video_ingest.extract_frames", return_value=[]):
            docs = VideoChunker().chunk([_video_raw_extract(str(video_file))], _meta())

        assert len(docs) == 1
        assert docs[0].structure["error_markers"] == []
        assert docs[0].structure["hallucination_risk"] == "low"
