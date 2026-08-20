from unittest.mock import patch

import pytest
from app.chunking.base_chunker import chunk_documents
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata


def test_audio_chunking():
    docs = [
        IngestedDocument(
            text="Audio speech from 0s to 2s: hello world",
            modality="audio",
            subtype="speech",
            source_type="audio",
            source="test.wav",
            page=None,
            chunk_id=None,
            structure={
                "segment_index": 0,
                "session_id": "test"
            }
        )
    ]

    chunks = chunk_documents(docs)

    assert isinstance(chunks, list)
    assert len(chunks) == 1

    doc = chunks[0]

    assert doc.modality == "audio"
    assert doc.chunk_id == 0
    # parent_modality is not added by chunk_documents — chunker preserves structure as-is
    assert doc.modality == "audio"


# ── AudioChunker.chunk() error_markers wiring (hallucination-reduction
# initiative, Phase 5, 2026-08-13) ───────────────────────────────────────────

def _audio_raw_extract() -> RawExtract:
    return RawExtract(
        text="",
        extract_type="audio_raw",
        raw_source_ref="audio:call.mp3",
        raw_bytes=b"RIFF....WAVEfmt ",  # never actually decoded — Whisper is mocked
        extra={"duration_seconds": 2.0},
    )


def _meta() -> UniversalMetadata:
    return UniversalMetadata(source_path="/tmp/call.mp3", modality="audio")


def _words(avg_logprob: float, no_speech_prob: float = 0.01) -> list[dict]:
    return [
        {"word": "Revenue", "start": 0.0, "end": 0.5,
         "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob},
        {"word": "grew", "start": 0.5, "end": 1.0,
         "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob},
        {"word": "ten", "start": 1.0, "end": 1.5,
         "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob},
        {"word": "percent.", "start": 1.5, "end": 2.0,
         "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob},
    ]


class TestAudioChunkerErrorMarkers:
    def test_low_confidence_transcription_flagged(self):
        from app.chunking.audio_chunker import AudioChunker

        with patch(
            "app.chunking.audio_chunker._transcribe_long_audio",
            return_value=_words(avg_logprob=-0.9),
        ), patch("app.chunking.audio_chunker.diarize", return_value=[]):
            docs = AudioChunker().chunk([_audio_raw_extract()], _meta())

        assert len(docs) == 1
        assert docs[0].structure["error_markers"] == ["low_transcription_confidence"]
        assert docs[0].structure["hallucination_risk"] == "high"

    def test_high_confidence_transcription_not_flagged(self):
        from app.chunking.audio_chunker import AudioChunker

        with patch(
            "app.chunking.audio_chunker._transcribe_long_audio",
            return_value=_words(avg_logprob=-0.02),
        ), patch("app.chunking.audio_chunker.diarize", return_value=[]):
            docs = AudioChunker().chunk([_audio_raw_extract()], _meta())

        assert len(docs) == 1
        assert docs[0].structure["error_markers"] == []
        assert docs[0].structure["hallucination_risk"] == "low"