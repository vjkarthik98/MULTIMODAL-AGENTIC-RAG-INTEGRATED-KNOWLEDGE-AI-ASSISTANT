"""Unit tests for app/chunking/av_shared.py::_assemble_chunks — the
hallucination-risk aggregation added in the hallucination-reduction
initiative (Phase 5, 2026-08-13). Shared by both audio_chunker.py and
video_chunker.py; both feed it words carrying avg_logprob/no_speech_prob
from their own (per-modality) Whisper-calling code.
"""

from __future__ import annotations

from app.chunking.av_shared import _assemble_chunks


def _word(text, start, end, avg_logprob=-0.05, no_speech_prob=0.01):
    return {
        "word": text,
        "start": start,
        "end": end,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
    }


class TestAssembleChunksConfidenceAggregation:
    def test_high_confidence_words_yield_low_risk(self):
        words = [_word(f"w{i}", i, i + 1, avg_logprob=-0.02, no_speech_prob=0.01) for i in range(5)]
        chunks = _assemble_chunks(words, diarization=[], role_map={}, min_words=1, max_words=10)
        assert len(chunks) == 1
        assert chunks[0]["hallucination_risk"] == "low"
        assert chunks[0]["confidence"] > 0.9

    def test_low_confidence_words_yield_high_risk(self):
        # avg_logprob=-0.8 -> confidence = 1.0 + (-0.8) = 0.2 < 0.4 threshold
        words = [_word(f"w{i}", i, i + 1, avg_logprob=-0.8, no_speech_prob=0.1) for i in range(5)]
        chunks = _assemble_chunks(words, diarization=[], role_map={}, min_words=1, max_words=10)
        assert len(chunks) == 1
        assert chunks[0]["hallucination_risk"] == "high"

    def test_worst_word_in_chunk_dominates_risk(self):
        # One badly-transcribed word amid otherwise-clean ones should still
        # drag the whole chunk's risk down (min confidence wins).
        words = [
            _word("clean1", 0, 1, avg_logprob=-0.02, no_speech_prob=0.01),
            _word("clean2", 1, 2, avg_logprob=-0.02, no_speech_prob=0.01),
            _word("garbled", 2, 3, avg_logprob=-0.9, no_speech_prob=0.05),
            _word("clean3", 3, 4, avg_logprob=-0.02, no_speech_prob=0.01),
        ]
        chunks = _assemble_chunks(words, diarization=[], role_map={}, min_words=1, max_words=10)
        assert len(chunks) == 1
        assert chunks[0]["hallucination_risk"] == "high"

    def test_missing_confidence_fields_default_to_low_risk(self):
        # Words without avg_logprob/no_speech_prob (e.g. an older caller, or
        # a fallback path) must not crash and must not be treated as risky.
        words = [{"word": f"w{i}", "start": i, "end": i + 1} for i in range(3)]
        chunks = _assemble_chunks(words, diarization=[], role_map={}, min_words=1, max_words=10)
        assert len(chunks) == 1
        assert chunks[0]["hallucination_risk"] == "low"
        assert chunks[0]["confidence"] == 1.0

    def test_confidence_resets_between_chunks(self):
        # A high-risk word in chunk 1 must not leak into chunk 2's aggregate
        # after a speaker-change flush.
        words = [
            _word("bad1", 0, 1, avg_logprob=-0.9, no_speech_prob=0.05),
            _word("bad2", 1, 2, avg_logprob=-0.9, no_speech_prob=0.05),
            _word("good1", 10, 11, avg_logprob=-0.02, no_speech_prob=0.01),
            _word("good2", 11, 12, avg_logprob=-0.02, no_speech_prob=0.01),
        ]
        # Force a speaker change between word 2 and 3 via diarization.
        diarization = [(0.0, 5.0, "spk_a"), (5.0, 20.0, "spk_b")]
        chunks = _assemble_chunks(words, diarization=diarization, role_map={}, min_words=1, max_words=10)
        assert len(chunks) == 2
        assert chunks[0]["hallucination_risk"] == "high"
        assert chunks[1]["hallucination_risk"] == "low"

    def test_empty_words_returns_empty(self):
        assert _assemble_chunks([], diarization=[], role_map={}) == []
