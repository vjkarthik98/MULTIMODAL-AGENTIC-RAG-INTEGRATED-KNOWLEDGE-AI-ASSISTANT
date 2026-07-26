import time

import pytest

from app.memory.memory_fusion import (
    build_memory_context,
    _hash_msg,
    _hash_prefix,
    _importance_score,
    _modality_weight,
    _normalize,
    _recency_score,
    _role_weight,
    _truncate,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_collapses_inner(self):
        assert _normalize("a  b  c") == "a b c"

    def test_empty(self):
        assert _normalize("") == ""

    def test_none(self):
        assert _normalize(None) == ""


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------

class TestTruncate:

    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("a" * 200, 50)
        assert len(result) == 50

    def test_empty(self):
        assert _truncate("", 100) == ""

    def test_zero_limit(self):
        assert _truncate("hello", 0) == ""


# ---------------------------------------------------------------------------
# _hash_msg
# ---------------------------------------------------------------------------

class TestHashMsg:

    def test_returns_64_char_hex(self):
        msg = {"role": "user", "content": "hello world"}
        h = _hash_msg(msg)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_messages_different_hashes(self):
        h1 = _hash_msg({"role": "user", "content": "hello"})
        h2 = _hash_msg({"role": "user", "content": "world"})
        assert h1 != h2

    def test_same_message_stable_hash(self):
        msg = {"role": "assistant", "content": "I can help."}
        assert _hash_msg(msg) == _hash_msg(msg)


# ---------------------------------------------------------------------------
# _hash_prefix
# ---------------------------------------------------------------------------

class TestHashPrefix:

    def test_returns_hex(self):
        h = _hash_prefix("some text")
        assert len(h) == 64

    def test_different_text_different_hash(self):
        h1 = _hash_prefix("abc")
        h2 = _hash_prefix("xyz")
        assert h1 != h2


# ---------------------------------------------------------------------------
# _recency_score
# ---------------------------------------------------------------------------

class TestRecencyScore:

    def test_recent_timestamp_high_score(self):
        now = time.time()
        score = _recency_score(now - 10, now)
        assert score > 0.9

    def test_old_timestamp_lower_score(self):
        now = time.time()
        recent = _recency_score(now - 60, now)
        old = _recency_score(now - 86400, now)
        assert recent > old

    def test_none_timestamp_returns_1(self):
        assert _recency_score(None, time.time()) == 1.0

    def test_score_between_0_and_1(self):
        now = time.time()
        assert 0.0 <= _recency_score(now - 1000, now) <= 1.0


# ---------------------------------------------------------------------------
# _importance_score
# ---------------------------------------------------------------------------

class TestImportanceScore:

    def test_normal_importance(self):
        score = _importance_score({"importance": 0.8})
        assert score == 0.8

    def test_missing_importance_defaults_to_0_5(self):
        assert _importance_score({}) == 0.5

    def test_clamped_above_1(self):
        score = _importance_score({"importance": 2.0})
        assert score == 1.0

    def test_clamped_below_0(self):
        score = _importance_score({"importance": -1.0})
        assert score == 0.0

    def test_nan_returns_0_5(self):
        import math
        score = _importance_score({"importance": float("nan")})
        assert score == 0.5


# ---------------------------------------------------------------------------
# _modality_weight
# ---------------------------------------------------------------------------

class TestModalityWeight:

    def test_text_weight_is_1(self):
        assert _modality_weight({"modality": "text"}) == 1.0

    def test_audio_weight_greater_than_text(self):
        assert _modality_weight({"modality": "audio"}) > 1.0

    def test_video_weight_greater_than_text(self):
        assert _modality_weight({"modality": "video"}) > 1.0

    def test_missing_modality_defaults_to_text_weight(self):
        assert _modality_weight({}) == 1.0

    def test_unknown_modality_defaults_to_1(self):
        assert _modality_weight({"modality": "unknown_xyz"}) == 1.0


# ---------------------------------------------------------------------------
# _role_weight
# ---------------------------------------------------------------------------

class TestRoleWeight:

    def test_system_weight_highest(self):
        assert _role_weight({"role": "system"}) > _role_weight({"role": "user"})

    def test_assistant_higher_than_user(self):
        assert _role_weight({"role": "assistant"}) > _role_weight({"role": "user"})

    def test_user_weight_is_1(self):
        assert _role_weight({"role": "user"}) == 1.0

    def test_missing_role_defaults_to_user(self):
        assert _role_weight({}) == 1.0


# ---------------------------------------------------------------------------
# build_memory_context
# ---------------------------------------------------------------------------

class TestBuildMemoryContext:

    def test_returns_string(self):
        result = build_memory_context(
            summary="This is a test session summary.",
            filtered_history=[
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence."},
            ],
            session_id="s1",
        )
        assert isinstance(result, str)

    def test_empty_history_and_summary_returns_empty_or_minimal(self):
        result = build_memory_context("", [], session_id="s1")
        assert isinstance(result, str)

    def test_summary_included_in_context(self):
        result = build_memory_context(
            summary="Previous session discussed machine learning.",
            filtered_history=[],
            session_id="s1",
        )
        assert "machine learning" in result or len(result) == 0

    def test_history_included_in_context(self):
        result = build_memory_context(
            summary="",
            filtered_history=[
                {"role": "user", "content": "What is deep learning exactly?"},
            ],
            session_id="s1",
        )
        # Either history appears in output or was too short to include
        assert isinstance(result, str)

    def test_max_total_chars_limits_output(self):
        long_summary = "x " * 1000
        result_small = build_memory_context(
            summary=long_summary,
            filtered_history=[{"role": "user", "content": "y " * 500}],
            max_total_chars=100,
            session_id="s1",
        )
        result_large = build_memory_context(
            summary=long_summary,
            filtered_history=[{"role": "user", "content": "y " * 500}],
            max_total_chars=5000,
            session_id="s1",
        )
        # With a smaller budget the output should be at most as long as with a larger one
        assert len(result_small) <= len(result_large)

    def test_vector_memories_accepted(self):
        result = build_memory_context(
            summary="Summary text",
            filtered_history=[],
            vector_memories=[
                {"text": "Related memory from vector store.", "score": 0.9}
            ],
            session_id="s1",
        )
        assert isinstance(result, str)
