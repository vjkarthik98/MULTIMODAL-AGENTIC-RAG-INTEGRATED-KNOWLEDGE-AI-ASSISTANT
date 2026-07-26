"""
Unit tests for pure helper functions in app/memory/memory_manager.py.
MemoryManager class itself requires live Redis/Mongo and is not tested here.
"""
import pytest

from app.memory.memory_manager import (
    _apply_sliding_window,
    _dedup,
    _hash_msg,
    _normalize,
    _valid_content,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_collapses_inner_spaces(self):
        assert _normalize("a  b  c") == "a b c"

    def test_none_returns_empty(self):
        assert _normalize(None) == ""

    def test_strips_null_bytes(self):
        result = _normalize("hel\x00lo")
        assert "\x00" not in result

    def test_strips_bom(self):
        result = _normalize("﻿hello")
        assert not result.startswith("﻿")


# ---------------------------------------------------------------------------
# _hash_msg
# ---------------------------------------------------------------------------

class TestHashMsg:

    def test_returns_64_char_hex(self):
        h = _hash_msg({"role": "user", "content": "hello"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self):
        h1 = _hash_msg({"role": "user", "content": "hello"})
        h2 = _hash_msg({"role": "user", "content": "world"})
        assert h1 != h2

    def test_same_message_stable_hash(self):
        msg = {"role": "assistant", "content": "I can help."}
        assert _hash_msg(msg) == _hash_msg(msg)

    def test_different_roles_different_hash(self):
        h1 = _hash_msg({"role": "user", "content": "hello"})
        h2 = _hash_msg({"role": "assistant", "content": "hello"})
        assert h1 != h2


# ---------------------------------------------------------------------------
# _valid_content
# ---------------------------------------------------------------------------

class TestValidContent:

    def test_valid_string(self):
        assert _valid_content("hello world") is True

    def test_empty_string_invalid(self):
        assert _valid_content("") is False

    def test_whitespace_only_invalid(self):
        assert _valid_content("   ") is False

    def test_too_short_invalid(self):
        assert _valid_content("ab") is False

    def test_non_string_invalid(self):
        assert _valid_content(None) is False
        assert _valid_content(123) is False


# ---------------------------------------------------------------------------
# _dedup
# ---------------------------------------------------------------------------

class TestDedup:

    def test_empty_list(self):
        assert _dedup([]) == []

    def test_no_duplicates_unchanged(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _dedup(msgs)
        assert len(result) == 2

    def test_duplicates_removed(self):
        msg = {"role": "user", "content": "duplicate message"}
        result = _dedup([msg, msg, msg])
        assert len(result) == 1

    def test_order_preserved(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = _dedup(msgs)
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"

    def test_returns_list(self):
        assert isinstance(_dedup([]), list)


# ---------------------------------------------------------------------------
# _apply_sliding_window
# ---------------------------------------------------------------------------

class TestApplySlidingWindow:

    def test_returns_list(self):
        assert isinstance(_apply_sliding_window([]), list)

    def test_empty_history(self):
        assert _apply_sliding_window([]) == []

    def test_short_history_kept(self):
        msgs = [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "reply"},
        ]
        result = _apply_sliding_window(msgs)
        assert len(result) >= 1

    def test_very_long_history_trimmed(self):
        # Create history that far exceeds the token budget
        msgs = [
            {"role": "user", "content": "word " * 500}
            for _ in range(50)
        ]
        result = _apply_sliding_window(msgs)
        assert len(result) < len(msgs)

    def test_most_recent_messages_kept(self):
        msgs = [
            {"role": "user", "content": f"message {i} " * 200}
            for i in range(20)
        ]
        result = _apply_sliding_window(msgs)
        if result:
            # Last message should be present (most recent)
            assert result[-1]["content"] == msgs[-1]["content"]
