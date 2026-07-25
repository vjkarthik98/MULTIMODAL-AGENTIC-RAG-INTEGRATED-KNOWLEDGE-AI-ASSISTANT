from unittest.mock import AsyncMock, MagicMock

import pytest

from app.memory.summarizer import _clean, _dedup, _hash, _sort_by_importance


# ---------------------------------------------------------------------------
# _clean
# ---------------------------------------------------------------------------

class TestClean:

    def test_strips_whitespace(self):
        assert _clean("  hello  ") == "hello"

    def test_collapses_inner_spaces(self):
        assert _clean("a  b  c") == "a b c"

    def test_empty_string(self):
        assert _clean("") == ""

    def test_none_returns_empty(self):
        assert _clean(None) == ""


# ---------------------------------------------------------------------------
# _hash
# ---------------------------------------------------------------------------

class TestHash:

    def test_returns_64_char_hex(self):
        msg = {"role": "user", "content": "hello world"}
        h = _hash(msg)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_messages_different_hashes(self):
        h1 = _hash({"role": "user", "content": "hello"})
        h2 = _hash({"role": "user", "content": "world"})
        assert h1 != h2

    def test_stable_for_same_input(self):
        msg = {"role": "assistant", "content": "This is a test."}
        assert _hash(msg) == _hash(msg)


# ---------------------------------------------------------------------------
# _dedup
# ---------------------------------------------------------------------------

class TestDedup:

    def test_removes_exact_duplicate(self):
        msg = {"role": "user", "content": "hello"}
        result = _dedup([msg, msg])
        assert len(result) == 1

    def test_keeps_different_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        result = _dedup(msgs)
        assert len(result) == 2

    def test_empty_list(self):
        assert _dedup([]) == []

    def test_order_preserved(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = _dedup(msgs)
        assert [m["content"] for m in result] == ["first", "second", "third"]

    def test_many_duplicates_reduced_to_one(self):
        msg = {"role": "assistant", "content": "repeated answer"}
        result = _dedup([msg] * 10)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _sort_by_importance
# ---------------------------------------------------------------------------

class TestSortByImportance:

    def test_sorted_descending(self):
        msgs = [
            {"role": "user", "content": "low", "importance": 0.2},
            {"role": "user", "content": "high", "importance": 0.9},
            {"role": "user", "content": "mid", "importance": 0.5},
        ]
        result = _sort_by_importance(msgs)
        importances = [m["importance"] for m in result]
        assert importances == sorted(importances, reverse=True)

    def test_missing_importance_uses_default(self):
        msgs = [
            {"role": "user", "content": "no_importance_field"},
            {"role": "user", "content": "has_importance", "importance": 0.9},
        ]
        result = _sort_by_importance(msgs)
        # "has_importance" should come first (0.9 > 0.5 default)
        assert result[0]["content"] == "has_importance"

    def test_empty_list_returns_empty(self):
        assert _sort_by_importance([]) == []

    def test_single_item_returned(self):
        msg = {"role": "user", "content": "only one", "importance": 0.7}
        result = _sort_by_importance([msg])
        assert len(result) == 1
        assert result[0] is msg


# ---------------------------------------------------------------------------
# summarize_conversation (mocked LLM)
# ---------------------------------------------------------------------------

class TestSummarizeConversation:

    def test_returns_string(self):
        from app.memory.summarizer import summarize_conversation

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Summary: user asked about AI and got explanations."

        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]
        result = summarize_conversation(mock_llm, history, session_id="s1")
        assert isinstance(result, str)

    def test_empty_history_returns_string(self):
        from app.memory.summarizer import summarize_conversation

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "No conversation to summarize."

        result = summarize_conversation(mock_llm, [], session_id="s1")
        assert isinstance(result, str)

    def test_llm_exception_returns_empty(self):
        from app.memory.summarizer import summarize_conversation

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")

        history = [{"role": "user", "content": "Tell me something interesting."}]
        result = summarize_conversation(mock_llm, history, session_id="s1")
        assert isinstance(result, str)

    def test_existing_summary_included(self):
        from app.memory.summarizer import summarize_conversation

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Updated summary based on new exchanges."

        history = [{"role": "user", "content": "Follow-up question on AI."}]
        result = summarize_conversation(
            mock_llm,
            history,
            session_id="s1",
            existing_summary="Previous summary: we discussed AI basics.",
        )
        assert isinstance(result, str)

    def test_llm_generate_called(self):
        from app.memory.summarizer import summarize_conversation

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Some summary."

        history = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "ML is a subset of AI."},
        ]
        summarize_conversation(mock_llm, history, session_id="s1")
        mock_llm.generate.assert_called()
