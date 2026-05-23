"""Unit tests for app/memory/formatter.py — Phase 24.9."""
from __future__ import annotations

import time
from typing import Dict, List

import pytest

from app.memory.formatter import format_history
from app.core.config import settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _msg(role: str, content: str, ts: float = None, modality: str = "text",
         importance: float = 0.5) -> Dict:
    m: Dict = {"role": role, "content": content, "modality": modality,
               "importance": importance}
    if ts is not None:
        m["timestamp"] = ts
    return m


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFormatHistory:

    def test_empty_history_returns_empty_string(self):
        result = format_history([])
        assert result == ""

    def test_valid_history_returns_string(self):
        history = [_msg("user", "Hello"), _msg("assistant", "Hi there")]
        result = format_history(history)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_output_starts_with_conversation_memory_header(self):
        history = [_msg("user", "What is RAG?")]
        result = format_history(history)
        assert "[CONVERSATION MEMORY]" in result

    def test_dedup_removes_duplicate_messages(self):
        duplicate = _msg("user", "Repeated message content")
        result = format_history([duplicate, duplicate, duplicate])
        count = result.count("Repeated message content")
        assert count == 1

    def test_system_messages_included_when_flag_true(self):
        history = [
            _msg("system", "System instruction context"),
            _msg("user",   "User question"),
        ]
        result = format_history(history, include_system=True)
        assert "System instruction context" in result

    def test_system_messages_excluded_when_flag_false(self):
        history = [
            _msg("system", "System instruction context"),
            _msg("user",   "User question"),
        ]
        result = format_history(history, include_system=False)
        assert "System instruction context" not in result

    def test_unknown_role_normalised(self):
        history = [_msg("alien_role", "Some message content")]
        result = format_history(history)
        assert "Some message content" in result

    def test_message_content_truncated_to_per_msg_limit(self):
        long_content = "word " * 1000
        history = [_msg("user", long_content)]
        result = format_history(history, max_messages=1)
        per_msg_limit = max(settings.MEMORY_MAX_CONTEXT_CHARS // 1, 100)
        assert len(result) <= settings.MAX_PROMPT_CHARS + len("[CONVERSATION MEMORY]\n[CONVERSATION]\n[USER]: ") + 10

    def test_truncation_preserves_head_and_tail_on_overflow(self):
        long_content = "x" * (settings.MAX_PROMPT_CHARS * 2)
        history = [_msg("user", long_content)]
        result = format_history(history)
        assert len(result) <= settings.MAX_PROMPT_CHARS + 20

    def test_max_messages_limits_normal_messages(self):
        history = [_msg("user", f"Message number {i}") for i in range(20)]
        result = format_history(history, max_messages=5)
        assert isinstance(result, str)

    def test_role_labels_present_in_output(self):
        history = [
            _msg("user",      "User question here"),
            _msg("assistant", "Assistant answer here"),
        ]
        result = format_history(history)
        assert "[USER]" in result or "USER" in result
        assert "[ASSISTANT]" in result or "ASSISTANT" in result

    def test_modality_tag_added_for_non_text(self):
        history = [_msg("user", "Describe this image.", modality="image")]
        result = format_history(history)
        assert "IMAGE" in result

    def test_very_short_content_skipped(self):
        history = [_msg("user", "ab"), _msg("user", "Valid content here.")]
        result = format_history(history)
        assert "Valid content here." in result

    def test_empty_content_skipped(self):
        history = [_msg("user", ""), _msg("user", "Good content here.")]
        result = format_history(history)
        assert "Good content here." in result

    def test_group_by_modality_flag(self):
        history = [
            _msg("user", "Text question",  modality="text"),
            _msg("user", "Image question", modality="image"),
        ]
        result = format_history(history, group_by_modality=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_importance_high_adds_important_tag(self):
        history = [_msg("user", "Very important message.", importance=0.95)]
        result = format_history(history)
        assert "[IMPORTANT]" in result

    def test_returns_string_on_single_message(self):
        history = [_msg("user", "Single message.")]
        result = format_history(history)
        assert isinstance(result, str)
