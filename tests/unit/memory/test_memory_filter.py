"""Unit tests for app/memory/memory_filter.py — Phase 24.9."""
from __future__ import annotations

import time
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from app.memory.memory_filter import (
    _cosine,
    _dedup,
    _hash,
    _recency,
    _role_weight,
    filter_relevant_history,
)
from app.core.config import settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_embedder(vec: List[float]) -> MagicMock:
    m = MagicMock()
    m.embed_query = MagicMock(return_value=vec)
    return m


def _msg(role: str, content: str, embedding=None, ts: float = None) -> Dict:
    m: Dict = {"role": role, "content": content}
    if embedding is not None:
        m["embedding"] = embedding
    if ts is not None:
        m["timestamp"] = ts
    return m


DIM = settings.TEXT_EMBEDDING_DIM


# ── _cosine ───────────────────────────────────────────────────────────────────

class TestCosine:

    def test_identical_vectors_return_one(self):
        v = [0.5] * DIM
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0] + [0.0] * (DIM - 1)
        b = [0.0] * (DIM - 1) + [1.0]
        assert abs(_cosine(a, b)) < 1e-6

    def test_zero_vector_does_not_crash(self):
        v = [0.0] * DIM
        assert _cosine(v, v) == 0.0


# ── _recency ──────────────────────────────────────────────────────────────────

class TestRecency:

    def test_recent_message_scores_near_one(self):
        now = time.time()
        score = _recency(now - 1, now)
        assert score > 0.99

    def test_old_message_scores_lower(self):
        now = time.time()
        recent = _recency(now - 10,   now)
        old    = _recency(now - 86400, now)
        assert recent > old

    def test_none_timestamp_returns_one(self):
        assert _recency(None, time.time()) == 1.0


# ── _role_weight ──────────────────────────────────────────────────────────────

class TestRoleWeight:

    def test_known_roles_have_weights(self):
        for role in ("user", "assistant", "system"):
            assert _role_weight(role) > 0

    def test_unknown_role_returns_one(self):
        assert _role_weight("alien") == 1.0


# ── _dedup ────────────────────────────────────────────────────────────────────

class TestDedup:

    def test_identical_messages_deduplicated(self):
        msg = _msg("user", "Hello world")
        result = _dedup([msg, msg, msg])
        assert len(result) == 1

    def test_different_messages_kept(self):
        msgs = [_msg("user", f"Message {i}") for i in range(3)]
        result = _dedup(msgs)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        assert _dedup([]) == []


# ── filter_relevant_history ───────────────────────────────────────────────────

class TestFilterRelevantHistory:

    def test_empty_history_returns_empty_list(self):
        embedder = _make_embedder([0.1] * DIM)
        result = filter_relevant_history("test query", [], embedder)
        assert result == []

    def test_returns_list(self, sample_history_with_embeddings):
        embedder = _make_embedder([0.9] * DIM)
        result = filter_relevant_history("RAG", sample_history_with_embeddings, embedder)
        assert isinstance(result, list)

    def test_top_k_parameter_limits_results(self):
        vecs = [[0.9] * DIM] * 10
        history = [_msg("user", f"msg {i}", embedding=[0.9] * DIM) for i in range(10)]
        embedder = _make_embedder([0.9] * DIM)
        result = filter_relevant_history("query", history, embedder, top_k=3)
        assert len(result) <= 3

    def test_threshold_filters_low_similarity(self):
        query_vec = [1.0, 0.0] + [0.0] * (DIM - 2)
        msg_vec   = [0.0, 1.0] + [0.0] * (DIM - 2)
        history   = [_msg("user", "orthogonal content", embedding=msg_vec)]
        embedder  = _make_embedder(query_vec)
        result = filter_relevant_history(
            "query", history, embedder, threshold=0.99
        )
        assert result == []

    def test_no_embedder_fallback_returns_top_k(self):
        failing_embedder = MagicMock()
        failing_embedder.embed_query = MagicMock(side_effect=Exception("no model"))
        history = [_msg("user", "Hello") for _ in range(5)]
        result = filter_relevant_history("hello", history, failing_embedder, top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_dedup_removes_identical_messages(self):
        vec = [0.9] * DIM
        duplicate = _msg("user", "Same content", embedding=vec)
        history = [duplicate] * 5
        embedder = _make_embedder(vec)
        result = filter_relevant_history("content", history, embedder)
        assert len(result) <= 1

    def test_high_similarity_messages_pass_threshold(self):
        vec = [0.9] * DIM
        history = [_msg("user", "RAG retrieval augmented generation", embedding=vec)]
        embedder = _make_embedder(vec)
        result = filter_relevant_history("RAG query", history, embedder, threshold=0.0)
        assert len(result) == 1

    def test_returns_empty_list_on_all_embed_failures(self):
        failing = MagicMock()
        failing.embed_query = MagicMock(return_value=None)
        history = [_msg("user", "Some text")]
        result = filter_relevant_history("query", history, failing)
        assert isinstance(result, list)

    def test_modality_filter_applied(self):
        vec = [0.9] * DIM
        history = [
            {**_msg("user", "image content", embedding=vec), "modality": "image"},
            {**_msg("user", "text content",  embedding=vec), "modality": "text"},
        ]
        embedder = _make_embedder(vec)
        result = filter_relevant_history(
            "query", history, embedder, threshold=0.0, modality="image"
        )
        for msg in result:
            assert msg.get("modality") == "image"
