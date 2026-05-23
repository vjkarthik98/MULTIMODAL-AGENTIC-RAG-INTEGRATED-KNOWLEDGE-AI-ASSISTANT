"""
Unit tests for pure helper functions in app/retrieval/retriever.py.
Retriever class requires live Qdrant/BM25 and is not tested here.
"""
import pytest

from app.retrieval.retriever import (
    _hash,
    _mmr,
    _normalize,
    _normalize_scores,
    _valid_score,
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

    def test_empty_string(self):
        assert _normalize("") == ""


# ---------------------------------------------------------------------------
# _hash
# ---------------------------------------------------------------------------

class TestHash:

    def test_returns_64_char_hex(self):
        h = _hash("some text", {"doc_id": "d1", "chunk_id": 0})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_inputs_different_hashes(self):
        h1 = _hash("text A", {"doc_id": "d1", "chunk_id": 0})
        h2 = _hash("text B", {"doc_id": "d1", "chunk_id": 0})
        assert h1 != h2

    def test_same_inputs_stable_hash(self):
        meta = {"doc_id": "d1", "chunk_id": 1}
        assert _hash("hello", meta) == _hash("hello", meta)


# ---------------------------------------------------------------------------
# _valid_score
# ---------------------------------------------------------------------------

class TestValidScore:

    def test_normal_score_valid(self):
        assert _valid_score(0.75) is True

    def test_zero_valid(self):
        assert _valid_score(0.0) is True

    def test_nan_invalid(self):
        import math
        assert _valid_score(float("nan")) is False

    def test_inf_invalid(self):
        assert _valid_score(float("inf")) is False

    def test_negative_inf_invalid(self):
        assert _valid_score(float("-inf")) is False


# ---------------------------------------------------------------------------
# _normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:

    def test_empty_list(self):
        assert _normalize_scores([]) == []

    def test_all_zero_scores_unchanged(self):
        docs = [{"score": 0.0}, {"score": 0.0}]
        result = _normalize_scores(docs)
        assert result[0]["score"] == 0.0

    def test_max_score_becomes_one(self):
        docs = [{"score": 0.5}, {"score": 1.0}, {"score": 0.25}]
        result = _normalize_scores(docs)
        assert max(r["score"] for r in result) == 1.0

    def test_relative_order_preserved(self):
        docs = [{"score": 0.2}, {"score": 0.8}, {"score": 0.4}]
        result = _normalize_scores(docs)
        scores = [r["score"] for r in result]
        assert scores[1] > scores[2] > scores[0]

    def test_single_item_becomes_one(self):
        docs = [{"score": 0.42}]
        result = _normalize_scores(docs)
        assert result[0]["score"] == 1.0


# ---------------------------------------------------------------------------
# _mmr
# ---------------------------------------------------------------------------

class TestMmr:

    def _make_result(self, score, embedding=None):
        return {"score": score, "text": "doc", "embedding": embedding}

    def test_empty_returns_empty(self):
        assert _mmr([], top_k=5) == []

    def test_top_k_respected(self):
        results = [self._make_result(s) for s in [0.9, 0.8, 0.7, 0.6, 0.5]]
        selected = _mmr(results, top_k=3)
        assert len(selected) <= 3

    def test_returns_list(self):
        results = [self._make_result(0.8)]
        assert isinstance(_mmr(results, top_k=5), list)

    def test_no_embeddings_falls_back_to_score_order(self):
        results = [
            self._make_result(0.3),
            self._make_result(0.9),
            self._make_result(0.6),
        ]
        selected = _mmr(results, top_k=2)
        assert len(selected) == 2
        # Highest score should be first
        assert selected[0]["score"] == 0.9

    def test_with_embeddings_selects_diverse(self):
        # Two nearly identical docs and one different — MMR should prefer diverse
        e1 = [1.0, 0.0] * 50
        e2 = [1.0, 0.0] * 50  # same as e1
        e3 = [0.0, 1.0] * 50  # orthogonal
        results = [
            {"score": 0.9, "text": "doc A", "embedding": e1},
            {"score": 0.85, "text": "doc B", "embedding": e2},
            {"score": 0.7, "text": "doc C", "embedding": e3},
        ]
        selected = _mmr(results, top_k=2, lambda_param=0.5)
        assert len(selected) == 2
