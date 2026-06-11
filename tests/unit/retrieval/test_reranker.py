"""Unit tests for app/retrieval/reranker.py — Phase 24.10."""
from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_reranker() -> "Reranker":
    """Create a Reranker with a mocked cross-encoder model."""
    from app.retrieval.reranker import Reranker
    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=np.array([0.8, 0.5, 0.3]))
    with patch("app.core.model_loader.model_loader.get_reranker",
               return_value=mock_model):
        return Reranker()


def _make_doc(text: str, score: float = 0.5, modality: str = "text",
              chunk_index: int = 0) -> Dict:
    return {
        "text": text,
        "score": score,
        "metadata": {
            "modality": modality,
            "structure": {"chunk_index": chunk_index},
        },
        "embedding": None,
    }


# ── Reranker initialization ───────────────────────────────────────────────────

class TestRerankerInit:

    def test_init_loads_model(self):
        r = _make_reranker()
        assert r.model is not None

    def test_top_k_configured(self):
        r = _make_reranker()
        from app.core.config import settings
        assert r.top_k == settings.RERANK_TOP_K

    def test_max_inputs_configured(self):
        r = _make_reranker()
        from app.core.config import settings
        assert r.max_inputs == settings.RERANK_MAX_INPUT


# ── _normalize_query ──────────────────────────────────────────────────────────

class TestRerankerNormalizeQuery:

    def test_strips_whitespace(self):
        r = _make_reranker()
        assert r._normalize_query("  hello  ") == "hello"

    def test_collapses_spaces(self):
        r = _make_reranker()
        assert r._normalize_query("hello   world") == "hello world"

    def test_empty_string_returns_empty(self):
        r = _make_reranker()
        assert r._normalize_query("") == ""


# ── _clean_text ───────────────────────────────────────────────────────────────

class TestRerankerCleanText:

    def test_strips_extra_spaces(self):
        r = _make_reranker()
        result = r._clean_text("hello   world")
        assert result == "hello world"

    def test_handles_none_as_empty(self):
        r = _make_reranker()
        result = r._clean_text(None)
        assert result == ""


# ── _context ──────────────────────────────────────────────────────────────────

class TestRerankerContext:

    def test_text_modality_no_header(self):
        r = _make_reranker()
        doc = {"text": "Plain text document content.", "metadata": {"modality": "text"}}
        result = r._context(doc)
        assert "[TEXT]" not in result
        assert "Plain text document content." in result

    def test_image_modality_has_header(self):
        r = _make_reranker()
        doc = {"text": "Image caption content.", "metadata": {"modality": "image"}}
        result = r._context(doc)
        assert "[IMAGE]" in result

    def test_audio_modality_has_header(self):
        r = _make_reranker()
        doc = {"text": "Audio transcript content.", "metadata": {"modality": "audio"}}
        result = r._context(doc)
        assert "[AUDIO]" in result

    def test_page_number_included(self):
        r = _make_reranker()
        doc = {"text": "PDF page content.", "metadata": {"modality": "pdf", "page": 5}}
        result = r._context(doc)
        assert "PG:5" in result

    def test_empty_text_doc_returns_empty(self):
        r = _make_reranker()
        doc = {"text": "", "metadata": {"modality": "text"}}
        result = r._context(doc)
        assert result == ""


# ── _filter ───────────────────────────────────────────────────────────────────

class TestRerankerFilter:

    def test_empty_text_filtered_out(self):
        r = _make_reranker()
        valid = "This is a valid document with sufficient length for reranking tests."
        docs = [{"text": "", "metadata": {}}, {"text": valid, "metadata": {}}]
        result = r._filter(docs)
        assert len(result) == 1

    def test_short_text_filtered_out(self):
        r = _make_reranker()
        docs = [{"text": "ab", "metadata": {}}]
        result = r._filter(docs)
        assert result == []

    def test_valid_docs_pass_through(self):
        r = _make_reranker()
        docs = [_make_doc("Valid text content for reranking test document with adequate length.")]
        result = r._filter(docs)
        assert len(result) == 1


# ── _normalize_scores ─────────────────────────────────────────────────────────

class TestRerankerNormalizeScores:

    def test_normalizes_to_zero_one(self):
        r = _make_reranker()
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        result = r._normalize_scores(scores)
        assert float(result.max()) <= 1.0 + 1e-6
        assert float(result.min()) >= 0.0 - 1e-6

    def test_empty_array_returned(self):
        r = _make_reranker()
        result = r._normalize_scores(np.array([]))
        assert len(result) == 0

    def test_all_equal_scores_clipped(self):
        r = _make_reranker()
        scores = np.array([5.0, 5.0, 5.0])
        result = r._normalize_scores(scores)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_nan_handled(self):
        r = _make_reranker()
        scores = np.array([float("nan"), 1.0, 2.0])
        result = r._normalize_scores(scores)
        assert not any(np.isnan(result))


# ── _valid_score ──────────────────────────────────────────────────────────────

class TestRerankerValidScore:

    def test_positive_score_valid(self):
        r = _make_reranker()
        assert r._valid_score(0.5) is True

    def test_zero_score_valid(self):
        # 0.0 is valid: the sigmoid floor filter (not _valid_score) handles removal
        # of off-topic documents. Silently dropping 0.0 would discard the
        # min-normalized chunk even when its sigmoid score passes the floor.
        r = _make_reranker()
        assert r._valid_score(0.0) is True

    def test_negative_score_invalid(self):
        r = _make_reranker()
        assert r._valid_score(-0.1) is False

    def test_nan_score_invalid(self):
        r = _make_reranker()
        assert r._valid_score(float("nan")) is False

    def test_inf_score_invalid(self):
        r = _make_reranker()
        assert r._valid_score(float("inf")) is False


# ── _dedup ─────────────────────────────────────────────────────────────────────

class TestRerankerDedup:

    def test_identical_docs_deduplicated(self):
        r = _make_reranker()
        doc = _make_doc("Identical content for dedup test.")
        result = r._dedup([doc, doc, doc], top_k=5)
        assert len(result) == 1

    def test_different_docs_kept(self):
        r = _make_reranker()
        docs = [_make_doc(f"Document {i} with unique content.") for i in range(3)]
        result = r._dedup(docs, top_k=5)
        assert len(result) == 3

    def test_top_k_limits_output(self):
        r = _make_reranker()
        docs = [_make_doc(f"Document {i} for top_k test.") for i in range(10)]
        result = r._dedup(docs, top_k=3)
        assert len(result) <= 3


# ── _text_overlap ─────────────────────────────────────────────────────────────

class TestRerankerTextOverlap:

    def test_identical_returns_one(self):
        r = _make_reranker()
        t = "hello world"
        assert abs(r._text_overlap(t, t) - 1.0) < 1e-6

    def test_no_overlap_returns_zero(self):
        r = _make_reranker()
        assert r._text_overlap("alpha beta", "gamma delta") == 0.0

    def test_empty_string_returns_zero(self):
        r = _make_reranker()
        assert r._text_overlap("", "hello world") == 0.0


# ── rerank — end to end ───────────────────────────────────────────────────────

class TestRerankerRerank:

    def test_empty_query_returns_empty(self):
        r = _make_reranker()
        docs = [_make_doc("Some document text content.")]
        result = r.rerank("", docs, session_id="s1")
        assert result == []

    def test_empty_docs_returns_empty(self):
        r = _make_reranker()
        result = r.rerank("query", [], session_id="s1")
        assert result == []

    def test_returns_list(self):
        r = _make_reranker()
        docs = [
            _make_doc("Retrieval augmented generation document."),
            _make_doc("Vector search embedding similarity."),
            _make_doc("BM25 keyword-based sparse retrieval."),
        ]
        r.model.predict = MagicMock(return_value=np.array([0.9, 0.6, 0.3]))
        result = r.rerank("retrieval", docs, session_id="s1")
        assert isinstance(result, list)

    def test_result_has_score_key(self):
        r = _make_reranker()
        docs = [_make_doc("Document with score key test.", score=0.8)]
        r.model.predict = MagicMock(return_value=np.array([0.9]))
        result = r.rerank("document", docs, session_id="s1")
        for hit in result:
            assert "score" in hit

    def test_result_sorted_by_score_descending(self):
        r = _make_reranker()
        docs = [
            _make_doc("Lower score document content.", score=0.3),
            _make_doc("Higher score document content.", score=0.9),
        ]
        r.model.predict = MagicMock(return_value=np.array([0.2, 0.8]))
        result = r.rerank("document", docs, session_id="s1")
        if len(result) > 1:
            scores = [d["score"] for d in result]
            assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self):
        r = _make_reranker()
        docs = [_make_doc(f"Document {i} for testing.", score=0.5) for i in range(8)]
        r.model.predict = MagicMock(return_value=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]))
        result = r.rerank("document", docs, top_k=3, session_id="s1")
        assert len(result) <= 3

    def test_model_failure_falls_back_to_score_sort(self):
        r = _make_reranker()
        r.model.predict = MagicMock(side_effect=RuntimeError("model failed"))
        docs = [
            _make_doc("High score doc.", score=0.9),
            _make_doc("Low score doc.", score=0.1),
        ]
        result = r.rerank("doc", docs, session_id="s1")
        assert isinstance(result, list)
        if len(result) > 1:
            assert result[0]["score"] >= result[1]["score"]


# ── _fallback ─────────────────────────────────────────────────────────────────

class TestRerankerFallback:

    def test_fallback_sorts_by_score(self):
        r = _make_reranker()
        docs = [
            _make_doc("Low score content.", score=0.2),
            _make_doc("High score content.", score=0.9),
            _make_doc("Mid score content.", score=0.5),
        ]
        result = r._fallback(docs, top_k=3)
        scores = [d["score"] for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_fallback_respects_top_k(self):
        r = _make_reranker()
        docs = [_make_doc(f"Doc {i}.", score=float(i) / 10) for i in range(10)]
        result = r._fallback(docs, top_k=3)
        assert len(result) <= 3


# ── health check ──────────────────────────────────────────────────────────────

class TestRerankerHealthCheck:

    def test_returns_dict(self):
        r = _make_reranker()
        result = r.health_check()
        assert isinstance(result, dict)

    def test_model_loaded_true(self):
        r = _make_reranker()
        assert r.health_check()["model_loaded"] is True

    def test_has_top_k_key(self):
        r = _make_reranker()
        assert "top_k" in r.health_check()

    def test_has_score_threshold_key(self):
        r = _make_reranker()
        assert "score_threshold" in r.health_check()
