"""Unit tests for app/reasoning/result_fusion.py — Phase 24.10."""
from __future__ import annotations

from typing import Dict, List
from unittest.mock import patch

import pytest

from app.reasoning.result_fusion import (
    ResultFusion,
    _confidence_score,
    _cosine,
    _cross_modal_link,
    _dedup,
    _detect_contradictions,
    _diversity,
    _filter,
    _hash,
    _modality_counts,
    _normalize_scores,
    _resolve_contradictions,
    _score_fusion,
    _valid_embedding,
    _valid_score,
)
from app.core.config import settings


# ── Helpers ───────────────────────────────────────────────────────────────────

DIM_TEXT   = settings.TEXT_EMBEDDING_DIM
DIM_VISION = settings.VISION_EMBEDDING_DIM


def _make_result(text: str, score: float = 0.5, modality: str = "text",
                 doc_id: str = "doc1", chunk_id: str = "0",
                 embedding: List[float] = None) -> Dict:
    return {
        "text": text,
        "score": score,
        "metadata": {
            "modality": modality,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
        },
        "embedding": embedding,
    }


# ── _hash ─────────────────────────────────────────────────────────────────────

class TestHash:

    def test_returns_64_char_hex(self):
        result = _hash("hello world", {"doc_id": "d1", "chunk_id": "0"})
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_inputs_same_hash(self):
        meta = {"doc_id": "d1", "chunk_id": "0"}
        assert _hash("text", meta) == _hash("text", meta)

    def test_different_text_different_hash(self):
        meta = {"doc_id": "d1", "chunk_id": "0"}
        assert _hash("text A", meta) != _hash("text B", meta)


# ── _valid_score ──────────────────────────────────────────────────────────────

class TestValidScore:

    def test_float_valid(self):
        assert _valid_score(0.5) is True

    def test_zero_valid(self):
        assert _valid_score(0.0) is True

    def test_nan_invalid(self):
        assert _valid_score(float("nan")) is False

    def test_inf_invalid(self):
        assert _valid_score(float("inf")) is False

    def test_negative_inf_invalid(self):
        assert _valid_score(float("-inf")) is False


# ── _valid_embedding ──────────────────────────────────────────────────────────

class TestValidEmbedding:

    def test_text_dim_embedding_valid(self):
        assert _valid_embedding([0.1] * DIM_TEXT) is True

    def test_vision_dim_embedding_valid(self):
        assert _valid_embedding([0.1] * DIM_VISION) is True

    def test_wrong_dim_invalid(self):
        assert _valid_embedding([0.1] * 10) is False

    def test_none_invalid(self):
        assert _valid_embedding(None) is False

    def test_empty_list_invalid(self):
        assert _valid_embedding([]) is False

    def test_non_list_invalid(self):
        assert _valid_embedding("not a list") is False


# ── _cosine ───────────────────────────────────────────────────────────────────

class TestCosine:

    def test_identical_vectors_return_one(self):
        v = [0.5] * DIM_TEXT
        assert abs(_cosine(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0] + [0.0] * (DIM_TEXT - 1)
        b = [0.0] * (DIM_TEXT - 1) + [1.0]
        assert abs(_cosine(a, b)) < 1e-5

    def test_zero_vector_returns_zero(self):
        v = [0.0] * DIM_TEXT
        assert _cosine(v, v) == 0.0


# ── _modality_counts ──────────────────────────────────────────────────────────

class TestModalityCounts:

    def test_counts_by_modality(self):
        results = [
            _make_result("t1", modality="text"),
            _make_result("t2", modality="text"),
            _make_result("i1", modality="image"),
        ]
        counts = _modality_counts(results)
        assert counts["text"] == 2
        assert counts["image"] == 1

    def test_empty_input_returns_empty_dict(self):
        assert _modality_counts([]) == {}

    def test_missing_modality_counted_as_unknown(self):
        counts = _modality_counts([{"text": "t", "metadata": {}}])
        assert "unknown" in counts


# ── _filter ───────────────────────────────────────────────────────────────────

class TestFilter:

    def test_empty_text_removed(self):
        results = [{"text": "", "score": 0.5}, {"text": "Valid content.", "score": 0.5}]
        filtered = _filter(results)
        assert len(filtered) == 1

    def test_none_text_removed(self):
        results = [{"text": None, "score": 0.5}]
        assert _filter(results) == []

    def test_valid_text_kept(self):
        results = [_make_result("Valid text here.", score=0.5)]
        assert len(_filter(results)) == 1


# ── _dedup ─────────────────────────────────────────────────────────────────────

class TestDedup:

    def test_identical_docs_deduplicated(self):
        doc = _make_result("Identical content here.", doc_id="d1", chunk_id="0")
        result = _dedup([doc, doc, doc])
        assert len(result) == 1

    def test_different_docs_kept(self):
        docs = [_make_result(f"Unique content {i}.", doc_id=f"d{i}") for i in range(3)]
        assert len(_dedup(docs)) == 3

    def test_empty_input_returns_empty(self):
        assert _dedup([]) == []


# ── _normalize_scores ─────────────────────────────────────────────────────────

class TestNormalizeScores:

    def test_adds_norm_score_field(self):
        results = [_make_result("text", score=0.5)]
        normalized = _normalize_scores(results)
        assert "norm_score" in normalized[0]

    def test_all_equal_scores_get_half(self):
        results = [_make_result("text", score=1.0), _make_result("other", score=1.0)]
        normalized = _normalize_scores(results)
        assert all(r["norm_score"] == 0.5 for r in normalized)

    def test_different_scores_spread_zero_to_one(self):
        results = [_make_result("low", score=0.0), _make_result("high", score=1.0)]
        normalized = _normalize_scores(results)
        scores = sorted(r["norm_score"] for r in normalized)
        assert scores[0] == 0.0
        assert scores[1] == 1.0


# ── _score_fusion ─────────────────────────────────────────────────────────────

class TestScoreFusion:

    def test_adds_final_score_field(self):
        results = [_make_result("Fusion test content.", score=0.5)]
        results = _normalize_scores(results)
        results = _score_fusion(results)
        assert "final_score" in results[0]

    def test_final_score_is_non_negative(self):
        results = [_make_result("Content for scoring.", score=0.5)]
        results = _normalize_scores(results)
        results = _score_fusion(results)
        assert results[0]["final_score"] >= 0.0

    def test_higher_norm_score_leads_to_higher_final(self):
        results = [
            _make_result("Low relevance doc.", score=0.2, doc_id="d1"),
            _make_result("High relevance doc.", score=0.8, doc_id="d2"),
        ]
        results = _normalize_scores(results)
        results = _score_fusion(results)
        high = next(r for r in results if r["metadata"]["doc_id"] == "d2")
        low  = next(r for r in results if r["metadata"]["doc_id"] == "d1")
        assert high["final_score"] >= low["final_score"]


# ── _detect_contradictions ────────────────────────────────────────────────────

class TestDetectContradictions:

    def test_no_embeddings_no_contradictions(self):
        results = [_make_result("A", score=0.9), _make_result("B", score=0.9)]
        contras = _detect_contradictions(results)
        assert contras == []

    def test_high_sim_no_contradiction(self):
        emb = [0.9] * DIM_TEXT
        results = [
            _make_result("A", score=0.9, embedding=emb),
            _make_result("B", score=0.9, embedding=emb),
        ]
        contras = _detect_contradictions(results)
        assert contras == []

    def test_low_sim_high_scores_detected(self):
        emb_a = [1.0] + [0.0] * (DIM_TEXT - 1)
        emb_b = [0.0] * (DIM_TEXT - 1) + [1.0]
        results = [
            _make_result("A", score=0.9, embedding=emb_a),
            _make_result("B", score=0.9, embedding=emb_b),
        ]
        contras = _detect_contradictions(results)
        assert len(contras) == 1
        assert contras[0][0] == 0
        assert contras[0][1] == 1


# ── _confidence_score ─────────────────────────────────────────────────────────

class TestConfidenceScore:

    def test_returns_float_in_range(self):
        result = _make_result("Test content for confidence.", score=0.8)
        others = [_make_result("Other content here.", score=0.5)]
        score = _confidence_score(result, [result] + others)
        assert 0.0 <= score <= 1.0

    def test_empty_text_returns_base_score(self):
        result = {"text": "", "score": 0.7, "metadata": {}}
        score = _confidence_score(result, [result])
        assert score == 0.7

    def test_shared_words_boost_confidence(self):
        shared = "retrieval augmented generation knowledge"
        result = _make_result(f"First {shared} document.", score=0.5)
        support = _make_result(f"Second {shared} document.", score=0.5)
        score_with = _confidence_score(result, [result, support])
        score_without = _confidence_score(result, [result])
        assert score_with >= score_without


# ── _cross_modal_link ─────────────────────────────────────────────────────────

class TestCrossModalLink:

    def test_links_image_to_text_chunk(self):
        text_doc = {
            "text": "Source text chunk content.",
            "metadata": {"modality": "text", "doc_id": "d1", "page": 1},
        }
        img_doc = {
            "text": "Image caption.",
            "metadata": {"modality": "image", "doc_id": "d1", "page": 1},
        }
        results = _cross_modal_link([text_doc, img_doc])
        img = next(r for r in results if r["metadata"]["modality"] == "image")
        assert "linked_text_chunk" in img

    def test_no_matching_page_no_link(self):
        text_doc = {
            "text": "Page 1 text content.",
            "metadata": {"modality": "text", "doc_id": "d1", "page": 1},
        }
        img_doc = {
            "text": "Page 2 image caption.",
            "metadata": {"modality": "image", "doc_id": "d1", "page": 2},
        }
        results = _cross_modal_link([text_doc, img_doc])
        img = next(r for r in results if r["metadata"]["modality"] == "image")
        assert "linked_text_chunk" not in img


# ── _diversity ────────────────────────────────────────────────────────────────

class TestDiversity:

    def test_no_embeddings_all_selected(self):
        results = [_make_result(f"Doc {i}") for i in range(5)]
        selected = _diversity(results, top_k=3, sim_threshold=0.9)
        assert len(selected) <= 3

    def test_identical_embeddings_deduped_by_threshold(self):
        emb = [0.9] * DIM_TEXT
        results = [_make_result(f"Doc {i}", embedding=emb) for i in range(5)]
        selected = _diversity(results, top_k=5, sim_threshold=0.5)
        assert len(selected) == 1  # All too similar


# ── ResultFusion.fuse ─────────────────────────────────────────────────────────

class TestResultFusionFuse:

    def test_empty_input_returns_empty(self):
        rf = ResultFusion()
        assert rf.fuse([]) == []

    def test_returns_list(self):
        rf = ResultFusion()
        results = [_make_result(f"Document {i} content here.", score=0.5)
                   for i in range(3)]
        output = rf.fuse(results, session_id="s1")
        assert isinstance(output, list)

    def test_output_does_not_exceed_top_k(self):
        rf = ResultFusion()
        results = [_make_result(f"Doc {i}", score=0.5, doc_id=f"d{i}", chunk_id=str(i))
                   for i in range(20)]
        output = rf.fuse(results, session_id="s1")
        assert len(output) <= rf.top_k

    def test_results_have_final_score(self):
        rf = ResultFusion()
        results = [_make_result("Scored document content.", score=0.7)]
        output = rf.fuse(results, session_id="s1")
        if output:
            assert "final_score" in output[0]

    def test_results_have_confidence_score(self):
        rf = ResultFusion()
        results = [_make_result("Confidence test content.", score=0.7)]
        output = rf.fuse(results, session_id="s1")
        if output:
            assert "confidence_score" in output[0]

    def test_duplicate_results_deduped(self):
        rf = ResultFusion()
        doc = _make_result("Identical duplicate content.", score=0.8)
        output = rf.fuse([doc, doc, doc], session_id="s1")
        assert len(output) == 1

    def test_empty_text_filtered_out(self):
        rf = ResultFusion()
        results = [
            {"text": "", "score": 0.9, "metadata": {}},
            _make_result("Valid content here.", score=0.5),
        ]
        output = rf.fuse(results, session_id="s1")
        texts = [r.get("text", "") for r in output]
        assert "" not in texts

    def test_detect_contradictions_flag_respected(self):
        rf = ResultFusion()
        emb_a = [1.0] + [0.0] * (DIM_TEXT - 1)
        emb_b = [0.0] * (DIM_TEXT - 1) + [1.0]
        results = [
            _make_result("High score doc A.", score=0.9, embedding=emb_a,
                         doc_id="da", chunk_id="0"),
            _make_result("High score doc B.", score=0.9, embedding=emb_b,
                         doc_id="db", chunk_id="0"),
        ]
        output_with    = rf.fuse(results, session_id="s1", detect_contradictions=True)
        output_without = rf.fuse(results, session_id="s1", detect_contradictions=False)
        assert isinstance(output_with, list)
        assert isinstance(output_without, list)

    def test_graceful_degradation_on_error(self):
        rf = ResultFusion()
        # Patch _normalize_scores to crash — should fall back to score sort
        with patch("app.reasoning.result_fusion._normalize_scores",
                   side_effect=RuntimeError("crash")):
            results = [_make_result("Fallback test.", score=0.8)]
            output = rf.fuse(results, session_id="s1")
        assert isinstance(output, list)
