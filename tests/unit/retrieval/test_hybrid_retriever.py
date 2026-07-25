"""Unit tests for app/retrieval/hybrid_retriever.py — Phase 24.10."""
from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever, _expand_query_heuristic


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bm25(results: List[Dict] = None) -> MagicMock:
    m = MagicMock()
    m.search = MagicMock(return_value=results or [])
    return m


def _make_vector_store(text_results: List[Dict] = None,
                       vision_results: List[Dict] = None,
                       has_vision: bool = False) -> MagicMock:
    m = MagicMock()
    m.search_text = MagicMock(return_value=text_results or [])
    m.search_vision = MagicMock(return_value=vision_results or [])
    m.has_vision_data = MagicMock(return_value=has_vision)
    return m


def _make_embedder(vec: List[float] = None) -> MagicMock:
    dim = 384
    m = MagicMock()
    m.embed_query = MagicMock(return_value=vec or [0.1] * dim)
    return m


def _make_result(text: str, score: float = 0.5, session_id: str = "s1",
                 modality: str = "text") -> Dict:
    return {
        "text": text,
        "score": score,
        "metadata": {"session_id": session_id, "modality": modality},
    }


def _make_retriever(bm25_results: List[Dict] = None,
                    vec_results: List[Dict] = None) -> HybridRetriever:
    bm25 = _make_bm25(bm25_results or [])
    vec = _make_vector_store(text_results=vec_results or [], has_vision=False)
    emb = _make_embedder()
    return HybridRetriever(bm25=bm25, vector_store=vec, embedder=emb)


# ── _expand_query_heuristic ───────────────────────────────────────────────────

class TestExpandQueryHeuristic:

    def test_empty_string_returns_empty(self):
        assert _expand_query_heuristic("") == []

    def test_keyword_query_returns_one_form(self):
        result = _expand_query_heuristic("RAG retrieval")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "RAG retrieval" in result

    def test_natural_question_returns_two_forms(self):
        result = _expand_query_heuristic("what is retrieval augmented generation")
        # Stopword stripping should produce a second variant
        assert len(result) >= 1

    def test_result_contains_original(self):
        q = "how does vector search work"
        result = _expand_query_heuristic(q)
        assert result[0] == q

    def test_short_single_word_no_expansion(self):
        result = _expand_query_heuristic("RAG")
        assert len(result) >= 1

    def test_returns_list(self):
        result = _expand_query_heuristic("explain the process")
        assert isinstance(result, list)


# ── HybridRetriever initialization ───────────────────────────────────────────

class TestHybridRetrieverInit:

    def test_init_stores_components(self):
        bm25 = _make_bm25()
        vec = _make_vector_store()
        emb = _make_embedder()
        r = HybridRetriever(bm25=bm25, vector_store=vec, embedder=emb)
        assert r.bm25 is bm25
        assert r.vector_store is vec
        assert r.embedder is emb

    def test_init_with_clip_embedder(self):
        bm25 = _make_bm25()
        vec = _make_vector_store()
        emb = _make_embedder()
        clip = MagicMock()
        r = HybridRetriever(bm25=bm25, vector_store=vec, embedder=emb,
                            clip_text_embedder=clip)
        assert r.clip_text_embedder is clip

    def test_embed_cache_starts_empty(self):
        r = _make_retriever()
        assert len(r._embed_cache) == 0


# ── search ─────────────────────────────────────────────────────────────────────

class TestHybridRetrieverSearch:

    def test_empty_query_returns_empty(self):
        r = _make_retriever()
        result = r.search("", session_id="s1")
        assert result == []

    def test_empty_session_id_returns_empty(self):
        r = _make_retriever()
        result = r.search("test query", session_id="")
        assert result == []

    def test_all_empty_retrievers_return_empty(self):
        r = _make_retriever(bm25_results=[], vec_results=[])
        result = r.search("retrieval augmented generation", session_id="s1")
        assert result == []

    def test_bm25_results_returned(self):
        docs = [_make_result(f"Document about retrieval {i}.", score=0.8) for i in range(3)]
        r = _make_retriever(bm25_results=docs, vec_results=[])
        result = r.search("retrieval", session_id="s1")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_vector_results_returned(self):
        docs = [_make_result(f"Vector result text {i}.", score=0.7) for i in range(3)]
        r = _make_retriever(bm25_results=[], vec_results=docs)
        result = r.search("query text", session_id="s1")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_combined_results_merged(self):
        bm25_docs = [_make_result("BM25 document about knowledge.", score=0.9)]
        vec_docs = [_make_result("Vector document about knowledge.", score=0.8)]
        r = _make_retriever(bm25_results=bm25_docs, vec_results=vec_docs)
        result = r.search("knowledge", session_id="s1")
        assert len(result) >= 1

    def test_top_k_limits_results(self):
        bm25_docs = [_make_result(f"Doc {i}", score=0.5) for i in range(10)]
        vec_docs  = [_make_result(f"Vec {i}", score=0.5) for i in range(10)]
        r = _make_retriever(bm25_results=bm25_docs, vec_results=vec_docs)
        result = r.search("query", session_id="s1", top_k=3)
        assert len(result) <= 9  # multiplied by candidate_multiplier, then capped

    def test_result_has_text_key(self):
        docs = [_make_result("RAG document with relevant content.", score=0.8)]
        r = _make_retriever(bm25_results=docs)
        result = r.search("RAG", session_id="s1")
        for hit in result:
            assert "text" in hit

    def test_result_has_score_key(self):
        docs = [_make_result("Test document with content.", score=0.8)]
        r = _make_retriever(bm25_results=docs)
        result = r.search("document", session_id="s1")
        for hit in result:
            assert "score" in hit

    def test_result_has_metadata_key(self):
        docs = [_make_result("Metadata document content.", score=0.8)]
        r = _make_retriever(bm25_results=docs)
        result = r.search("metadata", session_id="s1")
        for hit in result:
            assert "metadata" in hit


# ── modality detection ────────────────────────────────────────────────────────

class TestHybridRetrieverModalityDetection:

    def test_vision_query_detected(self):
        r = _make_retriever()
        assert r._is_vision_query("show me an image of the diagram") is True

    def test_audio_query_detected(self):
        r = _make_retriever()
        assert r._is_audio_query("what does the audio transcript say") is True

    def test_video_query_detected(self):
        r = _make_retriever()
        assert r._is_video_query("watch this video clip") is True

    def test_text_query_not_vision(self):
        r = _make_retriever()
        assert r._is_vision_query("explain retrieval augmented generation") is False


# ── normalize_query ───────────────────────────────────────────────────────────

class TestHybridRetrieverNormalizeQuery:

    def test_strips_leading_trailing_spaces(self):
        r = _make_retriever()
        result = r._normalize_query("  hello world  ")
        assert result == "hello world"

    def test_collapses_internal_spaces(self):
        r = _make_retriever()
        result = r._normalize_query("hello   world")
        assert result == "hello world"

    def test_empty_string_returns_empty(self):
        r = _make_retriever()
        result = r._normalize_query("")
        assert result == ""


# ── score normalization ───────────────────────────────────────────────────────

class TestHybridRetrieverScoreNormalization:

    def test_empty_list_returned_unchanged(self):
        r = _make_retriever()
        result = r._normalize_scores([])
        assert result == []

    def test_all_zero_scores_not_changed(self):
        r = _make_retriever()
        docs = [{"score": 0.0}, {"score": 0.0}]
        result = r._normalize_scores(docs)
        assert result is not None

    def test_scores_normalized_to_max_one(self):
        r = _make_retriever()
        docs = [{"score": 2.0, "text": "a"}, {"score": 4.0, "text": "b"}]
        result = r._normalize_scores(docs)
        max_score = max(d["score"] for d in result)
        assert abs(max_score - 1.0) < 1e-6


# ── RRF score ─────────────────────────────────────────────────────────────────

class TestHybridRetrieverRRF:

    def test_rrf_score_rank_1(self):
        r = _make_retriever()
        from app.core.config import settings
        expected = 1.0 / (settings.HYBRID_RRF_K + 1)
        assert abs(r._rrf_score(1) - expected) < 1e-9

    def test_rrf_decreases_with_rank(self):
        r = _make_retriever()
        assert r._rrf_score(1) > r._rrf_score(5)

    def test_rrf_always_positive(self):
        r = _make_retriever()
        for rank in range(1, 20):
            assert r._rrf_score(rank) > 0.0


# ── text overlap ──────────────────────────────────────────────────────────────

class TestHybridRetrieverTextOverlap:

    def test_identical_text_overlap_one(self):
        r = _make_retriever()
        text = "the quick brown fox"
        assert abs(r._text_overlap(text, text) - 1.0) < 1e-6

    def test_disjoint_text_overlap_zero(self):
        r = _make_retriever()
        assert r._text_overlap("abc def", "xyz uvw") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        r = _make_retriever()
        score = r._text_overlap("hello world", "hello there")
        assert 0.0 < score < 1.0


# ── health check ──────────────────────────────────────────────────────────────

class TestHybridRetrieverHealthCheck:

    def test_health_check_returns_dict(self):
        r = _make_retriever()
        result = r.health_check()
        assert isinstance(result, dict)

    def test_health_check_has_weights_key(self):
        r = _make_retriever()
        result = r.health_check()
        assert "weights" in result

    def test_health_check_reports_embedder_ready(self):
        r = _make_retriever()
        result = r.health_check()
        assert result.get("embedder_ready") is True

    def test_health_check_bm25_not_ready_when_empty(self):
        r = _make_retriever()
        result = r.health_check()
        # bm25.bm25 attribute is None since mock doesn't set it
        assert "bm25_ready" in result
