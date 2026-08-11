"""Unit tests for app/retrieval/bm25_retriever.py — Phase 24.10."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.bm25_retriever import BM25Retriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_retriever(user_id: str = "test_user", tmp_path: Path = None) -> BM25Retriever:
    r = BM25Retriever(user_id=user_id)
    return r


class _FakeDoc:
    """Minimal IngestedDocument-like object for BM25 tests."""
    def __init__(self, text: str, session_id: str = "sess1", i: int = 0):
        self.text = text
        self.modality = "text"
        self.subtype = None
        self.source = f"doc_{i}.txt"
        self.source_type = "file"
        self.chunk_id = i
        self.page = None
        self.structure = {
            "session_id": session_id,
            "doc_id": f"doc_{i}",
            "content_type": "text",
            "embedding_space": "text",
        }


def _make_docs(n: int = 5, session_id: str = "sess1") -> List[_FakeDoc]:
    return [
        _FakeDoc(
            text=f"Document number {i} about retrieval augmented generation.",
            session_id=session_id,
            i=i,
        )
        for i in range(n)
    ]


# ── Initialization ────────────────────────────────────────────────────────────

class TestBM25RetrieverInit:

    def test_init_without_user_id(self):
        r = BM25Retriever()
        assert r.user_id is None

    def test_init_with_user_id(self):
        r = BM25Retriever(user_id="alice")
        assert r.user_id == "alice"

    def test_bm25_none_at_cold_start(self):
        r = BM25Retriever(user_id="fresh_user")
        assert r.bm25 is None

    def test_documents_empty_at_cold_start(self):
        r = BM25Retriever(user_id="fresh_user")
        assert r.documents == []

    def test_tokenized_corpus_empty_at_cold_start(self):
        r = BM25Retriever()
        assert r.tokenized_corpus == []


# ── search — cold start ───────────────────────────────────────────────────────

class TestBM25RetrieverSearch:

    def test_search_cold_start_returns_empty(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "nonexistent.pkl"):
            r = BM25Retriever(user_id="cold_user")
            result = r.search("test query", session_id="s1")
        assert result == []

    def test_search_returns_list(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            docs = _make_docs(5)
            r.build_index(docs, user_id="u1")
            result = r.search("retrieval augmented generation", session_id="sess1")
        assert isinstance(result, list)

    def test_search_result_has_expected_keys(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.build_index(_make_docs(5), user_id="u1")
            result = r.search("document retrieval", session_id="sess1")
        if result:
            for hit in result:
                assert "text" in hit
                assert "score" in hit
                assert "metadata" in hit

    def test_search_top_k_limits_results(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            docs = _make_docs(10)
            r.build_index(docs, user_id="u1")
            result = r.search("retrieval generation", session_id="sess1", top_k=3)
        assert len(result) <= 3

    def test_search_empty_query_returns_empty(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            docs = _make_docs(5)
            r.build_index(docs, user_id="u1")
            result = r.search("", session_id="sess1")
        assert result == []

    def test_search_scores_are_non_negative(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            docs = _make_docs(5)
            r.build_index(docs, user_id="u1")
            result = r.search("retrieval augmented generation", session_id="sess1")
        for hit in result:
            assert hit["score"] >= 0.0


# ── build_index ───────────────────────────────────────────────────────────────

class TestBM25RetrieverBuildIndex:

    def test_build_index_populates_bm25(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.build_index(_make_docs(3), user_id="u1")
        assert r.bm25 is not None

    def test_build_index_empty_docs_leaves_bm25_none(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.build_index([], user_id="u1")
        assert r.bm25 is None

    def test_build_index_sets_documents(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            docs = _make_docs(4)
            r.build_index(docs, user_id="u1")
        assert len(r.documents) == 4

    def test_build_index_multiple_calls_replaces_index(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.build_index(_make_docs(3), user_id="u1")
            first_count = len(r.documents)
            r.build_index(_make_docs(7), user_id="u1")
            assert len(r.documents) == 7


# ── add_document ──────────────────────────────────────────────────────────────

class TestBM25RetrieverAddDocument:

    def test_add_document_increments_doc_count(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.add_document("First document text content here.", metadata={}, user_id="u1")
            r.add_document("Second document text content here.", metadata={}, user_id="u1")
        assert len(r.documents) == 2

    def test_add_document_rebuilds_bm25(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.add_document("Test document text content.", metadata={}, user_id="u1")
        assert r.bm25 is not None

    def test_add_empty_text_skipped(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.add_document("", metadata={}, user_id="u1")
        assert len(r.documents) == 0


# ── clear ─────────────────────────────────────────────────────────────────────

class TestBM25RetrieverClear:

    def test_clear_resets_bm25_to_none(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.build_index(_make_docs(3), user_id="u1")
            assert r.bm25 is not None
            r.clear(user_id="u1")
        assert r.bm25 is None

    def test_clear_resets_documents(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.build_index(_make_docs(3), user_id="u1")
            r.clear(user_id="u1")
        assert r.documents == []


# ── health_check ──────────────────────────────────────────────────────────────

class TestBM25RetrieverHealthCheck:

    def test_health_check_returns_dict(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            result = r.health_check(user_id="u1")
        assert isinstance(result, dict)

    def test_health_check_cold_start_not_ready(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "nonexistent.pkl"):
            r = BM25Retriever(user_id="u1")
            result = r.health_check(user_id="u1")
        assert result.get("ready") is False or result.get("bm25_ready") is False

    def test_health_check_after_build_is_ready(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            r.build_index(_make_docs(3), user_id="u1")
            result = r.health_check(user_id="u1")
        assert result.get("ready") is True or result.get("bm25_ready") is True


# ── session filter ────────────────────────────────────────────────────────────

class TestBM25RetrieverSessionFilter:

    def test_results_respect_session_id(self, tmp_path):
        with patch("app.retrieval.bm25_retriever.user_bm25_path",
                   return_value=tmp_path / "idx.pkl"):
            r = BM25Retriever(user_id="u1")
            docs_a = _make_docs(3, session_id="sessA")
            docs_b = _make_docs(3, session_id="sessB")
            r.build_index(docs_a + docs_b, user_id="u1")
            result = r.search("retrieval generation", session_id="sessA")
        for hit in result:
            assert hit["metadata"].get("session_id") == "sessA"


# ── per-modality BM25 constructor contract ─────────────────────────────────────
# Regression coverage for a real production bug (2026-08-08): DocxBM25 overrode
# __init__(self) with no user_id param, diverging from BaseBM25.__init__(self,
# user_id=None) — the contract every other per-modality class (TxtBM25, PdfBM25,
# XlsxBM25, ImageBM25, AudioBM25, VideoBM25) honors. BM25AggregatorRetriever
# instantiates every modality class uniformly with cls(user_id=uid), so this
# broke KB-file deletion for every modality, not just docx — confirmed via
# kb_delete_bm25_failed | file=fomc_dec2024.txt (a .txt file)
# | error=DocxBM25.__init__() got an unexpected keyword argument 'user_id'.
# There was previously zero test coverage of the per-modality classes or the
# aggregator at all, which is exactly why this went unnoticed.

class TestPerModalityBM25ConstructorContract:

    def test_every_modality_class_accepts_user_id(self):
        from app.retrieval.bm25_retriever import _MODALITY_TO_CLASS

        for modality, cls in set(_MODALITY_TO_CLASS.items()):
            instance = cls(user_id="regression_test_user")
            assert instance.user_id == "regression_test_user", (
                f"{cls.__name__} (modality={modality!r}) did not honor the "
                f"user_id passed to its constructor"
            )

    def test_every_modality_class_user_id_defaults_to_none(self):
        from app.retrieval.bm25_retriever import _MODALITY_TO_CLASS

        for cls in set(_MODALITY_TO_CLASS.values()):
            instance = cls()
            assert instance.user_id is None


class TestBM25AggregatorRetriever:
    """The exact code path that crashed in production: BM25AggregatorRetriever
    instantiating every per-modality index to purge a file by source name."""

    def test_all_indexes_instantiates_every_modality_without_raising(self):
        from app.retrieval.bm25_retriever import BM25AggregatorRetriever

        agg = BM25AggregatorRetriever(user_id="regression_test_user")
        indexes = agg._all_indexes()
        # One instance per distinct class (7 modality classes today), not one
        # per _MODALITY_TO_CLASS alias key (13 aliases, e.g. "txt"/"text" ->
        # same TxtBM25 class).
        assert len(indexes) >= 7
        for idx in indexes:
            assert idx.user_id == "regression_test_user"

    def test_delete_by_source_does_not_raise_on_any_modality(self, tmp_path):
        from app.retrieval.bm25_retriever import BM25AggregatorRetriever

        with patch("app.utils.paths.user_dir", return_value=tmp_path):
            agg = BM25AggregatorRetriever(user_id="regression_test_user")
            # Empty/nonexistent index on disk — must return 0, not raise.
            removed = agg.delete_by_source("some_file.txt", user_id="regression_test_user")
        assert removed == 0
