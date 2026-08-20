"""Tests for app/eval/jobs/shadow_sampler.py — the monitoring Phase 2 change:
sample_and_log() now also captures retrieval-quality signal (retrieval_count/
top1_score/mean_topk_score) derived from the SAME `sources` array every
caller already builds and passes in, so app/eval/jobs/drift_eval.py (a later
phase) has real retrieval-quality data to compare against a reference window
instead of only the query/answer/context text it had before.

Covers the pure helper directly (`_retrieval_stats`) and the end-to-end write
path with a mocked Mongo collection, plus the two hard safety requirements
this module's own docstring states: never raises, and returns silently
(no-op) whenever Mongo/sampling-rate conditions aren't met.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.eval.jobs.shadow_sampler import _retrieval_stats, sample_and_log


class TestRetrievalStats:
    def test_empty_sources(self):
        stats = _retrieval_stats([])
        assert stats == {"retrieval_count": 0, "top1_score": None, "mean_topk_score": None}

    def test_none_sources(self):
        stats = _retrieval_stats(None)
        assert stats == {"retrieval_count": 0, "top1_score": None, "mean_topk_score": None}

    def test_computes_top1_and_mean(self):
        sources = [
            {"text": "a", "score": 0.9},
            {"text": "b", "score": 0.6},
            {"text": "c", "score": 0.3},
        ]
        stats = _retrieval_stats(sources)
        assert stats["retrieval_count"] == 3
        assert stats["top1_score"] == 0.9
        assert stats["mean_topk_score"] == pytest.approx(0.6)

    def test_ignores_sources_missing_score(self):
        sources = [{"text": "a"}, {"text": "b", "score": 0.5}]
        stats = _retrieval_stats(sources)
        # retrieval_count still counts every source (that's a doc-count
        # signal); top1/mean only ever average the sources that DO carry a
        # real numeric score, e.g. a web-search source with no rerank score.
        assert stats["retrieval_count"] == 2
        assert stats["top1_score"] == 0.5
        assert stats["mean_topk_score"] == 0.5

    def test_all_sources_missing_score_returns_none(self):
        sources = [{"text": "a"}, {"text": "b"}]
        stats = _retrieval_stats(sources)
        assert stats["retrieval_count"] == 2
        assert stats["top1_score"] is None
        assert stats["mean_topk_score"] is None

    def test_non_numeric_score_ignored_not_raised(self):
        sources = [{"text": "a", "score": "not-a-number"}, {"text": "b", "score": 0.4}]
        stats = _retrieval_stats(sources)
        assert stats["top1_score"] == 0.4


class TestSampleAndLog:
    def _patch_mongo(self, monkeypatch, sample_rate: float = 1.0):
        from app.core.config import settings

        monkeypatch.setattr(settings, "ONLINE_EVAL_SAMPLE_RATE", sample_rate)

        mock_collection = MagicMock()
        mock_db = {settings.MONGO_EVAL_SHADOW_COLLECTION: mock_collection}
        mock_mongo = MagicMock()
        mock_mongo.db = mock_db

        fake_infra = MagicMock()
        fake_infra.get_mongo.return_value = mock_mongo
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)
        return mock_collection

    def test_writes_retrieval_stats_into_doc(self, monkeypatch):
        mock_collection = self._patch_mongo(monkeypatch)
        sources = [{"text": "ctx1", "score": 0.8}, {"text": "ctx2", "score": 0.4}]

        sample_and_log(
            session_id="s1",
            user_id="u1",
            query="what is revenue",
            answer="revenue was $1B",
            sources=sources,
            route="rag",
            latency_ms=123.4,
        )

        assert mock_collection.insert_one.called
        doc = mock_collection.insert_one.call_args[0][0]
        assert doc["retrieval_count"] == 2
        assert doc["top1_score"] == 0.8
        assert doc["mean_topk_score"] == pytest.approx(0.6)
        assert doc["route"] == "rag"
        assert doc["contexts"] == ["ctx1", "ctx2"]

    def test_sample_rate_zero_skips_write(self, monkeypatch):
        mock_collection = self._patch_mongo(monkeypatch, sample_rate=0.0)
        sample_and_log(
            session_id="s1",
            user_id="u1",
            query="q",
            answer="a",
            sources=[],
            route="rag",
            latency_ms=1.0,
        )
        assert not mock_collection.insert_one.called

    def test_empty_answer_skips_write(self, monkeypatch):
        mock_collection = self._patch_mongo(monkeypatch)
        sample_and_log(
            session_id="s1",
            user_id="u1",
            query="q",
            answer="",
            sources=[],
            route="rag",
            latency_ms=1.0,
        )
        assert not mock_collection.insert_one.called

    def test_mongo_unavailable_does_not_raise(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "ONLINE_EVAL_SAMPLE_RATE", 1.0)
        fake_infra = MagicMock()
        fake_infra.get_mongo.return_value = None
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        # Must not raise — this is the hard safety requirement the module
        # docstring states ("must NEVER raise into the request path").
        sample_and_log(
            session_id="s1",
            user_id="u1",
            query="q",
            answer="a",
            sources=[],
            route="rag",
            latency_ms=1.0,
        )

    def test_mongo_insert_raising_does_not_propagate(self, monkeypatch):
        mock_collection = self._patch_mongo(monkeypatch)
        mock_collection.insert_one.side_effect = RuntimeError("mongo write failed")

        sample_and_log(
            session_id="s1",
            user_id="u1",
            query="q",
            answer="a",
            sources=[],
            route="rag",
            latency_ms=1.0,
        )
