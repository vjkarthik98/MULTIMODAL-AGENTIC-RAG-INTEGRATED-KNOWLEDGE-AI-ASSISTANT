"""Tests for app/eval/jobs/online_eval.py — specifically the monitoring
Phase 6 addition: _score_batch()/_push_gauges() now also aggregate and push
the retrieval-quality fields (top1_score/mean_topk_score/retrieval_count)
shadow_sampler.py has stamped on every shadow-collection row since Phase 2,
alongside the faithfulness/hallucination/route aggregation this job already
did. No existing test file covered this module before this pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.eval.jobs.online_eval import _push_gauges, _score_batch


class TestScoreBatchRetrievalFields:
    def test_averages_retrieval_fields_across_rows(self):
        rows = [
            {
                "query": "q1",
                "answer": "a1",
                "contexts": ["c1"],
                "route": "rag",
                "latency_ms": 100,
                "top1_score": 0.8,
                "mean_topk_score": 0.6,
                "retrieval_count": 3,
            },
            {
                "query": "q2",
                "answer": "a2",
                "contexts": ["c2"],
                "route": "direct",
                "latency_ms": 50,
                "top1_score": 0.4,
                "mean_topk_score": 0.3,
                "retrieval_count": 1,
            },
        ]
        result = _score_batch(rows)
        assert result["top1_score"] == pytest.approx(0.6)
        assert result["mean_topk_score"] == pytest.approx(0.45)
        assert result["retrieval_count"] == pytest.approx(2.0)

    def test_rows_missing_retrieval_fields_are_excluded_not_zeroed(self):
        """A web-search row (no rerank score at all) must not drag the
        average toward 0 — it should be omitted from the mean entirely,
        same treatment shadow_sampler.py's own _retrieval_stats() already
        gives a doc with no numeric score."""
        rows = [
            {"query": "q1", "answer": "a1", "contexts": [], "route": "web_search"},
            {
                "query": "q2",
                "answer": "a2",
                "contexts": ["c2"],
                "route": "rag",
                "top1_score": 0.8,
                "mean_topk_score": 0.8,
                "retrieval_count": 2,
            },
        ]
        result = _score_batch(rows)
        assert result["top1_score"] == pytest.approx(0.8)
        assert result["mean_topk_score"] == pytest.approx(0.8)
        assert result["retrieval_count"] == pytest.approx(2.0)

    def test_no_rows_with_retrieval_fields_returns_none(self):
        rows = [{"query": "q1", "answer": "a1", "contexts": [], "route": "web_search"}]
        result = _score_batch(rows)
        assert result["top1_score"] is None
        assert result["mean_topk_score"] is None
        assert result["retrieval_count"] is None

    def test_non_numeric_retrieval_fields_ignored(self):
        rows = [
            {
                "query": "q1",
                "answer": "a1",
                "contexts": [],
                "route": "rag",
                "top1_score": "not-a-number",
            }
        ]
        result = _score_batch(rows)
        assert result["top1_score"] is None


class TestPushGauges:
    def test_sets_retrieval_gauges_when_present(self, monkeypatch):
        import app.core.metrics as m

        fake_top1 = MagicMock()
        fake_mean_topk = MagicMock()
        fake_count = MagicMock()
        monkeypatch.setattr(m, "eval_online_top1_score", fake_top1)
        monkeypatch.setattr(m, "eval_online_mean_topk_score", fake_mean_topk)
        monkeypatch.setattr(m, "eval_online_retrieval_count", fake_count)

        result = {
            "scored": 2,
            "faithfulness": None,
            "answer_relevancy": None,
            "hallucination_rate": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "route_counts": {},
            "top1_score": 0.6,
            "mean_topk_score": 0.45,
            "retrieval_count": 2.0,
        }
        _push_gauges(result)

        fake_top1.set.assert_called_once_with(0.6)
        fake_mean_topk.set.assert_called_once_with(0.45)
        fake_count.set.assert_called_once_with(2.0)

    def test_skips_none_retrieval_gauges(self, monkeypatch):
        import app.core.metrics as m

        fake_top1 = MagicMock()
        monkeypatch.setattr(m, "eval_online_top1_score", fake_top1)

        result = {
            "scored": 0,
            "faithfulness": None,
            "answer_relevancy": None,
            "hallucination_rate": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "route_counts": {},
            "top1_score": None,
            "mean_topk_score": None,
            "retrieval_count": None,
        }
        _push_gauges(result)

        fake_top1.set.assert_not_called()
