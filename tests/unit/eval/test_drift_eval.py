"""Tests for app/eval/jobs/drift_eval.py — monitoring Phase 3.

Covers the pure statistics helpers directly (no Mongo needed), the severity
classification's "correlated signal, not a single threshold" logic, and the
end-to-end run_drift_eval_once()/build_reference() paths with a mocked Mongo
collection and a real temp reference file.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.eval.jobs.drift_eval import (
    _classify_severity,
    _compare_column,
    _extract_series,
    _load_reference,
    _psi,
    _row_to_columns,
    build_reference,
    run_drift_eval_once,
)


class TestRowToColumns:
    def test_extracts_query_length_and_numeric_fields(self):
        row = {
            "query": "what is revenue",
            "top1_score": 0.8,
            "mean_topk_score": 0.5,
            "latency_ms": 120.0,
        }
        cols = _row_to_columns(row)
        assert cols["query_length"] == float(len("what is revenue"))
        assert cols["top1_score"] == 0.8
        assert cols["mean_topk_score"] == 0.5
        assert cols["latency_ms"] == 120.0

    def test_missing_fields_are_omitted_not_defaulted(self):
        row = {"query": "q"}
        cols = _row_to_columns(row)
        assert "top1_score" not in cols
        assert "mean_topk_score" not in cols
        assert "latency_ms" not in cols

    def test_non_numeric_score_ignored(self):
        row = {"query": "q", "top1_score": "not-a-number"}
        cols = _row_to_columns(row)
        assert "top1_score" not in cols


class TestExtractSeries:
    def test_builds_per_column_lists(self):
        rows = [
            {"query": "aa", "top1_score": 0.9},
            {"query": "bbbb", "top1_score": 0.5},
        ]
        series = _extract_series(rows)
        assert series["query_length"] == [2.0, 4.0]
        assert series["top1_score"] == [0.9, 0.5]
        assert series["mean_topk_score"] == []
        assert series["latency_ms"] == []


class TestPSI:
    def test_identical_distributions_near_zero(self):
        values = [0.5 + 0.01 * i for i in range(30)]
        psi = _psi(values, values)
        assert psi is not None
        assert psi < 0.01

    def test_shifted_distribution_has_positive_psi(self):
        reference = [0.5 + 0.01 * i for i in range(30)]
        current = [0.1 + 0.01 * i for i in range(30)]  # shifted well outside reference range
        psi = _psi(reference, current)
        assert psi is not None
        assert psi > 0.5

    def test_insufficient_rows_returns_none(self):
        assert _psi([0.1, 0.2], [0.1, 0.2]) is None


class TestCompareColumn:
    def test_insufficient_rows_not_tested(self):
        result = _compare_column([0.1, 0.2], [0.1, 0.2])
        assert result == {"tested": False, "reason": "insufficient_rows"}

    def test_same_distribution_not_drifted(self):
        reference = [0.7 + 0.001 * (i % 5) for i in range(40)]
        current = [0.7 + 0.001 * (i % 5) for i in range(40)]
        result = _compare_column(reference, current)
        assert result["tested"] is True
        assert result["drifted"] is False

    def test_clearly_shifted_distribution_is_drifted_and_degraded(self):
        reference = [0.9] * 40
        current = [0.2] * 40
        result = _compare_column(reference, current)
        assert result["tested"] is True
        assert result["drifted"] is True
        assert result["degraded"] is True  # current mean (0.2) < reference mean (0.9)

    def test_shifted_upward_is_drifted_but_not_degraded(self):
        reference = [0.2] * 40
        current = [0.9] * 40
        result = _compare_column(reference, current)
        assert result["drifted"] is True
        assert result["degraded"] is False


class TestClassifySeverity:
    def test_no_tested_columns_is_info(self):
        comparisons = {"top1_score": {"tested": False}}
        severity, score = _classify_severity(comparisons)
        assert severity == "info"
        assert score == 0.0

    def test_nothing_drifted_is_info(self):
        comparisons = {
            "top1_score": {"tested": True, "drifted": False, "degraded": False},
            "latency_ms": {"tested": True, "drifted": False, "degraded": False},
        }
        severity, score = _classify_severity(comparisons)
        assert severity == "info"
        assert score == 0.0

    def test_below_warning_threshold_is_info(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "DRIFT_WARNING_THRESHOLD", 0.5)
        monkeypatch.setattr(settings, "DRIFT_CRITICAL_THRESHOLD", 0.75)
        # 1 of 4 columns drifted = 0.25, below the 0.5 warning threshold
        comparisons = {
            "query_length": {"tested": True, "drifted": True, "degraded": False},
            "top1_score": {"tested": True, "drifted": False, "degraded": False},
            "mean_topk_score": {"tested": True, "drifted": False, "degraded": False},
            "latency_ms": {"tested": True, "drifted": False, "degraded": False},
        }
        severity, score = _classify_severity(comparisons)
        assert severity == "info"
        assert score == pytest.approx(0.25)

    def test_drift_without_quality_degradation_is_warning_not_critical(self, monkeypatch):
        """High dataset_score alone must NOT be critical — only a quality
        column drifting in the bad direction earns critical. This is the
        core 'correlated signal' rule this module's docstring states."""
        from app.core.config import settings

        monkeypatch.setattr(settings, "DRIFT_WARNING_THRESHOLD", 0.25)
        monkeypatch.setattr(settings, "DRIFT_CRITICAL_THRESHOLD", 0.5)
        comparisons = {
            "query_length": {"tested": True, "drifted": True, "degraded": False},
            "latency_ms": {"tested": True, "drifted": True, "degraded": False},
            "top1_score": {"tested": True, "drifted": False, "degraded": False},
            "mean_topk_score": {"tested": True, "drifted": False, "degraded": False},
        }
        severity, score = _classify_severity(comparisons)
        assert severity == "warning"
        assert score == pytest.approx(0.5)

    def test_quality_degradation_at_critical_volume_is_critical(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "DRIFT_WARNING_THRESHOLD", 0.25)
        monkeypatch.setattr(settings, "DRIFT_CRITICAL_THRESHOLD", 0.5)
        comparisons = {
            "query_length": {"tested": True, "drifted": True, "degraded": False},
            "latency_ms": {"tested": True, "drifted": True, "degraded": False},
            "top1_score": {"tested": True, "drifted": True, "degraded": True},
            "mean_topk_score": {"tested": True, "drifted": False, "degraded": False},
        }
        severity, score = _classify_severity(comparisons)
        assert severity == "critical"

    def test_quality_improvement_is_not_critical(self, monkeypatch):
        """A quality column drifting UPWARD (getting better) must never be
        classified critical, even at high dataset_score."""
        from app.core.config import settings

        monkeypatch.setattr(settings, "DRIFT_WARNING_THRESHOLD", 0.25)
        monkeypatch.setattr(settings, "DRIFT_CRITICAL_THRESHOLD", 0.5)
        comparisons = {
            "query_length": {"tested": True, "drifted": True, "degraded": False},
            "latency_ms": {"tested": True, "drifted": True, "degraded": False},
            "top1_score": {"tested": True, "drifted": True, "degraded": False},  # improved
            "mean_topk_score": {"tested": True, "drifted": False, "degraded": False},
        }
        severity, score = _classify_severity(comparisons)
        assert severity == "warning"


class TestLoadReference:
    def test_missing_file_returns_empty(self, tmp_path):
        rows = _load_reference(str(tmp_path / "does_not_exist.jsonl"))
        assert rows == []

    def test_parses_valid_jsonl(self, tmp_path):
        path = tmp_path / "ref.jsonl"
        path.write_text('{"query": "a"}\n{"query": "b"}\n')
        rows = _load_reference(str(path))
        assert rows == [{"query": "a"}, {"query": "b"}]

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "ref.jsonl"
        path.write_text('{"query": "a"}\nnot json\n{"query": "b"}\n')
        rows = _load_reference(str(path))
        assert rows == [{"query": "a"}, {"query": "b"}]


class TestRunDriftEvalOnce:
    def test_no_reference_file_skips(self, monkeypatch, tmp_path):
        from app.core.config import settings

        monkeypatch.setattr(settings, "DRIFT_REFERENCE_PATH", str(tmp_path / "missing.jsonl"))
        result = run_drift_eval_once()
        assert result == {"skipped": "no_reference_yet"}

    def test_mongo_unavailable_skips(self, monkeypatch, tmp_path):
        from app.core.config import settings

        ref_path = tmp_path / "ref.jsonl"
        ref_path.write_text('{"query": "a", "top1_score": 0.8}\n')
        monkeypatch.setattr(settings, "DRIFT_REFERENCE_PATH", str(ref_path))

        fake_infra = MagicMock()
        fake_infra.get_mongo.return_value = None
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        result = run_drift_eval_once()
        assert result == {"skipped": "mongo_unavailable"}

    def test_end_to_end_with_drifted_quality_column(self, monkeypatch, tmp_path):
        from app.core.config import settings

        # Reference: 40 rows with a healthy top1_score around 0.9.
        ref_rows = [
            {
                "query": f"reference query {i}",
                "top1_score": 0.9,
                "mean_topk_score": 0.7,
                "latency_ms": 100.0,
            }
            for i in range(40)
        ]
        ref_path = tmp_path / "ref.jsonl"
        ref_path.write_text("\n".join(json.dumps(r) for r in ref_rows) + "\n")
        monkeypatch.setattr(settings, "DRIFT_REFERENCE_PATH", str(ref_path))
        monkeypatch.setattr(settings, "DRIFT_WARNING_THRESHOLD", 0.25)
        monkeypatch.setattr(settings, "DRIFT_CRITICAL_THRESHOLD", 0.25)

        # Current: same query-length/latency shape, but top1_score collapsed.
        cur_rows = [
            {
                "query": f"reference query {i}",
                "top1_score": 0.1,
                "mean_topk_score": 0.7,
                "latency_ms": 100.0,
            }
            for i in range(40)
        ]
        mock_collection = MagicMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value = cur_rows
        mock_mongo = MagicMock()
        mock_mongo.db = {settings.MONGO_EVAL_SHADOW_COLLECTION: mock_collection}
        fake_infra = MagicMock()
        fake_infra.get_mongo.return_value = mock_mongo
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        result = run_drift_eval_once()
        assert result["severity"] == "critical"
        assert result["columns"]["top1_score"]["drifted"] is True
        assert result["columns"]["top1_score"]["degraded"] is True

    def test_never_raises_on_internal_error(self, monkeypatch, tmp_path):
        from app.core.config import settings

        ref_path = tmp_path / "ref.jsonl"
        ref_path.write_text('{"query": "a", "top1_score": 0.8}\n')
        monkeypatch.setattr(settings, "DRIFT_REFERENCE_PATH", str(ref_path))

        fake_infra = MagicMock()
        fake_infra.get_mongo.side_effect = RuntimeError("boom")
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        result = run_drift_eval_once()
        assert "error" in result


class TestBuildReference:
    def test_writes_jsonl_snapshot(self, monkeypatch, tmp_path):
        from app.core.config import settings

        out_path = tmp_path / "ref.jsonl"
        monkeypatch.setattr(settings, "DRIFT_REFERENCE_PATH", str(out_path))

        rows = [{"_id": "abc123", "query": "a", "top1_score": 0.8}]
        mock_collection = MagicMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value = rows
        mock_mongo = MagicMock()
        mock_mongo.db = {settings.MONGO_EVAL_SHADOW_COLLECTION: mock_collection}
        fake_infra = MagicMock()
        fake_infra.get_mongo.return_value = mock_mongo
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        result = build_reference()
        assert result["rows_written"] == 1
        written = _load_reference(str(out_path))
        assert written == [{"query": "a", "top1_score": 0.8}]  # _id stripped

    def test_no_traffic_returns_error(self, monkeypatch, tmp_path):
        from app.core.config import settings

        monkeypatch.setattr(settings, "DRIFT_REFERENCE_PATH", str(tmp_path / "ref.jsonl"))
        mock_collection = MagicMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value = []
        mock_mongo = MagicMock()
        mock_mongo.db = {settings.MONGO_EVAL_SHADOW_COLLECTION: mock_collection}
        fake_infra = MagicMock()
        fake_infra.get_mongo.return_value = mock_mongo
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        result = build_reference()
        assert result == {"error": "no_sampled_traffic"}
