"""Smoke test: verify the eval harness imports, loads gold, and metrics compute.

Does NOT require Qdrant/Redis/Mongo to be running. Uses purely offline logic.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# --- MetricResult ---

def test_metric_result_import():
    from app.eval.metrics.base import MetricResult, SuiteResult
    m = MetricResult(name="recall_at_5", value=0.72, n=40, notes="test")
    assert m.value == 0.72
    assert m.n == 40
    d = m.to_dict()
    assert d["name"] == "recall_at_5"
    assert d["value"] == 0.72


def test_metric_result_empty():
    from app.eval.metrics.base import MetricResult
    m = MetricResult.empty("recall_at_5", "no ground truth")
    assert math.isnan(m.value)
    assert m.n == 0


def test_suite_result_add_and_dict():
    from app.eval.metrics.base import MetricResult, SuiteResult
    s = SuiteResult(suite="retrieval")
    s.add(MetricResult(name="recall_at_5", value=0.65, n=10))
    s.add(MetricResult(name="mrr", value=0.50, n=10))
    d = s.to_dict()
    assert "recall_at_5" in d["metrics"]
    assert "mrr" in d["metrics"]


# --- Retrieval metrics ---

def test_recall_at_k_perfect():
    from app.eval.metrics.retrieval import recall_at_k
    m = recall_at_k(["a", "b", "c"], ["a", "b"], k=5)
    assert m.value == 1.0
    assert m.n == 2


def test_recall_at_k_miss():
    from app.eval.metrics.retrieval import recall_at_k
    m = recall_at_k(["x", "y", "z"], ["a", "b"], k=5)
    assert m.value == 0.0


def test_recall_at_k_partial():
    from app.eval.metrics.retrieval import recall_at_k
    m = recall_at_k(["a", "x", "y"], ["a", "b"], k=5)
    assert m.value == pytest.approx(0.5)


def test_recall_at_k_todo_returns_empty():
    from app.eval.metrics.retrieval import recall_at_k
    m = recall_at_k(["a", "b"], ["TODO_ingest_then_fill"], k=5)
    assert math.isnan(m.value)


def test_mrr_first_hit():
    from app.eval.metrics.retrieval import mrr
    m = mrr(["x", "a", "b"], ["a"])
    assert m.value == pytest.approx(0.5)  # rank 2


def test_mrr_no_hit():
    from app.eval.metrics.retrieval import mrr
    m = mrr(["x", "y"], ["a", "b"])
    assert m.value == 0.0


def test_ndcg_perfect():
    from app.eval.metrics.retrieval import ndcg_at_k
    m = ndcg_at_k(["a", "b", "c", "d", "e"], ["a", "b"], k=10)
    assert m.value == pytest.approx(1.0)


def test_context_precision():
    from app.eval.metrics.retrieval import context_precision
    docs = [
        {"metadata": {"chunk_id": "a"}, "text": "relevant"},
        {"metadata": {"chunk_id": "x"}, "text": "irrelevant"},
    ]
    m = context_precision(docs, ["a"])
    assert m.value == pytest.approx(0.5)


def test_hit_rate_hit():
    from app.eval.metrics.retrieval import hit_rate
    m = hit_rate(["a", "b", "c"], ["b"])
    assert m.value == 1.0


def test_hit_rate_miss():
    from app.eval.metrics.retrieval import hit_rate
    m = hit_rate(["x", "y"], ["a"])
    assert m.value == 0.0


def test_aggregate_retrieval_metrics():
    from app.eval.metrics.retrieval import aggregate_retrieval_metrics
    results = [
        {"retrieved_ids": ["a", "b", "c"], "relevant_ids": ["a", "b"], "query": "q1"},
        {"retrieved_ids": ["x", "y"], "relevant_ids": ["a"], "query": "q2"},
    ]
    agg = aggregate_retrieval_metrics(results)
    assert "recall_at_5" in agg
    assert "mrr" in agg
    assert "ndcg_at_10" in agg
    assert agg["recall_at_5"].n == 2  # both rows had real ground truth


# --- Routing metrics ---

def test_route_accuracy_perfect():
    from app.eval.metrics.routing import route_accuracy
    results = [
        {"expected_route": "rag", "actual_route": "rag", "query": "q1"},
        {"expected_route": "search", "actual_route": "search", "query": "q2"},
    ]
    m = route_accuracy(results)
    assert m.value == 1.0


def test_route_accuracy_partial():
    from app.eval.metrics.routing import route_accuracy
    results = [
        {"expected_route": "rag", "actual_route": "rag", "query": "q1"},
        {"expected_route": "search", "actual_route": "rag", "query": "q2"},
    ]
    m = route_accuracy(results)
    assert m.value == pytest.approx(0.5)


def test_hybrid_with_web_rate():
    from app.eval.metrics.routing import hybrid_with_web_rate
    results = [
        {"actual_route": "hybrid", "web_source_count": 3, "query": "q1"},
        {"actual_route": "hybrid", "web_source_count": 0, "query": "q2"},
    ]
    m = hybrid_with_web_rate(results)
    assert m.value == pytest.approx(0.5)


# --- Gold loader ---

def test_gold_loader_skips_todos():
    from app.eval.datasets.gold_loader import load_gold, GOLD_DIR
    rows = load_gold("txt", include_todos=False)
    for r in rows:
        assert r.get("relevant_chunk_ids") not in (
            "TODO_ingest_then_fill", ["TODO_ingest_then_fill"]
        ) or r.get("reference_answer") in ("SEARCH_REQUIRED", "INJECTION_PROBE")


def test_gold_loader_loads_routing():
    from app.eval.datasets.gold_loader import load_gold
    rows = load_gold("routing", include_todos=False)
    # routing rows have no chunk IDs — they're always "curated" if expected_route present
    assert len(rows) >= 8, f"Expected >=8 routing gold triples, got {len(rows)}"
    for r in rows:
        assert r.get("expected_route") in {"rag", "search", "memory", "direct", "hybrid"}


def test_gold_stats():
    from app.eval.datasets.gold_loader import gold_stats
    stats = gold_stats()
    assert "txt" in stats
    assert "routing" in stats
    assert stats["routing"]["total"] >= 8


# --- Config ---

def test_eval_config_loads():
    from app.eval.config import load_config
    cfg = load_config()
    assert cfg.user_id
    assert cfg.gold_dir.exists()
    assert cfg.thresholds_path.exists()


def test_weaken_spec_parse():
    from app.eval.config import WeakenSpec
    spec = WeakenSpec.parse("top_k=1,no_rerank,no_mmr")
    assert spec.top_k == 1
    assert spec.no_rerank is True
    assert spec.no_mmr is True
    assert spec.no_rrf is False
    assert spec.is_active()


def test_weaken_spec_empty():
    from app.eval.config import WeakenSpec
    spec = WeakenSpec.parse(None)
    assert not spec.is_active()


# --- Thresholds yaml ---

def test_thresholds_yaml_valid():
    import yaml
    from app.eval.config import THRESHOLDS_PATH
    assert THRESHOLDS_PATH.exists(), "thresholds.yaml must exist"
    with open(THRESHOLDS_PATH) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "thresholds.yaml must be a YAML dict"
    # Check at least one section is present
    assert any(k in data for k in ("retrieval", "generation", "hallucination", "routing"))


def test_thresholds_have_why_comments():
    """Verify all threshold entries include a 'why' key (documentation requirement)."""
    import yaml
    from app.eval.config import THRESHOLDS_PATH
    with open(THRESHOLDS_PATH) as f:
        data = yaml.safe_load(f)

    def _check_section(d, path=""):
        for k, v in d.items():
            if isinstance(v, dict):
                if "min" in v or "max" in v:
                    assert "why" in v, f"Threshold '{path}.{k}' is missing 'why' rationale"
                else:
                    _check_section(v, f"{path}.{k}")

    _check_section(data)


# --- Runner (offline, no infra) ---

def test_runner_instantiates():
    from app.eval.config import load_config
    from app.eval.runner import EvalRunner
    cfg = load_config()
    runner = EvalRunner(cfg)
    assert runner is not None


def test_runner_unknown_suite_does_not_crash():
    from app.eval.config import load_config
    from app.eval.runner import EvalRunner
    cfg = load_config()
    runner = EvalRunner(cfg)
    results = runner.run(["__nonexistent__"])
    assert "__nonexistent__" in results
    assert "unknown_suite" in results["__nonexistent__"].breached


def test_threshold_checker_no_breaches():
    from app.eval.metrics.base import MetricResult, SuiteResult
    from app.eval.runner import EvalRunner
    from app.eval.config import load_config

    cfg = load_config()
    runner = EvalRunner(cfg)

    # Perfect scores should never breach
    sr = SuiteResult(suite="retrieval")
    sr.add(MetricResult(name="recall_at_5", value=1.0, n=10))
    sr.add(MetricResult(name="mrr", value=1.0, n=10))
    code = runner.check_thresholds({"retrieval": sr})
    assert code == 0


def test_threshold_checker_detects_breach():
    from app.eval.metrics.base import MetricResult, SuiteResult
    from app.eval.runner import EvalRunner
    from app.eval.config import load_config

    cfg = load_config()
    runner = EvalRunner(cfg)

    # Clearly below minimum — should breach
    sr = SuiteResult(suite="test")
    sr.add(MetricResult(name="recall_at_5", value=0.01, n=10))
    code = runner.check_thresholds({"test": sr})
    assert code == 1


# --- Report writer ---

def test_report_writer_creates_files(tmp_path):
    from app.eval.metrics.base import MetricResult, SuiteResult
    from app.eval.report import write_reports
    from app.eval.config import load_config

    cfg = load_config()
    cfg.reports_dir = tmp_path

    sr = SuiteResult(suite="retrieval")
    sr.add(MetricResult(name="recall_at_5", value=0.72, n=10))

    json_p, md_p = write_reports({"retrieval": sr}, cfg=cfg)
    assert json_p.exists()
    assert md_p.exists()

    import json
    with open(json_p) as f:
        data = json.load(f)
    assert "suites" in data
    assert "retrieval" in data["suites"]
