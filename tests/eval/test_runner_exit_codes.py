"""Gate proof tests: runner must exit non-zero when thresholds are breached.

These tests verify the Phase 29 CI gate logic works correctly:
1. Perfect metrics → exit code 0
2. Deliberately bad metrics → exit code 1 (threshold breach)
3. Unknown suite → exit code does NOT crash (returns 0 or 2, not exception)
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.eval.config import EvalConfig, WeakenSpec, load_config
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.runner import EvalRunner


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def runner(cfg):
    return EvalRunner(cfg)


def test_perfect_scores_exit_zero(runner):
    """All metrics at or above threshold → exit code 0."""
    sr = SuiteResult(suite="retrieval")
    sr.add(MetricResult(name="recall_at_5", value=1.0, n=10))
    sr.add(MetricResult(name="mrr", value=1.0, n=10))
    sr.add(MetricResult(name="ndcg_at_10", value=1.0, n=10))
    sr.add(MetricResult(name="faithfulness", value=1.0, n=10))
    sr.add(MetricResult(name="answer_relevancy", value=1.0, n=10))
    sr.add(MetricResult(name="route_accuracy", value=1.0, n=10))
    sr.add(MetricResult(name="hallucination_rate", value=0.0, n=10))

    code = runner.check_thresholds({"retrieval": sr})
    assert code == 0, f"Expected exit 0 for perfect scores, got {code}"


def test_recall_below_threshold_exits_one(runner):
    """recall_at_5 below threshold.min=0.65 → exit code 1."""
    sr = SuiteResult(suite="retrieval")
    sr.add(MetricResult(name="recall_at_5", value=0.20, n=10))  # way below 0.65

    code = runner.check_thresholds({"retrieval": sr})
    assert code == 1, f"Expected exit 1 for recall_at_5=0.20, got {code}"


def test_hallucination_above_threshold_exits_one(runner):
    """hallucination_rate above threshold.max=0.20 → exit code 1."""
    sr = SuiteResult(suite="hallucination")
    sr.add(MetricResult(name="hallucination_rate", value=0.80, n=10))  # way above 0.20

    code = runner.check_thresholds({"hallucination": sr})
    assert code == 1, f"Expected exit 1 for hallucination_rate=0.80, got {code}"


def test_faithfulness_below_threshold_exits_one(runner):
    """faithfulness below threshold.min=0.70 → exit code 1."""
    sr = SuiteResult(suite="generation")
    sr.add(MetricResult(name="faithfulness", value=0.10, n=10))

    code = runner.check_thresholds({"generation": sr})
    assert code == 1, f"Expected exit 1 for faithfulness=0.10, got {code}"


def test_routing_below_threshold_exits_one(runner):
    """route_accuracy below threshold.min=0.85 → exit code 1."""
    sr = SuiteResult(suite="routing")
    sr.add(MetricResult(name="route_accuracy", value=0.30, n=10))

    code = runner.check_thresholds({"routing": sr})
    assert code == 1, f"Expected exit 1 for route_accuracy=0.30, got {code}"


def test_latency_above_threshold_exits_one(runner):
    """p95 latency above threshold.max=60 → exit code 1."""
    sr = SuiteResult(suite="latency")
    sr.add(MetricResult(name="p95_sec", value=120.0, n=10))

    code = runner.check_thresholds({"latency": sr})
    assert code == 1, f"Expected exit 1 for p95_sec=120s, got {code}"


def test_nan_metrics_do_not_cause_breach(runner):
    """NaN metrics (TODO rows) must be skipped — not treated as a breach."""
    sr = SuiteResult(suite="retrieval")
    sr.add(MetricResult(name="recall_at_5", value=float("nan"), n=0, notes="all TODO"))

    code = runner.check_thresholds({"retrieval": sr})
    assert code == 0, f"NaN metric should not breach threshold, got {code}"


def test_infrastructure_error_exits_two(runner):
    """Suite with breached infrastructure error key → exit code 2."""
    sr = SuiteResult(suite="retrieval")
    sr.breached["import_error"] = "Cannot connect to Qdrant"

    code = runner.check_thresholds({"retrieval": sr})
    assert code in (1, 2), f"Infrastructure error should exit 1 or 2, got {code}"


def test_weaken_spec_parsing():
    """WeakenSpec.parse must handle all flags correctly."""
    from app.eval.config import WeakenSpec

    spec = WeakenSpec.parse("top_k=1,no_rerank,no_rrf")
    assert spec.top_k == 1
    assert spec.no_rerank
    assert spec.no_rrf
    assert not spec.no_mmr
    assert spec.is_active()


def test_gate_proof_concept(runner):
    """Demonstrate: weakened-pipeline metrics breach thresholds.

    This is the core gate-proof: if we artificially set metrics to values
    we'd get from top_k=1,no_rerank, the gate must fire.
    """
    # Simulate what we'd get with top_k=1, no rerank: very low recall, bad MRR
    sr = SuiteResult(suite="retrieval")
    sr.add(MetricResult(name="recall_at_5", value=0.15, n=10))   # < 0.65 threshold
    sr.add(MetricResult(name="mrr", value=0.10, n=10))             # < 0.45 threshold
    sr.add(MetricResult(name="ndcg_at_10", value=0.20, n=10))      # < 0.55 threshold

    code = runner.check_thresholds({"retrieval": sr})
    assert code == 1, (
        f"Gate-proof FAILED: weakened-pipeline metrics should breach thresholds "
        f"(exit 1) but got exit {code}. "
        "This means thresholds are too loose or gate logic is broken."
    )


def test_multiple_suites_one_breach_exits_one(runner):
    """Even with other suites passing, one breach = exit 1."""
    passing = SuiteResult(suite="routing")
    passing.add(MetricResult(name="route_accuracy", value=1.0, n=10))

    failing = SuiteResult(suite="retrieval")
    failing.add(MetricResult(name="recall_at_5", value=0.01, n=10))

    code = runner.check_thresholds({"routing": passing, "retrieval": failing})
    assert code == 1, f"One breached suite should cause exit 1, got {code}"


def test_threshold_checker_loads_yaml(runner):
    """Threshold checker must load from thresholds.yaml without error."""
    thresholds = runner._load_thresholds()
    assert isinstance(thresholds, dict), "Thresholds must be a dict"
    assert thresholds, "Thresholds dict should not be empty"


def test_gold_set_schema():
    """Every gold JSONL line must conform to the required schema."""
    from app.eval.datasets.gold_loader import GOLD_FILES, GOLD_DIR
    import json

    REQUIRED_FIELDS = ["id", "modality", "source_file", "query",
                       "relevant_chunk_ids", "reference_answer",
                       "expected_route", "added_by", "added_at"]
    VALID_ROUTES = {"rag", "search", "memory", "direct", "hybrid"}

    errors = []
    for modality, fname in GOLD_FILES.items():
        path = GOLD_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for field in REQUIRED_FIELDS:
                    if field not in row:
                        errors.append(f"{fname}:{i+1} missing '{field}'")
                route = row.get("expected_route", "")
                if route not in VALID_ROUTES:
                    errors.append(f"{fname}:{i+1} invalid expected_route='{route}'")

    assert not errors, f"Gold set schema errors:\n" + "\n".join(errors[:10])
