"""app/eval/quality_push.py — report JSON -> Pushgateway text.

The dashboard must never show a score the badge script would refuse without
also showing why, so the honesty signals (judge_ok, coverage) are pinned here
against report shapes taken from the real writers.
"""

from __future__ import annotations

import pytest

from app.eval.quality_push import render

RAGAS_REPORT = {
    "mode": "production",
    "n_queries": 10,
    "n_errors": 1,
    "metrics": {
        "faithfulness": {"name": "faithfulness", "value": 0.82, "n": 8, "notes": "judge=qwen2.5_7b_direct_nli"},
        "answer_relevancy": {"name": "answer_relevancy", "value": 0.91, "n": 10, "notes": "judge=qwen2.5_7b"},
        "context_recall": {"name": "context_recall", "value": float("nan"), "n": 10, "notes": "judge=qwen2.5_7b"},
    },
    "finance_fidelity": 0.75,
    "hallucination_rate": 0.1,
}

DEEPEVAL_REPORT = {
    "summary": {
        "faithfulness": {"mean": 0.9, "n": 4},
        "answer_relevancy": {"mean": 0.8, "n": 4},
        "contextual_recall": {"mean": None, "n": 0},
    },
    "per_row": [{}, {}, {}, {}],
    "errors": [],
}


def _lines(text: str) -> set[str]:
    return {line for line in text.splitlines() if line and not line.startswith("#")}


def test_ragas_scores_and_honesty_signals():
    out = _lines(render(RAGAS_REPORT, "ragas", now=1700000000))
    assert 'magik_quality_metric{metric="faithfulness"} 0.82' in out
    assert 'magik_quality_metric{metric="answer_relevancy"} 0.91' in out
    assert 'magik_quality_metric{metric="finance_fidelity"} 0.75' in out
    assert 'magik_quality_metric_rows{metric="faithfulness"} 8' in out
    assert "magik_quality_coverage 0.8" in out
    assert "magik_quality_judge_ok 1" in out
    assert "magik_quality_rows_evaluated 10" in out
    assert "magik_quality_row_errors 1" in out
    assert "magik_quality_last_run_timestamp_seconds 1.7e+09" in out


def test_nan_is_skipped_not_zeroed():
    text = render(RAGAS_REPORT, "ragas")
    assert "context_recall" not in text
    assert not any(line.split()[-1].lower() == "nan" for line in _lines(text))


def test_lexical_fallback_marks_judge_not_ok():
    report = {
        **RAGAS_REPORT,
        "metrics": {
            "faithfulness": {"value": 0.87, "n": 10, "notes": "judge=lexical_fallback (ragas_error: x)"}
        },
    }
    assert "magik_quality_judge_ok 0" in _lines(render(report, "ragas"))


def test_deepeval_coverage_counts_metric_row_slots():
    out = _lines(render(DEEPEVAL_REPORT, "deepeval"))
    # 8 filled of 4 rows x 3 metrics = 12 slots.
    assert any(line.startswith("magik_quality_coverage 0.666") for line in out)
    assert "magik_quality_judge_ok 1" in out  # 2 of 3 metrics scored
    assert 'magik_quality_metric{metric="faithfulness"} 0.9' in out
    assert not any("contextual_recall" in line for line in out)


def test_deepeval_minority_scored_is_not_ok():
    report = {
        **DEEPEVAL_REPORT,
        "summary": {
            "a": {"mean": 1.0, "n": 1},
            "b": {"mean": None, "n": 0},
            "c": {"mean": None, "n": 0},
        },
    }
    assert "magik_quality_judge_ok 0" in _lines(render(report, "deepeval"))


def test_label_values_are_escaped():
    report = {"n_queries": 1, "metrics": {'we"ird\nname': {"value": 0.5, "n": 1, "notes": ""}}}
    assert 'magik_quality_metric{metric="we\\"ird name"} 0.5' in render(report, "ragas")


def test_unknown_tool_rejected():
    with pytest.raises(ValueError):
        render({}, "other")
