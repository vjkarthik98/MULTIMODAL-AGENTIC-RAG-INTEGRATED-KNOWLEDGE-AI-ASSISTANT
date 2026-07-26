"""Known-input → known-output tests for every metric module.

Each test provides a hand-crafted input where the expected output is computable
by inspection. This catches algorithmic regressions in the metric functions
themselves (independent of the pipeline).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Retrieval metrics ────────────────────────────────────────────────────────

from app.eval.metrics.retrieval import (
    context_precision,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        m = recall_at_k(["a", "b", "c"], ["a", "b"], k=5)
        assert m.value == pytest.approx(1.0)

    def test_zero_recall(self):
        m = recall_at_k(["x", "y"], ["a", "b"], k=5)
        assert m.value == pytest.approx(0.0)

    def test_partial_recall(self):
        m = recall_at_k(["a", "x", "y"], ["a", "b"], k=5)
        assert m.value == pytest.approx(0.5)

    def test_recall_respects_k(self):
        # 'b' is at position 3, which is beyond k=2
        m = recall_at_k(["a", "x", "b"], ["a", "b"], k=2)
        assert m.value == pytest.approx(0.5)

    def test_empty_relevant(self):
        m = recall_at_k(["a"], [], k=5)
        assert math.isnan(m.value)

    def test_empty_retrieved(self):
        m = recall_at_k([], ["a"], k=5)
        assert m.value == pytest.approx(0.0)


class TestMRR:
    def test_first_hit_position_1(self):
        m = mrr(["a", "b", "c"], ["a"])
        assert m.value == pytest.approx(1.0)

    def test_first_hit_position_2(self):
        m = mrr(["x", "a", "c"], ["a"])
        assert m.value == pytest.approx(0.5)

    def test_first_hit_position_3(self):
        m = mrr(["x", "y", "a"], ["a"])
        assert m.value == pytest.approx(1 / 3)

    def test_no_hit(self):
        m = mrr(["x", "y"], ["a"])
        assert m.value == pytest.approx(0.0)

    def test_empty_relevant(self):
        m = mrr(["a"], [])
        assert math.isnan(m.value)


class TestNDCG:
    def test_perfect_ndcg(self):
        m = ndcg_at_k(["a", "b"], ["a", "b"], k=5)
        assert m.value == pytest.approx(1.0)

    def test_swapped_reduces_ndcg(self):
        # ideal: a at pos 0, b at pos 1 → DCG = 1 + 1/log2(3)
        # actual: b at pos 0, a at pos 1 → same DCG (both relevant, equal binary gain)
        m_swapped = ndcg_at_k(["b", "a"], ["a", "b"], k=5)
        assert m_swapped.value == pytest.approx(1.0)

    def test_irrelevant_first_reduces_ndcg(self):
        # relevant item 'a' is at position 2 (index 1) instead of position 1 (index 0)
        m_delayed = ndcg_at_k(["x", "a"], ["a"], k=5)
        m_perfect = ndcg_at_k(["a", "x"], ["a"], k=5)
        assert m_delayed.value < m_perfect.value

    def test_zero_ndcg(self):
        m = ndcg_at_k(["x", "y"], ["a", "b"], k=5)
        assert m.value == pytest.approx(0.0)

    def test_empty_relevant(self):
        m = ndcg_at_k(["a"], [], k=5)
        assert math.isnan(m.value)


class TestPrecisionAtK:
    def test_all_relevant(self):
        m = precision_at_k(["a", "b"], ["a", "b", "c"], k=2)
        assert m.value == pytest.approx(1.0)

    def test_half_relevant(self):
        m = precision_at_k(["a", "x"], ["a", "b"], k=2)
        assert m.value == pytest.approx(0.5)

    def test_none_relevant(self):
        m = precision_at_k(["x", "y"], ["a", "b"], k=2)
        assert m.value == pytest.approx(0.0)


class TestContextPrecision:
    def test_all_on_topic(self):
        docs = [{"metadata": {"chunk_id": "a"}}, {"metadata": {"chunk_id": "b"}}]
        m = context_precision(docs, ["a", "b"])
        assert m.value == pytest.approx(1.0)

    def test_none_on_topic(self):
        docs = [{"metadata": {"chunk_id": "x"}}, {"metadata": {"chunk_id": "y"}}]
        m = context_precision(docs, ["a", "b"])
        assert m.value == pytest.approx(0.0)


class TestHitRate:
    def test_hit(self):
        m = hit_rate(["a", "b"], ["a"])
        assert m.value == pytest.approx(1.0)

    def test_miss(self):
        m = hit_rate(["x", "y"], ["a"])
        assert m.value == pytest.approx(0.0)


# ── OCR metrics ──────────────────────────────────────────────────────────────

from app.eval.metrics.ocr_metrics import (
    character_error_rate,
    exact_match,
    ocr_metrics_batch,
    word_error_rate,
)


class TestCER:
    def test_perfect(self):
        assert character_error_rate("hello", "hello") == pytest.approx(0.0)

    def test_one_substitution(self):
        # "hello" vs "hXllo" — 1 char edit, ref len 5
        assert character_error_rate("hXllo", "hello") == pytest.approx(1 / 5)

    def test_empty_hypothesis(self):
        # deletes all 5 reference chars → CER = 5/5 = 1.0
        assert character_error_rate("", "hello") == pytest.approx(1.0)

    def test_empty_reference(self):
        assert math.isnan(character_error_rate("hello", ""))

    def test_normalisation_ignored(self):
        # Normalisation lowercases — should still match
        assert character_error_rate("HELLO", "hello") == pytest.approx(0.0)


class TestWER:
    def test_perfect(self):
        assert word_error_rate("the cat sat", "the cat sat") == pytest.approx(0.0)

    def test_one_substitution(self):
        # "the dog sat" vs "the cat sat" — 1 word error, 3 reference words
        assert word_error_rate("the dog sat", "the cat sat") == pytest.approx(1 / 3)

    def test_all_wrong(self):
        # 3 deletions / 3 reference words
        assert word_error_rate("", "the cat sat") == pytest.approx(1.0)

    def test_empty_reference(self):
        assert math.isnan(word_error_rate("hello", ""))


class TestExactMatch:
    def test_match(self):
        assert exact_match("Hello World", "hello world") == pytest.approx(1.0)

    def test_no_match(self):
        assert exact_match("Hello World", "hello earth") == pytest.approx(0.0)


class TestOCRBatch:
    def test_batch_basic(self):
        results = ocr_metrics_batch(["hello world"], ["hello world"])
        assert results["ocr_cer"].value == pytest.approx(0.0)
        assert results["ocr_wer"].value == pytest.approx(0.0)
        assert results["ocr_exact_match"].value == pytest.approx(1.0)

    def test_batch_skips_todo(self):
        results = ocr_metrics_batch(["anything"], ["TODO_fill_after_ocr"])
        assert results["ocr_cer"].n == 0

    def test_batch_length_mismatch_returns_empty(self):
        results = ocr_metrics_batch(["a", "b"], ["x"])
        assert results["ocr_cer"].n == 0


# ── Audio metrics ────────────────────────────────────────────────────────────

from app.eval.metrics.audio_metrics import audio_wer_batch, compute_wer


class TestComputeWER:
    def test_perfect(self):
        assert compute_wer("revenue grew twenty percent", "revenue grew twenty percent") == pytest.approx(0.0, abs=0.05)

    def test_empty_hypothesis(self):
        result = compute_wer("", "revenue")
        assert math.isnan(result)

    def test_empty_reference(self):
        result = compute_wer("revenue", "")
        assert math.isnan(result)


class TestAudioWERBatch:
    def test_basic(self):
        m = audio_wer_batch(["hello world"], ["hello world"])
        assert m.value == pytest.approx(0.0, abs=0.05)

    def test_skips_todo(self):
        m = audio_wer_batch(["anything"], ["TODO_fill_from_official_transcript"])
        assert m.n == 0

    def test_empty_inputs(self):
        m = audio_wer_batch([], [])
        assert m.n == 0


# ── Video metrics ────────────────────────────────────────────────────────────

from app.eval.metrics.video_metrics import (
    caption_repetition_rate,
    frame_caption_recall,
)


class TestFrameCaptionRecall:
    def test_perfect(self):
        gen = ["Apple reported strong revenue growth this quarter"]
        gold = ["Apple reported strong revenue growth this quarter"]
        m = frame_caption_recall(gen, gold)
        assert m.value == pytest.approx(1.0)

    def test_zero_overlap(self):
        gen = ["completely unrelated sentence about weather"]
        gold = ["Apple revenue grew twenty percent"]
        m = frame_caption_recall(gen, gold)
        assert m.value == pytest.approx(0.0)

    def test_partial(self):
        gen = ["Apple revenue grew", "nothing useful at all"]
        gold = ["Apple revenue grew twenty percent", "JPMorgan reported strong earnings"]
        m = frame_caption_recall(gen, gold)
        # First pair passes BLEU-1 (3/4 words overlap), second fails
        assert 0.0 < m.value <= 1.0

    def test_skips_todo(self):
        gen = ["Apple revenue"]
        gold = ["TODO_fill_after_processing"]
        m = frame_caption_recall(gen, gold)
        assert m.n == 0

    def test_empty_gold(self):
        m = frame_caption_recall(["Apple"], [])
        assert m.n == 0


class TestCaptionRepetitionRate:
    def test_no_repetition(self):
        captions = [
            "Apple reported strong revenue growth",
            "JPMorgan earnings exceeded expectations",
        ]
        m = caption_repetition_rate(captions)
        assert m.value == pytest.approx(0.0)

    def test_detects_blip_loop(self):
        # P1-9: BLIP repetition bug — "invoice invoice invoice invoice"
        captions = ["invoice invoice invoice invoice invoice"]
        m = caption_repetition_rate(captions)
        assert m.value == pytest.approx(1.0)

    def test_mixed(self):
        captions = [
            "Apple reported strong revenue growth",
            "error error error error error",  # loopy
        ]
        m = caption_repetition_rate(captions)
        assert m.value == pytest.approx(0.5)

    def test_empty(self):
        m = caption_repetition_rate([])
        assert m.n == 0


# ── Latency metrics ──────────────────────────────────────────────────────────

from app.eval.metrics.latency import latency_stats


class TestLatencyStats:
    def test_percentiles(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = latency_stats(samples)
        assert stats["p50_sec"].value == pytest.approx(5.5, abs=0.1)
        assert stats["p95_sec"].value >= 9.0
        assert stats["p99_sec"].value >= 9.5

    def test_single_sample(self):
        stats = latency_stats([3.14])
        assert stats["p50_sec"].value == pytest.approx(3.14)

    def test_empty(self):
        stats = latency_stats([])
        assert math.isnan(stats["p50_sec"].value)

    def test_prefix(self):
        stats = latency_stats([1.0, 2.0], prefix="retrieval")
        assert "retrieval_p50_sec" in stats


# ── Routing metrics ──────────────────────────────────────────────────────────

from app.eval.metrics.routing import confusion_matrix, hybrid_with_web_rate, route_accuracy


class TestRouteAccuracy:
    def test_all_correct(self):
        rows = [
            {"actual_route": "rag", "expected_route": "rag"},
            {"actual_route": "search", "expected_route": "search"},
        ]
        m = route_accuracy(rows)
        assert m.value == pytest.approx(1.0)

    def test_all_wrong(self):
        rows = [
            {"actual_route": "search", "expected_route": "rag"},
            {"actual_route": "rag", "expected_route": "search"},
        ]
        m = route_accuracy(rows)
        assert m.value == pytest.approx(0.0)

    def test_half_correct(self):
        rows = [
            {"actual_route": "rag", "expected_route": "rag"},
            {"actual_route": "rag", "expected_route": "search"},
        ]
        m = route_accuracy(rows)
        assert m.value == pytest.approx(0.5)

    def test_empty(self):
        m = route_accuracy([])
        assert m.n == 0


class TestHybridWithWebRate:
    def test_hybrid_with_web(self):
        rows = [
            {"actual_route": "hybrid", "expected_route": "hybrid", "web_source_count": 2},
        ]
        m = hybrid_with_web_rate(rows)
        assert m.value == pytest.approx(1.0)

    def test_hybrid_without_web(self):
        rows = [
            {"actual_route": "hybrid", "expected_route": "hybrid", "web_source_count": 0},
        ]
        m = hybrid_with_web_rate(rows)
        assert m.value == pytest.approx(0.0)

    def test_non_hybrid_skipped(self):
        rows = [{"actual_route": "rag", "expected_route": "rag", "web_source_count": 0}]
        m = hybrid_with_web_rate(rows)
        assert m.n == 0


class TestConfusionMatrix:
    def test_diagonal(self):
        rows = [
            {"actual_route": "rag", "expected_route": "rag"},
            {"actual_route": "search", "expected_route": "search"},
        ]
        cm = confusion_matrix(rows)
        assert cm["rag"]["rag"] == 1
        assert cm["search"]["search"] == 1

    def test_off_diagonal(self):
        rows = [{"actual_route": "rag", "expected_route": "search"}]
        cm = confusion_matrix(rows)
        assert cm["search"]["rag"] == 1


# ── Hallucination metrics ────────────────────────────────────────────────────

from app.eval.metrics.hallucination import hallucination_flag_single, hallucination_rate


class TestHallucinationFlagSingle:
    def test_grounded_answer(self):
        result = hallucination_flag_single(
            answer="Revenue was $383 billion",
            contexts=["Apple reported revenue of $383 billion in FY2023"],
        )
        assert result["flagged"] is False

    def test_ungrounded_number(self):
        result = hallucination_flag_single(
            answer="Revenue was $999 billion",
            contexts=["Apple reported revenue of $383 billion in FY2023"],
        )
        # The number $999B is not in the context — should flag
        assert result["flagged"] is True

    def test_empty_contexts(self):
        result = hallucination_flag_single(
            answer="Revenue was $383 billion",
            contexts=[],
        )
        # Insufficient data to flag — returns False with "insufficient_data" reason
        assert result["flagged"] is False
        assert "insufficient_data" in result["reasons"]


class TestHallucinationRate:
    def test_zero_rate(self):
        rows = [
            {
                "answer": "Revenue was $383 billion",
                "contexts": ["Apple reported revenue of $383 billion in FY2023"],
            },
        ]
        m = hallucination_rate(rows)
        assert m.value == pytest.approx(0.0)

    def test_full_rate(self):
        rows = [
            {
                "answer": "Revenue was $999 billion",
                "contexts": ["Apple reported revenue of $383 billion"],
            },
        ]
        m = hallucination_rate(rows)
        assert m.value == pytest.approx(1.0)

    def test_empty_rows(self):
        m = hallucination_rate([])
        assert m.n == 0


# ── Generation metrics (lexical path) ────────────────────────────────────────

from app.eval.metrics.generation import template_leak_rate


class TestTemplateLeak:
    def test_clean_answer(self):
        m = template_leak_rate(["Apple's revenue grew 20% to $383 billion in FY2023."])
        assert m.value == pytest.approx(0.0)

    def test_detects_sic(self):
        m = template_leak_rate(["The answer is [sic] correct."])
        assert m.value == pytest.approx(1.0)

    def test_detects_sources_used(self):
        m = template_leak_rate(["Sources Used: 1 This is the answer."])
        assert m.value == pytest.approx(1.0)

    def test_detects_unfilled_template_var(self):
        m = template_leak_rate(["Revenue was {amount} billion."])
        assert m.value == pytest.approx(1.0)

    def test_mixed(self):
        m = template_leak_rate([
            "Apple's revenue grew 20%.",
            "Sources Used: 1 Some answer.",
        ])
        assert m.value == pytest.approx(0.5)

    def test_empty(self):
        m = template_leak_rate([])
        assert m.n == 0


# ── Verification metrics (Phase 32) ──────────────────────────────────────────

from app.eval.metrics.verification import compute_verification_metrics


def _report(verified, unsupported=None, bad_cites=None, n_attempts=1, duration_ms=1000.0):
    return {
        "verified": verified,
        "unsupported_claims": unsupported or [],
        "bad_citations": bad_cites or [],
        "attempts": [{} for _ in range(n_attempts)],
        "total_duration_ms": duration_ms,
    }


class TestComputeVerificationMetrics:

    def test_empty_reports_returns_nan_placeholders(self):
        metrics = compute_verification_metrics([])
        assert math.isnan(metrics["grounding_success_rate"].value)
        assert metrics["grounding_success_rate"].n == 0

    def test_all_grounded_and_cited(self):
        reports = [_report(True), _report(True)]
        metrics = compute_verification_metrics(reports)
        assert metrics["grounding_success_rate"].value == pytest.approx(1.0)
        assert metrics["citation_accuracy_v2"].value == pytest.approx(1.0)

    def test_grounding_success_rate_counts_unsupported_claims(self):
        reports = [_report(True), _report(False, unsupported=["fabricated: 999"])]
        metrics = compute_verification_metrics(reports)
        assert metrics["grounding_success_rate"].value == pytest.approx(0.5)

    def test_citation_accuracy_counts_bad_citations(self):
        reports = [_report(True), _report(False, bad_cites=["[wrong.pdf p.9]"])]
        metrics = compute_verification_metrics(reports)
        assert metrics["citation_accuracy_v2"].value == pytest.approx(0.5)

    def test_retry_success_rate_only_counts_retried_queries(self):
        reports = [
            _report(True, n_attempts=1),   # no retry
            _report(True, n_attempts=2),   # retried, passed
            _report(False, n_attempts=4),  # retried, exhausted
        ]
        metrics = compute_verification_metrics(reports)
        assert metrics["retry_success_rate"].n == 2  # only the 2 retried queries
        assert metrics["retry_success_rate"].value == pytest.approx(0.5)

    def test_retry_success_rate_nan_when_nothing_retried(self):
        reports = [_report(True, n_attempts=1), _report(True, n_attempts=1)]
        metrics = compute_verification_metrics(reports)
        assert math.isnan(metrics["retry_success_rate"].value)

    def test_avg_retry_count(self):
        reports = [_report(True, n_attempts=1), _report(True, n_attempts=3)]
        metrics = compute_verification_metrics(reports)
        # (0 retries + 2 retries) / 2 queries = 1.0
        assert metrics["avg_retry_count"].value == pytest.approx(1.0)

    def test_latency_percentiles(self):
        reports = [
            _report(True, duration_ms=500.0),
            _report(True, duration_ms=1000.0),
            _report(True, duration_ms=9000.0),
        ]
        metrics = compute_verification_metrics(reports)
        assert metrics["verification_latency_p50"].value == pytest.approx(1.0)
        assert metrics["verification_latency_p95"].value == pytest.approx(9.0)
