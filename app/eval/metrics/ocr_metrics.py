"""OCR quality metrics: CER, WER, exact-match for image/document OCR output.

Measures the P1-8 gap: poor OCR quality on dense rendered finance text.
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.eval.metrics.base import MetricResult


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\$\.\,\%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings (character level)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def character_error_rate(hypothesis: str, reference: str) -> float:
    """CER: edit distance at character level / len(reference)."""
    if not reference:
        return float("nan")
    hyp = _normalize(hypothesis)
    ref = _normalize(reference)
    if not ref:
        return float("nan")
    return _edit_distance(hyp, ref) / len(ref)


def word_error_rate(hypothesis: str, reference: str) -> float:
    """WER: edit distance at word level / len(reference_words)."""
    if not reference:
        return float("nan")
    hyp_words = _normalize(hypothesis).split()
    ref_words = _normalize(reference).split()
    if not ref_words:
        return float("nan")
    return _edit_distance(hyp_words, ref_words) / len(ref_words)


def exact_match(hypothesis: str, reference: str) -> float:
    """Exact match after normalization (1.0 or 0.0)."""
    return 1.0 if _normalize(hypothesis) == _normalize(reference) else 0.0


def ocr_metrics_batch(
    hypotheses: List[str],
    references: List[str],
) -> dict:
    """Compute CER, WER, exact_match over a batch of (hypothesis, reference) pairs.

    Returns {metric_name: MetricResult}.
    """
    from app.eval.metrics.base import MetricResult

    if not hypotheses or not references or len(hypotheses) != len(references):
        return {
            "ocr_cer": MetricResult.empty("ocr_cer", "no pairs or length mismatch"),
            "ocr_wer": MetricResult.empty("ocr_wer", "no pairs or length mismatch"),
            "ocr_exact_match": MetricResult.empty("ocr_exact_match", "no pairs"),
        }

    cers, wers, exacts = [], [], []
    for hyp, ref in zip(hypotheses, references):
        if not ref or ref == "TODO_fill_after_ocr":
            continue
        cer = character_error_rate(hyp, ref)
        wer = word_error_rate(hyp, ref)
        em = exact_match(hyp, ref)
        if cer == cer:  # skip nan
            cers.append(cer)
        if wer == wer:
            wers.append(wer)
        exacts.append(em)

    def _mean_result(vals: List[float], name: str) -> MetricResult:
        if not vals:
            return MetricResult.empty(name, "no valid reference pairs")
        return MetricResult(name=name, value=sum(vals) / len(vals), n=len(vals))

    return {
        "ocr_cer": _mean_result(cers, "ocr_cer"),
        "ocr_wer": _mean_result(wers, "ocr_wer"),
        "ocr_exact_match": _mean_result(exacts, "ocr_exact_match"),
    }
