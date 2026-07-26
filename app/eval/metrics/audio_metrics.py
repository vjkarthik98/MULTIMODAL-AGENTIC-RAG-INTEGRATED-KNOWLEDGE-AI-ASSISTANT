"""Audio ASR quality metrics: WER via jiwer (or fallback).

Measures Whisper transcript quality against gold transcripts from official earnings call sources.
"""

from __future__ import annotations

from app.eval.metrics.base import MetricResult


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate using jiwer when available, manual fallback otherwise."""
    if not reference or not hypothesis:
        return float("nan")
    try:
        import jiwer

        return float(jiwer.wer(reference, hypothesis))
    except ImportError:
        # Fallback: simple word-level edit distance
        from app.eval.metrics.ocr_metrics import word_error_rate

        return word_error_rate(hypothesis, reference)


def audio_wer_batch(
    transcripts: list[str],
    gold_transcripts: list[str],
) -> MetricResult:
    """Compute mean WER over a batch of (transcript, gold) pairs."""
    if not transcripts or not gold_transcripts:
        return MetricResult.empty("audio_wer", "no transcript pairs")

    wers = []
    for hyp, ref in zip(transcripts, gold_transcripts, strict=False):
        if not ref or ref in ("TODO_fill_from_official_transcript", "TODO"):
            continue
        wer = compute_wer(hyp, ref)
        if wer == wer:  # not nan
            wers.append(min(wer, 1.0))  # cap at 1.0 (deletion-only baseline)

    if not wers:
        return MetricResult.empty("audio_wer", "all gold transcripts are TODO")

    return MetricResult(
        name="audio_wer",
        value=sum(wers) / len(wers),
        n=len(wers),
        notes=f"min={min(wers):.3f} max={max(wers):.3f}",
    )
