"""Video quality metrics: frame-caption recall and BLIP repetition detection.

frame_caption_recall: fraction of gold key-frames with a matching caption (BLEU-1 ≥ τ).
caption_repetition_rate: detects P1-9 BLIP caption repetition loops.
"""

from __future__ import annotations

import re

from app.eval.metrics.base import MetricResult

_BLEU1_THRESHOLD = 0.40  # BLEU-1 precision to consider caption a "match"


def _bleu1_precision(candidate: str, reference: str) -> float:
    """Simple BLEU-1 unigram precision."""
    cand_tokens = re.findall(r"\b\w+\b", candidate.lower())
    ref_tokens = set(re.findall(r"\b\w+\b", reference.lower()))
    if not cand_tokens:
        return 0.0
    matches = sum(1 for t in cand_tokens if t in ref_tokens)
    return matches / len(cand_tokens)


def _has_repetition_loop(text: str, max_repeat: int = 3) -> bool:
    """Detect P1-9: same word appearing max_repeat+ consecutive times."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return False
    count = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            count += 1
            if count >= max_repeat:
                return True
        else:
            count = 1
    return False


def frame_caption_recall(
    generated_captions: list[str],
    gold_captions: list[str],
    threshold: float = _BLEU1_THRESHOLD,
) -> MetricResult:
    """Fraction of gold captions that have a matching generated caption (BLEU-1 ≥ threshold)."""
    if not gold_captions:
        return MetricResult.empty("frame_caption_recall", "no gold captions")

    valid_pairs = [
        (g, r)
        for g, r in zip(generated_captions, gold_captions)
        if r and r not in ("TODO_fill_after_processing", "TODO")
    ]
    if not valid_pairs:
        return MetricResult.empty("frame_caption_recall", "all gold captions are TODO")

    matches = sum(1 for gen, ref in valid_pairs if _bleu1_precision(gen, ref) >= threshold)
    return MetricResult(
        name="frame_caption_recall",
        value=matches / len(valid_pairs),
        n=len(valid_pairs),
        notes=f"matches={matches}/{len(valid_pairs)} at BLEU1≥{threshold}",
    )


def caption_repetition_rate(captions: list[str]) -> MetricResult:
    """Fraction of captions with a repetition loop (P1-9 BLIP bug detection)."""
    if not captions:
        return MetricResult.empty("caption_repetition_rate", "no captions")

    loopy = sum(1 for c in captions if _has_repetition_loop(c))
    return MetricResult(
        name="caption_repetition_rate",
        value=loopy / len(captions),
        n=len(captions),
        notes=f"loopy={loopy}/{len(captions)}",
    )
