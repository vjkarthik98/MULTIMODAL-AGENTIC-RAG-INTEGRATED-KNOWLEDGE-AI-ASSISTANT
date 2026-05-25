"""Hallucination detection metrics.

Implements cross-chunk consistency checks to surface the P0-3 gap:
hallucination guard silently passing on wrong answers (single-chunk or answer-contradicts-source).

This module MEASURES the gap — it does NOT fix the pipeline.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.eval.metrics.base import MetricResult


# Finance-specific number pattern: integers ≥ 4 digits or decimals
_NUMBER_RE = re.compile(r"\b\d{4,}(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(?:billion|million|%|bn|m)\b", re.IGNORECASE)


def _extract_numbers(text: str) -> List[str]:
    """Extract number-like tokens from text for grounding check."""
    return _NUMBER_RE.findall(text.lower())


def _numbers_grounded(answer: str, context_texts: List[str]) -> Tuple[bool, List[str]]:
    """Return (all_grounded, ungrounded_numbers).

    A number in the answer is considered grounded if it appears in at least one
    retrieved context chunk (exact or near-exact match).
    """
    ans_numbers = _extract_numbers(answer)
    if not ans_numbers:
        return True, []

    ctx_combined = " ".join(context_texts).lower()
    ungrounded = [n for n in ans_numbers if n.replace(",", "").strip() not in ctx_combined]
    return len(ungrounded) == 0, ungrounded


def hallucination_flag_single(
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """Check a single answer for hallucination signals.

    Returns:
        {
            "flagged": bool,
            "confidence": float (0=clean, 1=likely hallucination),
            "reasons": List[str],
            "ungrounded_numbers": List[str],
        }
    """
    reasons = []
    ungrounded_numbers: List[str] = []
    confidence = 0.0

    if not answer or not contexts:
        return {
            "flagged": False,
            "confidence": 0.0,
            "reasons": ["insufficient_data"],
            "ungrounded_numbers": [],
        }

    # Check 1: numeric grounding (P0-3: numbers not in retrieved context)
    grounded, ungrounded = _numbers_grounded(answer, contexts)
    if not grounded:
        reasons.append(f"ungrounded_numbers: {ungrounded}")
        ungrounded_numbers = ungrounded
        confidence = max(confidence, 0.8)

    # Check 2: cross-chunk consistency via reference answer (if available)
    if (
        reference_answer
        and reference_answer not in ("TODO", "")
        and "SEARCH_REQUIRED" not in reference_answer
    ):
        ref_numbers = _extract_numbers(reference_answer)
        ans_numbers = _extract_numbers(answer)
        # Check if key numbers in reference are absent from answer (wrong figure)
        ref_absent_from_ans = [n for n in ref_numbers if n not in answer.lower()]
        if ref_absent_from_ans and ans_numbers:
            # Answer has numbers but not the expected ones → likely wrong figure
            reasons.append(f"missing_reference_numbers: {ref_absent_from_ans[:3]}")
            confidence = max(confidence, 0.6)

    # Check 3: template leakage (P1-7)
    template_re = re.compile(r"\[sic\]|Sources Used: \d+|\{[a-zA-Z_]+\}", re.IGNORECASE)
    if template_re.search(answer):
        reasons.append("template_leakage")
        confidence = max(confidence, 0.4)

    flagged = confidence >= 0.3
    return {
        "flagged": flagged,
        "confidence": confidence,
        "reasons": reasons,
        "ungrounded_numbers": ungrounded_numbers,
    }


def hallucination_rate(eval_rows: List[Dict]) -> MetricResult:
    """Fraction of answers with at least one hallucination signal.

    eval_rows: [{"answer", "contexts", "reference_answer", "query"}]
    """
    if not eval_rows:
        return MetricResult.empty("hallucination_rate", "no eval rows")

    flagged_count = 0
    total = 0
    flagged_examples = []

    for row in eval_rows:
        answer = row.get("answer") or ""
        contexts = row.get("contexts") or []
        if isinstance(contexts, list) and contexts and isinstance(contexts[0], dict):
            contexts = [c.get("text") or str(c) for c in contexts]
        reference = row.get("reference_answer") or ""

        if not answer:
            continue
        total += 1

        result = hallucination_flag_single(answer, contexts, reference)
        if result["flagged"]:
            flagged_count += 1
            flagged_examples.append(f"  '{row.get('query', '')[:40]}': {result['reasons']}")

    if total == 0:
        return MetricResult.empty("hallucination_rate", "no valid answers to check")

    rate = flagged_count / total
    notes = f"flagged={flagged_count}/{total}"
    if flagged_examples:
        notes += " | examples: " + "; ".join(flagged_examples[:2])

    return MetricResult(name="hallucination_rate", value=rate, n=total, notes=notes)
