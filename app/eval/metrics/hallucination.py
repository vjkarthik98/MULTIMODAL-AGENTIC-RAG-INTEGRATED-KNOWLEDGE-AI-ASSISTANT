"""Hallucination detection metrics.

Implements cross-chunk consistency checks to surface the P0-3 gap:
hallucination guard silently passing on wrong answers (single-chunk or answer-contradicts-source).

This module MEASURES the gap — it does NOT fix the pipeline.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.eval.metrics.base import MetricResult


# Scale words → multiplier. Covers the common financial-report conventions.
_SCALE = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "trillions": 1e12, "tn": 1e12, "t": 1e12,
}

# A money/quantity token: optional $, a number with optional thousands commas and
# decimals, optionally followed by a scale word or % . We capture the number and
# the (optional) trailing unit so "$314,623 million", "314.6 billion", "2.5%" and
# bare "15,535" all parse.
_NUM_UNIT_RE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*"
    r"(thousand|thousands|million|millions|billion|billions|trillion|trillions|"
    r"bn|mn|tn|[kmbt])?\b\s*(%)?",
    re.IGNORECASE,
)

# Powers of 1000 to try when one side states an explicit scale (e.g. "314.6
# billion") and the other is a bare table figure printed "in millions".
_IMPLICIT_SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
_REL_TOL = 0.015  # 1.5% — absorbs rounding ("328.084M" reported as "328.1 billion").


class _Num:
    __slots__ = ("raw", "value", "digits", "has_unit", "is_year", "is_id", "is_pct")

    def __init__(self, raw, value, digits, has_unit, is_year, is_id, is_pct):
        self.raw, self.value, self.digits = raw, value, digits
        self.has_unit = has_unit
        self.is_year, self.is_id, self.is_pct = is_year, is_id, is_pct


def _parse_numbers(text: str) -> List[_Num]:
    """Parse quantity tokens into normalized magnitudes with metadata."""
    out: List[_Num] = []
    for m in _NUM_UNIT_RE.finditer(text or ""):
        int_part, frac_part, unit, pct = m.group(1), m.group(2), m.group(3), m.group(4)
        digits = (int_part + (frac_part or "")).replace(",", "")
        if not digits:
            continue
        try:
            base = float(int_part.replace(",", "") + ("." + frac_part if frac_part else ""))
        except ValueError:
            continue
        unit = (unit or "").lower()
        has_unit = bool(unit)
        value = base * _SCALE.get(unit, 1.0)
        int_digits = int_part.replace(",", "")
        # A bare 4-digit integer in [1900, 2099] with no unit/decimal/% is a year.
        is_year = (
            not has_unit and not frac_part and not pct
            and len(int_digits) == 4 and 1900 <= int(int_digits) <= 2099
        )
        # A bare integer with NO thousands separators, scale word, decimal or %
        # and >= 7 digits is an identifier (SEC accession/CIK, account/filing
        # no.), not a quantitative claim. Real monetary figures in prose carry
        # commas ("$1,219,355"), a scale word ("1.2 million") or a decimal — so
        # this never suppresses a genuine figure, and a bare id present in
        # context is still digit-matched anyway.
        is_id = (
            not has_unit and not frac_part and not pct
            and "," not in int_part and len(int_digits) >= 7
        )
        out.append(_Num(m.group(0).strip(), value, digits, has_unit, is_year, is_id, bool(pct)))
    return out


def _digit_match(a: str, b: str) -> bool:
    """Significant-digit containment, ignoring scale/format (handles rounding
    like 314.6 ↔ 314,623 where '3146' is a prefix of '314623')."""
    if len(a) < 2 or len(b) < 2:
        return a == b
    return a in b or b in a


def _value_match(a: float, b: float) -> bool:
    """Magnitude match within tolerance, trying implicit 1000^n scale factors so
    '314.6 billion' matches a bare '314,623' figure printed in millions."""
    if a <= 0 or b <= 0:
        return abs(a - b) < 1e-9
    for scale in _IMPLICIT_SCALES:
        for x, y in ((a, b * scale), (a * scale, b)):
            hi = max(x, y)
            if hi > 0 and abs(x - y) / hi <= _REL_TOL:
                return True
    return False


def _extract_numbers(text: str) -> List[str]:
    """Backwards-compatible helper: raw quantity tokens (used by reference check)."""
    return [n.raw.lower() for n in _parse_numbers(text)]


def _numbers_grounded(answer: str, context_texts: List[str]) -> Tuple[bool, List[str]]:
    """Return (all_grounded, ungrounded_numbers).

    A number in the answer is grounded when a context number matches it either
    by significant digits (scale/format independent) or by normalized magnitude
    within tolerance. Bare years and long identifier numbers are never flagged —
    they are not quantitative claims and previously drove false positives
    (e.g. '2023' and SEC accession '000121935523000039').
    """
    ans_numbers = _parse_numbers(answer)
    if not ans_numbers:
        return True, []

    ctx_numbers = _parse_numbers(" ".join(context_texts))
    ctx_digits = [c.digits for c in ctx_numbers]
    ctx_values = [c.value for c in ctx_numbers]

    ungrounded: List[str] = []
    for n in ans_numbers:
        if n.is_year or n.is_id:
            continue
        grounded = any(_digit_match(n.digits, cd) for cd in ctx_digits) or any(
            _value_match(n.value, cv) for cv in ctx_values
        )
        if not grounded:
            ungrounded.append(n.raw)

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
