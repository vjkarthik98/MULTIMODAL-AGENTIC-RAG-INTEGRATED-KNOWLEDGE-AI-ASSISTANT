"""Hallucination detection metrics.

Implements cross-chunk consistency checks to surface the P0-3 gap:
hallucination guard silently passing on wrong answers (single-chunk or answer-contradicts-source).

This module MEASURES the gap — it does NOT fix the pipeline.
"""

from __future__ import annotations

import re
from typing import Any

from app.eval.metrics.base import MetricResult

# Scale words → multiplier. Covers the common financial-report conventions.
_SCALE = {
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
    "tn": 1e12,
    "t": 1e12,
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

_REL_TOL = 0.015  # 1.5% — absorbs rounding for hallucination grounding checks.
_FIDELITY_TOL = 0.005  # 0.5% — strict tolerance for finance_fidelity (no scale bridging).


class _Num:
    __slots__ = ("raw", "value", "digits", "has_unit", "is_year", "is_id", "is_pct")

    def __init__(self, raw, value, digits, has_unit, is_year, is_id, is_pct):
        self.raw, self.value, self.digits = raw, value, digits
        self.has_unit = has_unit
        self.is_year, self.is_id, self.is_pct = is_year, is_id, is_pct


def _parse_numbers(text: str) -> list[_Num]:
    """Parse quantity tokens into normalized magnitudes with metadata."""
    out: list[_Num] = []
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
            not has_unit
            and not frac_part
            and not pct
            and len(int_digits) == 4
            and 1900 <= int(int_digits) <= 2099
        )
        # A bare integer with NO thousands separators, scale word, decimal or %
        # and >= 7 digits is an identifier (SEC accession/CIK, account/filing
        # no.), not a quantitative claim. Real monetary figures in prose carry
        # commas ("$1,219,355"), a scale word ("1.2 million") or a decimal — so
        # this never suppresses a genuine figure, and a bare id present in
        # context is still digit-matched anyway.
        is_id = (
            not has_unit
            and not frac_part
            and not pct
            and "," not in int_part
            and len(int_digits) >= 7
        )
        out.append(_Num(m.group(0).strip(), value, digits, has_unit, is_year, is_id, bool(pct)))
    return out


def _is_material_figure(n: _Num) -> bool:
    """True when `n` is a quantitative CLAIM rather than prose scaffolding.

    Reference answers are written as natural sentences, so they are full of
    numbers that are not the fact under test:

        "At its December 18, 2024 meeting, the FOMC lowered ... by 1/4
         percentage point (25 basis points) ... to 4.25-4.50 percent."

    `18` is a calendar day, `1` and `4` are the two halves of the fraction
    "1/4", and `2024` is a year. Requiring an answer to restate all of those
    verbatim tested prose style, not correctness — a perfectly right answer
    saying "December 2024 ... a quarter point" was flagged as a hallucination.
    Confirmed empirically against the txt-0001 gold row, which is one of the
    flagged examples in CD run 31139999120.

    A figure is material when it carries a scale word ("$3.2 billion"), a
    percent sign, or a decimal fraction ("4.50") — the forms real financial
    claims take. Bare small integers are deliberately NOT material: that gives
    up flagging on things like "25 basis points" (no unit token), which is the
    conservative direction to err, since Check 1 still verifies every answer
    number against the retrieved context regardless.
    """
    if n.is_year or n.is_id:
        return False
    return n.has_unit or n.is_pct or ("." in n.raw)


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
    hi = max(a, b)
    return abs(a - b) / hi <= _REL_TOL


def _value_match_strict(a: float, b: float, tol: float = _FIDELITY_TOL) -> bool:
    """Exact match within 0.5% tolerance — no scale bridging.

    Numbers at different scales (millions vs billions) must NOT match.
    Used by compute_finance_fidelity to catch scale hallucinations.
    """
    if b == 0:
        return a == 0
    return abs(a - b) / abs(b) <= tol


def _extract_numbers(text: str) -> list[str]:
    """Backwards-compatible helper: raw quantity tokens (used by reference check)."""
    return [n.raw.lower() for n in _parse_numbers(text)]


def _extract_finance_numbers(text: str) -> list[float]:
    """Extract normalized finance number magnitudes from text.

    Skips bare years and long identifiers — only returns genuine quantitative
    claims (dollar amounts, percentages, counts with scale words).
    """
    return [n.value for n in _parse_numbers(text) if not n.is_year and not n.is_id]


def compute_finance_fidelity(answer: str, contexts: list[str]) -> float:
    """Fraction of finance numbers in the answer that appear verbatim in context.

    Uses strict magnitude matching (0.5% tolerance, no scale bridging) so that
    $4.3B in the answer does NOT match $4.3M in context.

    Returns 1.0 when the answer contains no finance numbers (nothing to verify).

    Args:
        answer:   LLM-generated answer string.
        contexts: List of retrieved context chunk strings.

    Returns:
        Float in [0.0, 1.0].
    """
    answer_nums = _extract_finance_numbers(answer)
    if not answer_nums:
        return 1.0
    context_text = " ".join(contexts)
    context_nums = _extract_finance_numbers(context_text)
    matched = sum(1 for n in answer_nums if any(_value_match_strict(n, c) for c in context_nums))
    return matched / len(answer_nums)


def _numbers_grounded(answer: str, context_texts: list[str]) -> tuple[bool, list[str]]:
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

    ungrounded: list[str] = []
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
    contexts: list[str],
    reference_answer: str | None = None,
) -> dict[str, Any]:
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
    ungrounded_numbers: list[str] = []
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
    #
    # Compares reference numbers to answer numbers by VALUE, not by raw substring.
    # The previous implementation did `n not in answer.lower()` on the reference's
    # literal token, which made this a lexical-phrasing test rather than a factual
    # one — any answer that stated the same figure differently was flagged as a
    # hallucination. That became systematic once the conversational rewrap layer
    # (rag_pipeline._conversational_rewrap) started rephrasing final answers while
    # the gold `reference_answer` fields stayed in their original terse form:
    #
    #   reference: "...by 1/4 percentage point (25 basis points), ... 4.25-4.50 percent"
    #   answer:    "...a quarter-point cut, bringing the range to 4.25%-4.5%"
    #
    # Same facts, every number grounded — but "4.50" is not a substring of the
    # answer, so the row was flagged with confidence 0.6 (>= the 0.3 threshold).
    # This is what drove hallucination_rate to 0.76/0.79 on CD run 31139999120;
    # txt-0001 (the FOMC rate-cut row) appears in that run's own flagged examples.
    #
    # `_value_match` already implements the correct comparison and is what Check 1
    # uses; reusing it makes the two checks consistent. Years and identifiers are
    # excluded for the same reason they are in Check 1 — they are not claims.
    if (
        reference_answer
        and reference_answer not in ("TODO", "")
        and "SEARCH_REQUIRED" not in reference_answer
    ):
        ref_nums = [n for n in _parse_numbers(reference_answer) if _is_material_figure(n)]
        ans_nums = [n for n in _parse_numbers(answer) if not n.is_year and not n.is_id]
        if ref_nums and ans_nums:
            ans_digits = [a.digits for a in ans_nums]
            ans_values = [a.value for a in ans_nums]
            missing = [
                r.raw
                for r in ref_nums
                if not (
                    any(_digit_match(r.digits, ad) for ad in ans_digits)
                    or any(_value_match(r.value, av) for av in ans_values)
                )
            ]
            if missing:
                # Answer states figures, but not the ones the reference expects →
                # genuinely a different (likely wrong) number, not a rewording.
                reasons.append(f"missing_reference_numbers: {missing[:3]}")
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


def hallucination_rate(eval_rows: list[dict]) -> MetricResult:
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
