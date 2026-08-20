"""Unit tests for app/eval/metrics/hallucination.py — the fabrication/omission
split added in the hallucination-reduction initiative (Phase 1, 2026-08-13).

Splits the previously-blended `hallucination_rate` into `fabrication_rate`
(the model invented a number absent from context — real safety risk) and
`omission_rate` (the model omitted/misstated a number the gold reference has —
a completeness/retrieval signal, not fabrication). See the docstring on
`hallucination_flag_single` in the module under test for the full rationale.
"""

from __future__ import annotations

from app.eval.metrics.hallucination import (
    _numbers_grounded,
    fabrication_rate,
    hallucination_flag_single,
    hallucination_rate,
    omission_rate,
)

_CONTEXT = [
    "Apple reported net sales of $383,285 million and net income of "
    "$96,995 million for fiscal 2023."
]


class TestHallucinationFlagSingleSplit:
    def test_fabrication_only(self):
        # Answer states a number nowhere in context — true fabrication.
        result = hallucination_flag_single(
            "Apple's net sales were $999,999 million.", _CONTEXT, reference_answer=None
        )
        assert result["fabrication_flag"] is True
        assert result["omission_flag"] is False
        assert result["flagged"] is True

    def test_omission_only(self):
        # Answer states a DIFFERENT context-grounded figure (net income, not
        # fabricated) but never states the figure the reference expects (net
        # sales) — omission, not fabrication.
        result = hallucination_flag_single(
            "Apple's net income for fiscal 2023 was $96,995 million.",
            _CONTEXT,
            reference_answer="Apple's net sales for fiscal 2023 were $383,285 million.",
        )
        assert result["fabrication_flag"] is False
        assert result["omission_flag"] is True
        assert result["flagged"] is True

    def test_both_fabrication_and_omission(self):
        result = hallucination_flag_single(
            "Apple's net sales were $999,999 million.",
            _CONTEXT,
            reference_answer="Apple's net sales for fiscal 2023 were $383,285 million.",
        )
        assert result["fabrication_flag"] is True
        assert result["omission_flag"] is True

    def test_clean_answer_flags_neither(self):
        result = hallucination_flag_single(
            "Apple's net sales for fiscal 2023 were $383,285 million.",
            _CONTEXT,
            reference_answer="Apple's net sales for fiscal 2023 were $383,285 million.",
        )
        assert result["fabrication_flag"] is False
        assert result["omission_flag"] is False
        assert result["flagged"] is False

    def test_paraphrase_is_not_omission(self):
        # Regression fixture: same fact, different formatting — must NOT flag.
        # This is the exact case documented in hallucination.py's Check 2
        # comment as having driven hallucination_rate to 0.76/0.79 before the
        # value-based (not substring-based) comparison fix.
        result = hallucination_flag_single(
            "The FOMC cut rates by a quarter-point cut, bringing the range to 4.25%-4.5%.",
            [
                "At its December 18, 2024 meeting, the FOMC lowered the target range by "
                "1/4 percentage point (25 basis points), to 4.25-4.50 percent."
            ],
            reference_answer=(
                "At its December 18, 2024 meeting, the FOMC lowered the target range by "
                "1/4 percentage point (25 basis points), to 4.25-4.50 percent."
            ),
        )
        assert result["fabrication_flag"] is False
        assert result["omission_flag"] is False

    def test_template_leakage_folds_into_fabrication(self):
        result = hallucination_flag_single(
            "The answer is [sic] Sources Used: 3 something.", _CONTEXT, reference_answer=None
        )
        assert result["fabrication_flag"] is True
        assert result["omission_flag"] is False

    def test_insufficient_data_flags_neither(self):
        result = hallucination_flag_single("", [], reference_answer=None)
        assert result["fabrication_flag"] is False
        assert result["omission_flag"] is False
        assert result["flagged"] is False


class TestAggregateRates:
    _rows = [
        {  # fabrication only
            "query": "q1",
            "answer": "Apple's net sales were $999,999 million.",
            "contexts": _CONTEXT,
            "reference_answer": None,
        },
        {  # omission only
            "query": "q2",
            "answer": "Apple's net income for fiscal 2023 was $96,995 million.",
            "contexts": _CONTEXT,
            "reference_answer": "Apple's net sales for fiscal 2023 were $383,285 million.",
        },
        {  # clean
            "query": "q3",
            "answer": "Apple's net sales for fiscal 2023 were $383,285 million.",
            "contexts": _CONTEXT,
            "reference_answer": "Apple's net sales for fiscal 2023 were $383,285 million.",
        },
    ]

    def test_fabrication_rate_counts_only_fabrication_row(self):
        m = fabrication_rate(self._rows)
        assert m.n == 3
        assert m.value == 1 / 3

    def test_omission_rate_counts_only_omission_row(self):
        m = omission_rate(self._rows)
        assert m.n == 3
        assert m.value == 1 / 3

    def test_blended_hallucination_rate_counts_both(self):
        m = hallucination_rate(self._rows)
        assert m.n == 3
        assert m.value == 2 / 3

    def test_empty_rows_return_empty_metric(self):
        assert fabrication_rate([]).n == 0
        assert omission_rate([]).n == 0


class TestDateEmbeddedNumbersNotFlaggedAsFabrication:
    """Fabrication-rate follow-up (2026-08-17, N=3-averaged re-baseline pass).
    `is_year` alone only excludes the bare "2024" from a date; the day-of-
    month in "September 18, 2024" survives _parse_numbers as its own 2-digit
    "18" (is_year=False, is_id=False), indistinguishable from a real
    quantitative claim to _numbers_grounded — which then flags it fabricated
    whenever the retrieved context never restates the date as a phrase
    (routine for audio transcripts, where speakers say "at this meeting"
    rather than the calendar date). Live-reproduced: the SAME "18" from
    "September 18, 2024" independently flagged 3 separate audio rows in one
    hallucination-suite run, driving fabrication_rate from its v7 baseline
    (0.0653) to 0.0918 — enough to trip the gate. This is the answer-side
    twin of the reference-side fix already shipped for context_recall (see
    test_context_recall.py) — both now share app/eval/metrics/hallucination.
    py's one _CR_DATE_RE definition.
    """

    def test_day_of_month_in_date_not_flagged(self):
        answer = (
            "At the September 18, 2024 meeting, the FOMC lowered the target range "
            "for the federal funds rate by a half-percentage point to 4.75 percent "
            "to 5 percent."
        )
        context = ["The Committee lowered the target range to 4.75 to 5.00 percent."]
        grounded, ungrounded = _numbers_grounded(answer, context)
        assert grounded is True
        assert ungrounded == []

    def test_multiple_rows_with_same_date_all_clear(self):
        # The exact live-reproduced pattern: three DIFFERENT answers, same
        # meeting date, previously all independently flagged on "18".
        context = ["Chair Powell discussed the September rate decision."]
        answers = [
            "In the September 18, 2024 FOMC press conference, Chair Powell "
            "mentioned that 17 of the 19 participants wrote down three or more cuts.",
            "According to the FOMC Press Conference on September 18, 2024, "
            "Powell addressed the size of the rate cut.",
        ]
        for a in answers:
            grounded, ungrounded = _numbers_grounded(a, context)
            assert "18" not in ungrounded, f"'18' leaked through for: {a!r}"

    def test_genuine_fabricated_number_still_caught(self):
        # The fix must not become a blanket pass — a real fabricated figure,
        # unrelated to any date, must still be flagged.
        answer = "The target range was lowered to 4.75 percent to 5 percent on September 18, 2024, an unprecedented 999 basis point cut."
        context = ["The Committee lowered the target range to 4.75 to 5.00 percent."]
        grounded, ungrounded = _numbers_grounded(answer, context)
        assert grounded is False
        assert "999" in ungrounded
        assert "18" not in ungrounded

    def test_month_year_without_day_still_handled(self):
        # Regression guard matching generation.py's own equivalent test: a
        # bare "Month YYYY" (no day) must not leave a stray fragment behind.
        answer = "Total PCE prices rose 2.2 percent over the 12 months ending in August 2024."
        context = ["Total PCE prices rose 2.2 percent over the 12 months ending in August."]
        grounded, ungrounded = _numbers_grounded(answer, context)
        assert grounded is True
        assert ungrounded == []
