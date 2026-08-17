"""Unit tests for app/eval/metrics/generation.py::_deterministic_context_recall
— the year/date exclusion added in the per-modality quality pass (Phase C,
2026-08-13).

The old rule ("any token with >=2 digits is a specific fact the context must
contain") counted calendar years and day-of-month numbers as facts. That
systematically punished modalities whose CONTEXT does not restate dates:
measured on audio (FOMC press conferences, where speakers say "this year"
while the gold reference writes "September 18, 2024"), 8 of 10 measurable rows
were missing ONLY a year/date fragment, pinning audio's context_recall at
0.583 — worst of all 7 modalities — for a date convention rather than any
retrieval failure. Document modalities were unaffected because a 10-K repeats
the year throughout, making the cross-modality comparison apples-to-oranges.
"""

from __future__ import annotations

from app.eval.metrics.generation import _deterministic_context_recall as context_recall


class TestDateAndYearExclusion:
    def test_missing_year_alone_is_not_penalised(self):
        assert (
            context_recall(
                "At the September 18, 2024 meeting rates fell to 4.75-5.00 percent.",
                ["the Committee lowered the target range to 4.75 to 5.00 percent"],
            )
            == 1.0
        )

    def test_slash_date_not_penalised(self):
        assert (
            context_recall("On 9/28/24 Apple was ~$429.", ["Apple Inc.=~$429 at that tick"])
            == 1.0
        )

    def test_bare_standalone_year_not_a_fact(self):
        # Reference has ONLY a year as a numeric token -> nothing measurable.
        assert context_recall("The decision was made in 2024.", ["some unrelated context"]) is None


class TestGenuineFactsStillScored:
    def test_missing_real_figure_still_zero(self):
        assert (
            context_recall(
                "Payroll gains averaged 116,000 per month.",
                ["Payroll job gains have slowed in recent months."],
            )
            == 0.0
        )

    def test_bare_comma_integer_counts_as_recoverable_fact(self):
        # Guard against over-correcting via hallucination._is_material_figure,
        # which requires a unit/percent/decimal and would drop "116,000".
        assert (
            context_recall(
                "Payroll gains averaged 116,000 per month.",
                ["job gains averaged 116,000 per month"],
            )
            == 1.0
        )

    def test_participant_counts_kept(self):
        assert (
            context_recall(
                "All 19 participants wrote down cuts; 10 of the 19 wrote four.",
                ["of the 19 wrote down three or more cuts and 10 of the 19 wrote four"],
            )
            == 1.0
        )

    def test_partial_recall_still_partial(self):
        assert (
            context_recall(
                "Unemployment was 3.5 percent and inflation was 4.2 percent.",
                ["inflation was 4.2 percent at the time"],
            )
            == 0.5
        )

    def test_no_numeric_facts_is_unmeasurable(self):
        assert context_recall("The Committee held rates steady.", ["some context"]) is None

    def test_empty_inputs_unmeasurable(self):
        assert context_recall("", ["ctx"]) is None
        assert context_recall("ref with 4.2 percent", []) is None


class TestMonthYearWithoutDay:
    """Regression: 'August 2024' (month + year, NO day) must strip whole.

    The first version of _CR_DATE_RE required a day, so `\\d{1,2}` greedily
    consumed the '20' of '2024', leaving a stray '24' that was then scored as
    a specific fact the context had to contain. Measured on audio: 3 of the 6
    reference facts reported missing from the transcript were this artifact.
    """

    def test_month_year_no_day_not_scored(self):
        assert (
            context_recall(
                "Total PCE prices rose 2.2 percent over the 12 months ending in August 2024.",
                ["total PCE prices rose 2.2 percent over the 12 months ending in August"],
            )
            == 1.0
        )

    def test_fy_token_not_treated_as_date(self):
        # "FY2024" must NOT be stripped as a date; it is also not a fact
        # (is_year), so a reference with only FY tokens is unmeasurable.
        assert context_recall("Revenue grew in FY2024.", ["revenue grew"]) is None

    def test_month_day_year_still_strips(self):
        assert (
            context_recall(
                "At the September 18, 2024 meeting the rate fell to 4.75 percent.",
                ["the rate fell to 4.75 percent"],
            )
            == 1.0
        )
