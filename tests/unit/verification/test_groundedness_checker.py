"""Unit tests for app/verification/groundedness_checker.py — Responsibility 2.
Wraps reasoning_engine._hallucination_guard / _unsupported_numbers (both pure
functions — no LLM call, no GPU). Confirms the wrapper's scoring/penalty
logic, not the underlying guard (which has its own tests in
tests/unit/reasoning/).
"""

from app.verification.groundedness_checker import (
    GroundednessChecker,
    _find_tabular_row_premise,
    _narrow_tabular_premise,
)


def _doc(text):
    return {"text": text, "metadata": {}}


class TestGroundednessChecker:

    def test_empty_answer_scores_zero(self):
        gc = GroundednessChecker()
        result = gc.check("", [_doc("some context")])
        assert result.score == 0.0
        assert result.is_hallucinated is True

    def test_no_docs_scores_zero_not_free_pass(self):
        gc = GroundednessChecker()
        result = gc.check("An answer with no evidence to check.", [])
        assert result.score == 0.0
        assert result.is_hallucinated is True
        assert result.unsupported_claims

    def test_well_grounded_answer_scores_high(self):
        gc = GroundednessChecker()
        docs = [_doc("Apple reported net revenue of $94.9 billion in the fourth quarter of fiscal 2024.")]
        answer = "Apple reported net revenue of $94.9 billion in the fourth quarter of fiscal 2024."
        result = gc.check(answer, docs)
        assert result.score > 80.0
        assert result.is_hallucinated is False

    def test_fabricated_number_penalized(self):
        gc = GroundednessChecker()
        docs = [_doc("Apple reported net revenue of $94.9 billion in Q4 2024.")]
        answer = "Apple reported net revenue of $250.0 billion in Q4 2024."
        result = gc.check(answer, docs)
        assert "250.0" in result.unsupported_numbers
        assert result.is_hallucinated is True

    def test_year_numbers_not_treated_as_financial_claims(self):
        gc = GroundednessChecker()
        docs = [_doc("Apple's fiscal year 2024 results were released.")]
        answer = "Apple released its fiscal year 2024 results."
        result = gc.check(answer, docs)
        assert "2024" not in result.unsupported_numbers


class TestNarrowTabularPremise:
    """xlsx grounding follow-up (2026-08-17). A single xlsx retrieval chunk
    packs 40-60+ countries' rows into ~2000 chars; live-reproduced against the
    real NLI model that passing the WHOLE chunk as premise for a one-country
    sentence measures contradiction=0.87 for a verbatim-correct claim, purely
    from dozens of other countries' similarly-shaped rating tokens diluting
    the premise — narrowing to just the matching row measured 0.0003. This
    class tests the narrowing logic itself (pure function, no GPU); the NLI
    score change is reproduced end-to-end in
    test_groundedness_checker_nli.py::TestNarrowedPremiseFixesFalseContradiction.
    """

    _SHEET = (
        "Sheet: Sovereign Ratings (Moody's,S&P) (in units)\n"
        "Columns: Country | S&P Rating | Fitch rating | Moody's rating\n"
        "Burkina Faso | CCC+ | NR | Caa1\n"
        "Cambodia | N/A | NR | B2\n"
        "Canada | AAA | AA+ | Aaa\n"
        "Cape Verde | B | B | B2\n"
    )
    _COLON_SHEET = (
        "Sheet: Sovereign Ratings (in units)\n"
        "Columns: Country Risk Premiums\n"
        "Country: Jordan, S&P Rating: BB-, Moody's rating: Ba3\n"
        "Country: Kazakhstan, S&P Rating: BBB-, Moody's rating: Baa1\n"
    )

    def test_narrows_to_matching_pipe_row(self):
        out = _narrow_tabular_premise(self._SHEET, "Canada is rated S&P AAA and Moody's Aaa.")
        assert "Canada | AAA" in out
        assert "Cambodia" not in out
        assert "Cape Verde" not in out
        # Header kept for column-name context.
        assert "Columns: Country | S&P Rating" in out

    def test_narrows_to_matching_colon_style_row(self):
        out = _narrow_tabular_premise(self._COLON_SHEET, "What is Kazakhstan's S&P rating?")
        assert "Kazakhstan" in out
        assert "Jordan" not in out

    def test_multiple_matching_rows_all_kept(self):
        # A comparison sentence naming two entities keeps both rows.
        out = _narrow_tabular_premise(self._SHEET, "Canada is AAA while Cambodia is only B2.")
        assert "Canada | AAA" in out
        assert "Cambodia" in out

    def test_no_matching_row_falls_back_to_full_text_unchanged(self):
        out = _narrow_tabular_premise(self._SHEET, "What is Japan's credit rating?")
        assert out == self._SHEET

    def test_non_tabular_text_returned_unchanged(self):
        prose = "Apple reported net revenue of $94.9 billion in Q4 2024."
        assert _narrow_tabular_premise(prose, "Apple's revenue was $94.9 billion.") == prose

    def test_short_label_not_matched_spuriously(self):
        # A 2-char row label ("US", say) must not match on incidental
        # substring presence in unrelated sentences — the >=3 char guard.
        sheet = "Sheet: X\nColumns: Country | Rating\nUS | AA+\nUK | AA\n"
        out = _narrow_tabular_premise(sheet, "us dollar strength was discussed in the call.")
        assert out == sheet

    def test_malformed_metadata_row_echoing_column_name_excluded(self):
        # Live-reproduced: a real sheet had a malformed instructions row
        # baked into the data area whose first cell is literally "Country"
        # (echoing the vague "Columns: Country and Equity Risk Premiums"
        # header, not naming an actual country) — since "country" is generic
        # enough to appear in nearly every answer about country data, this
        # row matched spuriously and got used as the NLI premise in place of
        # the real entity's row.
        sheet = (
            "Sheet: ERPs by country (in units)\n"
            "Columns: Country and Equity Risk Premiums\n"
            "Country | Africa | Moody's rating | Rating-based Default Spread | "
            "Has to be sorted in ascending order\n"
            "Switzerland | Western Europe | Aaa | 0% | 4.23% | 0% | 0% | 4.23% | 0%\n"
        )
        out = _narrow_tabular_premise(
            sheet,
            "Switzerland carries a Moody's sovereign rating of Aaa. Rating-based approach: "
            "default spread 0%, country risk premium 0%.",
        )
        assert "Switzerland | Western Europe" in out
        assert "Has to be sorted" not in out

    def test_short_rating_code_does_not_substring_match_inside_longer_code(self):
        # Live-reproduced: a row labeled "Aa1" in an UNRELATED lookup sheet
        # spuriously matched an answer stating a "Caa1" rating, because "aa1"
        # is a contiguous substring of "caa1" — with no word-boundary check,
        # any bare `in` substring test treats that as a match.
        sheet = (
            "Sheet: Default Spreads for Ratings (in units)\n"
            "Columns: Rating | Default Spread\n"
            "Aa1 | 27\n"
            "Baa1 | 45\n"
        )
        out = _narrow_tabular_premise(
            sheet, "The Sovereign Ratings sheet lists Argentina at Moody's Caa1."
        )
        assert out == sheet  # no genuine match — falls back unchanged

    def test_word_boundary_match_still_finds_genuine_short_code(self):
        # The word-boundary fix must not become too strict: a rating code
        # that genuinely appears as its own word must still match.
        sheet = "Sheet: Lookup (in units)\nColumns: Rating | Spread\nAa1 | 27\nBaa1 | 45\n"
        out = _narrow_tabular_premise(sheet, "A rating of Aa1 carries a spread of 27 bps.")
        assert "Aa1 | 27" in out
        assert "Baa1" not in out

    def test_column_name_itself_excluded_as_row_label(self):
        # A sheet whose "Columns:" line DOES declare real column names — the
        # exclusion should also catch a row echoing one of THOSE, even
        # without relying on the generic-word denylist.
        sheet = (
            "Sheet: Ratings (in units)\n"
            "Columns: Country | S&P Rating | Region\n"
            "Region | NR | placeholder metadata row not real data\n"
            "France | AA | Europe\n"
        )
        out = _narrow_tabular_premise(sheet, "France is rated AA in the Region column.")
        assert "France | AA" in out
        assert "placeholder metadata" not in out


class TestFindTabularRowPremise:
    """_find_tabular_row_premise scans EVERY retrieved doc for a matching
    row, rather than trusting _best_matching_doc_text's single word-overlap
    pick — live-reproduced (2026-08-17): in a real 48-doc xlsx retrieval,
    Switzerland's actual row ranked 5th by retrieval score while an unrelated,
    longer/denser chunk ranked 1st purely on incidental word-overlap volume
    and was selected as the NLI premise, leaving the real row never
    considered even though narrowing works perfectly once it IS considered.
    """

    def test_finds_row_in_a_lower_ranked_doc(self):
        docs = [
            {"text": "Sheet: Other (in units)\nColumns: X | Y\nUnrelated | data | here\n"},
            {"text": "Sheet: Other2 (in units)\nColumns: X | Y\nAlso | unrelated | stuff\n"},
            {
                "text": (
                    "Sheet: ERPs by country (in units)\n"
                    "Columns: Country and Equity Risk Premiums\n"
                    "Switzerland | Western Europe | Aaa | 0% | 4.23% | 0%\n"
                )
            },
        ]
        answer = "Switzerland carries a Moody's sovereign rating of Aaa with a 0% country risk premium."
        premise = _find_tabular_row_premise(answer, docs)
        assert premise is not None
        assert "Switzerland | Western Europe" in premise

    def test_entity_name_preferred_over_unrelated_rating_code_table(self):
        # Live-reproduced: a generic "Rating | Default Spread" lookup table
        # ranked ABOVE Switzerland's actual country row and had its own row
        # labeled "Aaa" — a real word-boundary match against the answer's
        # "...rating of Aaa..." clause, but the WRONG table (Switzerland is
        # never mentioned in it at all). The entity-name pass must find
        # Switzerland's row instead, even though it's listed second.
        docs = [
            {
                "text": (
                    "Sheet: Default Spreads for Ratings (in units)\n"
                    "Columns: Rating | Default Spread\n"
                    "Aaa | 0\n"
                    "Aa1 | 12\n"
                )
            },
            {
                "text": (
                    "Sheet: ERPs by country (in units)\n"
                    "Columns: Country and Equity Risk Premiums\n"
                    "Switzerland | Western Europe | Aaa | 0% | 4.23% | 0%\n"
                )
            },
        ]
        answer = "Switzerland carries a Moody's sovereign rating of Aaa with a 0% country risk premium."
        premise = _find_tabular_row_premise(answer, docs)
        assert premise is not None
        assert "Switzerland | Western Europe" in premise
        assert "Default Spreads for Ratings" not in premise

    def test_pure_rating_lookup_question_still_falls_back_to_rating_code_pass(self):
        # No entity name in the answer at all — pass 2 must still find the
        # genuine rating-to-spread answer rather than returning None.
        docs = [
            {
                "text": (
                    "Sheet: Default Spreads for Ratings (in units)\n"
                    "Columns: Rating | Default Spread\n"
                    "Ba1 | 212.7\n"
                    "Caa1 | 637.2\n"
                )
            }
        ]
        answer = "A Ba1 rating carries a default spread of about 212.7 basis points."
        premise = _find_tabular_row_premise(answer, docs)
        assert premise is not None
        assert "Ba1 | 212.7" in premise

    def test_prefers_sheet_matching_most_of_the_answers_own_numbers(self):
        # Live-reproduced: "India" has a row in BOTH a detailed per-country
        # sheet and a regional weighted-average summary sheet. The summary
        # row ranked first in the doc list and was picked, but it's missing
        # most of the answer's actual figures (it has different columns
        # entirely) — the detailed sheet, which contains nearly every number
        # the answer states, must be preferred instead.
        docs = [
            {
                "text": (
                    "Sheet: Regional Weighted Averages (in millions)\n"
                    "Columns: Country | ERP | Default Spread | Tax Rate\n"
                    "India | 7.075% | 1.868% | 30%\n"
                )
            },
            {
                "text": (
                    "Sheet: ERPs by country (in units)\n"
                    "Columns: Country and Equity Risk Premiums\n"
                    "India | Asia | Baa3 | 1.868% | 7.075% | 2.845% | 1.005% | 5.235% | 0.66%\n"
                )
            },
        ]
        answer = (
            "India carries a Moody's sovereign rating of Baa3. Rating-based approach: "
            "default spread 1.868%, country risk premium 2.845%, total equity risk premium "
            "7.075%. CDS-based approach: country risk premium 1.005%, total equity risk "
            "premium 5.235%, net sovereign CDS spread (over Swiss) 0.66%."
        )
        premise = _find_tabular_row_premise(answer, docs)
        assert premise is not None
        assert "2.845%" in premise  # only present in the correct, detailed sheet
        assert "Regional Weighted Averages" not in premise

    def test_returns_none_when_no_doc_has_a_matching_row(self):
        docs = [{"text": "Sheet: X (in units)\nColumns: Country | Rating\nFrance | AA\n"}]
        assert _find_tabular_row_premise("What is Japan's rating?", docs) is None

    def test_returns_none_for_non_tabular_docs(self):
        docs = [{"text": "Apple reported net revenue of $94.9 billion in Q4 2024."}]
        assert _find_tabular_row_premise("Apple's revenue was $94.9 billion.", docs) is None

    def test_empty_docs_returns_none(self):
        assert _find_tabular_row_premise("Switzerland is rated Aaa.", []) is None
