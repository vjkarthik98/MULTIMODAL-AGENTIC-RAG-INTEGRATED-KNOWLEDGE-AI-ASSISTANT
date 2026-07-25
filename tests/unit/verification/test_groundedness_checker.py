"""Unit tests for app/verification/groundedness_checker.py — Responsibility 2.
Wraps reasoning_engine._hallucination_guard / _unsupported_numbers (both pure
functions — no LLM call, no GPU). Confirms the wrapper's scoring/penalty
logic, not the underlying guard (which has its own tests in
tests/unit/reasoning/).
"""

from app.verification.groundedness_checker import GroundednessChecker


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
