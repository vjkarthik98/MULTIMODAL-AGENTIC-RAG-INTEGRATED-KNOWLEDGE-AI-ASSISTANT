"""Unit tests for app/verification/confidence_scorer.py — PASS/FAIL decision
against settings.AGENT_VERIFY_* thresholds (default: retrieval/grounding/overall
>= 90, citation >= 95). No LLM, no network.
"""

from app.verification.confidence_scorer import ConfidenceScorer
from app.verification.verification_schema import (
    CitationCheckResult,
    CompletenessResult,
    GroundednessResult,
    RetrievalEvalResult,
)


def _perfect():
    return (
        RetrievalEvalResult(score=100.0),
        GroundednessResult(score=100.0),
        CitationCheckResult(score=100.0),
        CompletenessResult(aspects=["a"], covered=["a"]),
    )


class TestConfidenceScorer:

    def test_all_perfect_passes(self):
        scorer = ConfidenceScorer()
        retrieval, grounding, citation, completeness = _perfect()
        scores = scorer.score(retrieval, grounding, citation, completeness)
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "PASS"
        assert scores.overall == 100.0

    def test_weak_grounding_fails_even_with_perfect_others(self):
        scorer = ConfidenceScorer()
        retrieval, _, citation, completeness = _perfect()
        grounding = GroundednessResult(score=50.0, is_hallucinated=True)
        scores = scorer.score(retrieval, grounding, citation, completeness)
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "FAIL"
        assert "grounding" in reason or "hallucination" in reason

    def test_single_weak_dimension_drags_down_overall(self):
        # Weakest-link weighting: one bad dimension must not be diluted away
        # by three perfect ones (finance numeric errors are CRITICAL-class).
        scorer = ConfidenceScorer()
        retrieval, grounding, citation, completeness = _perfect()
        citation = CitationCheckResult(score=0.0, bad_citations=["[wrong.pdf p.9]"], checked_count=1)
        scores = scorer.score(retrieval, grounding, citation, completeness)
        # weakest=0 weighted 0.6 + mean(100,100,0,100)/4=75 weighted 0.4 = 30
        assert scores.overall < 50.0

    def test_unsupported_numbers_forces_fail(self):
        scorer = ConfidenceScorer()
        retrieval, _, citation, completeness = _perfect()
        grounding = GroundednessResult(score=95.0, is_hallucinated=False,
                                        unsupported_numbers=["120.5"])
        scores = scorer.score(retrieval, grounding, citation, completeness)
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "FAIL"
        assert "unsupported numbers" in reason

    def test_missing_aspects_forces_fail(self):
        scorer = ConfidenceScorer()
        retrieval, grounding, citation, _ = _perfect()
        completeness = CompletenessResult(aspects=["a", "b"], covered=["a"], missing=["b"])
        scores = scorer.score(retrieval, grounding, citation, completeness)
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "FAIL"
        assert "missing aspects" in reason

    def test_bad_citations_forces_fail(self):
        scorer = ConfidenceScorer()
        retrieval, grounding, _, completeness = _perfect()
        citation = CitationCheckResult(score=50.0, bad_citations=["[x.pdf p.1]"], checked_count=2)
        scores = scorer.score(retrieval, grounding, citation, completeness)
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "FAIL"
        assert "bad citations" in reason
