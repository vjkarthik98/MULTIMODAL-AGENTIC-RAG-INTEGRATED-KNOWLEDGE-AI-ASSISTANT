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

    def test_partial_completeness_no_longer_categorically_fails(self):
        # Changed 2026-08-13 (hallucination-reduction initiative Phase 3):
        # a single missing aspect used to force FAIL regardless of scores,
        # double-penalizing multi-part questions that were otherwise safe
        # (see confidence_scorer.py's decide() comment for the full
        # rationale + live evidence). completeness_score still costs real
        # points in `overall` (100*1/2=50 here -> overall=65, still above
        # the recalibrated AGENT_VERIFY_OVERALL_MIN=50) — proportional, not
        # categorical.
        scorer = ConfidenceScorer()
        retrieval, grounding, citation, _ = _perfect()
        completeness = CompletenessResult(aspects=["a", "b"], covered=["a"], missing=["b"])
        scores = scorer.score(retrieval, grounding, citation, completeness)
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "PASS"
        assert "missing aspects" not in reason

    def test_severe_incompleteness_still_fails_via_overall_score(self):
        # The proportional mechanism still catches genuinely bad completeness
        # — it just goes through the score, not a categorical override.
        scorer = ConfidenceScorer()
        retrieval, grounding, citation, _ = _perfect()
        completeness = CompletenessResult(
            aspects=["a", "b", "c", "d"], covered=["a"], missing=["b", "c", "d"]
        )
        scores = scorer.score(retrieval, grounding, citation, completeness)
        # completeness_score=25 -> weakest=25, mean=(100*3+25)/4=81.25,
        # overall=0.6*25+0.4*81.25=15+32.5=47.5 < 50 -> FAIL
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "FAIL"
        assert "overall" in reason

    def test_bad_citations_forces_fail(self):
        scorer = ConfidenceScorer()
        retrieval, grounding, _, completeness = _perfect()
        citation = CitationCheckResult(score=50.0, bad_citations=["[x.pdf p.1]"], checked_count=2)
        scores = scorer.score(retrieval, grounding, citation, completeness)
        decision, reason = scorer.decide(scores, grounding, citation, completeness)
        assert decision == "FAIL"
        assert "bad citations" in reason
