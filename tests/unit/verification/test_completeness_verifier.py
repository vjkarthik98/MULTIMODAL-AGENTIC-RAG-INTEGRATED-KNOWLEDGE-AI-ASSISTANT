"""Unit tests for app/verification/completeness_verifier.py — the generalized
Phase 32 successor to the retired app/agents/video_answer_agent.py. Migrated
verbatim from tests/unit/agents/test_video_answer_agent.py: the underlying
keyword-overlap heuristic is unchanged, only the return type
(CompletenessResult, not video-only AspectCoverage) and the modality scope
(all 7 modalities, not just video). Pure-function tests: no LLM, no
retriever, no network.
"""

from app.verification.completeness_verifier import (
    CompletenessVerifier,
    assemble_answer,
    verify_aspect_coverage,
)
from app.verification.verification_schema import CompletenessResult


# ---------------------------------------------------------------------------
# verify_aspect_coverage
# ---------------------------------------------------------------------------

class TestVerifyAspectCoverage:

    def test_all_aspects_covered(self):
        answer = ("Apple expects double-digit iPhone growth in the December "
                  "quarter as part of its fiscal 2025 guidance, is thrilled "
                  "with the iPhone Air reception, is building AI foundation "
                  "models in-house, and remains open to M&A opportunities.")
        aspects = [
            "December quarter 2025 guidance",
            "the iPhone Air reception",
            "AI foundation models",
            "M&A",
        ]
        cov = verify_aspect_coverage(answer, aspects)
        assert cov.missing == []
        assert set(cov.covered) == set(aspects)

    def test_one_aspect_missing(self):
        answer = "Apple expects double-digit iPhone growth and is thrilled with the iPhone Air reception."
        aspects = [
            "December quarter 2025 guidance",
            "the iPhone Air reception",
            "AI foundation models",
            "M&A",
        ]
        cov = verify_aspect_coverage(answer, aspects)
        assert "AI foundation models" in cov.missing
        assert "M&A" in cov.missing
        assert "the iPhone Air reception" in cov.covered

    def test_empty_answer_all_missing(self):
        cov = verify_aspect_coverage("", ["revenue growth", "gross margin outlook"])
        assert cov.missing == ["revenue growth", "gross margin outlook"]
        assert cov.covered == []

    def test_no_aspects_returns_empty_coverage(self):
        cov = verify_aspect_coverage("some answer text", [])
        assert cov.aspects == []
        assert cov.covered == []
        assert cov.missing == []

    def test_aspect_with_no_content_keywords_skipped(self):
        cov = verify_aspect_coverage("some answer", ["what and how"])
        assert cov.covered == []
        assert cov.missing == []

    def test_min_overlap_threshold_respected(self):
        answer = "Apple discussed iPhone Air."
        cov = verify_aspect_coverage(answer, ["the iPhone Air reception"], min_overlap=0.34)
        assert "the iPhone Air reception" in cov.covered
        cov_strict = verify_aspect_coverage(answer, ["the iPhone Air reception"], min_overlap=0.9)
        assert "the iPhone Air reception" in cov_strict.missing

    def test_is_complete_property(self):
        cov = CompletenessResult(aspects=["a"], covered=["a"], missing=[])
        assert cov.is_complete is True
        cov2 = CompletenessResult(aspects=["a"], covered=[], missing=["a"])
        assert cov2.is_complete is False


# ---------------------------------------------------------------------------
# assemble_answer
# ---------------------------------------------------------------------------

class TestAssembleAnswer:

    def test_no_missing_aspects_returns_primary_unchanged(self):
        cov = CompletenessResult(aspects=["a"], covered=["a"], missing=[])
        result = assemble_answer("The primary answer.", cov, followup_fn=lambda a: "should not be called")
        assert result == "The primary answer."

    def test_no_followup_fn_returns_primary_unchanged(self):
        cov = CompletenessResult(aspects=["a"], covered=[], missing=["a"])
        result = assemble_answer("The primary answer.", cov, followup_fn=None)
        assert result == "The primary answer."

    def test_missing_aspect_gets_appended(self):
        cov = CompletenessResult(aspects=["M&A"], covered=[], missing=["M&A"])
        result = assemble_answer(
            "Apple discussed guidance.", cov,
            followup_fn=lambda asp: "Apple remains open to M&A opportunities.")
        assert "Apple discussed guidance." in result
        assert "Apple remains open to M&A opportunities." in result

    def test_followup_returning_none_leaves_primary_unchanged(self):
        cov = CompletenessResult(aspects=["M&A"], covered=[], missing=["M&A"])
        result = assemble_answer("Primary answer.", cov, followup_fn=lambda asp: None)
        assert result == "Primary answer."

    def test_followup_exception_does_not_propagate(self):
        def _boom(asp):
            raise RuntimeError("retrieval failed")
        cov = CompletenessResult(aspects=["M&A"], covered=[], missing=["M&A"])
        result = assemble_answer("Primary answer.", cov, followup_fn=_boom)
        assert result == "Primary answer."

    def test_max_followups_bounds_extra_llm_calls(self):
        calls = []

        def _followup(asp):
            calls.append(asp)
            return f"addendum about {asp}"

        cov = CompletenessResult(aspects=["a", "b", "c"], covered=[], missing=["a", "b", "c"])
        assemble_answer("Primary.", cov, followup_fn=_followup, max_followups=1)
        assert len(calls) == 1
        assert calls[0] == "a"

    def test_multiple_followups_when_bound_allows(self):
        def _followup(asp):
            return f"about {asp}"
        cov = CompletenessResult(aspects=["a", "b"], covered=[], missing=["a", "b"])
        result = assemble_answer("Primary.", cov, followup_fn=_followup, max_followups=2)
        assert "about a" in result
        assert "about b" in result


# ---------------------------------------------------------------------------
# CompletenessVerifier — the class VerificationLoop actually calls
# ---------------------------------------------------------------------------

class TestCompletenessVerifier:

    def test_single_aspect_no_query_provided_skips_coverage_check(self):
        # <2 aspects AND no query given: nothing to score against, so it's
        # trivially "complete" — callers that don't pass `query` (e.g. the
        # old call sites before the live-smoke-test fix) keep the old,
        # permissive behavior.
        verifier = CompletenessVerifier()
        result = verifier.check("Some answer.", ["only one aspect"])
        assert result.is_complete
        assert result.covered == ["only one aspect"]

    def test_single_aspect_question_checks_answer_against_full_query(self):
        # Regression test (live smoke test, Phase 32): a single-fact question
        # ("What was Apple's Q4 2024 net revenue?") still needs a
        # completeness check — an answer that's fluent and grounded in SOME
        # retrieved text but never actually mentions revenue/the number asked
        # about must be flagged, not auto-passed just because there was only
        # one decomposed aspect. VerificationLoop now passes `query=` here so
        # the whole question is checked as a single aspect.
        verifier = CompletenessVerifier()
        result = verifier.check(
            "Both Apple's own internal advertising and licensing individually set records during the quarter.",
            aspects=[],  # _split_query_aspects returned <2 aspects for this simple question
            query="What was Apple's Q4 2024 net revenue?",
        )
        assert not result.is_complete

    def test_single_aspect_question_passes_when_answer_is_responsive(self):
        verifier = CompletenessVerifier()
        result = verifier.check(
            "Apple's Q4 2024 net revenue was $94.9 billion.",
            aspects=[],
            query="What was Apple's Q4 2024 net revenue?",
        )
        assert result.is_complete

    def test_multi_aspect_question_runs_coverage_check(self):
        verifier = CompletenessVerifier()
        result = verifier.check(
            "Discusses revenue growth only.",
            ["revenue growth", "operating margin", "capital returns"],
        )
        assert "revenue growth" in result.covered
        assert not result.is_complete

    def test_fill_gaps_delegates_to_assemble_answer(self):
        verifier = CompletenessVerifier(max_followups=1)
        cov = CompletenessResult(aspects=["a", "b"], covered=[], missing=["a", "b"])
        result = verifier.fill_gaps("Primary.", cov, followup_fn=lambda a: f"about {a}")
        assert "about a" in result
        assert "about b" not in result  # bounded to max_followups=1
