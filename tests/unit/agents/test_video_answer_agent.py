"""Unit tests for app/agents/video_answer_agent.py — the Phase-C VERIFY +
ASSEMBLE step for multi-part video answers. Pure-function tests: no LLM, no
retriever, no network.
"""

from app.agents.video_answer_agent import (
    AspectCoverage,
    assemble_answer,
    verify_aspect_coverage,
)


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
        # Answer covers guidance + iPhone Air but never mentions AI/foundation
        # models or M&A — the classic "4-part question, 1-part answer" failure.
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
        # An aspect that's entirely stopwords (after keyword extraction)
        # can't be scored either way — it's simply skipped, not counted as
        # missing (that would poison the coverage signal with noise).
        cov = verify_aspect_coverage("some answer", ["what and how"])
        assert cov.covered == []
        assert cov.missing == []

    def test_min_overlap_threshold_respected(self):
        answer = "Apple discussed iPhone Air."
        # "the iPhone Air reception" -> content words {iphone, air, reception}
        # only 2/3 present -> exactly the 0.34 default threshold boundary.
        cov = verify_aspect_coverage(answer, ["the iPhone Air reception"], min_overlap=0.34)
        assert "the iPhone Air reception" in cov.covered
        # A stricter threshold should now fail the same partial match.
        cov_strict = verify_aspect_coverage(answer, ["the iPhone Air reception"], min_overlap=0.9)
        assert "the iPhone Air reception" in cov_strict.missing


# ---------------------------------------------------------------------------
# assemble_answer
# ---------------------------------------------------------------------------

class TestAssembleAnswer:

    def test_no_missing_aspects_returns_primary_unchanged(self):
        cov = AspectCoverage(aspects=["a"], covered=["a"], missing=[])
        result = assemble_answer("The primary answer.", cov, followup_fn=lambda a: "should not be called")
        assert result == "The primary answer."

    def test_no_followup_fn_returns_primary_unchanged(self):
        cov = AspectCoverage(aspects=["a"], covered=[], missing=["a"])
        result = assemble_answer("The primary answer.", cov, followup_fn=None)
        assert result == "The primary answer."

    def test_missing_aspect_gets_appended(self):
        cov = AspectCoverage(aspects=["M&A"], covered=[], missing=["M&A"])
        result = assemble_answer(
            "Apple discussed guidance.", cov,
            followup_fn=lambda asp: "Apple remains open to M&A opportunities.")
        assert "Apple discussed guidance." in result
        assert "Apple remains open to M&A opportunities." in result

    def test_followup_returning_none_leaves_primary_unchanged(self):
        cov = AspectCoverage(aspects=["M&A"], covered=[], missing=["M&A"])
        result = assemble_answer("Primary answer.", cov, followup_fn=lambda asp: None)
        assert result == "Primary answer."

    def test_followup_exception_does_not_propagate(self):
        def _boom(asp):
            raise RuntimeError("retrieval failed")
        cov = AspectCoverage(aspects=["M&A"], covered=[], missing=["M&A"])
        result = assemble_answer("Primary answer.", cov, followup_fn=_boom)
        assert result == "Primary answer."  # degrades gracefully, never raises

    def test_max_followups_bounds_extra_llm_calls(self):
        calls = []

        def _followup(asp):
            calls.append(asp)
            return f"addendum about {asp}"

        cov = AspectCoverage(aspects=["a", "b", "c"], covered=[], missing=["a", "b", "c"])
        assemble_answer("Primary.", cov, followup_fn=_followup, max_followups=1)
        assert len(calls) == 1
        assert calls[0] == "a"

    def test_multiple_followups_when_bound_allows(self):
        def _followup(asp):
            return f"about {asp}"
        cov = AspectCoverage(aspects=["a", "b"], covered=[], missing=["a", "b"])
        result = assemble_answer("Primary.", cov, followup_fn=_followup, max_followups=2)
        assert "about a" in result
        assert "about b" in result
