"""Unit tests for app/verification/citation_verifier.py — Responsibilities 3+4:
does the cited chunk actually contain the claim it's cited for. No LLM, no
network; exercises the real cite-tag extraction from reasoning_engine.py.
"""

from app.verification.citation_verifier import CitationVerifier


def _doc(text, chunk_id="c1"):
    return {"text": text, "metadata": {"chunk_id": chunk_id}}


def _source(cite_key, chunk_id="c1", snippet=""):
    return {"cite_key": cite_key, "chunk_id": chunk_id, "snippet": snippet}


class TestCitationVerifier:

    def test_no_answer_or_sources_scores_perfect(self):
        cv = CitationVerifier()
        result = cv.check("", [], [])
        assert result.score == 100.0
        assert result.checked_count == 0

    def test_no_citations_in_answer_is_soft_pass(self):
        cv = CitationVerifier()
        result = cv.check(
            "An answer with no bracket tags at all.",
            [_doc("Apple reported revenue of $94.9 billion.")],
            [_source("[apple.pdf p.4]")],
        )
        assert result.checked_count == 0
        assert result.score == 100.0

    def test_correct_citation_passes(self):
        cv = CitationVerifier()
        answer = "Apple reported net revenue of $94.9 billion in Q4 2024. [apple.pdf p.4]"
        result = cv.check(
            answer,
            [_doc("Apple reported net revenue of $94.9 billion in Q4 2024.")],
            [_source("[apple.pdf p.4]")],
        )
        assert result.checked_count == 1
        assert result.bad_citations == []
        assert result.score == 100.0

    def test_citation_pointing_at_unrelated_chunk_fails(self):
        cv = CitationVerifier()
        # The cite tag exists in `sources`, but the chunk it maps to has
        # nothing to do with the claim right before the tag — the "wrong
        # page cited" failure mode.
        answer = "Apple reported net revenue of $94.9 billion in Q4 2024. [apple.pdf p.9]"
        result = cv.check(
            answer,
            [_doc("The board of directors met quarterly to review governance policy.", chunk_id="c9")],
            [_source("[apple.pdf p.9]", chunk_id="c9")],
        )
        assert "[apple.pdf p.9]" in result.bad_citations
        assert result.score == 0.0

    def test_cite_key_with_no_matching_doc_fails(self):
        cv = CitationVerifier()
        answer = "Some claim here. [missing.pdf p.1]"
        result = cv.check(
            answer,
            [_doc("Unrelated text.", chunk_id="other")],
            [_source("[missing.pdf p.1]", chunk_id="nomatch", snippet="")],
        )
        assert "[missing.pdf p.1]" in result.bad_citations

    def test_hallucinated_cite_key_not_in_valid_set_ignored(self):
        # A tag the LLM invented that isn't in the closed set of retrieved
        # sources is filtered out by _extract_cite_tags upstream — it simply
        # isn't "cited" for verification purposes (this differs from output
        # guard's job of stripping it from the visible answer).
        cv = CitationVerifier()
        answer = "Some claim. [invented_source.pdf p.99]"
        result = cv.check(
            answer,
            [_doc("Real content.", chunk_id="c1")],
            [_source("[apple.pdf p.4]", chunk_id="c1")],
        )
        assert result.checked_count == 0

    def test_partial_batch_mixed_good_and_bad(self):
        cv = CitationVerifier()
        answer = (
            "Revenue was $94.9 billion. [apple.pdf p.4] "
            "The company also discussed unrelated governance topics. [apple.pdf p.9]"
        )
        result = cv.check(
            answer,
            [
                _doc("Revenue was $94.9 billion.", chunk_id="c4"),
                _doc("Board members reviewed audit committee charters.", chunk_id="c9"),
            ],
            [
                _source("[apple.pdf p.4]", chunk_id="c4"),
                _source("[apple.pdf p.9]", chunk_id="c9"),
            ],
        )
        assert result.checked_count == 2
        assert "[apple.pdf p.9]" in result.bad_citations
        assert "[apple.pdf p.4]" not in result.bad_citations
        assert result.score == 50.0


class TestSameTagCitedMultipleTimes:
    """citation_accuracy_v2 cross-modality follow-up (2026-08-17). A single
    source cited more than once, for different claims, in one answer — a
    common pattern for finance answers reusing one document/sheet across
    several stated facts. Two bugs live-reproduced here, both now fixed:
    (1) answer.find() alone only ever locates the FIRST occurrence, so a
    fabricated/unrelated LATER claim under the same tag was invisible to the
    check; (2) fixing (1) naively still let a close-together second
    citation's 160-char lookback window bleed backward across the sentence
    boundary into the FIRST (genuinely supported) claim's text, diluting the
    word-overlap score with borrowed, irrelevant support.
    """

    def test_second_claim_under_same_tag_about_unrelated_topic_fails(self):
        cv = CitationVerifier()
        answer = (
            "Apple's gross margin was 46.2% in FY2024 [1]. "
            "iPhone unit shipments increased 15% year over year [1]."
        )
        result = cv.check(
            answer,
            [_doc("Apple's gross margin was 46.2% in FY2024.", chunk_id="c1")],
            [_source("[1]", chunk_id="c1")],
        )
        assert "[1]" in result.bad_citations
        assert result.score == 0.0

    def test_both_claims_under_same_tag_genuinely_supported_passes(self):
        cv = CitationVerifier()
        answer = (
            "Gross margin was 46.2% in FY2024 [1]. "
            "It rose from the 44.1% prior year figure [1]."
        )
        result = cv.check(
            answer,
            [_doc("Gross margin was 46.2% in FY2024, up from 44.1% in FY2023.", chunk_id="c1")],
            [_source("[1]", chunk_id="c1")],
        )
        assert result.bad_citations == []
        assert result.score == 100.0

    def test_single_use_of_a_tag_still_works(self):
        # Regression guard: the multi-occurrence loop must not change
        # behavior for the (overwhelmingly common) single-citation case.
        cv = CitationVerifier()
        answer = "Gross margin was 46.2% in FY2024 [1]."
        result = cv.check(
            answer,
            [_doc("Gross margin was 46.2% in FY2024.", chunk_id="c1")],
            [_source("[1]", chunk_id="c1")],
        )
        assert result.bad_citations == []
        assert result.score == 100.0

    def test_three_uses_second_bad_third_good_all_checked(self):
        # Neither the first nor the last occurrence being fine should mask a
        # bad claim in the middle.
        cv = CitationVerifier()
        answer = (
            "Gross margin was 46.2% in FY2024 [1]. "
            "iPhone shipments rose 15% [1]. "
            "It was up from 44.1% the prior year [1]."
        )
        result = cv.check(
            answer,
            [_doc("Gross margin was 46.2% in FY2024, up from 44.1% the prior year.", chunk_id="c1")],
            [_source("[1]", chunk_id="c1")],
        )
        assert "[1]" in result.bad_citations


class TestCommaFormattedNumbersTokenizeAsOneWord:
    """citation_accuracy_v2 cross-modality follow-up (2026-08-17). The old
    tokenizer regex `[a-zA-Z0-9%$.]+` did not include ",", so a comma-grouped
    figure like "$391,035" split into "$391" (4 chars) and "035" (3 chars) —
    BOTH under the len(w) > 4 significance filter, making the single most
    common financial-figure format in this domain invisible to the support
    check. Live-reproduced against the real apple_10k.pdf corpus: a citation
    whose chunk verbatim contained "$391,035 | $383,285 | $394,328" — exactly
    the answer's own stated figures — scored 0.0 (flagged bad) because none
    of those three numbers survived tokenization to be counted as a match.
    """

    def test_comma_grouped_dollar_figure_recognized_as_supporting_evidence(self):
        cv = CitationVerifier()
        answer = (
            "Apple's total net sales were $391,035 million, up from $383,285 million "
            "the prior year [apple_10k.pdf p.35]."
        )
        result = cv.check(
            answer,
            [_doc("Total net sales | $391,035 | $383,285 | $394,328", chunk_id="c35")],
            [_source("[apple_10k.pdf p.35]", chunk_id="c35")],
        )
        assert result.bad_citations == []
        assert result.score == 100.0

    def test_comma_grouped_figure_genuinely_absent_still_flagged(self):
        # The fix must not become a blanket pass — a comma-formatted figure
        # that really isn't in the chunk must still fail.
        cv = CitationVerifier()
        answer = "Revenue was $999,999,999 million [apple_10k.pdf p.35]."
        result = cv.check(
            answer,
            [_doc("Total net sales | $391,035 | $383,285 | $394,328", chunk_id="c35")],
            [_source("[apple_10k.pdf p.35]", chunk_id="c35")],
        )
        assert "[apple_10k.pdf p.35]" in result.bad_citations

    def test_percent_figures_still_tokenize_correctly(self):
        # Regression guard: decimal/percent numbers (no commas) already
        # worked before this fix and must keep working.
        cv = CitationVerifier()
        answer = "Gross margin was 46.2% in FY2024 [1]."
        result = cv.check(
            answer,
            [_doc("Gross Margin % | 46.2% | 44.1%", chunk_id="c1")],
            [_source("[1]", chunk_id="c1")],
        )
        assert result.bad_citations == []
