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
