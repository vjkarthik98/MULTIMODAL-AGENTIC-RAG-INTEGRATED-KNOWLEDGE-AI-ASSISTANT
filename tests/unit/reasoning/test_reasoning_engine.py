"""
Unit tests for pure helper functions in app/reasoning/reasoning_engine.py.
ReasoningEngine class requires a live LLM and is not tested here.
"""
import pytest

from app.reasoning.reasoning_engine import (
    _extract_cite_tags,
    _extract_numbers,
    _hallucination_guard,
    _normalize,
    _number_in_context,
    _strip_cite_tags,
    _unsupported_numbers,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert _normalize("a  b  c") == "a b c"

    def test_none_returns_empty(self):
        assert _normalize(None) == ""

    def test_empty_string(self):
        assert _normalize("") == ""


# ---------------------------------------------------------------------------
# _strip_cite_tags
# ---------------------------------------------------------------------------

class TestStripCiteTags:

    def test_removes_bracketed_citation(self):
        text = "The revenue was $1M [report.pdf p.4]."
        result = _strip_cite_tags(text)
        assert "[report.pdf p.4]" not in result
        assert "revenue" in result

    def test_clean_text_unchanged(self):
        text = "No citations here at all."
        result = _strip_cite_tags(text)
        assert "No citations" in result

    def test_empty_string(self):
        assert _strip_cite_tags("") == ""

    def test_none_returns_empty(self):
        assert _strip_cite_tags(None) == ""

    def test_multiple_citations_stripped(self):
        text = "Fact one [a.pdf p.1] and fact two [b.pdf p.2]."
        result = _strip_cite_tags(text)
        assert "[a.pdf p.1]" not in result
        assert "[b.pdf p.2]" not in result


# ---------------------------------------------------------------------------
# _extract_cite_tags
# ---------------------------------------------------------------------------

class TestExtractCiteTags:

    def test_known_key_in_text_returned(self):
        valid = ["report.pdf p.4", "doc.pdf p.1"]
        text  = "Revenue [report.pdf p.4] was high."
        result = _extract_cite_tags(text, valid)
        assert "report.pdf p.4" in result

    def test_unknown_key_not_returned(self):
        valid  = ["known.pdf p.1"]
        text   = "Some invented [fake.pdf p.99] citation."
        result = _extract_cite_tags(text, valid)
        assert "fake.pdf p.99" not in result

    def test_empty_valid_returns_empty(self):
        assert _extract_cite_tags("some text", []) == []

    def test_empty_text_returns_empty(self):
        assert _extract_cite_tags("", ["key"]) == []

    def test_order_of_appearance_preserved(self):
        valid  = ["b.pdf", "a.pdf"]
        text   = "First a.pdf then b.pdf."
        result = _extract_cite_tags(text, valid)
        assert result.index("a.pdf") < result.index("b.pdf")


# ---------------------------------------------------------------------------
# _extract_numbers
# ---------------------------------------------------------------------------

class TestExtractNumbers:

    def test_integer_extracted(self):
        assert "42" in _extract_numbers("The answer is 42.")

    def test_decimal_extracted(self):
        assert "3.14" in _extract_numbers("Pi is approximately 3.14.")

    def test_percentage_extracted(self):
        assert "31.4%" in _extract_numbers("Growth was 31.4%.")

    def test_empty_text_returns_empty(self):
        assert _extract_numbers("") == []

    def test_no_numbers_returns_empty(self):
        assert _extract_numbers("no digits here") == []

    def test_multiple_numbers_all_extracted(self):
        result = _extract_numbers("Revenue 100 and profit 25.5%")
        assert "100" in result
        assert "25.5%" in result


# ---------------------------------------------------------------------------
# _number_in_context
# ---------------------------------------------------------------------------

class TestNumberInContext:

    def test_exact_match(self):
        assert _number_in_context("42", "The answer is 42 units.") is True

    def test_percentage_stripped_match(self):
        assert _number_in_context("31.4%", "The value was 31.4 percent.") is True

    def test_number_not_in_context(self):
        assert _number_in_context("999", "The answer is 42 units.") is False

    def test_empty_number_returns_false(self):
        assert _number_in_context("", "some context") is False

    def test_empty_context_returns_false(self):
        assert _number_in_context("42", "") is False


# ---------------------------------------------------------------------------
# _unsupported_numbers
# ---------------------------------------------------------------------------

class TestUnsupportedNumbers:

    def test_supported_number_not_flagged(self):
        docs   = [{"text": "Revenue was 100 million."}]
        answer = "Revenue was 100 million."
        result = _unsupported_numbers(answer, docs)
        assert "100" not in result

    def test_unsupported_number_flagged(self):
        docs   = [{"text": "Revenue was 100 million."}]
        answer = "Revenue was 999 million."
        result = _unsupported_numbers(answer, docs)
        assert "999" in result

    def test_no_numbers_returns_empty(self):
        docs   = [{"text": "some context"}]
        answer = "no numbers here at all"
        assert _unsupported_numbers(answer, docs) == []

    def test_empty_docs_returns_empty(self):
        assert _unsupported_numbers("answer with 42", []) == ["42"]


# ---------------------------------------------------------------------------
# _hallucination_guard
# ---------------------------------------------------------------------------

class TestHallucinationGuard:

    def test_returns_tuple(self):
        docs   = [{"text": "Machine learning is a subset of AI."}]
        answer = "Machine learning is a subset of AI."
        result = _hallucination_guard(answer, docs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_supported_answer_not_hallucinated(self):
        docs   = [{"text": "Machine learning uses statistical models to learn from data patterns."}]
        answer = "Machine learning uses statistical models to learn from data patterns."
        is_hallucinated, score = _hallucination_guard(answer, docs)
        assert is_hallucinated is False
        assert score > 0.5

    def test_empty_answer_not_hallucinated(self):
        docs = [{"text": "some context here"}]
        is_hallucinated, score = _hallucination_guard("", docs)
        assert is_hallucinated is False

    def test_empty_docs_not_hallucinated(self):
        is_hallucinated, score = _hallucination_guard("some answer", [])
        assert is_hallucinated is False

    def test_score_between_0_and_1(self):
        docs   = [{"text": "The sky is blue and clouds are white."}]
        answer = "The sky is blue."
        _, score = _hallucination_guard(answer, docs)
        assert 0.0 <= score <= 1.0

    def test_completely_unsupported_may_be_hallucinated(self):
        docs   = [{"text": "The capital of France is Paris."}]
        answer = "Quantum entanglement enables faster than light communication between particles."
        is_hallucinated, score = _hallucination_guard(answer, docs, threshold=0.9)
        assert is_hallucinated is True
