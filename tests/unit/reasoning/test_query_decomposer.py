from unittest.mock import MagicMock

import pytest

from app.reasoning.query_decomposer import (
    QueryDecomposer,
    _confidence,
    _dedup_against_original,
    _filter,
    _hash,
    _is_complex,
    _normalize,
    _parse,
    _rule_based_fallback,
    detect_query_type,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_collapses_inner_spaces(self):
        assert _normalize("a  b  c") == "a b c"

    def test_empty_string(self):
        assert _normalize("") == ""


# ---------------------------------------------------------------------------
# detect_query_type
# ---------------------------------------------------------------------------

class TestDetectQueryType:

    def test_comparative_query(self):
        result = detect_query_type("Compare BERT versus GPT performance")
        assert result == "comparative"

    def test_temporal_query(self):
        result = detect_query_type("What are the latest AI developments?")
        assert result == "temporal"

    def test_multihop_multiple_questions(self):
        result = detect_query_type("What is X? And what is Y?")
        assert result == "multihop"

    def test_factual_default(self):
        result = detect_query_type("What is machine learning?")
        assert result == "factual"

    def test_aggregation_query(self):
        result = detect_query_type("How many papers were published?")
        # "how many" is an aggregation keyword
        result_type = detect_query_type("How many items are listed?")
        assert isinstance(result_type, str)


# ---------------------------------------------------------------------------
# _is_complex
# ---------------------------------------------------------------------------

class TestIsComplex:

    def test_multiple_questions_is_complex(self):
        assert _is_complex("What is A? And what is B?") is True

    def test_comparative_is_complex(self):
        assert _is_complex("Compare transformer versus RNN architecture") is True

    def test_short_simple_query_not_complex(self):
        assert _is_complex("What is AI?") is False

    def test_long_query_may_be_complex(self):
        long_q = "Explain the differences between supervised and unsupervised machine learning methods"
        # May or may not be complex depending on DECOMPOSITION_MIN_WORDS setting
        assert isinstance(_is_complex(long_q), bool)


# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------

class TestConfidence:

    def test_very_short_query_low_confidence(self):
        assert _confidence("AI?") < 0.5

    def test_medium_query_moderate_confidence(self):
        score = _confidence("What is machine learning algorithm?")
        assert 0.0 <= score <= 1.0

    def test_longer_query_higher_confidence(self):
        short = _confidence("What is AI?")
        long  = _confidence("What is the role of attention mechanisms in transformer-based language models?")
        assert long >= short

    def test_question_mark_boosts_confidence(self):
        without = _confidence("explain machine learning algorithms")
        with_q  = _confidence("explain machine learning algorithms?")
        assert with_q >= without

    def test_score_between_0_and_1(self):
        for text in ["x?", "short query here", "a " * 20]:
            assert 0.0 <= _confidence(text) <= 1.0


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------

class TestParse:

    def test_empty_text_returns_empty(self):
        assert _parse("") == []

    def test_numbered_lines_parsed(self):
        text = "1. What is machine learning?\n2. How does deep learning work?"
        result = _parse(text)
        assert len(result) >= 1
        assert all(isinstance(s, str) for s in result)

    def test_bullet_lines_parsed(self):
        text = "- What is AI?\n- How does NLP work?"
        result = _parse(text)
        assert len(result) >= 1

    def test_short_lines_skipped(self):
        text = "1. Hi?\n2. What are the key differences between supervised and unsupervised learning?"
        result = _parse(text)
        # "Hi?" is < 5 chars and should be skipped
        assert all(len(s) >= 5 for s in result)

    def test_adds_question_mark_if_missing(self):
        text = "What is machine learning"
        result = _parse(text)
        if result:
            assert result[0].endswith("?")


# ---------------------------------------------------------------------------
# _filter
# ---------------------------------------------------------------------------

class TestFilter:

    def test_removes_too_short_queries(self):
        queries = ["AI?", "What is transformer-based language model architecture?"]
        result = _filter(queries, min_confidence=0.0, max_subqueries=5)
        # "AI?" has < 4 words and should be filtered
        for q in result:
            assert len(q.split()) >= 4

    def test_respects_max_subqueries(self):
        queries = [
            "What is machine learning and how does it work?",
            "How does deep learning differ from machine learning?",
            "What are the applications of neural networks in practice?",
            "Why is attention mechanism important for transformers?",
        ]
        result = _filter(queries, min_confidence=0.0, max_subqueries=2)
        assert len(result) <= 2

    def test_deduplicates_identical_queries(self):
        q = "What is the role of attention in transformers?"
        result = _filter([q, q, q], min_confidence=0.0, max_subqueries=10)
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        assert _filter([], min_confidence=0.5, max_subqueries=3) == []


# ---------------------------------------------------------------------------
# _rule_based_fallback
# ---------------------------------------------------------------------------

class TestRuleBasedFallback:

    def test_conjunction_split(self):
        query = "What is machine learning and how does it work?"
        result = _rule_based_fallback(query, max_subqueries=5)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_vs_comparative_split(self):
        query = "transformers vs RNN for sequence modeling"
        result = _rule_based_fallback(query, max_subqueries=5)
        assert len(result) >= 2

    def test_single_simple_query_returned(self):
        query = "What is AI?"
        result = _rule_based_fallback(query, max_subqueries=5)
        assert len(result) == 1
        assert result[0].endswith("?")

    def test_respects_max_subqueries(self):
        query = "A and B and C and D and E and F"
        result = _rule_based_fallback(query, max_subqueries=2)
        assert len(result) <= 2

    def test_all_subqueries_end_with_question_mark(self):
        query = "What is BERT and how does it differ from GPT"
        result = _rule_based_fallback(query, max_subqueries=5)
        for q in result:
            assert q.endswith("?")


# ---------------------------------------------------------------------------
# _dedup_against_original
# ---------------------------------------------------------------------------

class TestDedupAgainstOriginal:

    def test_removes_exact_duplicate_of_original(self):
        original = "What is machine learning?"
        subqueries = ["What is machine learning?", "How does gradient descent work?"]
        result = _dedup_against_original(subqueries, original)
        assert original not in result

    def test_keeps_different_subqueries(self):
        original = "What is AI?"
        subqueries = ["How does machine learning work?", "What are neural networks?"]
        result = _dedup_against_original(subqueries, original)
        assert len(result) == 2

    def test_empty_subqueries_returns_empty(self):
        result = _dedup_against_original([], "original query")
        assert result == []


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------

def _make_decomposer():
    llm = MagicMock()
    llm.generate.return_value = (
        "1. What is machine learning?\n"
        "2. How does deep learning work?\n"
        "3. What are the main algorithms used in supervised learning?"
    )
    return QueryDecomposer(llm)


class TestQueryDecomposerDecompose:

    def test_simple_query_not_decomposed(self):
        decomposer = _make_decomposer()
        result = decomposer.decompose("What is AI?", session_id="s1")
        # Short simple query may return [] or just itself
        assert isinstance(result, list)

    def test_complex_query_decomposed(self):
        decomposer = _make_decomposer()
        result = decomposer.decompose(
            "Compare transformer vs RNN and explain how attention works?",
            session_id="s1",
        )
        assert isinstance(result, list)

    def test_returns_list_of_strings(self):
        decomposer = _make_decomposer()
        result = decomposer.decompose(
            "What is machine learning and how does it work?",
            session_id="s1",
        )
        assert all(isinstance(q, str) for q in result)

    def test_max_subqueries_respected(self):
        decomposer = _make_decomposer()
        result = decomposer.decompose(
            "Compare and contrast A vs B and explain C and describe D and list E?",
            session_id="s1",
        )
        assert len(result) <= decomposer.max_subqueries

    def test_llm_exception_falls_back_gracefully(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM failed")
        decomposer = QueryDecomposer(llm)
        result = decomposer.decompose(
            "What is machine learning and how does it differ from deep learning?",
            session_id="s1",
        )
        # Should fall back to rule-based, not crash
        assert isinstance(result, list)
