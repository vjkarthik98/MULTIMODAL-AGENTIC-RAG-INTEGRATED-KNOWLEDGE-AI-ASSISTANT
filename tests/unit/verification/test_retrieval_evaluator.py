"""Unit tests for app/verification/retrieval_evaluator.py — Responsibility 1:
is the retrieved evidence relevant/sufficient/non-conflicting. No LLM, no
network. Aspect-coverage uses the real _split_query_aspects() from
rag_pipeline.py, so single-fact queries are exercised (2+ aspect splitting is
covered indirectly through completeness_verifier's own tests).
"""

from app.verification.retrieval_evaluator import RetrievalEvaluator


def _doc(text, score=0.8):
    return {"text": text, "score": score, "metadata": {}}


class TestRetrievalEvaluator:

    def test_no_docs_is_insufficient(self):
        ev = RetrievalEvaluator()
        result = ev.evaluate("What was Q4 revenue?", [])
        assert result.score == 0.0
        assert result.insufficient_context is True

    def test_relevant_docs_score_high(self):
        ev = RetrievalEvaluator()
        docs = [_doc("Apple reported net revenue of $94.9 billion.", score=0.9) for _ in range(3)]
        result = ev.evaluate("What was Q4 revenue?", docs)
        assert result.score > 50.0
        assert result.insufficient_context is False

    def test_low_score_docs_flagged_insufficient(self):
        ev = RetrievalEvaluator()
        docs = [_doc("Irrelevant text.", score=0.01) for _ in range(3)]
        result = ev.evaluate("What was Q4 revenue?", docs)
        assert result.insufficient_context is True

    def test_conflicting_numbers_detected(self):
        ev = RetrievalEvaluator()
        docs = [
            _doc("Net revenue of $94.9 billion was reported for the quarter."),
            _doc("Net revenue of $34.2 billion was reported for the quarter."),
        ]
        result = ev.evaluate("What was net revenue?", docs)
        assert result.conflicting_evidence is True

    def test_consistent_numbers_not_flagged_as_conflicting(self):
        ev = RetrievalEvaluator()
        docs = [
            _doc("Net revenue of $94.9 billion was reported for the quarter."),
            _doc("The reported net revenue of $94.9 billion beat expectations."),
        ]
        result = ev.evaluate("What was net revenue?", docs)
        assert result.conflicting_evidence is False

    def test_score_of_handles_missing_score_field(self):
        ev = RetrievalEvaluator()
        docs = [{"text": "some text", "metadata": {}}]  # no "score" key
        result = ev.evaluate("query", docs)
        assert isinstance(result.score, float)
