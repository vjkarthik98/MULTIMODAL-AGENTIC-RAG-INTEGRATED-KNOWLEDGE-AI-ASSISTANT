"""Refusals must never be scored as hallucinations (per-modality quality pass,
Phase A1, 2026-08-13).

A refusal asserts no factual claim, so it can be neither grounded nor
ungrounded. Before this fix, live xlsx gold rows returning the
"I couldn't generate a reliable answer." fallback (reasoning_engine
._fallback_response) or "No relevant information was found in your knowledge
base…" scored grounding=0.0 with NLI contradiction 0.99, which:
  * dragged xlsx grounding_success_rate to 0.417 for reasons unrelated to
    answer quality (and image's to 0.000 in combination with Phase A2's bug),
  * marked verified=False, so VerificationLoop appended "…treat the figures
    above with caution" to a refusal containing NO figures — a user-facing bug.

Covers all three layers the fix touches: the shared primitive
(reasoning_engine.is_refusal_answer), the checker
(GroundednessChecker.check), and the loop's user-facing notice
(VerificationLoop.run).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.reasoning.reasoning_engine import is_refusal_answer
from app.verification.groundedness_checker import GroundednessChecker
from app.verification.verification_loop import VerificationLoop

# The two literal strings observed live on xlsx gold rows, plus the generic
# document-doesn't-contain phrasing the model emits on its own.
_REFUSALS = [
    "I couldn't generate a reliable answer.",
    "No relevant information was found in your knowledge base to answer this question.",
    "The provided documents do not contain the information needed.",
    "I don't know.",
]

_CONTEXT = [{"text": "Apple reported net sales of $383,285 million for fiscal 2023.", "metadata": {}}]


class TestIsRefusalAnswer:
    @pytest.mark.parametrize("text", _REFUSALS)
    def test_refusal_strings_detected(self, text):
        assert is_refusal_answer(text) is True

    def test_empty_is_refusal(self):
        assert is_refusal_answer("") is True
        assert is_refusal_answer(None) is True

    def test_real_answer_is_not_refusal(self):
        assert (
            is_refusal_answer("Apple's net sales for fiscal 2023 were $383,285 million.") is False
        )

    def test_substantive_answer_mentioning_a_gap_is_still_refusal_by_design(self):
        # Documented limitation of substring matching: a long answer that gives
        # real content AND notes a gap trips the sentinel. Accepted here because
        # this list is used ONLY to skip groundedness scoring (fail-safe: we
        # skip a check we could have run), NOT to suppress an answer. The
        # streaming path deliberately uses a different detector
        # (rag_pipeline._is_llm_refusal) that DOES handle short-vs-long.
        text = (
            "Apple's net sales were $383,285 million for fiscal 2023. "
            "The document does not contain the segment breakdown."
        )
        assert is_refusal_answer(text) is True


class TestGroundednessCheckerRefusal:
    @pytest.mark.parametrize("text", _REFUSALS)
    def test_refusal_not_flagged_as_hallucination(self, text):
        result = GroundednessChecker().check(text, _CONTEXT)
        assert result.is_hallucinated is False
        assert result.score == 100.0
        assert result.unsupported_claims == []
        assert result.unsupported_numbers == []

    def test_refusal_short_circuits_before_empty_docs_check(self):
        # A refusal with NO retrieved docs must also pass — previously this hit
        # the "no retrieved evidence to verify against" branch and flagged.
        result = GroundednessChecker().check("I couldn't generate a reliable answer.", [])
        assert result.is_hallucinated is False
        assert result.score == 100.0

    def test_genuinely_ungrounded_answer_still_flagged(self):
        # Regression guard: the short-circuit must not weaken real detection.
        result = GroundednessChecker().check(
            "Apple's net sales for fiscal 2023 were $999,999 million.", _CONTEXT
        )
        assert result.is_hallucinated is True


def _doc(text, chunk_id="c1", score=0.8):
    return {"text": text, "metadata": {"chunk_id": chunk_id}, "score": score}


class TestVerificationLoopRefusalNotice:
    _NOTICE_FRAGMENT = "could not be fully verified"

    def test_refusal_answer_gets_no_limitation_notice(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["xlsx"])
        engine = MagicMock()
        engine.generate_answer.return_value = {
            "answer": "I couldn't generate a reliable answer."
        }

        answer, report = loop_run(engine, modality_hint="xlsx")

        assert self._NOTICE_FRAGMENT not in answer
        assert report.limitation_notice is None

    def test_unverified_real_answer_still_gets_notice(self, monkeypatch):
        # Regression guard: suppressing the notice for refusals must not
        # suppress it for a genuinely unverified factual answer.
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["xlsx"])
        engine = MagicMock()
        engine.generate_answer.return_value = {
            "answer": "Totally unrelated fabricated claim about $999,999 million in revenue."
        }

        answer, report = loop_run(engine, modality_hint="xlsx")

        if not report.verified:
            assert self._NOTICE_FRAGMENT in answer
            assert report.limitation_notice is not None


def loop_run(engine, modality_hint):
    return VerificationLoop().run(
        "What was the equity risk premium?",
        "sess-refusal",
        "user1",
        retriever=MagicMock(),
        reasoning_engine=engine,
        initial_docs=[_doc("The mature-market equity risk premium is 4.33 percent.")],
        modality_hint=modality_hint,
    )
