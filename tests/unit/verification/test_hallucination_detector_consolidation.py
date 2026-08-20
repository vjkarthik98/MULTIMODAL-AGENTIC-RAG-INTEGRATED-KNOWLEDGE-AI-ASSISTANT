"""Cross-cutting test for the hallucination-reduction initiative's Phase 4
consolidation (2026-08-13): reasoning_engine.generate_answer(),
GroundednessChecker.check() (used by VerificationLoop), and
output_guard._check_groundedness() previously ran THREE independent
hallucination-detection implementations that could (and did) disagree with
each other on the same answer. All three now share one implementation
(app.verification.groundedness_checker.lexical_and_nli_verdict for the
lexical+NLI portion, app.reasoning.reasoning_engine._unsupported_numbers for
numeric grounding). This feeds the same answer/context into all three call
sites and asserts they agree, per the Phase 4 plan's testing requirement.

Slow (real NLI model, matches test_groundedness_checker_nli.py's marker).
"""

from __future__ import annotations

import pytest

from app.guardrails.output_guard import _check_groundedness
from app.verification.groundedness_checker import GroundednessChecker, lexical_and_nli_verdict


def _nli_available() -> bool:
    try:
        from app.core.model_loader import model_loader

        model_loader.get_nli_model()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _nli_available(), reason="NLI model unavailable (no network / no GPU)"),
]


class TestThreeSitesAgree:
    def test_clean_grounded_answer_agrees_across_all_three_sites(self):
        answer = "Apple's net sales for fiscal 2023 were $383,285 million."
        context_text = "Apple reported net sales of $383,285 million for fiscal 2023."
        docs = [{"text": context_text, "metadata": {}}]

        # Site 1: reasoning_engine.generate_answer()'s hallucination guard.
        site1_hallucinated, _, _, _, _ = lexical_and_nli_verdict(answer, docs)

        # Site 2: VerificationLoop's GroundednessChecker.
        site2_result = GroundednessChecker().check(answer, docs)

        # Site 3: output_guard's warn-only check.
        site3_flagged, _ = _check_groundedness(answer, [context_text], "test-session")

        assert site1_hallucinated is False
        assert site2_result.is_hallucinated is False
        assert site3_flagged is False

    def test_semantic_inversion_agrees_across_all_three_sites(self):
        # The genuine incremental NLI capability (see
        # test_groundedness_checker_nli.py::test_semantic_inversion_caught_by_nli_not_lexical) —
        # confirms it fires consistently everywhere, not just in GroundednessChecker.
        context_text = (
            "Apple's revenue increased significantly in the fourth quarter "
            "compared to the prior year."
        )
        answer = (
            "Apple's revenue decreased significantly in the fourth quarter "
            "compared to the prior year."
        )
        docs = [{"text": context_text, "metadata": {}}]

        site1_hallucinated, _, _, _, _ = lexical_and_nli_verdict(answer, docs)
        site2_result = GroundednessChecker().check(answer, docs)
        site3_flagged, _ = _check_groundedness(answer, [context_text], "test-session")

        assert site1_hallucinated is True
        assert site2_result.is_hallucinated is True
        assert site3_flagged is True

    def test_fabricated_number_agrees_across_sites_2_and_3(self):
        # Site 1 deliberately excludes numeric grounding (reasoning_engine
        # handles it separately, with its own retry-then-citation-bypass flow
        # — see lexical_and_nli_verdict's docstring), so it is NOT expected to
        # flag this on its own. Sites 2 and 3 both run the numeric check
        # (via GroundednessChecker) and must agree with each other.
        context_text = "Apple reported net sales of $383,285 million for fiscal 2023."
        answer = "Apple's net sales for fiscal 2023 were $999,999 million."
        docs = [{"text": context_text, "metadata": {}}]

        site2_result = GroundednessChecker().check(answer, docs)
        site3_flagged, _ = _check_groundedness(answer, [context_text], "test-session")

        assert site2_result.is_hallucinated is True
        assert site3_flagged is True
