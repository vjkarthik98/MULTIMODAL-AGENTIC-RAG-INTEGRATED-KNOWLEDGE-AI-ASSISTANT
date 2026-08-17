"""Unit tests for the NLI contradiction pass in
app/verification/groundedness_checker.py (hallucination-reduction initiative,
Phase 3, 2026-08-13). Unlike test_groundedness_checker.py (pure-function,
no GPU), these exercise the REAL cross-encoder/nli-deberta-v3-base model —
slow (model download/load on first run) and needs a working
sentence-transformers/torch install, so marked `slow` and skipped if the
model can't be loaded in this environment (e.g. no network).

Tests the design actually shipped, not the original plan: NLI contributes an
ADDITIVE contradiction penalty on top of the unchanged lexical guard, not a
replacement of support_score with raw entailment probability. See the
module docstring under test for why (entailment probability was empirically
found unreliable — attribution phrasing/pronoun shifts push a genuinely
correct restatement into "neutral", not "entailment" — while contradiction
probability is cleanly separable).
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.reasoning.reasoning_engine import _hallucination_guard
from app.verification.groundedness_checker import GroundednessChecker


def _doc(text):
    return {"text": text, "metadata": {}}


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


@pytest.fixture(autouse=True)
def _enable_nli(monkeypatch):
    monkeypatch.setattr(settings, "NLI_GROUNDEDNESS_ENABLED", True)


class TestNLIGroundedness:
    def test_true_restatement_scores_high(self):
        gc = GroundednessChecker()
        docs = [_doc("Apple reported net sales of $383,285 million for fiscal 2023.")]
        answer = "Apple's net sales for fiscal 2023 were $383,285 million."
        result = gc.check(answer, docs)
        assert result.score > 70.0
        assert result.is_hallucinated is False

    def test_contradicted_number_flagged(self):
        gc = GroundednessChecker()
        docs = [_doc("Apple reported net sales of $383,285 million for fiscal 2023.")]
        answer = "Apple's net sales for fiscal 2023 were $999,999 million."
        result = gc.check(answer, docs)
        assert result.is_hallucinated is True
        # Caught by BOTH the pre-existing numeric-fidelity check (999,999
        # absent from context) and the NLI contradiction penalty.
        assert "999,999" in " ".join(result.unsupported_numbers)
        assert any("NLI contradiction" in c for c in result.unsupported_claims)

    def test_unrelated_claim_flagged(self):
        # Caught by the LEXICAL guard (near-zero word overlap) — NLI's
        # contradiction signal is not expected to fire here (an unrelated
        # claim is "neutral", not "contradicted"); confirms NLI's additive
        # penalty doesn't need to fire for this case to still be caught.
        gc = GroundednessChecker()
        docs = [_doc("Apple reported net sales of $383,285 million for fiscal 2023.")]
        answer = "The Federal Reserve held interest rates steady in December."
        result = gc.check(answer, docs)
        assert result.is_hallucinated is True

    def test_semantic_inversion_caught_by_nli_not_lexical(self):
        # The genuine, verified incremental value of the additive NLI pass:
        # a claim that inverts the source's meaning ("increased" ->
        # "decreased") shares enough vocabulary to pass the lexical
        # word-overlap check outright (confirmed below) and contains no
        # numbers for _unsupported_numbers to catch — only NLI's semantic
        # contradiction check catches it.
        docs = [
            _doc(
                "Apple's revenue increased significantly in the fourth quarter "
                "compared to the prior year."
            )
        ]
        answer = (
            "Apple's revenue decreased significantly in the fourth quarter "
            "compared to the prior year."
        )

        lexical_hallucinated, lexical_support = _hallucination_guard(answer, docs)
        assert lexical_hallucinated is False  # confirms lexical alone misses this
        assert lexical_support == 1.0

        gc = GroundednessChecker()
        result = gc.check(answer, docs)
        assert result.is_hallucinated is True
        assert any("NLI contradiction" in c for c in result.unsupported_claims)

    def test_nli_cannot_remove_a_lexical_flag(self):
        # Documents the known, deliberate scope limit: NLI is additive-only
        # and cannot rescue a paraphrase the lexical guard flags via low word
        # overlap, even when the paraphrase is factually correct — single-
        # premise NLI can't reliably distinguish that case from a genuinely
        # unrelated/fabricated claim (see module docstring).
        docs = [
            _doc(
                "At its December 18, 2024 meeting, the FOMC lowered the target range by "
                "1/4 percentage point (25 basis points), to 4.25-4.50 percent."
            )
        ]
        answer = "The FOMC cut rates by a quarter-point, bringing the range to 4.25%-4.5%."

        lexical_hallucinated, _ = _hallucination_guard(answer, docs)
        gc = GroundednessChecker()
        result = gc.check(answer, docs)
        # Whatever the lexical guard decided stands — NLI never overrides it.
        assert result.is_hallucinated == (lexical_hallucinated or result.is_hallucinated)
        if lexical_hallucinated:
            assert result.is_hallucinated is True

    def test_disabled_flag_uses_lexical_path_unchanged(self, monkeypatch):
        monkeypatch.setattr(settings, "NLI_GROUNDEDNESS_ENABLED", False)
        gc = GroundednessChecker()
        docs = [_doc("Apple reported net sales of $383,285 million for fiscal 2023.")]
        answer = "Apple's net sales for fiscal 2023 were $383,285 million."
        result = gc.check(answer, docs)
        assert result.is_hallucinated is False


class TestDeterministicAnswerExemption:
    """`deterministic_answer=True` skips ONLY the NLI entailment judgment, for
    answers mechanically parsed out of the retrieved context
    (rag_pipeline._synthesize_image_chart_answer). Added 2026-08-13 after
    single-premise NLI returned false contradictions on 11/14 image rows.
    """

    _CHART_DOC = [
        _doc(
            "line_chart: COMPARISON OF 5-YEAR CUMULATIVE TOTAL RETURN Among Apple Inc., "
            "the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index. "
            "CHART VALUES - pixel-calibrated reads: 9/28/24: Apple Inc.=~$429, "
            "S&P 500 Index=~$210, Dow Jones U.S. Technology Supersector Index=~$322"
        )
    ]

    def test_multiclause_chart_answer_passes_when_deterministic(self):
        answer = (
            "On 9/28/24, ranked from highest to lowest cumulative total return: "
            "Apple Inc. at approximately $429; Dow Jones U.S. Technology Supersector "
            "Index at approximately $322; S&P 500 Index at approximately $210."
        )
        gc = GroundednessChecker()
        exempt = gc.check(answer, self._CHART_DOC, deterministic_answer=True)
        assert exempt.is_hallucinated is False
        assert not any("NLI contradiction" in c for c in exempt.unsupported_claims)

    def test_numeric_grounding_still_enforced_for_deterministic_answers(self):
        # The exemption must NOT become a blanket pass: a figure absent from
        # context is still caught (this is what would catch a synthesizer bug).
        answer = "Apple Inc. was approximately $99999 on 9/28/24 per this chart."
        result = GroundednessChecker().check(
            answer, self._CHART_DOC, deterministic_answer=True
        )
        assert result.is_hallucinated is True
        assert any("99999" in n for n in result.unsupported_numbers)

    def test_default_is_not_exempt(self):
        # A normal generated answer keeps full NLI scrutiny.
        answer = "Apple Inc. revenue decreased significantly compared to the prior year."
        docs = [_doc("Apple Inc. revenue increased significantly compared to the prior year.")]
        result = GroundednessChecker().check(answer, docs)
        assert result.is_hallucinated is True
        assert any("NLI contradiction" in c for c in result.unsupported_claims)


class TestNarrowedPremiseFixesFalseContradiction:
    """xlsx grounding follow-up (2026-08-17), end-to-end with the real NLI
    model. Live-reproduced against the actual retrieved chunk from
    ctryprem.xlsx's "Sovereign Ratings" sheet (58 countries, ~2000 chars):
    scoring a verbatim-correct, single-country answer against the WHOLE chunk
    measured contradiction=0.87 and flagged a correct answer as hallucinated
    — this drove xlsx's grounding_success_rate down to 0.417, the worst of
    any modality. _narrow_tabular_premise (see the pure-function tests in
    test_groundedness_checker.py) fixes it by isolating the matching row
    before scoring.
    """

    _DENSE_SHEET = _doc(
        "Sheet: Sovereign Ratings (Moody's,S&P) (in units)\n"
        "Columns: Country | S&P Rating | Fitch rating | Moody's rating | Moody's Rating (unadj)\n"
        "Burkina Faso | CCC+ | NR | Caa1 | NR | C- | C3\n"
        "Cambodia | N/A | NR | B2 | B2 | C+ | C1\n"
        "Cameroon | B- | B | Caa1 | Caa1 | CC | Ca2\n"
        "Canada | AAA | AA+ | Aaa | Aaa | CC- | Ca3\n"
        "Cape Verde | B | B | B2 | NR | CC+ | Ca1\n"
        "Cayman Islands | 0 | #N/A | Aa3 | Aa3 | CCC | Caa2\n"
        "Chile | A | A- | A2 | A2 | CCC- | Caa3\n"
        "China | A+ | A | A1 | A1 | CCC+ | Caa1\n"
        "Colombia | BB | BB | Baa2 | Baa2 | CCC | Caa2\n"
        + "\n".join(
            f"Country{i} | BB{i % 3} | BB{i % 3} | Baa{i % 3} | Baa{i % 3}" for i in range(45)
        )
    )

    def test_verbatim_correct_country_row_not_flagged_as_contradiction(self):
        answer = (
            "The Sovereign Ratings sheet lists Canada at S&P AAA, Fitch AA+, and Moody's Aaa."
        )
        result = GroundednessChecker().check(answer, [self._DENSE_SHEET])
        assert result.is_hallucinated is False
        assert not any("NLI contradiction" in c for c in result.unsupported_claims)

    def test_wrong_rating_for_named_country_still_caught(self):
        # The narrowing must not become a blanket pass — a genuinely wrong
        # rating for the SAME named country (wrong S&P grade) must still be
        # flagged once the premise is narrowed to Canada's actual row.
        answer = "The Sovereign Ratings sheet lists Canada at S&P CCC+."
        result = GroundednessChecker().check(answer, [self._DENSE_SHEET])
        assert result.is_hallucinated is True
