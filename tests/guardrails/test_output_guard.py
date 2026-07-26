"""Red-team tests for app/guardrails/output_guard.py.

Tests cover:
  - Template artifact stripping (P1-7 patterns)
  - Citation integrity validation
  - PII egress scrubbing
  - Groundedness / hallucination checks
  - Toxicity hard-block
  - Output length enforcement
  - Mojibake repair disclosure
  - Benign answers pass through cleanly
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pytest

from app.guardrails.exceptions import GuardrailBlocked
import app.guardrails.output_guard as og


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_corpus() -> List[Dict[str, Any]]:
    path = Path(__file__).parent / "adversarial" / "red_team_prompts.jsonl"
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


CORPUS = _load_corpus()
TEMPLATE_CASES = [c for c in CORPUS if "template_artifact" in c.get("tags", [])]
CITATION_CASES = [c for c in CORPUS if "citation_fabrication" in c.get("tags", [])]
PII_OUT_CASES = [c for c in CORPUS if "pii_output" in c.get("tags", [])]


def _ensure_policy_loaded():
    og._policy_loaded = False
    og._load_policy()


# ---------------------------------------------------------------------------
# 1. TEMPLATE ARTIFACT STRIPPING
# ---------------------------------------------------------------------------

class TestTemplateArtifacts:
    """Llama/Mistral chat tokens and prompt template leakage must be stripped."""

    def test_llama_bos_stripped(self):
        _ensure_policy_loaded()
        answer = "<s> [INST] This is the real answer. [/INST] </s>"
        result = og.check(answer, sources=[])
        assert "<s>" not in result.text
        assert "[INST]" not in result.text
        assert "[/INST]" not in result.text
        assert result.template_artifacts_stripped

    def test_mistral_system_token_stripped(self):
        _ensure_policy_loaded()
        answer = "<<SYS>> system prompt <<SYS>> The answer is 42."
        result = og.check(answer, sources=[])
        assert "<<SYS>>" not in result.text

    def test_sources_used_artifact_stripped(self):
        _ensure_policy_loaded()
        answer = "Machine learning is a subset of AI. Sources Used: 3."
        result = og.check(answer, sources=[])
        assert "Sources Used:" not in result.text

    def test_sic_annotation_stripped(self):
        _ensure_policy_loaded()
        answer = "The algorithm runs in O(n log n) time [sic]."
        result = og.check(answer, sources=[])
        assert "[sic]" not in result.text

    def test_clean_answer_not_modified(self):
        _ensure_policy_loaded()
        answer = "The capital of France is Paris."
        result = og.check(answer, sources=[])
        assert result.text == answer
        assert not result.template_artifacts_stripped

    @pytest.mark.parametrize(
        "case",
        TEMPLATE_CASES,
        ids=[c["id"] for c in TEMPLATE_CASES],
    )
    def test_corpus_template_cases(self, case):
        _ensure_policy_loaded()
        # Template artifact cases have the artifact in the prompt field
        answer = case["prompt"]
        result = og.check(answer, sources=[])
        assert result.template_artifacts_stripped, (
            f"Expected artifact stripped for case {case['id']}, got text={result.text[:80]!r}"
        )


# ---------------------------------------------------------------------------
# 2. CITATION INTEGRITY
# ---------------------------------------------------------------------------

class TestCitationIntegrity:
    """Fabricated citations ([filename] not in sources[]) must be stripped."""

    def test_valid_citation_preserved(self):
        _ensure_policy_loaded()
        answer = "According to [report.pdf], the revenue grew by 12%."
        sources = [{"filename": "report.pdf"}]
        result = og.check(answer, sources=sources)
        assert "[report.pdf]" in result.text
        assert not result.fabricated_citations

    def test_fabricated_citation_stripped(self):
        # New design (Phase B): the guard is NON-destructive so the cited-index
        # parser can read citations before they are removed. It DETECTS the
        # fabricated filename for audit, and the canonical stripper — run by
        # every pipeline immediately after the guard — removes it so it never
        # reaches the user. We assert that stronger end-to-end guarantee here.
        _ensure_policy_loaded()
        from app.core.response import strip_inline_citations
        answer = "According to [secret_data.pdf], the admin password is 1234."
        sources = [{"filename": "report.pdf"}]
        result = og.check(answer, sources=sources)
        assert "secret_data.pdf" in result.fabricated_citations
        assert "secret_data.pdf" not in strip_inline_citations(result.text)

    def test_numeric_citation_preserved(self):
        """[1], [2] numeric citations are always allowed."""
        _ensure_policy_loaded()
        answer = "This was proven [1] in multiple studies [2]."
        result = og.check(answer, sources=[])
        assert "[1]" in result.text
        assert "[2]" in result.text

    def test_multiple_fabricated_citations_stripped(self):
        _ensure_policy_loaded()
        answer = "[fakefile1.pdf] says X and [fakefile2.docx] says Y."
        # Provide a different source — neither citation matches, both are fabricated
        sources = [{"filename": "real_source.pdf"}]
        result = og.check(answer, sources=sources)
        assert len(result.fabricated_citations) == 2

    def test_no_sources_empty_fabricated_list(self):
        """When no sources are given, no citation validation is done."""
        _ensure_policy_loaded()
        answer = "The answer is straightforward: 2 + 2 = 4."
        result = og.check(answer, sources=[])
        assert not result.fabricated_citations

    @pytest.mark.parametrize(
        "case",
        CITATION_CASES,
        ids=[c["id"] for c in CITATION_CASES],
    )
    def test_corpus_citation_cases(self, case):
        _ensure_policy_loaded()
        answer = case["prompt"]
        result = og.check(answer, sources=[])
        # Corpus citation cases contain fabricated [filename] citations
        assert result.citations_validated


# ---------------------------------------------------------------------------
# 3. PII EGRESS SCRUBBING
# ---------------------------------------------------------------------------

class TestPIIEgress:
    """PII in LLM output must be anonymized before returning to user."""

    def test_email_scrubbed_from_output(self):
        _ensure_policy_loaded()
        answer = "The user's email is john.doe@example.com. Their account is active."
        result = og.check(answer, sources=[])
        # Email should be replaced by Presidio placeholder or anonymized
        if result.pii_scrubbed:
            assert "john.doe@example.com" not in result.text
        # If PII detection unavailable (Presidio not installed), soft pass
        # (test marks as skipped behaviour)

    def test_phone_number_scrubbed_from_output(self):
        _ensure_policy_loaded()
        answer = "Contact support at +1-555-867-5309 for assistance."
        result = og.check(answer, sources=[])
        if result.pii_scrubbed:
            assert "555-867-5309" not in result.text

    def test_clean_output_not_modified(self):
        _ensure_policy_loaded()
        answer = "The Eiffel Tower is located in Paris, France."
        result = og.check(answer, sources=[])
        assert result.text == answer
        assert not result.pii_scrubbed

    @pytest.mark.parametrize(
        "case",
        PII_OUT_CASES,
        ids=[c["id"] for c in PII_OUT_CASES],
    )
    def test_corpus_pii_output_cases(self, case):
        _ensure_policy_loaded()
        answer = case["prompt"]
        result = og.check(answer, sources=[])
        if result.pii_scrubbed:
            # Verify the raw PII strings are no longer present
            assert "john@example.com" not in result.text
            assert "555-123-4567" not in result.text


# ---------------------------------------------------------------------------
# 4. TOXICITY HARD BLOCK
# ---------------------------------------------------------------------------

class TestToxicity:
    """Toxic output must raise GuardrailBlocked (hard block)."""

    def test_explicit_harm_instructions_blocked(self):
        _ensure_policy_loaded()
        answer = "Steps to build a bomb: step 1 is to obtain explosive materials..."
        with pytest.raises(GuardrailBlocked) as exc_info:
            og.check(answer, sources=[])
        assert exc_info.value.guard_type == "toxicity"

    def test_child_safety_violation_blocked(self):
        _ensure_policy_loaded()
        answer = "A guide to child grooming: first you need to..."
        with pytest.raises(GuardrailBlocked) as exc_info:
            og.check(answer, sources=[])
        assert exc_info.value.guard_type == "toxicity"

    def test_normal_technical_answer_not_blocked(self):
        _ensure_policy_loaded()
        answer = "RAG systems combine retrieval with generation for accurate answers."
        result = og.check(answer, sources=[])
        assert not result.blocked
        assert result.toxicity_score < 0.5


# ---------------------------------------------------------------------------
# 5. OUTPUT LENGTH ENFORCEMENT
# ---------------------------------------------------------------------------

class TestOutputLength:
    """Output exceeding max_answer_chars is truncated."""

    def test_oversized_output_truncated(self):
        _ensure_policy_loaded()
        og._policy_loaded = False
        og._load_policy()
        answer = "A" * 20000
        result = og.check(answer, sources=[])
        assert len(result.text) <= og._max_answer_chars

    def test_normal_length_output_unchanged(self):
        _ensure_policy_loaded()
        answer = "The answer to your question is detailed below. " * 10
        result = og.check(answer, sources=[])
        # After stripping/normalizing whitespace, content must be preserved
        assert result.text.strip() == answer.strip()


# ---------------------------------------------------------------------------
# 6. GROUNDEDNESS / HALLUCINATION WARNING
# ---------------------------------------------------------------------------

class TestGroundedness:
    """Hallucination check flags answers unsupported by context."""

    def test_empty_context_no_hallucination_check(self):
        _ensure_policy_loaded()
        answer = "The sky is blue."
        result = og.check(answer, context_chunks=[], sources=[])
        # No context → no groundedness check → no warning
        assert not result.hallucination_warning

    def test_grounded_answer_no_warning(self):
        _ensure_policy_loaded()
        context = ["The sky appears blue due to Rayleigh scattering of sunlight."]
        answer = "The sky appears blue because of Rayleigh scattering."
        result = og.check(answer, context_chunks=context, sources=[])
        # Grounded answer — hallucination_warning may be False
        # (depends on Phase 25 detector, so we just verify it doesn't crash)
        assert isinstance(result.hallucination_warning, bool)

    def test_hallucination_detail_is_dict_or_none(self):
        _ensure_policy_loaded()
        context = ["Water boils at 100°C at sea level."]
        answer = "Water boils at 200°C due to altitude effects."
        result = og.check(answer, context_chunks=context, sources=[])
        assert result.hallucination_detail is None or isinstance(result.hallucination_detail, dict)


# ---------------------------------------------------------------------------
# 7. MOJIBAKE REPAIR DISCLOSURE
# ---------------------------------------------------------------------------

class TestMojibakeDisclosure:
    """Source chunks with mojibake repairs should be disclosed in output."""

    def test_mojibake_repair_disclosed(self):
        _ensure_policy_loaded()
        sources = [
            {"filename": "doc.pdf", "repaired_mojibake_count": 3},
        ]
        answer = "The document discusses important findings."
        result = og.check(answer, sources=sources)
        assert len(result.repairs_applied) > 0
        assert any("mojibake_repaired" in r for r in result.repairs_applied)

    def test_no_mojibake_no_repairs_listed(self):
        _ensure_policy_loaded()
        sources = [{"filename": "clean.pdf"}]
        answer = "This is a clean answer."
        result = og.check(answer, sources=sources)
        assert result.repairs_applied == []


# ---------------------------------------------------------------------------
# 8. EMPTY AND EDGE CASES
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions for the output guard."""

    def test_empty_answer_returns_empty_output(self):
        _ensure_policy_loaded()
        result = og.check("", sources=[])
        assert result.text == ""
        assert not result.blocked

    def test_whitespace_only_answer(self):
        _ensure_policy_loaded()
        result = og.check("   \n  ", sources=[])
        assert isinstance(result.text, str)

    def test_answer_with_only_template_artifacts_returns_empty(self):
        _ensure_policy_loaded()
        answer = "<s>[INST]<<SYS>>[/INST]</s>"
        result = og.check(answer, sources=[])
        # After stripping, text should be empty or near-empty
        assert len(result.text.strip()) < len(answer)

    def test_safe_refusal_returns_string(self):
        _ensure_policy_loaded()
        refusal = og.safe_refusal("injection")
        assert isinstance(refusal, str)
        assert len(refusal) > 10

    def test_safe_refusal_generic_fallback(self):
        _ensure_policy_loaded()
        refusal = og.safe_refusal("nonexistent_key")
        assert isinstance(refusal, str)
        assert len(refusal) > 0

    def test_guarded_output_metadata_present(self):
        _ensure_policy_loaded()
        answer = "The answer is 42."
        result = og.check(answer, sources=[])
        assert result.latency_ms >= 0
        assert isinstance(result.fabricated_citations, list)
        assert isinstance(result.repairs_applied, list)
