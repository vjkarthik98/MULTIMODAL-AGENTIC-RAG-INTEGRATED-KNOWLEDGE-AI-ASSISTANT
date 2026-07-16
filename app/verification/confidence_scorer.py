"""ConfidenceScorer — Responsibility 6 (docs/Phase_32_Agentic_Answer_Verification.md).

Combines RetrievalEvaluator + GroundednessChecker + CitationVerifier +
CompletenessVerifier outputs into the four 0-100 ConfidenceScores and the
PASS/FAIL decision. Thresholds come from settings.AGENT_VERIFY_* — never
hardcoded (CLAUDE.md: no literals inlined, all via settings.*).
"""

from __future__ import annotations

from typing import List, Tuple

from app.core.config import settings
from app.verification.verification_schema import (
    CitationCheckResult,
    CompletenessResult,
    ConfidenceScores,
    GroundednessResult,
    RetrievalEvalResult,
    VerifyDecision,
)


class ConfidenceScorer:
    def score(self, retrieval: RetrievalEvalResult, grounding: GroundednessResult,
              citation: CitationCheckResult, completeness: CompletenessResult) -> ConfidenceScores:
        # Overall is the weakest-link-weighted mean, not a plain average — a
        # single very-wrong dimension (e.g. a fabricated number) must not be
        # diluted by three fine ones, since finance numeric errors are a
        # CRITICAL-severity class on their own (engineering-standards.md §2).
        completeness_score = 100.0 if completeness.is_complete else (
            100.0 * len(completeness.covered) / max(len(completeness.aspects), 1)
        )
        weakest = min(retrieval.score, grounding.score, citation.score, completeness_score)
        mean = (retrieval.score + grounding.score + citation.score + completeness_score) / 4.0
        overall = round(0.6 * weakest + 0.4 * mean, 2)

        return ConfidenceScores(
            retrieval=retrieval.score,
            grounding=grounding.score,
            citation=citation.score,
            overall=overall,
        )

    def decide(self, scores: ConfidenceScores, grounding: GroundednessResult,
               citation: CitationCheckResult, completeness: CompletenessResult) -> Tuple[VerifyDecision, str]:
        reasons: List[str] = []

        if scores.retrieval < settings.AGENT_VERIFY_RETRIEVAL_MIN:
            reasons.append(f"retrieval {scores.retrieval:.1f} < {settings.AGENT_VERIFY_RETRIEVAL_MIN}")
        if scores.grounding < settings.AGENT_VERIFY_GROUNDING_MIN:
            reasons.append(f"grounding {scores.grounding:.1f} < {settings.AGENT_VERIFY_GROUNDING_MIN}")
        if scores.citation < settings.AGENT_VERIFY_CITATION_MIN:
            reasons.append(f"citation {scores.citation:.1f} < {settings.AGENT_VERIFY_CITATION_MIN}")
        if scores.overall < settings.AGENT_VERIFY_OVERALL_MIN:
            reasons.append(f"overall {scores.overall:.1f} < {settings.AGENT_VERIFY_OVERALL_MIN}")
        if grounding.is_hallucinated:
            reasons.append("hallucination guard fired")
        if grounding.unsupported_numbers:
            reasons.append(f"unsupported numbers: {grounding.unsupported_numbers[:3]}")
        if citation.bad_citations:
            reasons.append(f"bad citations: {citation.bad_citations[:3]}")
        if not completeness.is_complete:
            reasons.append(f"missing aspects: {completeness.missing[:3]}")

        if reasons:
            return "FAIL", "; ".join(reasons)
        return "PASS", "all thresholds met"
