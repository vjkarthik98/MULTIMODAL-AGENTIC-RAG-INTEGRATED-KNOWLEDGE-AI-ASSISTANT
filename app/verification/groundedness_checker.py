"""GroundednessChecker — Responsibility 2 (docs/Phase_32_Agentic_Answer_Verification.md).

Wraps app/reasoning/reasoning_engine.py's existing numeric-faithfulness guard
(_hallucination_guard, _unsupported_numbers) rather than duplicating it — that
guard already runs a one-shot hardened retry inside generate_answer(). This
module adds the 0-100 confidence score VerificationLoop needs on top of the
guard's own (bool, float) / (List[str]) outputs.

Imports reasoning_engine lazily (function-scoped), matching this codebase's
existing convention for cross-module calls between app/reasoning/,
app/pipeline/, and app/agents/ — there are no module-level cross-imports
among those packages today, and this file must not become the first one.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.verification.verification_schema import GroundednessResult


class GroundednessChecker:
    """Every important statement in the answer must be supported by retrieved docs."""

    def check(self, answer: str, docs: List[Dict[str, Any]], query: str = "") -> GroundednessResult:
        if not answer:
            return GroundednessResult(score=0.0, is_hallucinated=True)
        if not docs:
            # Empty-retrieval path is handled upstream (explicit "no information"
            # answer) — a groundedness check with no evidence to check against
            # is meaningless, not automatically passing.
            return GroundednessResult(score=0.0, is_hallucinated=True,
                                       unsupported_claims=["no retrieved evidence to verify against"])

        from app.reasoning.reasoning_engine import _hallucination_guard, _unsupported_numbers

        is_hallucinated, support_score = _hallucination_guard(answer, docs)
        bad_numbers = _unsupported_numbers(answer, docs, query=query)

        # Blend sentence-level support (0-1) with a numeric-fidelity penalty:
        # each unsupported number is a harder failure than a paraphrase miss,
        # since finance numbers are the domain's "sacred" correctness bar
        # (engineering-standards.md §4.1).
        numeric_penalty = min(len(bad_numbers) * 0.15, 0.6)
        raw_score = max(0.0, support_score - numeric_penalty)
        score = round(raw_score * 100.0, 2)

        unsupported_claims: List[str] = []
        if is_hallucinated:
            unsupported_claims.append(
                f"answer support score {support_score:.2f} below hallucination threshold"
            )

        return GroundednessResult(
            score=score,
            is_hallucinated=is_hallucinated or bool(bad_numbers),
            unsupported_claims=unsupported_claims,
            unsupported_numbers=bad_numbers,
        )
