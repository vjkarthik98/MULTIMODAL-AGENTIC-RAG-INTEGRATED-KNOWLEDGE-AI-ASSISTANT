"""Public dispatch for the answer-verification package (Phase 32).

Mirrors app/chunking/__init__.py and app/embeddings/__init__.py's lazy-dispatch
convention: import the package, not the submodules, from pipeline code.

Two entry points:
- `VerificationLoop` — the full generate->verify->retry loop. Used by
  app/pipeline/query_pipeline.py and app/pipeline/rag_pipeline.py.
- `verify()` — a single-shot scoring pass over an ALREADY-generated answer,
  with no retrieval/retry side effects. Used by the eval harness and tests
  that need a VerificationReport without paying for the full loop.
"""

from __future__ import annotations

from typing import Any

from app.verification.citation_verifier import CitationVerifier
from app.verification.completeness_verifier import CompletenessVerifier
from app.verification.confidence_scorer import ConfidenceScorer
from app.verification.groundedness_checker import GroundednessChecker
from app.verification.retrieval_evaluator import RetrievalEvaluator
from app.verification.verification_loop import VerificationLoop, normalize_modality
from app.verification.verification_schema import VerificationReport

__all__ = ["VerificationLoop", "verify", "VerificationReport", "normalize_modality"]


def verify(
    query: str,
    docs: list[dict[str, Any]],
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    modality: str | None = None,
) -> VerificationReport:
    """Score an already-generated answer once — no retrieval, no retries.

    For the eval harness (grounding_success_rate, citation_accuracy) and unit
    tests that need a report without constructing a retriever/reasoning_engine.
    """
    from app.pipeline.rag_pipeline import _split_query_aspects

    sources = sources or []
    try:
        aspects = _split_query_aspects(query)
    except Exception:
        aspects = []

    retrieval_res = RetrievalEvaluator().evaluate(query, docs)
    grounding_res = GroundednessChecker().check(answer, docs, query=query)
    citation_res = CitationVerifier().check(answer, docs, sources)
    completeness_res = CompletenessVerifier().check(answer, aspects, query=query)

    scorer = ConfidenceScorer()
    scores = scorer.score(retrieval_res, grounding_res, citation_res, completeness_res)
    decision, reason = scorer.decide(scores, grounding_res, citation_res, completeness_res)

    return VerificationReport(
        verified=decision == "PASS",
        scores=scores,
        unsupported_claims=grounding_res.unsupported_claims
        + [f"unsupported number: {n}" for n in grounding_res.unsupported_numbers],
        bad_citations=citation_res.bad_citations,
        missing_aspects=completeness_res.missing,
        degraded=decision != "PASS",
        limitation_notice=None if decision == "PASS" else reason,
    )
