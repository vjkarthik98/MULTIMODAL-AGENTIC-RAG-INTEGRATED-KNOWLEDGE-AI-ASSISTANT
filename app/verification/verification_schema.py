"""Pydantic contracts for the Phase 32 answer-verification loop.

Same convention as app/agents/agent_schema.py: all cross-boundary data is a
typed model, never a raw dict. See docs/Phase_32_Agentic_Answer_Verification.md
for the full design.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RetryStrategy = Literal[
    "baseline", "expand_retrieval", "query_rewrite", "increase_depth", "decomposition",
]

VerifyDecision = Literal["PASS", "FAIL"]


class ConfidenceScores(BaseModel):
    """0-100 scores. See app/core/config.py AGENT_VERIFY_*_MIN for gate thresholds."""

    retrieval: float = Field(ge=0.0, le=100.0)
    grounding: float = Field(ge=0.0, le=100.0)
    citation:  float = Field(ge=0.0, le=100.0)
    overall:   float = Field(ge=0.0, le=100.0)

    def to_dict(self) -> Dict[str, float]:
        return {
            "retrieval": self.retrieval,
            "grounding": self.grounding,
            "citation":  self.citation,
            "overall":   self.overall,
        }


class RetryAttempt(BaseModel):
    attempt_number: int
    strategy:        RetryStrategy
    scores:          ConfidenceScores
    decision:        VerifyDecision
    reason:          str
    duration_ms:     float
    answer_preview:  str = Field(default="", max_length=200)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "strategy":        self.strategy,
            "scores":          self.scores.to_dict(),
            "decision":        self.decision,
            "reason":          self.reason,
            "duration_ms":     self.duration_ms,
        }


class VerificationReport(BaseModel):
    verified:           bool
    scores:              ConfidenceScores
    unsupported_claims:  List[str] = Field(default_factory=list)
    bad_citations:       List[str] = Field(default_factory=list)
    missing_aspects:     List[str] = Field(default_factory=list)
    attempts:            List[RetryAttempt] = Field(default_factory=list)
    total_duration_ms:   float = 0.0
    degraded:            bool = False
    limitation_notice:   Optional[str] = None
    created_at:          float = Field(default_factory=time.time)
    # Citation-filtered sources from the winning attempt's own
    # reasoning_engine.generate_answer() call — NOT the full retrieval
    # candidate pool. Empty for a refusal answer, cited-only otherwise
    # (mirrors generate_answer()'s own `parsed["sources"]` semantics).
    # Callers must use this instead of the raw initial_sources/candidate
    # pool, or citation transparency regresses to "show everything retrieved."
    cited_sources:       List[Dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified":           self.verified,
            "scores":              self.scores.to_dict(),
            "unsupported_claims":  self.unsupported_claims,
            "bad_citations":       self.bad_citations,
            "missing_aspects":     self.missing_aspects,
            "attempts":            [a.to_dict() for a in self.attempts],
            "total_duration_ms":   self.total_duration_ms,
            "degraded":            self.degraded,
            "limitation_notice":   self.limitation_notice,
            "cited_sources":       self.cited_sources,
        }


class RetrievalEvalResult(BaseModel):
    """Output of RetrievalEvaluator — is the retrieved evidence usable at all."""

    score:                float = Field(ge=0.0, le=100.0)
    aspect_coverage_frac: float = Field(ge=0.0, le=1.0, default=1.0)
    conflicting_evidence: bool = False
    insufficient_context: bool = False
    reason:               str = ""


class GroundednessResult(BaseModel):
    """Output of GroundednessChecker."""

    score:               float = Field(ge=0.0, le=100.0)
    is_hallucinated:      bool = False
    unsupported_claims:   List[str] = Field(default_factory=list)
    unsupported_numbers:  List[str] = Field(default_factory=list)


class CitationCheckResult(BaseModel):
    """Output of CitationVerifier."""

    score:          float = Field(ge=0.0, le=100.0)
    bad_citations:   List[str] = Field(default_factory=list)
    checked_count:   int = 0


class CompletenessResult(BaseModel):
    """Output of CompletenessVerifier — ex-video_answer_agent.py, generalized."""

    aspects:  List[str] = Field(default_factory=list)
    covered:  List[str] = Field(default_factory=list)
    missing:  List[str] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing
