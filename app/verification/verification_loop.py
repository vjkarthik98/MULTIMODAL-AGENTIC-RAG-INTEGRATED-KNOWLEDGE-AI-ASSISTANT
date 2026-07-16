"""VerificationLoop — orchestrates generate -> verify -> decide -> retry.

Entry point for the whole Phase 32 subsystem
(docs/Phase_32_Agentic_Answer_Verification.md). Called from
app/pipeline/query_pipeline.py and app/pipeline/rag_pipeline.py::stream() in
place of the direct `reasoning_engine.generate_answer()` call.

Failure-propagation contract (architect review, Phase 32): a HARD failure on
the baseline (attempt 0) generation — LLM unavailable, OOM, etc. — is
re-raised so the caller's existing exception handling (query_pipeline.py's
GGUF fallback chain) still fires exactly as it did before this loop existed.
A hard failure on a RETRY attempt (1-4) does not crash the request — the loop
already has a working baseline answer, so that attempt is scored as FAIL and
the loop proceeds to stop/retry normally.

Owns its own wall-clock timeout (StoppingCriteria) independent of
AgentController's, because AgentController's timeout wraps routing, not
generation (CLAUDE.md hard rule: bounded loops need max-steps + timeout +
token budget simultaneously; this loop's "steps" are verification attempts,
its "budget" is AGENT_VERIFY_MAX_RETRIES).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import Counter, Histogram

from app.core.config import settings
from app.utils.logger import get_logger
from app.verification.citation_verifier import CitationVerifier
from app.verification.completeness_verifier import CompletenessVerifier
from app.verification.confidence_scorer import ConfidenceScorer
from app.verification.groundedness_checker import GroundednessChecker
from app.verification.retrieval_evaluator import RetrievalEvaluator
from app.verification.retry_controller import RetryController
from app.verification.stopping_criteria import StoppingCriteria
from app.verification.verification_schema import (
    ConfidenceScores,
    RetryAttempt,
    VerificationReport,
)

logger = get_logger(__name__)

# video_chunker.py deliberately tags frame/vision-collection chunks
# modality="mp4" (see qdrant_store.py:499's own "video","mp4" pairing) while
# transcript/text-collection chunks from the SAME file are tagged "video" —
# by design, for the dual text_collection/vision_collection split. Any
# modality-gate check against settings.AGENT_VERIFY_MODALITIES (canonical
# 7 names: txt/pdf/docx/xlsx/image/audio/video) must normalize this alias
# first, or verification silently never fires whenever the top-ranked doc
# happens to be a frame/vision chunk (confirmed via live smoke test,
# Phase 32) — same alias set the pre-Phase-32 code defensively checked
# (`m in ("audio", "mp3", "video", "mp4")`).
_MODALITY_ALIASES = {"mp4": "video", "mp3": "audio"}


def normalize_modality(modality: Optional[str]) -> str:
    m = str(modality or "").lower()
    return _MODALITY_ALIASES.get(m, m)


_LIMITATION_NOTICE = (
    "This answer could not be fully verified against the source material — "
    "treat the figures above with caution."
)

_loop_total = Counter(
    "magik_verification_loop_total", "Verification loop outcomes", ["decision"],
)
_retry_total = Counter(
    "magik_verification_retry_total", "Verification retry attempts by strategy", ["strategy"],
)
_confidence_hist = Histogram(
    "magik_verification_confidence", "Verification confidence scores", ["score_type"],
    buckets=(0, 20, 40, 60, 75, 85, 90, 95, 100),
)
_duration_hist = Histogram(
    "magik_verification_duration_seconds", "Verification loop total duration",
)


class VerificationLoop:
    def __init__(self) -> None:
        self.retrieval_evaluator = RetrievalEvaluator()
        self.groundedness_checker = GroundednessChecker()
        self.citation_verifier = CitationVerifier()
        self.completeness_verifier = CompletenessVerifier()
        self.confidence_scorer = ConfidenceScorer()
        self.stopping = StoppingCriteria()

    def run(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str],
        retriever: Any,
        reasoning_engine: Any,
        initial_docs: List[Dict[str, Any]],
        initial_sources: Optional[List[Dict[str, Any]]] = None,
        llm: Any = None,
        modality_hint: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        memory_context: str = "",
    ) -> Tuple[str, VerificationReport]:

        modality_hint = normalize_modality(modality_hint) if modality_hint else modality_hint
        if not settings.AGENT_VERIFY_ENABLED or (
            modality_hint and modality_hint not in settings.AGENT_VERIFY_MODALITIES
        ):
            return self._passthrough(
                query, session_id, user_id, reasoning_engine,
                initial_docs, initial_sources, memory_context,
            )

        start = time.time()
        retry_ctrl = RetryController()
        aspects = self._split_aspects(query)

        docs = list(initial_docs or [])
        sources = list(initial_sources or [])
        effective_query = query

        attempts: List[RetryAttempt] = []
        answers: List[str] = []
        cited_sources_per_attempt: List[List[Dict[str, Any]]] = []
        grounding_results = []
        citation_results = []
        completeness_results = []
        strategy = "baseline"
        attempt_number = 0

        while True:
            t0 = time.time()

            if attempt_number > 0:
                try:
                    docs, effective_query = retry_ctrl.execute(
                        strategy, effective_query, session_id, user_id,
                        retriever, llm, filters, docs,
                    )
                except Exception:
                    # A retry's retrieval/merge step failing must not crash
                    # the request either — same contract as a retry
                    # generation failure. RetryController._search already
                    # catches retriever.search() exceptions internally; this
                    # is the outer guard for anything else in execute()
                    # (e.g. a malformed doc breaking _dedup_docs/_normalize_docs).
                    logger.warning(
                        event="verification_retry_execute_failed",
                        attempt_number=attempt_number, strategy=strategy, session_id=session_id,
                    )
                    break
                sources = self._rebuild_sources(docs, fallback=sources)
                _retry_total.labels(strategy=strategy).inc()

            try:
                answer, cited_sources = self._generate(
                    reasoning_engine, effective_query, docs, memory_context,
                    session_id, sources, user_id,
                )
            except Exception:
                if attempt_number == 0:
                    # Baseline hard failure — let the caller's own fallback
                    # chain handle it, exactly as it did before this loop.
                    raise
                # A retry-attempt failure must not crash the request, but it
                # also must NOT be scored: re-checking a STALE answer (from a
                # prior attempt) against this attempt's freshly-retried docs
                # would produce a meaningless — and potentially a false-PASS —
                # attempt record, since the score would reflect a coincidental
                # match between unrelated answer/doc pairs, not a real
                # verification of anything this attempt actually generated.
                # Stop here and fall back to the best attempt already scored
                # (guaranteed non-empty: the baseline always produces one).
                logger.warning(
                    event="verification_retry_generation_failed",
                    attempt_number=attempt_number, strategy=strategy, session_id=session_id,
                )
                break

            retrieval_res = self.retrieval_evaluator.evaluate(effective_query, docs)
            grounding_res = self.groundedness_checker.check(answer, docs, query=effective_query)
            citation_res = self.citation_verifier.check(answer, docs, sources)
            completeness_res = self.completeness_verifier.check(answer, aspects, query=effective_query)

            scores = self.confidence_scorer.score(
                retrieval_res, grounding_res, citation_res, completeness_res,
            )
            decision, reason = self.confidence_scorer.decide(
                scores, grounding_res, citation_res, completeness_res,
            )

            duration_ms = round((time.time() - t0) * 1000, 1)
            attempt = RetryAttempt(
                attempt_number=attempt_number, strategy=strategy, scores=scores,
                decision=decision, reason=reason, duration_ms=duration_ms,
                answer_preview=answer[:200],
            )
            attempts.append(attempt)
            answers.append(answer)
            cited_sources_per_attempt.append(cited_sources)
            grounding_results.append(grounding_res)
            citation_results.append(citation_res)
            completeness_results.append(completeness_res)

            logger.info(
                event="verification_iteration",
                attempt_number=attempt_number, strategy=strategy,
                scores=scores.to_dict(), decision=decision, reason=reason,
                duration_ms=duration_ms, session_id=session_id,
            )

            should_stop, _stop_reason = self.stopping.should_stop(attempts, start)
            if should_stop:
                break

            next_strategy = retry_ctrl.next_strategy()
            if next_strategy is None:
                break
            strategy = next_strategy
            attempt_number += 1

        best = self.stopping.best_attempt(attempts) or attempts[-1]
        best_idx = attempts.index(best)
        final_answer = answers[best_idx]
        verified = best.decision == "PASS"
        best_grounding = grounding_results[best_idx]
        best_citation = citation_results[best_idx]
        best_completeness = completeness_results[best_idx]

        total_ms = round((time.time() - start) * 1000, 1)

        report = VerificationReport(
            verified=verified,
            scores=best.scores,
            unsupported_claims=[] if verified else (
                best_grounding.unsupported_claims
                + [f"unsupported number: {n}" for n in best_grounding.unsupported_numbers]
            ),
            bad_citations=[] if verified else best_citation.bad_citations,
            missing_aspects=[] if verified else best_completeness.missing,
            attempts=attempts,
            total_duration_ms=total_ms,
            degraded=not verified,
            limitation_notice=None if verified else _LIMITATION_NOTICE,
            cited_sources=cited_sources_per_attempt[best_idx],
        )

        if not verified and final_answer:
            final_answer = f"{final_answer.rstrip()}\n\n{_LIMITATION_NOTICE}"

        self._record_metrics(verified, attempts, total_ms)
        logger.info(
            event="verification_final", verified=verified, attempts=len(attempts),
            total_duration_ms=total_ms, session_id=session_id,
        )

        return final_answer, report

    # INTERNALS

    def _passthrough(self, query, session_id, user_id, reasoning_engine,
                      docs, sources, memory_context) -> Tuple[str, VerificationReport]:
        """Verification disabled globally or opted out for this modality."""
        answer, cited_sources = self._generate(reasoning_engine, query, docs or [], memory_context,
                                                 session_id, sources, user_id)
        scores = ConfidenceScores(retrieval=100.0, grounding=100.0, citation=100.0, overall=100.0)
        report = VerificationReport(verified=True, scores=scores, degraded=False,
                                     cited_sources=cited_sources)
        return answer, report

    @staticmethod
    def _generate(reasoning_engine: Any, query: str, docs: List[Dict[str, Any]],
                   memory_context: str, session_id: str,
                   sources: Optional[List[Dict[str, Any]]], user_id: Optional[str],
                   ) -> Tuple[str, List[Dict[str, Any]]]:
        """Returns (answer, cited_sources). `cited_sources` is
        generate_answer()'s OWN citation-filtered subset (empty for a
        refusal, cited-only otherwise) — NOT the raw candidate pool passed
        in as `sources`. Callers must propagate this, not `sources`/
        `initial_sources`, or citation transparency silently regresses to
        "show everything retrieved" (caught in code review, Phase 32)."""
        out = reasoning_engine.generate_answer(
            query=query, retrieved_docs=docs, memory_context=memory_context,
            session_id=session_id, sources=sources or None, user_id=user_id or "",
        )
        answer = (out.get("answer") or "").strip()
        cited = out.get("sources")
        return answer, (cited if isinstance(cited, list) else list(sources or []))

    @staticmethod
    def _split_aspects(query: str) -> List[str]:
        try:
            from app.pipeline.rag_pipeline import _split_query_aspects
            return _split_query_aspects(query)
        except Exception:
            return []

    @staticmethod
    def _rebuild_sources(docs: List[Dict[str, Any]],
                          fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            from app.core.response import build_sources
            return build_sources(docs)
        except Exception:
            return fallback

    @staticmethod
    def _record_metrics(verified: bool, attempts: List[RetryAttempt], total_ms: float) -> None:
        try:
            _loop_total.labels(decision="PASS" if verified else "FAIL").inc()
            _duration_hist.observe(total_ms / 1000.0)
            if attempts:
                s = attempts[-1].scores
                _confidence_hist.labels(score_type="retrieval").observe(s.retrieval)
                _confidence_hist.labels(score_type="grounding").observe(s.grounding)
                _confidence_hist.labels(score_type="citation").observe(s.citation)
                _confidence_hist.labels(score_type="overall").observe(s.overall)
        except Exception:
            pass
