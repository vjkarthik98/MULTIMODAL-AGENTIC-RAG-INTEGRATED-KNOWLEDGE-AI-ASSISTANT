"""CompletenessVerifier — generalized from the retired app/agents/video_answer_agent.py.

The original module scoped this to the streaming AV answer path only. The
underlying heuristic (keyword-overlap coverage per decomposed aspect) was
never actually video-specific — a 4-part PDF/XLSX "compare X, Y, and Z"
question drops aspects the same way a multi-part video question does. This
version runs for every modality via VerificationLoop.

Deliberately does NOT do per-aspect answer synthesis (split the question,
generate N separate short answers, concatenate them) — that approach was
tried and reverted upstream: stitching independently-generated fragments
mixes facts and leaks boilerplate. Instead the primary answer is generated
ONCE over unioned context (unchanged), and this module adds a lightweight
coverage check + bounded follow-up calls for whichever aspects the primary
answer visibly dropped — never unbounded, and it only ever appends, never
edits, the primary answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from app.verification.verification_schema import CompletenessResult

_STOPWORDS = frozenset(
    {
        "what",
        "when",
        "which",
        "were",
        "with",
        "that",
        "this",
        "your",
        "about",
        "did",
        "does",
        "the",
        "and",
        "for",
        "was",
        "are",
        "how",
        "why",
        "who",
        "call",
        "quarter",
        "these",
        "they",
        "their",
        "from",
        "into",
        "said",
        "say",
        "says",
        "during",
        "regarding",
        "whether",
        "tell",
        "company",
        "results",
        "reported",
        "report",
        "also",
        "year",
        "over",
    }
)


def _aspect_keywords(aspect: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9&]+", aspect.lower()) if len(w) > 2 and w not in _STOPWORDS
    }


def verify_aspect_coverage(
    answer: str, aspects: list[str], min_overlap: float = 0.34
) -> CompletenessResult:
    """For each aspect phrase, check whether its distinguishing keywords
    actually appear in the generated answer. An aspect counts as "covered"
    once at least `min_overlap` of its content keywords show up. Pure
    keyword-presence heuristic — cheap, deterministic, no extra LLM call —
    sufficient to flag an obviously-dropped aspect (a 4-part question
    answered on only 1 part) without adding latency to every answer.
    """
    al = (answer or "").lower()
    result = CompletenessResult(aspects=list(aspects))
    for asp in aspects:
        kws = _aspect_keywords(asp)
        if not kws:
            continue
        hits = sum(1 for w in kws if w in al)
        frac = hits / len(kws)
        (result.covered if frac >= min_overlap else result.missing).append(asp)
    return result


def assemble_answer(
    primary_answer: str,
    coverage: CompletenessResult,
    followup_fn: Callable[[str], str | None] | None = None,
    max_followups: int = 3,
) -> str:
    """If VERIFY found aspects the primary answer dropped, ask `followup_fn`
    (a single-aspect generation call, injected by the caller) for a short
    addendum on each missing aspect (bounded by `max_followups`) and append
    it. A bad decomposition can never cascade into unbounded extra
    latency/cost. Never removes or edits the primary answer; only appends
    clearly-additive sentences.
    """
    if not coverage.missing or followup_fn is None:
        return primary_answer
    out = primary_answer
    for asp in coverage.missing[:max_followups]:
        try:
            addendum = followup_fn(asp)
        except Exception:
            addendum = None
        if addendum and addendum.strip():
            sep = " " if out.rstrip().endswith((".", "!", "?")) else ". "
            out = f"{out.rstrip()}{sep}{addendum.strip()}"
    return out


class CompletenessVerifier:
    """Responsibility 5 — is every part of a multi-part question answered."""

    def __init__(self, min_overlap: float = 0.34, max_followups: int = 3) -> None:
        self.min_overlap = min_overlap
        self.max_followups = max_followups

    def check(self, answer: str, aspects: list[str], query: str = "") -> CompletenessResult:
        # A single-aspect (or zero-aspect) question still needs a check: the
        # multi-aspect keyword-overlap heuristic reused here as-is, treating
        # the whole ORIGINAL QUERY as the one aspect. Auto-passing here
        # (the pre-live-smoke-test behavior) let a fluent, topically-related
        # but non-responsive answer slip through — grounded and cited, but
        # never actually mentioning the thing asked about (e.g. a revenue
        # question answered with an unrelated "advertising and licensing"
        # sentence that happens to share a couple of words with the
        # transcript). Caught by a live end-to-end smoke test, Phase 32.
        if len(aspects) < 2:
            if not query:
                return CompletenessResult(aspects=list(aspects), covered=list(aspects))
            return verify_aspect_coverage(answer, [query], min_overlap=self.min_overlap)
        return verify_aspect_coverage(answer, aspects, min_overlap=self.min_overlap)

    def fill_gaps(
        self,
        primary_answer: str,
        coverage: CompletenessResult,
        followup_fn: Callable[[str], str | None] | None,
    ) -> str:
        return assemble_answer(
            primary_answer, coverage, followup_fn=followup_fn, max_followups=self.max_followups
        )
