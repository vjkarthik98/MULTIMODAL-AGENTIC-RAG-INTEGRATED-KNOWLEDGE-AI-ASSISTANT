"""Query-decomposition agent for multimodal VIDEO answers (Phase C).

Formalizes a plan -> retrieve -> generate -> verify -> assemble loop as a
reusable, testable unit, scoped to the streaming AV answer path
(app/pipeline/rag_pipeline.py). This is additive scaffolding for the eventual
full agentic upgrade (a larger model at 48GB VRAM) — the per-aspect retrieval
here already runs fine on the current LLM; only the VERIFY/ASSEMBLE steps are
new relative to the earlier context-only decomposition
(rag_pipeline._build_av_stream_context).

Deliberately does NOT do per-aspect answer synthesis (split the question,
generate N separate short answers, concatenate them) — that approach was
tried and reverted: stitching independently-generated fragments mixes facts
and leaks boilerplate (e.g. the IR safe-harbor intro bleeding into an
unrelated aspect's answer). Instead the primary answer is generated ONCE over
a unioned, aspect-retrieved context (the safe approach), and this module adds
a lightweight coverage check + a SINGLE bounded follow-up call for whichever
aspect (if any) the primary answer visibly dropped — never more than one
extra LLM call, and it only ever appends, never edits, the primary answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

_STOPWORDS = frozenset({
    "what", "when", "which", "were", "with", "that", "this", "your", "about",
    "did", "does", "the", "and", "for", "was", "are", "how", "why", "who",
    "apple", "call", "quarter", "these", "they", "their", "from", "into", "said",
    "say", "says", "during", "regarding", "whether", "tell", "company", "results",
    "reported", "report", "also", "year", "over",
})


def _aspect_keywords(aspect: str) -> set:
    return {w for w in re.findall(r"[a-z0-9&]+", aspect.lower())
            if len(w) > 2 and w not in _STOPWORDS}


@dataclass
class AspectCoverage:
    aspects: List[str] = field(default_factory=list)
    covered: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


# VERIFY

def verify_aspect_coverage(answer: str, aspects: List[str],
                           min_overlap: float = 0.34) -> AspectCoverage:
    """For each aspect phrase, check whether its distinguishing keywords
    actually appear in the generated answer. An aspect counts as "covered"
    once at least `min_overlap` of its content keywords show up. Pure
    keyword-presence heuristic — cheap, deterministic, no extra LLM call —
    sufficient to flag an obviously-dropped aspect (a 4-part question
    answered on only 1 part) without adding latency to every video answer.
    """
    al = (answer or "").lower()
    result = AspectCoverage(aspects=list(aspects))
    for asp in aspects:
        kws = _aspect_keywords(asp)
        if not kws:
            continue
        hits = sum(1 for w in kws if w in al)
        frac = hits / len(kws)
        (result.covered if frac >= min_overlap else result.missing).append(asp)
    return result


# ASSEMBLE

def assemble_answer(primary_answer: str, coverage: AspectCoverage,
                    followup_fn: Optional[Callable[[str], Optional[str]]] = None,
                    max_followups: int = 1) -> str:
    """If VERIFY found aspects the primary answer dropped, ask `followup_fn`
    (a single-aspect generation call, injected by the caller) for a short
    addendum on the highest-priority missing aspect and append it. Bounded to
    `max_followups` (default 1) so a bad decomposition can never cascade into
    unbounded extra latency/cost. Never removes or edits the primary answer;
    only appends a clearly-additive sentence.
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
