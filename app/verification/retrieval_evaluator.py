"""RetrievalEvaluator — Responsibility 1 (docs/Phase_32_Agentic_Answer_Verification.md).

Is the retrieved evidence even usable before we bother generating an answer
from it: relevant, sufficient for a multi-part question, and not internally
contradictory. Reuses hybrid_retriever's own fusion/rerank scores (already
computed — no re-scoring) and rag_pipeline's query-aspect splitter.
"""

from __future__ import annotations

import re
from typing import Any

from app.verification.verification_schema import RetrievalEvalResult

_NUM_WITH_CONTEXT_RE = re.compile(
    r"([A-Za-z][A-Za-z \-]{2,30}?)\s+(?:of\s+|was\s+|is\s+|at\s+)?\$?([\d,]+\.?\d*)\s*(billion|million|thousand|%)?",
    re.IGNORECASE,
)
_MIN_SCORE_FOR_RELEVANT = 0.15


class RetrievalEvaluator:
    def evaluate(self, query: str, docs: list[dict[str, Any]]) -> RetrievalEvalResult:
        if not docs:
            return RetrievalEvalResult(
                score=0.0,
                insufficient_context=True,
                reason="no documents retrieved",
            )

        relevant = [d for d in docs if self._score_of(d) >= _MIN_SCORE_FOR_RELEVANT]
        relevance_frac = len(relevant) / len(docs) if docs else 0.0

        aspect_frac = self._aspect_coverage(query, docs)
        conflicting = self._has_conflicting_numbers(docs)

        # Weighted blend: relevance dominates (it's the direct fusion/rerank
        # signal already computed upstream), aspect coverage catches
        # multi-part questions under-retrieved, conflict is a hard penalty.
        base = 0.65 * relevance_frac + 0.35 * aspect_frac
        if conflicting:
            base *= 0.7
        score = round(base * 100.0, 2)

        insufficient = relevance_frac < 0.34 or (aspect_frac < 0.5 and aspect_frac > 0)
        reason = ""
        if insufficient:
            reason = f"relevance_frac={relevance_frac:.2f} aspect_frac={aspect_frac:.2f}"
        elif conflicting:
            reason = "conflicting numeric evidence across top-k docs"

        return RetrievalEvalResult(
            score=score,
            aspect_coverage_frac=aspect_frac,
            conflicting_evidence=conflicting,
            insufficient_context=insufficient,
            reason=reason,
        )

    @staticmethod
    def _score_of(doc: dict[str, Any]) -> float:
        val = doc.get("score")
        return float(val) if isinstance(val, (int, float)) else 0.0

    @staticmethod
    def _aspect_coverage(query: str, docs: list[dict[str, Any]]) -> float:
        try:
            from app.pipeline.rag_pipeline import _split_query_aspects
        except Exception:
            return 1.0
        aspects = _split_query_aspects(query)
        if len(aspects) < 2:
            return 1.0
        text = " ".join(str(d.get("text", "") or "").lower() for d in docs)
        hits = 0
        for asp in aspects:
            words = [w for w in re.findall(r"[a-z0-9]+", asp.lower()) if len(w) > 3]
            if not words:
                continue
            if sum(1 for w in words if w in text) / len(words) >= 0.3:
                hits += 1
        return hits / len(aspects) if aspects else 1.0

    @staticmethod
    def _has_conflicting_numbers(docs: list[dict[str, Any]]) -> bool:
        """Cheap conflict signal: same labeled metric (e.g. 'net revenue')
        with materially different scaled values across top docs. False
        positives are acceptable here — this only nudges the score down and
        adds a reason string, it never blocks on its own."""
        seen: dict[str, set] = {}
        for d in docs[:5]:
            text = str(d.get("text", "") or "")
            for m in _NUM_WITH_CONTEXT_RE.finditer(text):
                label = m.group(1).strip().lower()
                if len(label) < 4:
                    continue
                try:
                    val = float(m.group(2).replace(",", ""))
                except ValueError:
                    continue
                seen.setdefault(label, set()).add(val)
        for vals in seen.values():
            if len(vals) > 1:
                lo, hi = min(vals), max(vals)
                if lo > 0 and (hi / lo) > 1.5:
                    return True
        return False
