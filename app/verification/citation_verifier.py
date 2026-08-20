"""CitationVerifier — Responsibilities 3+4 (docs/Phase_32_Agentic_Answer_Verification.md).

Verifies that every cite_key the answer emits (a) exists in the closed set of
sources actually retrieved for this query, and (b) the text surrounding that
citation tag in the answer is actually supported by the specific chunk it
points at — not just "a source with this key exists somewhere," which would
pass even if the model cited the wrong page/worksheet/timestamp for a claim
that lives in a different chunk of the same file.

Real field names on RetrievedSource (app/core/response.py::build_retrieved_source),
NOT the generic page_number/worksheet/start_time/frame names from the original
spec: `page`, `sheet`, `timestamp_start`, `timestamp_end`, `subtype` (video
frames use subtype="frame"), `asset_path`, `chunk_id`, `cite_key`.
"""

from __future__ import annotations

import re
from typing import Any

from app.verification.verification_schema import CitationCheckResult

# Two alternatives: a run of letters (plain words), OR a numeric token that
# may carry a leading $, thousands-separator commas, a decimal part, and a
# trailing % — e.g. "$391,035" or "46.2%". The old single character class
# `[a-zA-Z0-9%$.]+` did not include ",", so a comma-grouped figure like
# "$391,035" split into "$391" (4 chars) and "035" (3 chars) — BOTH under the
# `len(w) > 4` significance filter below, making the most common financial
# number format in this domain invisible to the support check regardless of
# whether the cited chunk actually contains it. Live-reproduced (2026-08-17,
# citation_accuracy_v2 cross-modality follow-up): a citation whose chunk
# verbatim contained "$391,035 | $383,285 | $394,328" — exactly the answer's
# own stated figures — still scored 0.0 (flagged bad) because none of those
# three numbers survived tokenization to be counted as a match.
_CLAIM_WORD_RE = re.compile(r"[a-zA-Z]+|\$?\d[\d,]*\.?\d*%?")
_CLAIM_WINDOW_CHARS = 160
_MIN_CLAIM_SUPPORT_FRAC = 0.20


class CitationVerifier:
    """Every cited chunk must actually contain the claim it's cited for."""

    def check(
        self, answer: str, docs: list[dict[str, Any]], sources: list[dict[str, Any]]
    ) -> CitationCheckResult:
        if not answer or not sources:
            return CitationCheckResult(score=100.0, checked_count=0)

        from app.reasoning.reasoning_engine import _extract_cite_tags

        valid_keys = [s.get("cite_key") for s in sources if s.get("cite_key")]
        cited = _extract_cite_tags(answer, valid_keys)
        if not cited:
            # Nothing cited. Not automatically wrong — some answers are
            # synthesized/refusal text — but it's not evidence of GOOD
            # citations either, so this is a soft, non-blocking default.
            return CitationCheckResult(score=100.0, checked_count=0)

        cite_to_text = self._cite_key_to_chunk_text(docs, sources)

        bad: list[str] = []
        for key in cited:
            chunk_text = cite_to_text.get(key, "")
            if not chunk_text:
                bad.append(key)
                continue
            if not self._all_occurrences_supported(answer, key, chunk_text):
                bad.append(key)

        checked = len(cited)
        score = round(100.0 * (checked - len(bad)) / checked, 2) if checked else 100.0
        return CitationCheckResult(score=score, bad_citations=bad, checked_count=checked)

    @staticmethod
    def _cite_key_to_chunk_text(
        docs: list[dict[str, Any]], sources: list[dict[str, Any]]
    ) -> dict[str, str]:
        chunk_id_to_doc: dict[str, dict[str, Any]] = {}
        for d in docs or []:
            cid = str((d.get("metadata") or {}).get("chunk_id") or "")
            if cid:
                chunk_id_to_doc[cid] = d

        out: dict[str, str] = {}
        for s in sources or []:
            key = s.get("cite_key")
            if not key:
                continue
            cid = str(s.get("chunk_id") or "")
            doc = chunk_id_to_doc.get(cid)
            # Fall back to the source's own snippet if the full chunk text
            # isn't reachable (e.g. docs list doesn't carry this chunk_id) —
            # a shorter but still real check beats skipping verification.
            out[key] = (doc.get("text") if doc else None) or s.get("snippet") or ""
        return out

    @classmethod
    def _all_occurrences_supported(cls, answer: str, cite_key: str, chunk_text: str) -> bool:
        """Every USE of this cite_key must be supported, not just the first.

        Two bugs, both live-reproduced (2026-08-17, citation_accuracy_v2
        cross-modality follow-up) with a source cited twice for two
        different claims under the same tag ("Gross margin was 46.2% [1].
        iPhone shipments increased 15% [1]."), the second claim being about
        a topic the cited chunk never mentions at all:
        1. answer.find() alone only ever locates the FIRST occurrence, so
           only the first claim's window was ever checked — a fabricated or
           unrelated second claim under the same tag was invisible to this
           check entirely (scored 100.0, zero bad_citations).
        2. Fixing (1) alone by scanning every occurrence still wasn't enough
           when two citations of the same key are close together: the fixed
           160-char lookback window for the SECOND citation still reached
           backward across the sentence boundary into the FIRST claim's
           (genuinely supported) text, diluting the word-overlap score with
           borrowed support that has nothing to do with the second claim.
           window_start (anchored to just after the PREVIOUS occurrence of
           this same key) stops each window at its own claim's boundary.
        """
        idx = answer.find(cite_key)
        if idx < 0:
            return True
        window_start = 0
        while idx >= 0:
            if not cls._claim_supported(answer, idx, chunk_text, window_start):
                return False
            window_start = idx + len(cite_key)
            idx = answer.find(cite_key, window_start)
        return True

    @staticmethod
    def _claim_supported(answer: str, idx: int, chunk_text: str, window_start: int = 0) -> bool:
        window = answer[max(window_start, idx - _CLAIM_WINDOW_CHARS) : idx]
        words = [w.lower() for w in _CLAIM_WORD_RE.findall(window) if len(w) > 4]
        if not words:
            # No substantive claim text before the tag (e.g. tag opens the
            # sentence) — nothing concrete to check against; don't penalize.
            return True
        chunk_lower = chunk_text.lower()
        hits = sum(1 for w in words if w in chunk_lower)
        return (hits / len(words)) >= _MIN_CLAIM_SUPPORT_FRAC
