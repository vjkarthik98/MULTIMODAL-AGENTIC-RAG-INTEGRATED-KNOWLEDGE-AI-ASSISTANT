"""txt_bm25.py — BM25 index for plain-text / transcript chunks."""
from __future__ import annotations

from typing import Any, List

from app.bm25.base_bm25 import BaseBM25


class TxtBM25(BaseBM25):
    """BM25 index for .txt documents and earnings call transcripts.

    Enrichment over base:
    - section_title repeated ×2 (already in base ×2 — add once more for TXT)
    - FLS amplification: forward-looking chunks get guidance/outlook tokens ×2
    - Speaker prefix: "speaker {role}" token for transcript queries
    """

    modality = "txt"

    def _build_indexed_text(self, doc: Any) -> str:
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = list(self._base_text(doc))

        # Extra section_title boost for TXT (base already added ×2; add once more)
        section_title = (s.get("section_title") or "").strip()
        if section_title:
            parts.append(section_title)

        # Speaker prefix for transcript speaker-turn chunks
        speaker = (s.get("speaker") or s.get("speaker_name") or "").strip()
        role    = (s.get("speaker_role") or "").strip()
        if speaker:
            parts.append(f"speaker {speaker}")
        if role:
            parts.append(f"speaker role {role}")

        # FLS amplification: forward-looking statements are high-value for guidance queries
        if s.get("is_forward_looking"):
            fls_tokens = "guidance outlook forecast projection expects anticipated"
            parts.append(fls_tokens)
            parts.append(fls_tokens)

        # Call section token
        call_section = (s.get("call_section") or "").strip()
        if call_section:
            parts.append(call_section.replace("_", " "))

        return " ".join(parts)
