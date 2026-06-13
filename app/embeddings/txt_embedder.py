"""txt_embedder.py — Finance-grade embedder for plain-text / transcript chunks."""
from __future__ import annotations

from typing import Any

from app.embeddings.base_embedder import BaseEmbedder
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_FLS_SUFFIX = " guidance outlook forecast projection guidance outlook"


class TxtEmbedder(BaseEmbedder):
    """Embedder for TXT chunks (earnings call transcripts, news, filings).

    Enrichment:
      [CALL: {call_section}] [Speaker: {name}] [Section: {title}] {text}
      + FLS amplification when is_forward_looking=True
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        s = getattr(doc, "structure", {}) or {}

        parts = []

        call_section = (s.get("call_section") or "").strip()
        if call_section:
            parts.append(f"[CALL: {call_section}]")

        speaker = (
            s.get("speaker_name")
            or s.get("speaker")
            or s.get("speaker_role")
            or ""
        ).strip()
        if speaker:
            parts.append(f"[Speaker: {speaker}]")

        section = (s.get("section_title") or "").strip()
        if section and len(section) <= 120:
            parts.append(f"[Section: {section}]")

        prefix = (" ".join(parts) + " ") if parts else ""
        text   = prefix + cleaned_text

        if s.get("is_forward_looking"):
            text += _FLS_SUFFIX

        # Transcript heading: repeat section title for BM25-style weight in dense space
        if s.get("chunk_type") == "section" and section:
            text += f" {section} {section}"

        result = text[:settings.MAX_PROMPT_CHARS]
        logger.debug(event="embed_text_built", modality="txt", chars=len(result))
        return result
