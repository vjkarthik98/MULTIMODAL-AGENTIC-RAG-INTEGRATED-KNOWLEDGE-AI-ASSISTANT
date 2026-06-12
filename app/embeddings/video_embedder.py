"""video_embedder.py — Finance-grade embedder for video/webcast chunks."""
from __future__ import annotations

from typing import Any, List

from app.embeddings.base_embedder import BaseEmbedder, sanitize_text, normalize_finance_numbers
from app.core.config import settings


class VideoEmbedder(BaseEmbedder):
    """Embedder for video chunks (investor day webcasts, earnings presentations).

    Enrichment:
      Uses combined_text (transcript + "[VISUAL AT Xs]: caption [ON-SCREEN]: ocr")
      already assembled by VideoChunker — bakes visual context into the vector.

      Slide bullets amplified ×3 so "slide 23: revenue breakdown" queries hit
      the right chunk even when spoken text doesn't repeat the slide title.

      Speaker + timestamp header mirrors AudioEmbedder for consistent retrieval.
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        s = getattr(doc, "structure", {}) or {}

        # Prefer combined_text (transcript + visual frames) over plain text
        combined = (s.get("combined_text") or "").strip()
        base_text = combined if combined and len(combined) > len(cleaned_text) else cleaned_text
        if combined:
            base_text = sanitize_text(combined) or cleaned_text
            base_text = normalize_finance_numbers(base_text)

        header_parts: List[str] = []

        name = (s.get("speaker_name") or s.get("speaker") or "").strip()
        role = (s.get("speaker_role") or "").strip()
        if name and role:
            header_parts.append(f"[Speaker: {name} - {role}]")
        elif name:
            header_parts.append(f"[Speaker: {name}]")

        ts_s = s.get("start_timestamp") or s.get("timestamp_start")
        ts_e = s.get("end_timestamp")   or s.get("timestamp_end")
        if ts_s is not None and ts_e is not None:
            header_parts.append(f"[{float(ts_s):.0f}s-{float(ts_e):.0f}s]")

        call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
        if call_section:
            header_parts.append(f"[{call_section}]")

        header = (" ".join(header_parts) + " ") if header_parts else ""

        # Slide bullets — amplify ×3 so slide-title queries retrieve correctly
        slide_bullets: List[str] = s.get("slide_bullets") or []
        bullet_block = ""
        if slide_bullets:
            bullet_text = " ".join(str(b) for b in slide_bullets[:10])
            bullet_block = f" {bullet_text} {bullet_text} {bullet_text}"

        # Finance entity amplification
        fin_entities: dict = s.get("finance_entities") or {}
        entity_tokens: List[str] = []
        if isinstance(fin_entities, dict):
            for v in fin_entities.values():
                if isinstance(v, list):
                    entity_tokens.extend(str(x) for x in v[:4])
        entity_suffix = (f" [ENTITIES: {', '.join(entity_tokens)}]") if entity_tokens else ""

        result = f"{header}{base_text}{bullet_block}{entity_suffix}"
        return result[:settings.MAX_PROMPT_CHARS]
