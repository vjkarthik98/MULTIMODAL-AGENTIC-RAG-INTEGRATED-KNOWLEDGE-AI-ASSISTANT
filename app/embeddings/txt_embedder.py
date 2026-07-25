"""txt_embedder.py — Finance-grade embedder for plain-text / transcript chunks."""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter

from app.core.config import settings
from app.embeddings.base_embedder import BaseEmbedder
from app.utils.logger import get_logger

logger = get_logger(__name__)

_FLS_SUFFIX = " guidance outlook forecast projection guidance outlook"

_EMBED_BUILT = Counter(
    "magik_txt_embed_text_built_total",
    "Embed texts successfully built for txt",
)
_EMBED_ERRORS = Counter(
    "magik_txt_embed_text_errors_total",
    "Errors building embed text for txt",
)


class TxtEmbedder(BaseEmbedder):
    """Embedder for TXT chunks (earnings call transcripts, news, filings).

    Enrichment:
      [CALL: {call_section}] [Speaker: {name}] [Section: {title}] {text}
      + FLS amplification when is_forward_looking=True
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}

            parts = []

            call_section = (s.get("call_section") or "").strip()
            if call_section:
                parts.append(f"[CALL: {call_section}]")

            speaker = (
                s.get("speaker_name") or s.get("speaker") or s.get("speaker_role") or ""
            ).strip()
            if speaker:
                parts.append(f"[Speaker: {speaker}]")

            section = (s.get("section_title") or "").strip()
            if section and len(section) <= 120:
                parts.append(f"[Section: {section}]")

            prefix = (" ".join(parts) + " ") if parts else ""
            text = prefix + cleaned_text

            if s.get("is_forward_looking"):
                text += _FLS_SUFFIX

            # Transcript heading: repeat section title for BM25-style weight in dense space
            if s.get("chunk_type") == "section" and section:
                text += f" {section} {section}"

            # Repeat this chunk's own detected finance figures once more at the
            # end. A numeric query ("what was the rate", "how much did X rise")
            # embeds close to the exact figure it's asking about; chunks that
            # only mention a number once, mid-sentence, among unrelated prose
            # can under-rank against chunks that happen to repeat it. This
            # boosts a chunk's own real, already-extracted numbers — it never
            # introduces a number that isn't verbatim in the chunk.
            fin_entities = s.get("finance_entities")
            if fin_entities:
                text += " " + " ".join(str(e) for e in fin_entities[:12])

            result = text[: settings.MAX_PROMPT_CHARS]
            logger.debug(event="embed_text_built", modality="txt", chars=len(result))
            _EMBED_BUILT.inc()
            return result
        except Exception as _exc:
            _EMBED_ERRORS.inc()
            logger.error(event="embed_text_build_failed", modality="txt", error=str(_exc))
            return cleaned_text  # safe fallback to unenriched text

    def health_check(self) -> dict:
        return {
            "modality": "txt",
            "status": "ok",
            "class": self.__class__.__name__,
        }
