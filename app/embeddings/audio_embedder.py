"""audio_embedder.py — Finance-grade embedder for audio/transcript chunks."""
from __future__ import annotations

from typing import Any, List

from app.embeddings.base_embedder import BaseEmbedder
from app.core.config import settings
from app.utils.logger import get_logger
from prometheus_client import Counter

logger = get_logger(__name__)

_EMBED_BUILT = Counter(
    "magik_audio_embed_text_built_total",
    "Embed texts successfully built for audio",
)
_EMBED_ERRORS = Counter(
    "magik_audio_embed_text_errors_total",
    "Errors building embed text for audio",
)


class AudioEmbedder(BaseEmbedder):
    """Embedder for audio chunks (earnings calls, analyst days, podcasts).

    Enrichment:
      [Speaker: {name}] [{role}] [{ts_start}s-{ts_end}s] [{call_section}] {text}
      + finance entity amplification at end

    Entity amplification appends tickers, dollar amounts, and company names
    so queries like "AAPL Q3 revenue" hit the right speaker segment.
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}

            header_parts: List[str] = []

            # Section prefix — bakes call context into the vector.
            # "[Q&A SESSION]" / "[PREPARED REMARKS]" from spec Phase 2.6.
            call_section = (s.get("call_section") or "").strip()
            if call_section:
                section_label = call_section.upper().replace("_", " ")
                header_parts.append(f"[{section_label}]")

            # Speaker identity — avoid "Name - Role - Role" when name already embeds role.
            name = (s.get("speaker_name") or s.get("speaker") or "").strip()
            role = (s.get("speaker_role") or "").strip()
            role_in_name = role and role.lower() in name.lower()
            if name and role and not role_in_name:
                header_parts.append(f"[Speaker: {name} - {role}]")
            elif name:
                header_parts.append(f"[Speaker: {name}]")
            elif role:
                header_parts.append(f"[Speaker: {role}]")

            # Timestamps at millisecond precision — spec Phase 2.6 critical field.
            ts_s = s.get("start_timestamp") or s.get("timestamp_start")
            ts_e = s.get("end_timestamp")   or s.get("timestamp_end")
            if ts_s is not None and ts_e is not None:
                header_parts.append(f"[{float(ts_s):.3f}s-{float(ts_e):.3f}s]")
            elif ts_s is not None:
                header_parts.append(f"[{float(ts_s):.3f}s]")

            header = (" ".join(header_parts) + " ") if header_parts else ""

            # Finance entity amplification — append tickers/amounts/companies.
            fin_entities: dict = s.get("finance_entities") or {}
            entity_tokens: List[str] = []
            if isinstance(fin_entities, dict):
                for v in fin_entities.values():
                    if isinstance(v, list):
                        entity_tokens.extend(str(x) for x in v[:4])

            suffix = (f" [ENTITIES: {', '.join(entity_tokens)}]") if entity_tokens else ""

            # Q&A section token for section-aware retrieval.
            qa_token = ""
            if s.get("is_question"):
                qa_token = " [analyst question]"
            elif s.get("is_answer"):
                qa_token = " [management answer]"

            result = f"{header}{cleaned_text}{qa_token}{suffix}"
            result = result[:settings.MAX_PROMPT_CHARS]
            logger.debug(event="embed_text_built", modality="audio", chars=len(result))
            _EMBED_BUILT.inc()
            return result
        except Exception as _exc:
            _EMBED_ERRORS.inc()
            logger.error(event="embed_text_build_failed", modality="audio", error=str(_exc))
            return cleaned_text  # safe fallback to unenriched text

    def health_check(self) -> dict:
        return {
            "modality": "audio",
            "status": "ok",
            "class": self.__class__.__name__,
        }
