"""audio_embedder.py — Finance-grade embedder for audio/transcript chunks."""
from __future__ import annotations

from typing import Any, List

from app.embeddings.base_embedder import BaseEmbedder
from app.core.config import settings


class AudioEmbedder(BaseEmbedder):
    """Embedder for audio chunks (earnings calls, analyst days, podcasts).

    Enrichment:
      [Speaker: {name}] [{role}] [{ts_start}s-{ts_end}s] [{call_section}] {text}
      + finance entity amplification at end

    Entity amplification appends tickers, dollar amounts, and company names
    so queries like "AAPL Q3 revenue" hit the right speaker segment.
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        s = getattr(doc, "structure", {}) or {}

        header_parts: List[str] = []

        name = (
            s.get("speaker_name") or s.get("speaker") or ""
        ).strip()
        role = (s.get("speaker_role") or "").strip()
        if name and role:
            header_parts.append(f"[Speaker: {name} - {role}]")
        elif name:
            header_parts.append(f"[Speaker: {name}]")
        elif role:
            header_parts.append(f"[Speaker: {role}]")

        ts_s = s.get("timestamp_start") or s.get("start_timestamp")
        ts_e = s.get("timestamp_end")   or s.get("end_timestamp")
        if ts_s is not None and ts_e is not None:
            header_parts.append(f"[{float(ts_s):.0f}s-{float(ts_e):.0f}s]")
        elif ts_s is not None:
            header_parts.append(f"[{float(ts_s):.0f}s]")

        call_section = (s.get("call_section") or "").strip()
        if call_section:
            header_parts.append(f"[{call_section}]")

        header = (" ".join(header_parts) + " ") if header_parts else ""

        # Finance entity amplification — append tickers/amounts/companies
        fin_entities: dict = s.get("finance_entities") or {}
        entity_tokens: List[str] = []
        if isinstance(fin_entities, dict):
            for v in fin_entities.values():
                if isinstance(v, list):
                    entity_tokens.extend(str(x) for x in v[:4])

        suffix = (f" [ENTITIES: {', '.join(entity_tokens)}]") if entity_tokens else ""

        # Q&A section: add "question" or "answer" token for section-aware retrieval
        qa_token = ""
        if s.get("is_question"):
            qa_token = " [analyst question]"
        elif s.get("is_answer"):
            qa_token = " [management answer]"

        result = f"{header}{cleaned_text}{qa_token}{suffix}"
        return result[:settings.MAX_PROMPT_CHARS]
