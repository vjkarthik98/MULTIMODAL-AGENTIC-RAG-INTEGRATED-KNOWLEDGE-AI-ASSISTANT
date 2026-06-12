"""audio_bm25.py — BM25 index for audio/earnings call transcript chunks."""
from __future__ import annotations

from typing import Any, List

from app.bm25.base_bm25 import BaseBM25


class AudioBM25(BaseBM25):
    """BM25 index for audio transcript chunks (earnings calls, investor days).

    Enrichment over base:
    - speaker_name + role tokens: "speaker cfo john smith"
    - timestamp token: "time 28" (minute) so "at 28 minutes" queries work
    - call_section label: "prepared_remarks", "qa_session" etc.
    - finance_entities values: tickers, amounts, dates amplified
    - Q&A role tokens: "analyst_question" / "management_answer"
    """

    modality = "mp3"

    def _build_indexed_text(self, doc: Any) -> str:
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = list(self._base_text(doc))

        # Speaker identity
        name = (s.get("speaker_name") or s.get("speaker") or
                getattr(doc, "speaker", None) or "").strip()
        role = (s.get("speaker_role") or "").strip()
        if name:
            parts.append(f"speaker {name}")
        if role:
            parts.append(f"speaker role {role}")
        if name and role:
            parts.append(f"speaker {name} {role}")

        # Timestamp token — convert to minute for natural "at 28 minutes" queries
        ts_s = s.get("timestamp_start") or s.get("start_timestamp") or \
               getattr(doc, "timestamp_start", None)
        if ts_s is not None:
            try:
                minute = int(float(ts_s) / 60)
                parts.append(f"time {minute}")
                parts.append(f"minute {minute}")
            except (ValueError, TypeError):
                pass

        # Call section
        call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
        if call_section:
            parts.append(call_section.replace("_", " "))

        # Q&A role
        if s.get("is_question"):
            parts.append("analyst question analyst query")
        elif s.get("is_answer"):
            parts.append("management answer management response")

        # Finance entity amplification
        fin_entities: dict = s.get("finance_entities") or {}
        if isinstance(fin_entities, dict):
            for key, vals in fin_entities.items():
                if isinstance(vals, list):
                    for v in vals[:4]:
                        parts.append(str(v))

        return " ".join(parts)
