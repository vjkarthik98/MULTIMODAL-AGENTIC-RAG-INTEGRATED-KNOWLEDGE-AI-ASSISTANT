"""audio_bm25.py — BM25 index for audio/earnings call transcript chunks."""
from __future__ import annotations

from typing import Any, List

from prometheus_client import Counter

from app.bm25.base_bm25 import BaseBM25
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BM25_INDEXED = Counter(
    "magik_audio_bm25_indexed_total",
    "Documents indexed in audio BM25",
)
_BM25_INDEX_ERRORS = Counter(
    "magik_audio_bm25_index_errors_total",
    "Errors in audio BM25 _build_indexed_text",
)


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
        try:
            s = getattr(doc, "structure", {}) or {}
            parts: List[str] = list(self._base_text(doc))

            # Speaker identity — name, role, composite "speaker CFO john smith"
            name = (s.get("speaker_name") or s.get("speaker") or
                    getattr(doc, "speaker", None) or "").strip()
            role = (s.get("speaker_role") or "").strip()
            if name:
                parts.append(f"speaker {name}")
            if role:
                parts.append(f"speaker role {role}")
            if name and role:
                parts.append(f"speaker {name} {role}")

            # Timestamps — three token forms (spec Phase 3.6):
            #   "timestamp_1842"  → "find audio near 1842 seconds"
            #   "time 30"         → "at 30 minutes"
            #   "at_30_42"        → MM_SS format for "at 30:42"
            ts_s = s.get("start_timestamp") or s.get("timestamp_start") or \
                   getattr(doc, "timestamp_start", None)
            if ts_s is not None:
                try:
                    total_sec = int(float(ts_s))
                    minute    = total_sec // 60
                    second    = total_sec % 60
                    parts.append(f"timestamp_{total_sec}")
                    parts.append(f"time {minute}")
                    parts.append(f"minute {minute}")
                    parts.append(f"at_{minute:02d}_{second:02d}")
                except (ValueError, TypeError):
                    pass

            # Call section + composite "prepared remarks CFO" / "qa session analyst"
            call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
            if call_section:
                parts.append(call_section.replace("_", " "))
                if role:
                    parts.append(f"{call_section.replace('_', ' ')} {role.lower()}")

            # Earnings call detection flag
            if s.get("is_earnings_call", False):
                parts.append("earnings call conference call quarterly results")

            # Q&A role tokens
            if s.get("is_question"):
                parts.append("analyst question analyst query")
            elif s.get("is_answer"):
                parts.append("management answer management response")

            # Topic section (distinct from call_section)
            topic_section = (s.get("topic_section") or "").strip()
            if topic_section and topic_section != call_section:
                parts.append(topic_section.replace("_", " "))

            # Finance entity amplification
            fin_entities: dict = s.get("finance_entities") or {}
            if isinstance(fin_entities, dict):
                for key, vals in fin_entities.items():
                    if isinstance(vals, list):
                        for v in vals[:4]:
                            parts.append(str(v))

            _BM25_INDEXED.inc()
            return " ".join(parts)
        except Exception as _exc:
            _BM25_INDEX_ERRORS.inc()
            logger.warning(
                event="bm25_index_text_failed",
                modality=self.modality,
                error=str(_exc),
            )
            return getattr(doc, "text", "") or ""

    def health_check(self, user_id=None) -> dict:
        base = super().health_check(user_id)
        base["class"] = self.__class__.__name__
        return base
