"""video_bm25.py — BM25 index for video/webcast chunks."""
from __future__ import annotations

from typing import Any, List

from app.bm25.base_bm25 import BaseBM25


class VideoBM25(BaseBM25):
    """BM25 index for video chunks (investor day presentations, webcasts, demos).

    Enrichment over base:
    - All audio tokens (speaker, timestamp, call_section, entities) — same as AudioBM25
    - slide_bullets amplified ×3 — slide titles are the primary retrieval anchor for
      video queries like "what did slide 23 show about revenue"
    - frame_index token so "at minute 62 the chart showed" queries land correctly
    - combined_text (transcript + visual captions) included if available
    """

    modality = "mp4"

    def _build_indexed_text(self, doc: Any) -> str:
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = list(self._base_text(doc))

        # ── Audio track tokens (same enrichment as AudioBM25) ─────────────────
        name = (s.get("speaker_name") or s.get("speaker") or
                getattr(doc, "speaker", None) or "").strip()
        role = (s.get("speaker_role") or "").strip()
        if name:
            parts.append(f"speaker {name}")
        if role:
            parts.append(f"speaker role {role}")
        if name and role:
            parts.append(f"speaker {name} {role}")

        ts_s = s.get("timestamp_start") or s.get("start_timestamp") or \
               getattr(doc, "timestamp_start", None)
        if ts_s is not None:
            try:
                minute = int(float(ts_s) / 60)
                parts.append(f"time {minute}")
                parts.append(f"minute {minute}")
            except (ValueError, TypeError):
                pass

        call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
        if call_section:
            parts.append(call_section.replace("_", " "))

        if s.get("is_question"):
            parts.append("analyst question analyst query")
        elif s.get("is_answer"):
            parts.append("management answer management response")

        fin_entities: dict = s.get("finance_entities") or {}
        if isinstance(fin_entities, dict):
            for vals in fin_entities.values():
                if isinstance(vals, list):
                    for v in vals[:4]:
                        parts.append(str(v))

        # ── Visual track tokens ───────────────────────────────────────────────

        # Slide bullets amplified ×3 — slide content is the dominant retrieval signal
        slide_bullets: List[str] = s.get("slide_bullets") or []
        if slide_bullets:
            bullet_text = " ".join(str(b) for b in slide_bullets[:10])
            parts.append(bullet_text)
            parts.append(bullet_text)
            parts.append(bullet_text)  # ×3

        # Frame index token
        frame_index = s.get("frame_index") or getattr(doc, "frame_index", None)
        if frame_index is not None:
            parts.append(f"frame {frame_index}")

        # Frame captions from the visual track
        frame_captions: List[Any] = s.get("frame_captions") or []
        for fc in frame_captions[:5]:
            if isinstance(fc, dict):
                cap = (fc.get("caption") or "").strip()
                ocr = (fc.get("ocr_text") or "").strip()
                if cap:
                    parts.append(cap[:200])
                if ocr:
                    parts.append(ocr[:100])
            elif isinstance(fc, str):
                parts.append(fc[:200])

        return " ".join(parts)
