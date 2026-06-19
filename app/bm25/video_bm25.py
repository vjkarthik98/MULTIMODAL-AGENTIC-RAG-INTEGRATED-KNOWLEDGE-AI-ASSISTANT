"""video_bm25.py — BM25 index for video/webcast chunks.

Implements the main combined index plus three sub-indexes required by the
MAGIK spec (Phase 3.7):
  - bm25_mp4_audio.pkl    → transcript only
  - bm25_mp4_visual.pkl   → frame captions + OCR only
  - bm25_mp4_combined.pkl → all combined (the main index file)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional

from prometheus_client import Counter
from rank_bm25 import BM25Plus

from app.bm25.base_bm25 import BaseBM25, _INDEX_VERSION
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BM25_INDEXED = Counter(
    "magik_video_bm25_indexed_total",
    "Documents indexed in video BM25",
)
_BM25_INDEX_ERRORS = Counter(
    "magik_video_bm25_index_errors_total",
    "Errors in video BM25 _build_indexed_text",
)


class VideoBM25(BaseBM25):
    """BM25 index for video chunks (investor day presentations, webcasts, demos).

    Enrichment over base:
    - All audio tokens (speaker, timestamp, call_section, entities) — same as AudioBM25
    - slide_bullets amplified ×3 — slide titles are the primary retrieval anchor for
      video queries like "what did slide 23 show about revenue"
    - frame_index token so "at minute 62 the chart showed" queries land correctly
    - combined_text (transcript + visual captions) included if available

    Sub-indexes (Phase 3.7 requirement):
    - _bm25_audio: transcript-only index for "what did X say" queries
    - _bm25_visual: frame-caption+OCR index for "what was shown on screen" queries
    - _bm25_combined: same as main index (combined)
    """

    modality = "mp4"

    def __init__(self, user_id: Optional[str] = None) -> None:
        super().__init__(user_id=user_id)
        self._sub_audio_corpus: List[List[str]] = []
        self._sub_visual_corpus: List[List[str]] = []
        self._sub_audio_docs: List[Any] = []
        self._sub_visual_docs: List[Any] = []
        self._bm25_audio: Optional[BM25Plus] = None
        self._bm25_visual: Optional[BM25Plus] = None

    def _build_indexed_text(self, doc: Any) -> str:
        try:
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

            call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
            if call_section:
                parts.append(call_section.replace("_", " "))
                # Composite call_section + role phrase (mirrors AudioBM25)
                if role:
                    parts.append(f"{call_section.replace('_', ' ')} {role.lower()}")

            if s.get("is_question"):
                parts.append("analyst question analyst query")
            elif s.get("is_answer"):
                parts.append("management answer management response")

            # Earnings call detection token
            if s.get("is_earnings_call", False):
                parts.append("earnings call conference call quarterly results")

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

            # Frame captions — scene change tokens, frame_caption key, numeric repeat
            frame_captions: List[Any] = s.get("frame_captions") or []
            for fc in frame_captions[:5]:
                if isinstance(fc, dict):
                    cap = (fc.get("frame_caption") or "").strip()
                    ocr = (fc.get("ocr_text") or "").strip()
                    if fc.get("scene_change"):
                        parts.append("new_slide")
                    if cap:
                        parts.append(cap[:200])
                        cap_lower = cap.lower()
                        if any(kw in cap_lower for kw in ("chart", "graph", "bar chart", "line chart", "pie")):
                            parts.append("chart_appears")
                        if any(kw in cap_lower for kw in ("table", " row", " column", "spreadsheet")):
                            parts.append("table_shown")
                    if ocr:
                        parts.append(ocr[:100])
                elif isinstance(fc, str):
                    parts.append(fc[:200])

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

    # ── Audio-only indexed text ───────────────────────────────────────────────

    def _build_audio_text(self, doc: Any) -> str:
        """Transcript-only text for the audio sub-index."""
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = []

        # Transcript / base text
        text = getattr(doc, "text", "") or ""
        if text:
            parts.append(text[:2000])

        name = (s.get("speaker_name") or s.get("speaker") or "").strip()
        role = (s.get("speaker_role") or "").strip()
        if name:
            parts.append(f"speaker {name}")
        if role:
            parts.append(f"speaker role {role}")

        ts_s = s.get("start_timestamp") or s.get("timestamp_start")
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

        call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
        if call_section:
            parts.append(call_section.replace("_", " "))

        if s.get("is_earnings_call", False):
            parts.append("earnings call conference call quarterly results")

        fin_entities: dict = s.get("finance_entities") or {}
        if isinstance(fin_entities, dict):
            for vals in fin_entities.values():
                if isinstance(vals, list):
                    for v in vals[:3]:
                        parts.append(str(v))

        return " ".join(parts)

    # ── Visual-only indexed text ──────────────────────────────────────────────

    def _build_visual_text(self, doc: Any) -> str:
        """Frame caption + OCR text for the visual sub-index."""
        s = getattr(doc, "structure", {}) or {}
        parts: List[str] = []

        slide_bullets: List[str] = s.get("slide_bullets") or []
        if slide_bullets:
            bullet_text = " ".join(str(b) for b in slide_bullets[:10])
            # ×2 amplification for visual content
            parts.append(bullet_text)
            parts.append(bullet_text)

        frame_captions: List[Any] = s.get("frame_captions") or []
        for fc in frame_captions[:8]:
            if isinstance(fc, dict):
                cap = (fc.get("frame_caption") or "").strip()
                ocr = (fc.get("ocr_text") or "").strip()
                slide_num = fc.get("slide_number")
                if fc.get("scene_change"):
                    parts.append("new_slide")
                if slide_num is not None:
                    parts.append(f"slide {slide_num}")
                if cap:
                    parts.append(cap[:300])
                    cap_lower = cap.lower()
                    if any(kw in cap_lower for kw in ("chart", "graph", "bar chart", "line chart", "pie")):
                        parts.append("chart_appears")
                    if any(kw in cap_lower for kw in ("table", " row", " column", "spreadsheet")):
                        parts.append("table_shown")
                if ocr:
                    # OCR numbers repeated for weight
                    parts.append(ocr[:200])
                    parts.append(ocr[:200])
            elif isinstance(fc, str):
                parts.append(fc[:300])

        frame_index = s.get("frame_index")
        if frame_index is not None:
            parts.append(f"frame {frame_index}")

        return " ".join(parts)

    # ── Sub-index path helpers ────────────────────────────────────────────────

    def _sub_path(self, tag: str, user_id: Optional[str] = None) -> Path:
        from app.utils.paths import user_dir
        uid = user_id or self._loaded_user_id or "default"
        p = user_dir(uid) / "bm25_index"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"mp4_{tag}.pkl"

    # ── Override add_documents to also populate sub-indexes ──────────────────

    def add_documents(
        self,
        documents: List[Any],
        session_id: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        # Main (combined) index via parent
        super().add_documents(documents, session_id=session_id, user_id=user_id)

        for doc in documents:
            audio_tokens = self.tokenize(self._build_audio_text(doc))
            visual_tokens = self.tokenize(self._build_visual_text(doc))

            if audio_tokens:
                self._sub_audio_corpus.append(audio_tokens)
                self._sub_audio_docs.append(doc)
            if visual_tokens:
                self._sub_visual_corpus.append(visual_tokens)
                self._sub_visual_docs.append(doc)

        self._rebuild_sub_indexes(user_id)

    def _rebuild_sub_indexes(self, user_id: Optional[str] = None) -> None:
        for tag, corpus, docs, attr_bm25 in [
            ("audio",  self._sub_audio_corpus,  self._sub_audio_docs,  "_bm25_audio"),
            ("visual", self._sub_visual_corpus, self._sub_visual_docs, "_bm25_visual"),
        ]:
            if not corpus:
                continue
            bm25 = BM25Plus(corpus, k1=1.5, b=0.75)
            setattr(self, attr_bm25, bm25)
            path = self._sub_path(tag, user_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump({
                    "index_version": _INDEX_VERSION,
                    "documents": docs,
                    "tokenized_corpus": corpus,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            logger.info(event="bm25_video_sub_saved", tag=tag, docs=len(corpus))

    def search_sub(
        self,
        query: str,
        sub_index: str,
        top_k: int = 10,
    ) -> List[Any]:
        """Search a video sub-index.

        sub_index: 'audio' | 'visual' | 'combined'
        Returns list of (doc, score) pairs sorted descending.
        Falls back to main search if sub-index empty.
        """
        if sub_index == "combined":
            return self.search(query, top_k=top_k)

        bm25 = getattr(self, f"_bm25_{sub_index}", None)
        docs = getattr(self, f"_sub_{sub_index}_docs", [])
        if bm25 is None or not docs:
            return self.search(query, top_k=top_k)

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = bm25.get_scores(tokens)
        import numpy as _np
        ranked = _np.argsort(scores)[::-1][:top_k]
        return [(docs[i], float(scores[i])) for i in ranked if float(scores[i]) > 0]
