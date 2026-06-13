"""video_embedder.py — Finance-grade embedder for video/webcast chunks."""
from __future__ import annotations

from typing import Any, List

from app.embeddings.base_embedder import (
    BaseEmbedder, sanitize_text, normalize_finance_numbers, valid_embedding,
)
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoEmbedder(BaseEmbedder):
    """Embedder for video chunks (investor day webcasts, earnings presentations).

    Phase 2.7 THREE named vectors per chunk (MAGIK spec):
      doc.embedding           = combined (transcript + visual) — primary retrieval
      doc.embedding_audio     = audio-only (transcript + speaker header)
      doc.embedding_visual    = visual-only (frame captions + slide bullets + OCR)

    All three are stored in Qdrant as named vectors so retrieval can query
    audio, visual, or combined independently.
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        """Primary (combined) embedding text."""
        return self._combined_text(doc, cleaned_text)

    def _combined_text(self, doc: Any, cleaned_text: str) -> str:
        s = getattr(doc, "structure", {}) or {}

        combined = (s.get("combined_text") or "").strip()
        base_text = combined if combined and len(combined) > len(cleaned_text) else cleaned_text
        if combined:
            base_text = sanitize_text(combined) or cleaned_text
            base_text = normalize_finance_numbers(base_text)

        header = self._speaker_header(s)

        slide_bullets: List[str] = s.get("slide_bullets") or []
        bullet_block = ""
        if slide_bullets:
            bullet_text = " ".join(str(b) for b in slide_bullets[:10])
            bullet_block = f" {bullet_text} {bullet_text} {bullet_text}"

        entity_suffix = self._entity_suffix(s)
        result = f"{header}{base_text}{bullet_block}{entity_suffix}"
        return result[:settings.MAX_PROMPT_CHARS]

    def _audio_only_text(self, doc: Any) -> str:
        """Audio-only embedding: transcript + speaker/timestamp header."""
        s          = getattr(doc, "structure", {}) or {}
        transcript = (s.get("transcript") or "").strip()
        if not transcript:
            return ""
        base_text = sanitize_text(transcript) or transcript
        base_text = normalize_finance_numbers(base_text)
        header    = self._speaker_header(s)
        entity_suffix = self._entity_suffix(s)
        result    = f"{header}{base_text}{entity_suffix}"
        return result[:settings.MAX_PROMPT_CHARS]

    def _visual_only_text(self, doc: Any) -> str:
        """Visual-only embedding: frame captions + slide bullets + on-screen OCR."""
        s            = getattr(doc, "structure", {}) or {}
        frame_parts: List[str] = []
        for fc in (s.get("frame_captions") or []):
            if isinstance(fc, dict):
                caption  = (fc.get("caption")  or "").strip()
                ocr_text = (fc.get("ocr_text") or "").strip()
                ts       = fc.get("timestamp")
                ts_tag   = f"[{float(ts):.0f}s]" if ts is not None else ""
                if caption:
                    frame_parts.append(f"{ts_tag} {caption}")
                if ocr_text:
                    frame_parts.append(f"[ON-SCREEN]: {ocr_text}")
        slide_bullets: List[str] = s.get("slide_bullets") or []
        if slide_bullets:
            bullet_text = " ".join(str(b) for b in slide_bullets[:10])
            frame_parts.extend([bullet_text] * 3)  # amplify ×3
        visual_text = " ".join(frame_parts).strip()
        if not visual_text:
            return ""
        visual_text = normalize_finance_numbers(visual_text)
        return visual_text[:settings.MAX_PROMPT_CHARS]

    @staticmethod
    def _speaker_header(s: dict) -> str:
        parts: List[str] = []
        name = (s.get("speaker_name") or s.get("speaker") or "").strip()
        role = (s.get("speaker_role") or "").strip()
        if name and role:
            parts.append(f"[Speaker: {name} - {role}]")
        elif name:
            parts.append(f"[Speaker: {name}]")
        ts_s = s.get("start_timestamp") or s.get("timestamp_start")
        ts_e = s.get("end_timestamp")   or s.get("timestamp_end")
        if ts_s is not None and ts_e is not None:
            parts.append(f"[{float(ts_s):.0f}s-{float(ts_e):.0f}s]")
        call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
        if call_section:
            parts.append(f"[{call_section}]")
        return (" ".join(parts) + " ") if parts else ""

    @staticmethod
    def _entity_suffix(s: dict) -> str:
        fin_entities: dict = s.get("finance_entities") or {}
        entity_tokens: List[str] = []
        if isinstance(fin_entities, dict):
            for v in fin_entities.values():
                if isinstance(v, list):
                    entity_tokens.extend(str(x) for x in v[:4])
        return (f" [ENTITIES: {', '.join(entity_tokens)}]") if entity_tokens else ""

    def embed_documents(self, docs: List[Any], session_id: str = "default") -> List[Any]:
        """Embed video docs with THREE named vectors (Phase 2.7).

        1. Primary (combined) via parent.embed_documents — sets doc.embedding
        2. Audio-only — sets doc.embedding_audio
        3. Visual-only — sets doc.embedding_visual
        """
        results = super().embed_documents(docs, session_id=session_id)
        if not results:
            return results

        embedder = self._get_model()

        for vector_name, text_fn, attr in [
            ("audio",   self._audio_only_text,   "embedding_audio"),
            ("visual",  self._visual_only_text,   "embedding_visual"),
        ]:
            texts: List[str] = []
            rdocs: List[Any] = []
            for doc in results:
                t = text_fn(doc)
                if t:
                    texts.append(t)
                    rdocs.append(doc)
            if not texts:
                continue
            for i in range(0, len(texts), embedder.batch_size):
                batch_texts = texts[i:i + embedder.batch_size]
                batch_docs  = rdocs[i:i + embedder.batch_size]
                try:
                    embs = embedder._encode_with_retry(embedder.model, batch_texts)
                    for doc, emb in zip(batch_docs, embs):
                        if valid_embedding(emb, embedder.expected_dim):
                            setattr(doc, attr, emb)
                            struct = dict(getattr(doc, "structure", {}) or {})
                            struct[f"has_{vector_name}_embedding"] = True
                            doc.structure = struct
                except Exception as exc:
                    logger.debug(
                        event="video_alt_embed_skip",
                        vector=vector_name,
                        error=str(exc),
                    )

        return results
