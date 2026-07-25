"""video_embedder.py — Finance-grade embedder for video/webcast chunks."""

from __future__ import annotations

import re
from typing import Any

from prometheus_client import Counter

from app.core.config import settings
from app.embeddings.base_embedder import (
    BaseEmbedder,
    normalize_finance_numbers,
    sanitize_text,
    valid_embedding,
)
from app.utils.logger import get_logger

_HAS_DIGITS_RE = re.compile(r"\d")

logger = get_logger(__name__)

_EMBED_BUILT = Counter(
    "magik_video_embed_text_built_total",
    "Embed texts successfully built for video",
)
_EMBED_ERRORS = Counter(
    "magik_video_embed_text_errors_total",
    "Errors building embed text for video",
)


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
        try:
            result = self._combined_text(doc, cleaned_text)
            logger.debug(event="embed_text_built", modality="video", chars=len(result))
            _EMBED_BUILT.inc()
            return result
        except Exception as _exc:
            _EMBED_ERRORS.inc()
            logger.error(event="embed_text_build_failed", modality="video", error=str(_exc))
            return cleaned_text  # safe fallback to unenriched text

    def _combined_text(self, doc: Any, cleaned_text: str) -> str:
        s = getattr(doc, "structure", {}) or {}

        combined = (s.get("combined_text") or "").strip()
        base_text = combined if combined and len(combined) > len(cleaned_text) else cleaned_text
        if combined:
            base_text = sanitize_text(combined) or cleaned_text
            base_text = normalize_finance_numbers(base_text)

        header = self._speaker_header(s)

        # Build [SLIDE N AT ts]: caption blocks from frame_captions
        slide_parts: list[str] = []
        for fc in s.get("frame_captions") or []:
            if not isinstance(fc, dict):
                continue
            caption = (fc.get("frame_caption") or "").strip()
            ocr_text = (fc.get("ocr_text") or "").strip()
            ts = fc.get("frame_timestamp")
            slide_num = fc.get("slide_number")
            if slide_num is not None and ts is not None:
                tag = f"[SLIDE {slide_num} AT {float(ts):.1f}s]"
            elif ts is not None:
                tag = f"[{float(ts):.3f}s]"
            else:
                tag = ""
            if caption:
                line = f"{tag}: {caption}" if tag else caption
                slide_parts.append(line)
                # Numeric doubling — spec: repeat caption TWICE if it contains numbers
                if _HAS_DIGITS_RE.search(caption):
                    slide_parts.append(line)
            if ocr_text:
                slide_parts.append(f"[ON-SCREEN TEXT]: {ocr_text}")

        # Slide bullets amplified ×3 (primary slide retrieval anchor)
        slide_bullets: list[str] = s.get("slide_bullets") or []
        bullet_block = ""
        if slide_bullets:
            bullet_text = " ".join(str(b) for b in slide_bullets[:10])
            bullet_block = f" {bullet_text} {bullet_text} {bullet_text}"

        slide_block = (" " + " ".join(slide_parts)) if slide_parts else ""
        entity_suffix = self._entity_suffix(s)
        result = f"{header}{base_text}{slide_block}{bullet_block}{entity_suffix}"
        return result[: settings.MAX_PROMPT_CHARS]

    def _audio_only_text(self, doc: Any) -> str:
        """Audio-only embedding: transcript + speaker/timestamp header."""
        s = getattr(doc, "structure", {}) or {}
        transcript = (s.get("transcript") or "").strip()
        if not transcript:
            return ""
        base_text = sanitize_text(transcript) or transcript
        base_text = normalize_finance_numbers(base_text)
        header = self._speaker_header(s)
        entity_suffix = self._entity_suffix(s)
        result = f"{header}{base_text}{entity_suffix}"
        return result[: settings.MAX_PROMPT_CHARS]

    def _visual_only_text(self, doc: Any) -> str:
        """Visual-only embedding: frame captions + slide bullets + on-screen OCR."""
        s = getattr(doc, "structure", {}) or {}
        frame_parts: list[str] = []
        for fc in s.get("frame_captions") or []:
            if isinstance(fc, dict):
                caption = (fc.get("frame_caption") or "").strip()
                ocr_text = (fc.get("ocr_text") or "").strip()
                ts = fc.get("frame_timestamp")
                slide_num = fc.get("slide_number")
                if slide_num is not None and ts is not None:
                    tag = f"[SLIDE {slide_num} AT {float(ts):.1f}s]"
                elif ts is not None:
                    tag = f"[{float(ts):.3f}s]"
                else:
                    tag = ""
                if caption:
                    line = f"{tag}: {caption}" if tag else caption
                    frame_parts.append(line)
                    if _HAS_DIGITS_RE.search(caption):
                        frame_parts.append(line)
                if ocr_text:
                    frame_parts.append(f"[ON-SCREEN TEXT]: {ocr_text}")
        slide_bullets: list[str] = s.get("slide_bullets") or []
        if slide_bullets:
            bullet_text = " ".join(str(b) for b in slide_bullets[:10])
            frame_parts.extend([bullet_text] * 3)  # amplify ×3
        visual_text = " ".join(frame_parts).strip()
        if not visual_text:
            return ""
        visual_text = normalize_finance_numbers(visual_text)
        return visual_text[: settings.MAX_PROMPT_CHARS]

    @staticmethod
    def _speaker_header(s: dict) -> str:
        # [VIDEO] prefix per Phase 2.7 spec format
        parts: list[str] = ["[VIDEO]"]
        name = (s.get("speaker_name") or s.get("speaker") or "").strip()
        role = (s.get("speaker_role") or "").strip()
        if name and role:
            parts.append(f"[SPEAKER: {name} - {role}]")
        elif name:
            parts.append(f"[SPEAKER: {name}]")
        elif role:
            parts.append(f"[SPEAKER: {role}]")
        # Millisecond-precision timestamps per spec
        ts_s = s.get("start_timestamp") or s.get("timestamp_start")
        ts_e = s.get("end_timestamp") or s.get("timestamp_end")
        if ts_s is not None and ts_e is not None:
            parts.append(f"[{float(ts_s):.3f}s-{float(ts_e):.3f}s]")
        elif ts_s is not None:
            parts.append(f"[{float(ts_s):.3f}s]")
        call_section = (s.get("call_section") or s.get("topic_section") or "").strip()
        if call_section:
            parts.append(f"[{call_section.upper().replace('_', ' ')}]")
        return " ".join(parts) + " "

    @staticmethod
    def _entity_suffix(s: dict) -> str:
        fin_entities: dict = s.get("finance_entities") or {}
        entity_tokens: list[str] = []
        if isinstance(fin_entities, dict):
            for v in fin_entities.values():
                if isinstance(v, list):
                    entity_tokens.extend(str(x) for x in v[:4])
        return (f" [ENTITIES: {', '.join(entity_tokens)}]") if entity_tokens else ""

    def embed_documents(self, docs: list[Any], session_id: str = "default") -> list[Any]:
        """Embed video docs with THREE named vectors (Phase 2.7) + SigLIP for frames.

        transcript_frame docs (embedding_space="text"):
          doc.embedding        = combined BGE 1024-dim → text_collection
          doc.embedding_audio  = audio-only BGE 1024-dim (named vector)
          doc.embedding_visual = visual-only BGE 1024-dim (named vector)

        frame docs (embedding_space="vision", BUG-2 fix):
          doc.embedding        = SigLIP 1152-dim from frame image → vision_collection
        """
        from pathlib import Path as _Path

        # Split by embedding_space before BGE encoding to avoid dimension mismatch.
        vision_docs = [
            d
            for d in docs
            if (getattr(d, "structure", None) or {}).get("embedding_space") == "vision"
        ]
        text_docs = [d for d in docs if d not in vision_docs]

        # ── Text / transcript docs: existing BGE combined + audio/visual path ──
        text_results: list[Any] = []
        if text_docs:
            text_results = super().embed_documents(text_docs, session_id=session_id)

        if text_results:
            embedder = self._get_model()
            for vector_name, text_fn, attr in [
                ("audio", self._audio_only_text, "embedding_audio"),
                ("visual", self._visual_only_text, "embedding_visual"),
            ]:
                texts: list[str] = []
                rdocs: list[Any] = []
                for doc in text_results:
                    t = text_fn(doc)
                    if t:
                        texts.append(t)
                        rdocs.append(doc)
                if not texts:
                    continue
                for i in range(0, len(texts), embedder.batch_size):
                    batch_texts = texts[i : i + embedder.batch_size]
                    batch_docs = rdocs[i : i + embedder.batch_size]
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

        # ── Vision / frame docs: SigLIP 1152-dim from frame image file ──────
        vision_results: list[Any] = list(vision_docs)  # return even if embed fails
        if vision_docs:
            vision_results = self._embed_vision_frames(vision_docs, session_id, _Path)

        return text_results + vision_results

    def _embed_vision_frames(self, docs: list[Any], session_id: str, _Path: Any) -> list[Any]:
        """Embed video frame images with SigLIP → vision_collection (1152-dim).

        Docs without a valid asset_path or where SigLIP is unavailable are
        returned as-is (doc.embedding remains None and qdrant_store skips them).
        """
        try:
            from app.core.model_loader import model_loader

            # get_siglip() returns (processor, model, device)
            siglip_processor, siglip_model, siglip_device = model_loader.get_siglip()
            from app.embeddings.image_embedder import ImageEmbedder

            ie = ImageEmbedder(
                model=siglip_model,
                processor=siglip_processor,
                device=siglip_device,
            )
        except Exception as exc:
            logger.warning(event="siglip_unavailable_for_video_frames", error=str(exc))
            return docs

        for doc in docs:
            s = getattr(doc, "structure", {}) or {}
            asset_path = s.get("asset_path", "")
            if not asset_path or not _Path(asset_path).exists():
                continue
            try:
                emb = ie.embed(str(asset_path), session_id=session_id)
                doc.embedding = emb
                struct = dict(s)
                struct["has_visual_embedding"] = True
                doc.structure = struct
            except Exception as exc:
                logger.warning(
                    event="siglip_frame_embed_failed",
                    path=asset_path,
                    error=str(exc),
                )

        return docs

    def health_check(self) -> dict:
        return {
            "modality": "video",
            "status": "ok",
            "class": self.__class__.__name__,
        }
