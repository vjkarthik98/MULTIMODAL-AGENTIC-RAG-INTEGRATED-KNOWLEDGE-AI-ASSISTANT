"""video_chunker.py

Finance-grade Video chunker — audio pipeline + frame captioning.

Frame-extraction code (FrameMetadata, extract_frames, PySceneDetect/OpenCV
path) lives in app/ingestion/video_ingest.py because frame extraction is
called from within the ingestor before RawExtract objects exist.
"""
from __future__ import annotations

import bisect
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess

from PIL import Image as PILImage

from app.chunking.audio_chunker import (
    _assemble_chunks,
    _map_speaker_roles,
    _run_whisper,
)
from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import deterministic_chunk_id, extract_finance_entities
from app.core.config import settings
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

import time
from prometheus_client import Counter, Histogram

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_video_chunks_total",
    "Total chunks produced by video chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_video_chunk_errors_total",
    "Total errors in video chunker",
)

_FRAME_WINDOW_S = 5.0          # attach frames within ±5s of audio chunk
_FINANCIAL_TRIGGER_RE = re.compile(r"[$%]|\bbillion\b|\brevenue\b|\bearnings\b", re.I)

# LLaVA-1.5-7B INT8 = ~3.5 GB; semaphore limits concurrent instances to keep
# total VRAM within A10G 24 GB budget alongside other resident models.
_LLAVA_SEMAPHORE = threading.Semaphore(settings.VIDEO_CAPTION_CONCURRENCY)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO EXTRACTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _extract_audio(video_path: str, wav_path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
            capture_output=True, timeout=600,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.warning(event="video_ffmpeg_audio_failed", error=str(exc))
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS  (merged from app/models/llava_captioner.py)
# ══════════════════════════════════════════════════════════════════════════════

_VIDEO_FRAME_PROMPT = (
    "USER: <image>\n"
    "Analyze this financial presentation frame. Report verbatim:\n"
    "1) Slide or chart title\n"
    "2) All bullet points\n"
    "3) Every number visible with its label\n"
    "4) Chart type and all axis labels\n"
    "5) Speaker name and title if shown in lower-third\n"
    "6) Any table headers and cell values\n"
    "7) Slide number if visible\n"
    "Be extremely precise about numbers — do not round or paraphrase.\n"
    "ASSISTANT:"
)


def caption_frame(image: "PILImage.Image", prompt: Optional[str] = None) -> str:
    """Caption a single video frame using LLaVA-1.5-7b. Returns '' on failure."""
    try:
        from app.core.model_loader import model_loader
        processor, model, device = model_loader.get_llava()
    except Exception as exc:
        logger.warning(event="llava_unavailable", error=str(exc))
        return ""
    try:
        import torch as _torch
        text   = prompt or _VIDEO_FRAME_PROMPT
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
        with _torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=settings.LLAVA_MAX_TOKENS)
        decoded = processor.decode(out[0], skip_special_tokens=True).strip()
        if "ASSISTANT:" in decoded:
            decoded = decoded.split("ASSISTANT:", 1)[-1].strip()
        return decoded
    except Exception as exc:
        logger.warning(event="llava_caption_failed", error=str(exc))
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# FRAME CAPTIONING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _caption_and_ocr_frame(frame_dict: Dict) -> Dict:
    """Run LLaVA + TrOCR on a frame given its FrameMetadata dict."""
    result = {
        "frame_timestamp": frame_dict["timestamp_start"],
        "scene_change":    frame_dict["is_scene_boundary"],
        "frame_caption":   "",
        "ocr_text":        "",
        "slide_number":    None,
    }
    frame_path = frame_dict.get("path", "")
    if not frame_path or not Path(frame_path).exists():
        return result

    try:
        from PIL import Image as _PIL
        img = _PIL.open(frame_path).convert("RGB")
    except Exception:
        return result

    try:
        result["frame_caption"] = caption_frame(img)
    except Exception as exc:
        logger.warning(event="llava_caption_failed", error=str(exc))

    try:
        from app.chunking.image_chunker import ocr as _ocr
        result["ocr_text"] = _ocr(img)
    except Exception as exc:
        logger.warning(event="trocr_frame_failed", error=str(exc))

    ocr_text = result["ocr_text"]
    m = re.search(r"slide\s*(\d+)", ocr_text, re.I)
    result["slide_number"] = int(m.group(1)) if m else None
    return result


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

class VideoChunker(BaseChunker):
    """Finance-grade chunker for video files (investor days, earnings webcasts, CNBC segments).

    Pipeline:
      ffmpeg audio → Whisper → pyannote diarization → audio chunks
      PySceneDetect/OpenCV frame extraction → LLaVA+TrOCR per frame
      audio-visual sync alignment (±5 s window) → IngestedDocuments.
    """

    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        source  = Path(meta.source_path).name or "unknown.mp4"
        surface = "video_chunker"
        modality_var.set("video")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="video", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="video", source=source)
            return []
        try:
            docs: List[IngestedDocument] = []

            for ext in extracts:
                if ext.extract_type != "video_raw":
                    continue

                video_path = ext.extra.get("file_path", "") or meta.source_path
                if not video_path or not Path(video_path).exists():
                    logger.warning(event="video_chunker_missing_file", source=source)
                    continue

                with tempfile.TemporaryDirectory() as tmpdir:
                    wav_path = os.path.join(tmpdir, "audio.wav")

                    # 1. Extract audio track.
                    if not _extract_audio(video_path, wav_path):
                        logger.warning(event="video_no_audio", source=source)
                        continue

                    # 2. Whisper transcription.
                    words = _run_whisper(wav_path)

                    # 3. Diarization.
                    diarization: List[Tuple[float, float, str]] = []
                    try:
                        from app.chunking.audio_chunker import diarize as _diarize
                        diarization = _diarize(wav_path)
                    except Exception:
                        pass

                    full_transcript = " ".join(w["word"] for w in words)
                    role_map        = _map_speaker_roles(diarization, full_transcript)
                    audio_chunks    = _assemble_chunks(words, diarization, role_map)

                    # 4. Frame extraction (import from video_ingest where it lives).
                    try:
                        from app.ingestion.video_ingest import extract_frames
                        frame_dicts = extract_frames(
                            video_path=video_path,
                            interval_sec=settings.VIDEO_FRAME_INTERVAL_SEC,
                            session_id=meta.custom_fields.get("session_id", "internal"),
                        )
                    except Exception as exc:
                        logger.warning(event="frame_extraction_error", error=str(exc))
                        frame_dicts = []

                    # 5. Caption + OCR each frame — concurrent, VRAM-bounded.
                    def _caption_safe(fd: Dict) -> Tuple[float, Dict]:
                        with _LLAVA_SEMAPHORE:
                            return fd["timestamp_start"], _caption_and_ocr_frame(fd)

                    captioned_frames: Dict[float, Dict] = {}
                    with ThreadPoolExecutor(
                        max_workers=settings.VIDEO_CAPTION_CONCURRENCY
                    ) as _caption_pool:
                        for ts, result in _caption_pool.map(_caption_safe, frame_dicts):
                            captioned_frames[ts] = result

                    # 6. Build IngestedDocuments (one per audio chunk).
                    # Pre-sort frame timestamps once for O(log f) bisect window
                    # lookup instead of O(f) linear dict scan per chunk.
                    _sorted_ts: List[float] = sorted(captioned_frames)

                    for chunk_idx, ch in enumerate(audio_chunks):
                        transcript = ch.get("transcript", "")
                        if not transcript.strip():
                            continue

                        t_start = ch["start"]
                        t_end   = ch["end"]

                        _lo = bisect.bisect_left (_sorted_ts, t_start - _FRAME_WINDOW_S)
                        _hi = bisect.bisect_right(_sorted_ts, t_end   + _FRAME_WINDOW_S)
                        chunk_frames = [captioned_frames[ts] for ts in _sorted_ts[_lo:_hi]]

                        visual_ctx = ""
                        slide_bullets: List[str] = []
                        for cf in chunk_frames:
                            if cf.get("frame_caption"):
                                visual_ctx += f"\n[VISUAL AT {cf['frame_timestamp']:.1f}s]: {cf['frame_caption']}"
                            if cf.get("ocr_text"):
                                visual_ctx += f"\n[ON-SCREEN]: {cf['ocr_text']}"
                            if cf.get("slide_number") is not None:
                                slide_bullets.append(f"Slide {cf['slide_number']}")

                        combined_text = transcript
                        if visual_ctx:
                            combined_text += visual_ctx

                        fin_entities = extract_finance_entities(combined_text)
                        has_finance  = bool(_FINANCIAL_TRIGGER_RE.search(transcript))
                        chunk_hash   = deterministic_chunk_id(source, f"v_{t_start:.1f}", chunk_idx)

                        # Extract slide numbers from slide_bullets like ["Slide 3", "Slide 4"] (MD 1.7)
                        slide_numbers_covered: List[int] = []
                        for sb in slide_bullets:
                            m = re.search(r"\bslide\s*(\d+)\b", sb, re.IGNORECASE)
                            if m:
                                slide_numbers_covered.append(int(m.group(1)))

                        structure = {
                            "chunk_hash_id":        chunk_hash,
                            "source_file":          source,
                            "chunk_index":          chunk_idx,
                            "start_timestamp":      round(t_start, 3),
                            "end_timestamp":        round(t_end, 3),
                            "duration_seconds":     round(t_end - t_start, 3),
                            "speaker_label":        ch.get("speaker_label"),
                            "speaker_name":         ch.get("speaker_name"),
                            "speaker_role":         ch.get("speaker_role"),
                            "topic_section":        ch.get("topic_section"),
                            "call_section":         ch.get("call_section"),
                            "transcript":           transcript,
                            "frame_captions":       chunk_frames,
                            "combined_text":        combined_text,
                            "slide_bullets":        slide_bullets,
                            "has_slide_content":    bool(slide_bullets),
                            "slide_numbers_covered": slide_numbers_covered,
                            "finance_entities":     fin_entities,
                            "has_finance_signal":   has_finance,
                            "is_question":          ch.get("is_question", False),
                            "is_answer":            ch.get("is_answer", False),
                        }

                        doc = self._make_doc(
                            text=combined_text,
                            modality="mp4",
                            subtype="transcript_frame",
                            source=source,
                            page=None,
                            chunk_idx=chunk_idx,
                            structure=structure,
                            meta=meta,
                            surface=surface,
                        )
                        if doc:
                            docs.append(doc)

            logger.info(event="video_chunking_done", source=source, chunks=len(docs))
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="video", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "video",
            "status": "ok",
            "class": self.__class__.__name__,
        }
