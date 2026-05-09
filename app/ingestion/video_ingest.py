
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytesseract
from PIL import Image

from app.core.config import settings
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.ingestion.video_frames import extract_frames
from app.utils.logger import get_logger

logger = get_logger(__name__)


# SUPPORTED FORMATS

SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


# HASH

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# FFMPEG PATH

def _resolve_ffmpeg() -> str:
    configured = Path(settings.FFMPEG_PATH)
    if configured.exists():
        return str(configured)

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    raise FileNotFoundError("FFMPEG_NOT_FOUND")


# FFPROBE DURATION

def _probe_duration(file_path: str) -> Optional[float]:
    try:
        ffprobe = shutil.which("ffprobe") or "ffprobe"
        result  = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# FRAME OCR

def _extract_frame_ocr(image_path: str) -> str:
    try:
        img  = Image.open(image_path).convert("RGB")
        text = (pytesseract.image_to_string(img) or "").strip()
        return text if len(text) > 10 else ""
    except Exception:
        return ""


# BLUR SCORE

def _blur_score(image_path: str) -> float:
    try:
        import cv2
        import numpy as np
        img       = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1.0
        laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
        return float(min(laplacian / 100.0, 1.0))
    except Exception:
        return 1.0


# SPEECH ALIGNMENT

def _link_speech(
    timestamp: float,
    speech_segments: List[Dict],
) -> Optional[Dict]:
    for seg in speech_segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg
    return None


# ALIGNMENT SCORE

def _alignment_score(caption: str, speech: Optional[Dict]) -> float:
    if not speech or not caption:
        return 0.0
    speech_text = speech.get("text", "")
    if not speech_text:
        return 0.0
    caption_words = set(caption.lower().split())
    speech_words  = set(speech_text.lower().split())
    if not speech_words:
        return 0.0
    overlap = caption_words & speech_words
    return round(len(overlap) / len(speech_words), 3)


# AUDIO EXTRACTION

def _extract_audio(file_path: str, audio_path: str) -> None:
    ffmpeg = _resolve_ffmpeg()
    cmd    = [
        ffmpeg, "-y",
        "-i", file_path,
        "-vn",
        "-ar", str(settings.AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-f", "wav",
        audio_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.FFMPEG_TIMEOUT_SEC,
    )

    if result.returncode != 0:
        logger.error(
            event="ffmpeg_audio_extract_failed",
            stderr=result.stderr[-500:] if result.stderr else "",
        )
        raise RuntimeError("AUDIO_EXTRACTION_FAILED")


# BASE STRUCTURE

def _base_structure(
    doc_id: str,
    session_id: str,
    file_hash: str,
    source_path: str,
    **extra,
) -> Dict:
    return {
        "doc_id":      doc_id,
        "session_id":  session_id,
        "file_hash":   file_hash,
        "source_path": source_path,
        **extra,
    }


# MAIN

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError(f"UNSUPPORTED_VIDEO_FORMAT: {ext}")

    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError("EMPTY_FILE")

    if file_size > settings.MAX_FILE_SIZE_VIDEO:
        raise ValueError(
            f"VIDEO_TOO_LARGE: {file_size} bytes exceeds {settings.MAX_FILE_SIZE_VIDEO} bytes"
        )

    # DURATION PRE-CHECK
    video_duration = _probe_duration(file_path)
    if video_duration is not None and video_duration > settings.MAX_VIDEO_DURATION_SEC:
        raise ValueError(
            f"VIDEO_TOO_LONG: {video_duration:.1f}s exceeds {settings.MAX_VIDEO_DURATION_SEC}s"
        )

    start       = time.time()
    doc_id      = str(uuid.uuid4())
    file_hash   = _file_hash(file_path)
    source_name = path.name
    source_path = str(path.resolve())

    audio_path     = None
    frame_temp_dir = None

    logger.info(
        event="video_ingest_start",
        file=source_name,
        size=file_size,
        duration=video_duration,
        session_id=session_id,
    )

    try:
        documents: List[IngestedDocument] = []

        # AUDIO EXTRACTION
        staging = settings.UPLOAD_STAGING_DIR
        staging.mkdir(parents=True, exist_ok=True)

        fd, audio_path = tempfile.mkstemp(suffix=".wav", dir=str(staging))
        os.close(fd)

        _extract_audio(file_path, audio_path)

        # AUDIO INGESTION
        audio_docs      = audio_ingest(audio_path, session_id)
        speech_segments: List[Dict] = []

        for i, doc in enumerate(audio_docs):
            s       = doc.structure or {}
            start_t = s.get("timestamp_start")
            end_t   = s.get("timestamp_end")

            if start_t is None or end_t is None or end_t <= start_t:
                continue

            speech_segments.append({
                "index":      i,
                "start":      start_t,
                "end":        end_t,
                "text":       doc.text,
                "confidence": s.get("confidence", 1.0),
                "language":   s.get("language"),
            })

            documents.append(
                IngestedDocument(
                    text=doc.text,
                    modality="video",
                    subtype="speech",
                    source_type="video",
                    source=source_name,
                    chunk_id=i,
                    structure=_base_structure(
                        doc_id, session_id, file_hash, source_path,
                        timestamp_start=start_t,
                        timestamp_end=end_t,
                        confidence=s.get("confidence"),
                        language=s.get("language"),
                        hallucination_risk=s.get("hallucination_risk", "low"),
                        snr=s.get("snr"),
                        snr_degraded=s.get("snr_degraded", False),
                        content_type="video_speech",
                        ingestion_time=time.time(),
                    ),
                    extra_metadata={
                        "importance_score":   s.get("confidence", 1.0),
                        "modality_weight":    1.2,
                        "data_quality_score": s.get("confidence", 1.0),
                    },
                ).finalize()
            )

        # FRAME EXTRACTION
        frames: List[Dict] = []
        try:
            frames = extract_frames(
                file_path,
                settings.VIDEO_FRAME_INTERVAL_SEC,
                session_id,
            )
        except Exception as e:
            logger.warning(event="frame_extract_failed", file=source_name, error=str(e))

        if frames:
            frame_temp_dir = Path(frames[0]["path"]).parent

        # FRAME PROCESSING
        for frame in frames[:settings.MAX_VIDEO_FRAMES]:
            try:
                ts      = frame["timestamp_start"]
                f_path  = frame["path"]

                caption  = generate_caption(f_path, session_id) or f"Scene at {ts}s"
                ocr_text = _extract_frame_ocr(f_path)
                blur     = _blur_score(f_path)
                linked   = _link_speech(ts, speech_segments)
                align    = _alignment_score(caption, linked)

                conflict_flag = bool(linked and ocr_text and align < 0.1)

                documents.append(
                    IngestedDocument(
                        text=caption,
                        modality="video",
                        subtype="frame",
                        source_type="video",
                        source=source_name,
                        chunk_id=frame["frame_index"],
                        structure=_base_structure(
                            doc_id, session_id, file_hash, source_path,
                            asset_path=f_path,
                            timestamp_start=ts,
                            timestamp_end=frame.get("timestamp_end", ts),
                            frame_index=frame["frame_index"],
                            linked_speech=linked,
                            conflict_flag=conflict_flag,
                            alignment_score=align,
                            blur_score=blur,
                            fps=frame.get("fps"),
                            video_duration=frame.get("video_duration"),
                            content_type="video_frame",
                            ingestion_time=time.time(),
                        ),
                        extra_metadata={
                            "importance_score":   blur,
                            "modality_weight":    1.0,
                            "data_quality_score": blur,
                        },
                    ).finalize()
                )

                if ocr_text:
                    documents.append(
                        IngestedDocument(
                            text=ocr_text,
                            modality="video",
                            subtype="ocr",
                            source_type="video",
                            source=source_name,
                            structure=_base_structure(
                                doc_id, session_id, file_hash, source_path,
                                timestamp_start=ts,
                                frame_index=frame["frame_index"],
                                content_type="video_ocr",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "importance_score":   0.7,
                                "modality_weight":    0.9,
                                "data_quality_score": 0.7,
                            },
                        ).finalize()
                    )

            except Exception as e:
                logger.warning(
                    event="frame_process_error",
                    frame_index=frame.get("frame_index"),
                    error=str(e),
                )

        if not documents:
            raise ValueError("NO_VIDEO_CONTENT_EXTRACTED")

        latency = round(time.time() - start, 2)

        logger.info(
            event="video_ingest_success",
            file=source_name,
            docs=len(documents),
            speech_segments=len(speech_segments),
            frames_extracted=len(frames),
            video_duration=video_duration,
            latency=latency,
            session_id=session_id,
        )

        return documents

    except Exception as e:
        logger.error(
            event="video_ingest_failed",
            file=source_name,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 2),
        )
        raise

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        if frame_temp_dir and frame_temp_dir.exists():
            shutil.rmtree(frame_temp_dir, ignore_errors=True)