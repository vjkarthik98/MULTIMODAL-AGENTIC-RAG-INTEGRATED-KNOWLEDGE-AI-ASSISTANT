import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Dict

import pytesseract
from PIL import Image

from app.core.config import settings
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.ingestion.video_frames import extract_frames
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  HASH 
def _generate_file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


#  FFMPEG 
def _resolve_ffmpeg_path() -> str:
    configured = Path(settings.FFMPEG_PATH)
    if configured.exists():
        return str(configured)

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    raise FileNotFoundError("FFMPEG_NOT_FOUND")


#  OCR 
def _extract_frame_text(image_path: str) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img) or ""
        text = text.strip()
        return text if len(text) > 10 else ""
    except Exception:
        return ""


#  ALIGNMENT 
def _link_speech(timestamp: float, segments: List[Dict]) -> Dict:
    for seg in segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg
    return None


#  MAIN 
def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError("VIDEO_TOO_LARGE")

    start = time.time()

    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    source_name = path.name
    source_path = str(path.resolve())

    audio_path = None
    frame_temp_dir = None

    try:
        logger.info(event="video_ingest_start", file=file_path)

        documents: List[IngestedDocument] = []

        #  AUDIO 
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        cmd = [
            _resolve_ffmpeg_path(),
            "-y",
            "-i", file_path,
            "-vn",
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
            "-ac", "1",
            audio_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.FFMPEG_TIMEOUT_SEC,
        )

        if result.returncode != 0:
            raise RuntimeError("AUDIO_EXTRACTION_FAILED")

        audio_docs = audio_ingest(audio_path, session_id)

        speech_segments = []
        for i, doc in enumerate(audio_docs):

            s = doc.structure or {}
            start_t = s.get("timestamp_start")
            end_t = s.get("timestamp_end")

            if start_t is None or end_t is None or end_t <= start_t:
                continue

            speech_segments.append({
                "index": i,
                "start": start_t,
                "end": end_t,
                "confidence": s.get("confidence", 1.0),
            })

            documents.append(
                IngestedDocument(
                    text=doc.text,
                    modality="video",
                    subtype="speech",
                    source_type="video",
                    source=source_name,
                    chunk_id=i,
                    structure={
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "file_hash": file_hash,
                        "source_path": source_path,
                        "timestamp_start": start_t,
                        "timestamp_end": end_t,
                        "confidence": s.get("confidence"),
                        "content_type": "video_speech",
                    },
                    extra_metadata={
                        "importance_score": s.get("confidence", 1.0),
                        "modality_weight": 1.2,
                    },
                ).finalize()
            )

        #  FRAMES 
        frames = []
        try:
            frames = extract_frames(
                file_path,
                settings.VIDEO_FRAME_INTERVAL_SEC,
                session_id
            )
        except Exception as e:
            logger.warning(event="frame_extract_failed", error=str(e))

        if frames:
            frame_temp_dir = Path(frames[0]["path"]).parent

        #  FRAME PROCESS 
        for frame in frames[:settings.MAX_VIDEO_FRAMES]:

            try:
                ts = frame["timestamp_start"]

                caption = generate_caption(frame["path"], session_id) or f"Scene at {ts}s"
                ocr_text = _extract_frame_text(frame["path"])

                linked = _link_speech(ts, speech_segments)

                conflict_flag = False
                if linked and ocr_text:
                    if ocr_text.lower() not in caption.lower():
                        conflict_flag = True

                documents.append(
                    IngestedDocument(
                        text=caption,
                        modality="video",
                        subtype="frame",
                        source_type="video",
                        source=source_name,
                        chunk_id=frame["frame_index"],
                        structure={
                            "doc_id": doc_id,
                            "session_id": session_id,
                            "file_hash": file_hash,
                            "source_path": source_path,
                            "asset_path": frame["path"],
                            "timestamp": ts,
                            "linked_speech": linked,
                            "conflict_flag": conflict_flag,
                            "content_type": "video_frame",
                        },
                        extra_metadata={
                            "importance_score": 1.0,
                            "modality_weight": 1.0,
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
                            structure={
                                "doc_id": doc_id,
                                "timestamp": ts,
                                "content_type": "video_ocr",
                            },
                            extra_metadata={
                                "importance_score": 0.7,
                                "modality_weight": 0.9,
                            },
                        ).finalize()
                    )

            except Exception as e:
                logger.warning(event="frame_process_error", error=str(e))

        if not documents:
            raise ValueError("NO_VIDEO_CONTENT")

        latency = round(time.time() - start, 2)

        logger.info(
            event="video_ingest_success",
            docs=len(documents),
            latency=latency
        )

        return documents

    except Exception as e:
        logger.error(event="video_ingest_failed", error=str(e))
        raise

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        if frame_temp_dir and os.path.exists(frame_temp_dir):
            shutil.rmtree(frame_temp_dir, ignore_errors=True)