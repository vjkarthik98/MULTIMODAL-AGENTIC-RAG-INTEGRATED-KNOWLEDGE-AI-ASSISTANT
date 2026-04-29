import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

from app.core.config import settings
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.ingestion.video_frames import extract_frames
from app.utils.logger import get_logger


logger = get_logger(__name__)


# GENERATE FILE HASH
def _generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# RESOLVE FFMPEG
def _resolve_ffmpeg_path() -> str:
    configured = Path(settings.FFMPEG_PATH)
    if configured.exists():
        return str(configured)

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    raise FileNotFoundError("FFMPEG NOT FOUND")


# SAFE OCR FOR FRAME
def _extract_frame_text(image_path: str) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img) or ""
        text = text.strip()

        if len(text) < 10:
            return ""

        return text[:settings.MAX_PROMPT_CHARS]

    except Exception:
        return ""


# MAIN INGEST FUNCTION
def ingest(file_path: str, session_id: str = "default") -> List[IngestedDocument]:

    # VALIDATION
    if not session_id:
        raise ValueError("SESSION_ID REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} NOT FOUND")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"VIDEO TOO LARGE: {size_mb:.2f}MB")

    start = time.time()

    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    source_name = path.name
    source_path = str(path.resolve())

    audio_path = None
    frame_temp_dir = None

    try:
        logger.info("[VideoIngest][START] session_id=%s", session_id)

        documents: List[IngestedDocument] = []

        # AUDIO EXTRACTION
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        ffmpeg_cmd = [
            _resolve_ffmpeg_path(),
            "-y",
            "-i", file_path,
            "-vn",
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
            "-ac", "1",
            audio_path,
        ]

        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=settings.FFMPEG_TIMEOUT_SEC,
        )

        if result.returncode != 0:
            raise RuntimeError("FFMPEG AUDIO EXTRACTION FAILED")

        # AUDIO INGEST
        audio_docs = audio_ingest(audio_path, session_id=session_id)

        speech_segments = []

        for i, doc in enumerate(audio_docs[:settings.MAX_AUDIO_SEGMENTS]):

            structure = dict(doc.structure or {})
            start_t = structure.get("start_time")
            end_t = structure.get("end_time")

            # VALIDATE SEGMENT
            if start_t is None or end_t is None or end_t <= start_t:
                continue

            speech_segments.append({
                "index": i,
                "start": start_t,
                "end": end_t
            })

            text = f"Video speech from {start_t}s to {end_t}s: {doc.text}"
            text = text[:settings.MAX_PROMPT_CHARS]

            documents.append(
                IngestedDocument(
                    text=text,
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
                        "start_time": start_t,
                        "end_time": end_t,
                        "content_type": "video_speech",
                        "embedding_space": "text",
                    },
                ).finalize()
            )

        # FRAME EXTRACTION
        try:
            frames = extract_frames(
                file_path,
                interval_sec=settings.VIDEO_FRAME_INTERVAL_SEC,
                session_id=session_id,
            )
        except Exception as e:
            logger.warning("[VideoIngest] FRAME EXTRACTION FAILED | %s", str(e))
            frames = []

        if frames:
            frame_temp_dir = frames[0].get("temp_dir")

        # PROCESS FRAMES
        for frame in frames[:settings.MAX_VIDEO_FRAMES]:

            try:
                timestamp = frame.get("timestamp")

                # CAPTION
                caption = generate_caption(frame["path"], session_id=session_id)
                if not caption:
                    caption = f"Scene at {timestamp}s"

                # OCR FROM FRAME
                ocr_text = _extract_frame_text(frame["path"])

                # LINK TO SPEECH
                linked_segment = None
                for seg in speech_segments:
                    if seg["start"] <= timestamp <= seg["end"]:
                        linked_segment = seg["index"]
                        break

                text = f"Frame at {timestamp}s shows: {caption}"
                text = text[:settings.MAX_PROMPT_CHARS]

                documents.append(
                    IngestedDocument(
                        text=text,
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
                            "timestamp": timestamp,
                            "linked_segment_index": linked_segment,
                            "content_type": "video_frame",
                            "embedding_space": "vision",
                        },
                    ).finalize()
                )

                # OCR DOCUMENT (OPTIONAL BUT IMPORTANT)
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
                                "timestamp": timestamp,
                                "content_type": "video_ocr",
                            },
                        ).finalize()
                    )

            except Exception as e:
                logger.warning("[VideoIngest][FRAME_FAIL] %s", str(e))
                continue

        # GLOBAL LIMIT
        if len(documents) > settings.MAX_INGESTED_DOCS:
            documents = documents[:settings.MAX_INGESTED_DOCS]

        if not documents:
            raise ValueError("NO CONTENT EXTRACTED")

        logger.info(
            "[VideoIngest][SUCCESS] session_id=%s | docs=%s | latency=%.2fs",
            session_id,
            len(documents),
            time.time() - start
        )

        return documents

    except Exception as e:
        logger.error("[VideoIngest][FAILED] %s", str(e))
        raise

    finally:
        # CLEANUP TEMP FILES (CRITICAL)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        if frame_temp_dir and os.path.exists(frame_temp_dir):
            shutil.rmtree(frame_temp_dir, ignore_errors=True)