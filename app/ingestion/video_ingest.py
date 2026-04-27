import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

from app.core.config import settings
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.ingestion.video_frames import extract_frames
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _resolve_ffmpeg_path() -> str:
    configured = Path(settings.FFMPEG_PATH)
    if configured.exists():
        return str(configured)

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    raise FileNotFoundError("ffmpeg not found")


def ingest(file_path: str, session_id: str = "default") -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("session_id required")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    # File size validation
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"Video too large: {size_mb:.2f}MB")

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
            raise RuntimeError("FFmpeg audio extraction failed")

        # AUDIO INGEST 
        audio_docs = audio_ingest(audio_path, session_id=session_id)

        max_segments = settings.MAX_AUDIO_SEGMENTS
        speech_segments = []

        for i, doc in enumerate(audio_docs[:max_segments]):

            structure = dict(doc.structure or {})
            start_t = structure.get("start_time")
            end_t = structure.get("end_time")

            speech_segments.append({"index": i, "start": start_t, "end": end_t})

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
                        "segment_index": i,
                        "start_time": start_t,
                        "end_time": end_t,
                        "parent_modality": "video",
                        "content_type": "video_speech",
                        "embedding_space": "text",
                        "ingestion_time": time.time(),
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
            logger.warning("[VideoIngest] frame extraction failed | %s", str(e))
            frames = []
        

        if frames:
            frame_temp_dir = frames[0].get("temp_dir")

        max_frames = settings.MAX_VIDEO_FRAMES

        for frame in frames[:max_frames]:

            try:
                caption = generate_caption(frame["path"], session_id=session_id)
                if not caption:
                    caption = "visual scene from video"
                    

                timestamp = frame.get("timestamp")

                linked_segment = None
                for seg in speech_segments:
                    if seg["start"] <= timestamp <= seg["end"]:
                        linked_segment = seg["index"]
                        break

                text = f"Video frame at {timestamp}s shows: {caption}"
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
                            "ingestion_time": time.time(),
                        },
                    ).finalize()
                )

            except Exception as e:
                logger.warning("[VideoIngest][FRAME_FAIL] %s", str(e))
                continue

        # Global doc limit
        if len(documents) > settings.MAX_INGESTED_DOCS:
            logger.warning("[VideoIngest] doc limit applied")
            documents = documents[:settings.MAX_INGESTED_DOCS]

        if not documents:
            raise ValueError("No content extracted")

        latency = round(time.time() - start, 2)

        logger.info(
            "[VideoIngest][SUCCESS] session_id=%s | docs=%s | latency=%ss",
            session_id,
            len(documents),
            latency
        )

        return documents

    except Exception as e:
        logger.error("[VideoIngest][FAILED] %s", str(e))
        raise

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        if frame_temp_dir and os.path.exists(frame_temp_dir):
            """shutil.rmtree(frame_temp_dir, ignore_errors=True)"""