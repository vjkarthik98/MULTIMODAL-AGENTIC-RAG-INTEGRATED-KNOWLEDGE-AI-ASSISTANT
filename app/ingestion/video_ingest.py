import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from app.core.config import settings
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.schema import IngestedDocument
from app.ingestion.video_frames import extract_frames
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _generate_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as file_handle:
        return hashlib.md5(file_handle.read()).hexdigest()


def _resolve_ffmpeg_path() -> str:
    configured = Path(settings.FFMPEG_PATH)
    if configured.exists():
        return str(configured)

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    raise FileNotFoundError("ffmpeg executable was not found")


def ingest(file_path: str, session_id: str = "default"):
    if not session_id:
        raise ValueError("session_id is required")
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} not found")

    start_time = time.time()
    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)
    source_name = os.path.basename(file_path)
    source_path = os.path.abspath(file_path)
    audio_path = None
    frame_temp_dir = None

    try:
        logger.info("[VideoIngest][START] session_id=%s | file=%s", session_id, file_path)
        documents = []

        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        result = subprocess.run(
            [
                _resolve_ffmpeg_path(),
                "-y",
                "-i",
                file_path,
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error("[VideoIngest][FFMPEG_FAIL] session_id=%s | %s", session_id, result.stderr)
            raise RuntimeError("Audio extraction failed")

        audio_docs = audio_ingest(audio_path, session_id=session_id)
        speech_segments = []
        total_segments = len(audio_docs)

        for index, doc in enumerate(audio_docs):
            structure = dict(doc.structure or {})
            start = structure.get("start_time")
            end = structure.get("end_time")

            speech_segments.append({"index": index, "start": start, "end": end})
            documents.append(
                IngestedDocument(
                    text=f"Video speech from {start}s to {end}s: {doc.text}",
                    modality="video",
                    subtype="speech",
                    source_type="video",
                    source=source_name,
                    chunk_id=index,
                    structure={
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "file_hash": file_hash,
                        "source_path": source_path,
                        "segment_index": index,
                        "start_time": start,
                        "end_time": end,
                        "linked_frames": [],
                        "parent_modality": "video",
                        "modality_source": "audio",
                        "content_type": "video_speech",
                        "total_segments": total_segments,
                        "ingestion_time": time.time(),
                    },
                )
            )

        frames = extract_frames(
            file_path,
            interval_sec=settings.FRAME_INTERVAL_SECONDS,
            session_id=session_id,
        )
        if frames:
            frame_temp_dir = frames[0].get("temp_dir")

        total_frames = len(frames)
        logger.info("[VideoIngest] session_id=%s | frames_extracted=%s", session_id, total_frames)

        for frame in frames:
            caption = generate_caption(frame["path"], session_id=session_id)
            if not caption:
                continue

            timestamp = frame.get("timestamp")
            linked_segment_index = None
            for segment in speech_segments:
                start = segment.get("start")
                end = segment.get("end")
                if start is not None and end is not None and start <= timestamp <= end:
                    linked_segment_index = segment["index"]
                    break

            documents.append(
                IngestedDocument(
                    text=f"Video frame at {timestamp}s shows: {caption}",
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
                        "frame_index": frame["frame_index"],
                        "timestamp": timestamp,
                        "linked_segment_index": linked_segment_index,
                        "parent_modality": "video",
                        "modality_source": "image",
                        "content_type": "video_frame",
                        "total_frames": total_frames,
                        "ingestion_time": time.time(),
                    },
                )
            )

        if not documents:
            logger.error("[VideoIngest][EMPTY] session_id=%s | No content extracted", session_id)
            raise ValueError("No content extracted from video")

        latency = time.time() - start_time
        logger.info(
            "[VideoIngest][SUCCESS] session_id=%s | docs=%s | latency=%.2fs",
            session_id,
            len(documents),
            latency,
        )
        return documents

    except Exception as exc:
        logger.error("[VideoIngest][FAILED] session_id=%s | error=%s", session_id, exc)
        raise RuntimeError(f"Video ingestion failed: {exc}") from exc

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if frame_temp_dir and os.path.exists(frame_temp_dir):
            shutil.rmtree(frame_temp_dir, ignore_errors=True)
