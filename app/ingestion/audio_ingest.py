import hashlib
import os
import time
import uuid
from typing import List

from app.core.config import settings
from app.core.model_loader import model_loader
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def ingest(file_path: str, session_id: str = "default") -> List[IngestedDocument]:
    if not session_id:
        raise ValueError("session_id is required")

    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} not found")

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({file_size_mb:.2f}MB)")

    start_time = time.time()

    source_name = os.path.basename(file_path)
    source_path = os.path.abspath(file_path)

    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    try:
        logger.info("[AudioIngest][START] session_id=%s | file=%s", session_id, file_path)

        whisper_model = model_loader.get_whisper()

        segments_iter, info = whisper_model.transcribe(file_path)

        documents: List[IngestedDocument] = []

        language = getattr(info, "language", None)

        max_segments = getattr(settings, "MAX_AUDIO_SEGMENTS", 500)
        max_duration = settings.MAX_AUDIO_DURATION_SEC

        for index, segment in enumerate(segments_iter):

            if index >= max_segments:
                logger.warning("[AudioIngest] segment limit reached")
                break

            text = (getattr(segment, "text", "") or "").strip()
            if len(text) < 2:
                continue

            start = round(float(getattr(segment, "start", 0.0)), 2)
            end = round(float(getattr(segment, "end", start)), 2)

            if end > max_duration:
                logger.warning("[AudioIngest] max duration reached")
                break

            duration = round(max(end - start, 0.0), 2)

            documents.append(
                IngestedDocument(
                    text=f"Audio speech from {start}s to {end}s: {text}",
                    modality="audio",
                    subtype="speech",
                    source_type="audio",
                    source=source_name,
                    structure={
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "file_hash": file_hash,
                        "source_path": source_path,
                        "segment_index": index,
                        "start_time": start,
                        "end_time": end,
                        "duration": duration,
                        "language": language,
                        "model": settings.WHISPER_MODEL,
                        "modality_source": "audio",
                        "content_type": "speech_segment",
                        "ingestion_time": time.time(),
                    },
                )
            )

        if not documents:
            logger.error("[AudioIngest][EMPTY] session_id=%s", session_id)
            raise ValueError("No valid audio segments extracted")

        latency = round(time.time() - start_time, 2)

        logger.info(
            "[AudioIngest][SUCCESS] session_id=%s | segments=%s | latency=%.2fs",
            session_id,
            len(documents),
            latency
        )

        return documents

    except Exception as exc:
        logger.error("[AudioIngest][FAILED] session_id=%s | error=%s", session_id, str(exc))
        raise RuntimeError(f"Audio ingestion failed: {exc}") from exc