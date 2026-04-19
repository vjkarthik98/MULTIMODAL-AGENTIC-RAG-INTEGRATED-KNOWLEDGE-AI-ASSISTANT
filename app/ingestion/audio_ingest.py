import hashlib
import os
import time
import uuid

from app.core.model_loader import model_loader
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = get_logger(__name__)


def _generate_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as file_handle:
        return hashlib.md5(file_handle.read()).hexdigest()


def ingest(file_path: str, session_id: str = "default") -> list[IngestedDocument]:
    if not session_id:
        raise ValueError("session_id is required")
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} not found")

    start_time = time.time()
    source_name = os.path.basename(file_path)
    source_path = os.path.abspath(file_path)
    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    try:
        logger.info("[AudioIngest][START] session_id=%s | file=%s", session_id, file_path)

        audio_model = model_loader.get_whisper()
        segments_iter, info = audio_model.transcribe(file_path)
        segments = list(segments_iter)

        documents: list[IngestedDocument] = []
        total_segments = len(segments)
        language = getattr(info, "language", None)

        for index, segment in enumerate(segments):
            text = (getattr(segment, "text", "") or "").strip()
            if len(text) < 2:
                continue

            start = round(float(getattr(segment, "start", 0.0)), 2)
            end = round(float(getattr(segment, "end", start)), 2)
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
                        "model": model_loader.MODEL_CONFIG["whisper"],
                        "modality_source": "audio",
                        "content_type": "speech_segment",
                        "total_segments": total_segments,
                        "ingestion_time": time.time(),
                    },
                )
            )

        if not documents:
            logger.error("[AudioIngest][EMPTY] session_id=%s | No valid segments", session_id)
            raise ValueError("No valid audio segments extracted")

        latency = time.time() - start_time
        logger.info(
            "[AudioIngest][SUCCESS] session_id=%s | segments=%s | latency=%.2fs",
            session_id,
            len(documents),
            latency,
        )
        return documents

    except Exception as exc:
        logger.error("[AudioIngest][FAILED] session_id=%s | error=%s", session_id, exc)
        raise RuntimeError(f"Audio ingestion failed: {exc}") from exc
