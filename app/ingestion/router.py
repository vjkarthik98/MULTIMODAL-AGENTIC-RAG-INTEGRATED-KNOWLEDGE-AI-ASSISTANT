import os
import time
from pathlib import Path
from typing import Callable, Dict, List

from app.core.config import settings
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.document_ingest import ingest as document_ingest
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.ingestion.text_ingest import ingest as text_ingest
from app.ingestion.video_ingest import ingest as video_ingest
from app.utils.logger import get_logger


logger = get_logger(__name__)


# Config-driven extensions
TEXT_EXTENSIONS = set(getattr(settings, "TEXT_EXTENSIONS", [".txt", ".md"]))
DOCUMENT_EXTENSIONS = set(getattr(settings, "DOCUMENT_EXTENSIONS", [".pdf", ".docx", ".xlsx", ".xls"]))
IMAGE_EXTENSIONS = set(getattr(settings, "IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]))
AUDIO_EXTENSIONS = set(getattr(settings, "AUDIO_EXTENSIONS", [".mp3", ".wav", ".m4a", ".flac"]))
VIDEO_EXTENSIONS = set(getattr(settings, "VIDEO_EXTENSIONS", [".mp4", ".avi", ".mov", ".mkv"]))


INGESTION_HANDLERS: Dict[str, Callable[[str, str], List[IngestedDocument]]] = {
    "text": text_ingest,
    "document": document_ingest,
    "image": image_ingest,
    "audio": audio_ingest,
    "video": video_ingest,
}


def detect_modality(file_path: str, session_id: str = "default") -> str:
    if not file_path or not file_path.strip():
        raise ValueError("file_path is required")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    # File size validation (important)
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.2f}MB")

    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"

    raise ValueError(f"Unsupported file type: {ext}")


def _validate_documents(
    documents: List[IngestedDocument],
    session_id: str
) -> List[IngestedDocument]:

    validated_documents: List[IngestedDocument] = []

    for index, doc in enumerate(documents):
        try:
            if isinstance(doc, IngestedDocument):
                validated = doc
            elif isinstance(doc, dict):
                validated = IngestedDocument(**doc)
            else:
                logger.warning(
                    "[Router][SKIP] session_id=%s | index=%s | invalid_type=%s",
                    session_id,
                    index,
                    type(doc)
                )
                continue

            validated_documents.append(validated)

        except Exception as exc:
            logger.warning(
                "[Router][VALIDATION_FAIL] session_id=%s | index=%s | error=%s",
                session_id,
                index,
                exc,
            )
            continue

    return validated_documents


def route_ingestion(
    file_path: str,
    session_id: str = "default"
) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("session_id required")

    start = time.time()

    modality = detect_modality(file_path, session_id=session_id)

    logger.info(
        "[Router][START] session_id=%s | modality=%s",
        session_id,
        modality
    )

    try:
        handler = INGESTION_HANDLERS.get(modality)
        if not handler:
            raise ValueError(f"No handler for modality={modality}")

        documents = handler(file_path, session_id=session_id)

        validated_documents = _validate_documents(documents, session_id)

        if not validated_documents:
            raise ValueError("No valid documents after validation")

        # Limit ingestion output
        max_docs = getattr(settings, "MAX_INGESTED_DOCS", 5000)

        if len(validated_documents) > max_docs:
            logger.warning(
                "[Router] limiting docs %s -> %s",
                len(validated_documents),
                max_docs
            )
            validated_documents = validated_documents[:max_docs]

        latency = round(time.time() - start, 2)

        logger.info(
            "[Router][SUCCESS] session_id=%s | docs=%s | latency=%ss",
            session_id,
            len(validated_documents),
            latency
        )

        return validated_documents

    except Exception as exc:
        logger.error(
            "[Router][FAILED] session_id=%s | error=%s",
            session_id,
            str(exc)
        )
        raise RuntimeError(f"Ingestion routing failed: {exc}") from exc