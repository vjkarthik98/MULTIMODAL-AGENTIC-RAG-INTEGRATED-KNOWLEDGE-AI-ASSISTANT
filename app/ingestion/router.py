import os
from typing import Callable, Dict, List

from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.document_ingest import ingest as document_ingest
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.ingestion.text_ingest import ingest as text_ingest
from app.ingestion.video_ingest import ingest as video_ingest
from app.utils.logger import get_logger


logger = get_logger(__name__)

TEXT_EXTENSIONS = {".txt", ".md"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

INGESTION_HANDLERS: Dict[str, Callable[[str, str], List[IngestedDocument]]] = {
    "text": text_ingest,
    "document": document_ingest,
    "image": image_ingest,
    "audio": audio_ingest,
    "video": video_ingest,
}


def detect_modality(file_path: str, session_id: str = "default") -> str:
    if not file_path or not file_path.strip():
        logger.error("[Router] session_id=%s | Empty file path", session_id)
        raise ValueError("file_path is required")

    if not os.path.exists(file_path):
        logger.error("[Router] session_id=%s | File not found | %s", session_id, file_path)
        raise FileNotFoundError(f"{file_path} not found")

    ext = os.path.splitext(file_path)[1].lower()
    logger.debug(
        "[Router] session_id=%s | Detecting modality | file=%s | ext=%s",
        session_id,
        file_path,
        ext,
    )

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

    logger.error(
        "[Router] session_id=%s | Unsupported file type | ext=%s",
        session_id,
        ext,
    )
    raise ValueError(f"Unsupported file type: {ext}")


def _validate_documents(documents: List[IngestedDocument], session_id: str) -> List[IngestedDocument]:
    validated_documents: List[IngestedDocument] = []

    for index, doc in enumerate(documents):
        try:
            if isinstance(doc, IngestedDocument):
                validated = doc
            elif isinstance(doc, dict):
                validated = IngestedDocument(**doc)
            else:
                raise TypeError(f"Unsupported document type at index {index}: {type(doc)!r}")

            validated_documents.append(validated)
        except Exception as exc:
            logger.error(
                "[Router][VALIDATION_FAIL] session_id=%s | index=%s | error=%s",
                session_id,
                index,
                exc,
            )
            raise ValueError(f"Invalid document structure at index {index}: {exc}") from exc

    return validated_documents


def route_ingestion(file_path: str, session_id: str = "default") -> List[IngestedDocument]:
    modality = detect_modality(file_path, session_id=session_id)
    logger.info(
        "[Router][START] session_id=%s | modality=%s | file=%s",
        session_id,
        modality,
        file_path,
    )

    try:
        documents = INGESTION_HANDLERS[modality](file_path, session_id=session_id)
        validated_documents = _validate_documents(documents, session_id=session_id)

        if not validated_documents:
            logger.error("[Router][EMPTY] session_id=%s | modality=%s", session_id, modality)
            raise ValueError("No documents returned from ingestion")

        logger.info(
            "[Router][SUCCESS] session_id=%s | modality=%s | docs=%s",
            session_id,
            modality,
            len(validated_documents),
        )
        return validated_documents

    except Exception as exc:
        logger.error(
            "[Router][ERROR] session_id=%s | modality=%s | error=%s",
            session_id,
            modality,
            exc,
        )
        raise RuntimeError(f"Ingestion routing failed: {exc}") from exc
