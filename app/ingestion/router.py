import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Set

from app.core.config import settings
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.document_ingest import ingest as document_ingest
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.ingestion.text_ingest import ingest as text_ingest
from app.ingestion.video_ingest import ingest as video_ingest
from app.utils.logger import get_logger


logger = get_logger(__name__)


# EXTENSION CONFIG
TEXT_EXTENSIONS = set(getattr(settings, "TEXT_EXTENSIONS", [".txt", ".md"]))
DOCUMENT_EXTENSIONS = set(getattr(settings, "DOCUMENT_EXTENSIONS", [".pdf", ".docx", ".xlsx", ".xls"]))
IMAGE_EXTENSIONS = set(getattr(settings, "IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]))
AUDIO_EXTENSIONS = set(getattr(settings, "AUDIO_EXTENSIONS", [".mp3", ".wav", ".m4a", ".flac"]))
VIDEO_EXTENSIONS = set(getattr(settings, "VIDEO_EXTENSIONS", [".mp4", ".avi", ".mov", ".mkv"]))


# HANDLER REGISTRY
INGESTION_HANDLERS: Dict[str, Callable[[str, str], List[IngestedDocument]]] = {
    "text": text_ingest,
    "document": document_ingest,
    "image": image_ingest,
    "audio": audio_ingest,
    "video": video_ingest,
}


# DETECT FILE MODALITY
def detect_modality(file_path: str) -> str:

    if not file_path or not file_path.strip():
        raise ValueError("FILE_PATH REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} NOT FOUND")

    # EMPTY FILE CHECK
    if path.stat().st_size == 0:
        raise ValueError("EMPTY FILE")

    # FILE SIZE CHECK
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"FILE TOO LARGE: {size_mb:.2f}MB")

    ext = path.suffix.lower()

    # ENV-LEVEL VALIDATION
    allowed_types = getattr(settings, "ALLOWED_FILE_TYPES", [])
    if allowed_types and ext.replace(".", "") not in allowed_types:
        raise ValueError(f"FILE TYPE NOT ALLOWED: {ext}")

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

    raise ValueError(f"UNSUPPORTED FILE TYPE: {ext}")


# VALIDATE AND NORMALIZE DOCUMENTS
def _validate_documents(
    documents: List[IngestedDocument],
    session_id: str
) -> List[IngestedDocument]:

    validated_documents: List[IngestedDocument] = []
    seen_hashes: Set[str] = set()

    for index, doc in enumerate(documents):
        try:
            # NORMALIZE INPUT TYPE
            if isinstance(doc, dict):
                doc = IngestedDocument(**doc)

            if not isinstance(doc, IngestedDocument):
                continue

            # FORCE FINALIZATION (CRITICAL)
            doc = doc.finalize()

            # ENSURE SESSION CONSISTENCY
            doc.structure["session_id"] = session_id

            # DEDUPLICATION
            content_hash = hash(doc.text)

            if content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)

            validated_documents.append(doc)

        except Exception as exc:
            logger.warning(
                "[Router][VALIDATION_FAIL] session_id=%s | index=%s | error=%s",
                session_id,
                index,
                str(exc),
            )
            continue

    return validated_documents


# MAIN ROUTER
def route_ingestion(
    file_path: str,
    session_id: str = "default"
) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID REQUIRED")

    start = time.time()

    try:
        modality = detect_modality(file_path)

        logger.info(
            "[Router][START] session_id=%s | modality=%s | file=%s",
            session_id,
            modality,
            file_path
        )

        handler = INGESTION_HANDLERS.get(modality)

        if not handler:
            raise ValueError(f"NO HANDLER FOR MODALITY={modality}")

        # RUN INGESTION
        documents = handler(file_path, session_id=session_id)

        # VALIDATE OUTPUT
        validated_documents = _validate_documents(documents, session_id)

        if not validated_documents:
            raise ValueError("NO VALID DOCUMENTS AFTER VALIDATION")

        # LIMIT OUTPUT
        max_docs = getattr(settings, "MAX_INGESTED_DOCS", 5000)

        if len(validated_documents) > max_docs:
            logger.warning(
                "[Router] LIMITING DOCS %s -> %s",
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
            "[Router][FAILED] session_id=%s | file=%s | error=%s",
            session_id,
            file_path,
            str(exc)
        )
        raise RuntimeError(f"INGESTION ROUTING FAILED: {exc}") from exc