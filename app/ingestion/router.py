import hashlib
import mimetypes
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.core.response import ErrorCode, Severity, UniversalErrorResponse, Modality
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.document_ingest import ingest as document_ingest
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.ingestion.text_ingest import ingest as text_ingest
from app.ingestion.video_ingest import ingest as video_ingest
from app.utils.logger import get_logger

logger = get_logger(__name__)


# EXTENSION MAPS

TEXT_EXTENSIONS: Set[str]     = {".txt", ".md"}
DOCUMENT_EXTENSIONS: Set[str] = {".pdf", ".docx", ".xlsx", ".xls"}
IMAGE_EXTENSIONS: Set[str]    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTENSIONS: Set[str]    = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS: Set[str]    = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

EXT_TO_MODALITY: Dict[str, str] = {
    **{ext: "text"     for ext in TEXT_EXTENSIONS},
    **{ext: "document" for ext in DOCUMENT_EXTENSIONS},
    **{ext: "image"    for ext in IMAGE_EXTENSIONS},
    **{ext: "audio"    for ext in AUDIO_EXTENSIONS},
    **{ext: "video"    for ext in VIDEO_EXTENSIONS},
}

# MODALITY TO RESPONSE MODALITY
MODALITY_LABEL: Dict[str, str] = {
    "text":     Modality.TEXT,
    "document": Modality.PDF,
    "image":    Modality.IMAGE,
    "audio":    Modality.AUDIO,
    "video":    Modality.VIDEO,
}

# PER-MODALITY FILE SIZE LIMITS (bytes)
MODALITY_SIZE_LIMITS: Dict[str, int] = {
    "text":     settings.MAX_FILE_SIZE_TEXT,
    "document": settings.MAX_FILE_SIZE_PDF,
    "image":    settings.MAX_FILE_SIZE_IMAGE,
    "audio":    settings.MAX_FILE_SIZE_AUDIO,
    "video":    settings.MAX_FILE_SIZE_VIDEO,
}

# INGESTION HANDLERS
INGESTION_HANDLERS: Dict[str, Callable[[str, str], List[IngestedDocument]]] = {
    "text":     text_ingest,
    "document": document_ingest,
    "image":    image_ingest,
    "audio":    audio_ingest,
    "video":    video_ingest,
}

MAX_INGESTED_DOCS: int = 5000


# HASH

def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# MIME DETECTION

def _detect_mime(path: Path) -> Tuple[Optional[str], Optional[str]]:
    mime, encoding = mimetypes.guess_type(path.name)
    return mime, encoding


def _validate_mime(path: Path, mime: Optional[str]) -> None:
    allowed = settings.ALLOWED_MIME_TYPES
    if allowed and mime and mime not in allowed:
        raise ValueError(f"MIME_NOT_ALLOWED_{mime}")


# MODALITY DETECTION

def detect_modality(file_path: str) -> str:

    if not file_path:
        raise ValueError("FILE_PATH_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    stat = path.stat()

    if stat.st_size == 0:
        raise ValueError("EMPTY_FILE")

    ext      = path.suffix.lower()
    modality = EXT_TO_MODALITY.get(ext)

    if not modality:
        raise ValueError(f"UNSUPPORTED_TYPE_{ext}")

    # PER-MODALITY SIZE CHECK
    limit = MODALITY_SIZE_LIMITS.get(modality, settings.MAX_FILE_SIZE_MB * 1024 * 1024)
    if stat.st_size > limit:
        raise ValueError(
            f"FILE_TOO_LARGE: {stat.st_size} bytes exceeds {limit} bytes for {modality}"
        )

    # ALLOWED FILE TYPES CHECK
    allowed_types = settings.ALLOWED_FILE_TYPES
    if allowed_types and ext.lstrip(".") not in allowed_types:
        raise ValueError(f"EXT_NOT_ALLOWED_{ext}")

    # MIME VALIDATION
    mime, _ = _detect_mime(path)
    _validate_mime(path, mime)

    return modality


# DOCUMENT VALIDATION

def _validate_documents(
    documents: List[IngestedDocument],
    session_id: str,
    modality: str,
) -> List[IngestedDocument]:

    validated: List[IngestedDocument] = []
    seen: Set[str]                    = set()
    skipped: int                      = 0

    for i, doc in enumerate(documents):
        try:
            if isinstance(doc, dict):
                doc = IngestedDocument(**doc)

            if not isinstance(doc, IngestedDocument):
                skipped += 1
                continue

            doc = doc.finalize()

            # ENFORCE SESSION
            doc.structure["session_id"] = session_id

            # STABLE DEDUP
            h = _stable_hash(doc.text)
            if h in seen:
                continue

            seen.add(h)
            validated.append(doc)

        except Exception as e:
            skipped += 1
            logger.warning(
                event="doc_validation_failed",
                index=i,
                modality=modality,
                error=str(e),
            )

    if skipped:
        logger.warning(
            event="docs_skipped",
            skipped=skipped,
            accepted=len(validated),
            modality=modality,
        )

    return validated


# ROUTE INGESTION

def route_ingestion(
    file_path: str,
    session_id: str,
) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not file_path:
        raise ValueError("FILE_PATH_REQUIRED")

    start = time.time()

    try:
        modality = detect_modality(file_path)

        logger.info(
            event="ingestion_start",
            modality=modality,
            file=os.path.basename(file_path),
            session_id=session_id,
        )

        handler = INGESTION_HANDLERS.get(modality)

        if not handler:
            raise ValueError(f"HANDLER_NOT_FOUND_{modality}")

        docs = handler(file_path, session_id)

        if not docs:
            raise ValueError("NO_DOCS_RETURNED")

        docs = _validate_documents(docs, session_id, modality)

        if not docs:
            raise ValueError("NO_VALID_DOCS")

        if len(docs) > MAX_INGESTED_DOCS:
            docs = docs[:MAX_INGESTED_DOCS]
            logger.warning(
                event="doc_limit_applied",
                limit=MAX_INGESTED_DOCS,
                modality=modality,
            )

        latency = round(time.time() - start, 2)

        logger.info(
            event="ingestion_success",
            modality=modality,
            docs=len(docs),
            latency=latency,
            session_id=session_id,
        )

        return docs

    except Exception as e:
        logger.error(
            event="ingestion_failed",
            file=os.path.basename(file_path),
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 2),
        )
        raise