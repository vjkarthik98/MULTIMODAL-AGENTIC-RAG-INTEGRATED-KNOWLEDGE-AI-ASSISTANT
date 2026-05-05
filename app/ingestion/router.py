import hashlib
import mimetypes
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


#  EXTENSIONS 
TEXT_EXTENSIONS = set(getattr(settings, "TEXT_EXTENSIONS", [".txt", ".md"]))
DOCUMENT_EXTENSIONS = set(getattr(settings, "DOCUMENT_EXTENSIONS", [".pdf", ".docx", ".xlsx", ".xls"]))
IMAGE_EXTENSIONS = set(getattr(settings, "IMAGE_EXTENSIONS", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]))
AUDIO_EXTENSIONS = set(getattr(settings, "AUDIO_EXTENSIONS", [".mp3", ".wav", ".m4a", ".flac"]))
VIDEO_EXTENSIONS = set(getattr(settings, "VIDEO_EXTENSIONS", [".mp4", ".avi", ".mov", ".mkv"]))


#  HANDLERS 
INGESTION_HANDLERS: Dict[str, Callable[[str, str], List[IngestedDocument]]] = {
    "text": text_ingest,
    "document": document_ingest,
    "image": image_ingest,
    "audio": audio_ingest,
    "video": video_ingest,
}


#  HASH 
def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#  MIME VALIDATION 
def _validate_mime(path: Path):
    mime, _ = mimetypes.guess_type(path.name)

    if not mime:
        return  # allow fallback

    allowed = getattr(settings, "ALLOWED_MIME_TYPES", [])
    if allowed and mime not in allowed:
        raise ValueError(f"MIME_NOT_ALLOWED_{mime}")


#  DETECT 
def detect_modality(file_path: str) -> str:

    if not file_path:
        raise ValueError("FILE_PATH_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(file_path)

    if path.stat().st_size == 0:
        raise ValueError("EMPTY_FILE")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError("FILE_TOO_LARGE")

    _validate_mime(path)

    ext = path.suffix.lower()

    allowed_types = getattr(settings, "ALLOWED_FILE_TYPES", [])
    if allowed_types and ext.replace(".", "") not in allowed_types:
        raise ValueError(f"EXT_NOT_ALLOWED_{ext}")

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

    raise ValueError(f"UNSUPPORTED_TYPE_{ext}")


#  VALIDATION 
def _validate_documents(
    documents: List[IngestedDocument],
    session_id: str
) -> List[IngestedDocument]:

    validated: List[IngestedDocument] = []
    seen: Set[str] = set()

    for i, doc in enumerate(documents):
        try:
            if isinstance(doc, dict):
                doc = IngestedDocument(**doc)

            if not isinstance(doc, IngestedDocument):
                continue

            doc = doc.finalize()

            # enforce session
            doc.structure["session_id"] = session_id

            # stable dedup
            h = _stable_hash(doc.text)

            if h in seen:
                continue

            seen.add(h)

            validated.append(doc)

        except Exception as e:
            logger.warning(
                event="doc_validation_failed",
                index=i,
                error=str(e)
            )

    return validated


#  ROUTER 
def route_ingestion(
    file_path: str,
    session_id: str
) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    start = time.time()

    try:
        modality = detect_modality(file_path)

        logger.info(
            event="ingestion_start",
            modality=modality,
            file=file_path
        )

        handler = INGESTION_HANDLERS.get(modality)

        if not handler:
            raise ValueError("HANDLER_NOT_FOUND")

        docs = handler(file_path, session_id)

        docs = _validate_documents(docs, session_id)

        if not docs:
            raise ValueError("NO_VALID_DOCS")

        max_docs = getattr(settings, "MAX_INGESTED_DOCS", 5000)

        if len(docs) > max_docs:
            docs = docs[:max_docs]
            logger.warning(event="doc_limit_applied", limit=max_docs)

        latency = round(time.time() - start, 2)

        logger.info(
            event="ingestion_success",
            docs=len(docs),
            latency=latency
        )

        return docs

    except Exception as e:
        logger.error(
            event="ingestion_failed",
            file=file_path,
            error=str(e)
        )
        raise