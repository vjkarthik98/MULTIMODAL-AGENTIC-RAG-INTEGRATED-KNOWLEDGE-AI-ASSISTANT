import hashlib
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import chardet

from app.chunking.chunker import chunk_text
from app.core.config import settings
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


# SUPPORTED EXTENSIONS

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".log", ".json", ".yaml", ".yml"}


# HASH

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ENCODING DETECTION

def _detect_encoding(file_path: Path) -> Tuple[str, float]:
    with open(file_path, "rb") as f:
        raw = f.read(10_000)

    result     = chardet.detect(raw)
    encoding   = result.get("encoding") or "utf-8"
    confidence = float(result.get("confidence") or 0.0)

    return encoding, confidence


# TEXT LOADING

def _load_text(file_path: Path) -> str:
    encoding, confidence = _detect_encoding(file_path)

    if confidence < 0.7:
        logger.warning(
            event="encoding_low_confidence",
            encoding=encoding,
            confidence=confidence,
            file=file_path.name,
        )

    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()

    except Exception:
        logger.warning(event="encoding_fallback_latin1", file=file_path.name)
        with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


# NORMALIZE

def _normalize_text(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


# LANGUAGE DETECTION

def _detect_language(text: str) -> Optional[str]:
    try:
        from langdetect import detect
        return detect(text[:2000])
    except Exception:
        return None


# SUBTYPE DETECTION

def _detect_subtype(chunk: str) -> str:
    lines      = [l.strip() for l in chunk.split("\n") if l.strip()]
    if not lines:
        return "paragraph"

    first_line = lines[0]

    # MARKDOWN HEADING
    if first_line.startswith("#"):
        return "heading"

    # SHORT SINGLE-LINE LIKELY HEADING
    if len(lines) == 1 and len(first_line.split()) <= 8 and not first_line.endswith("."):
        return "heading"

    return "paragraph"


# QUALITY SCORE

def _quality_score(chunk: str) -> float:
    length     = len(chunk)
    word_count = len(chunk.split())

    if length < settings.CHUNK_MIN_SIZE:
        return 0.1

    if length < 100 or word_count < 10:
        return 0.3

    if length < 300 or word_count < 30:
        return 0.6

    return 1.0


# MAIN

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    ext = path.suffix.lower()
    if ext and ext not in SUPPORTED_TEXT_EXTENSIONS:
        logger.warning(event="text_ingest_unknown_ext", ext=ext, file=path.name)

    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError("EMPTY_FILE")

    if file_size > settings.MAX_FILE_SIZE_TEXT:
        raise ValueError(
            f"FILE_TOO_LARGE: {file_size} bytes exceeds {settings.MAX_FILE_SIZE_TEXT} bytes"
        )

    start = time.time()

    logger.info(
        event="text_ingest_start",
        file=path.name,
        size=file_size,
        session_id=session_id,
    )

    try:
        # LOAD AND NORMALIZE
        raw_text = _load_text(path)
        text     = _normalize_text(raw_text)

        if not text:
            raise ValueError("EMPTY_TEXT_AFTER_NORMALIZE")

        if len(text) < 50:
            raise ValueError("TEXT_TOO_SHORT")

        # METADATA
        file_hash   = _file_hash(file_path)
        doc_id      = str(uuid.uuid4())
        source_name = path.name
        source_path = str(path.resolve())
        line_count  = text.count("\n") + 1
        word_count  = len(text.split())
        language    = _detect_language(text)

        # CHUNKING
        chunks = chunk_text(text)

        if not chunks:
            raise ValueError("NO_CHUNKS_PRODUCED")

        if len(chunks) > settings.MAX_CHUNKS:
            logger.warning(
                event="chunk_limit_applied",
                original=len(chunks),
                limited=settings.MAX_CHUNKS,
                file=path.name,
            )
            chunks = chunks[:settings.MAX_CHUNKS]

        total_chunks = len(chunks)
        documents: List[IngestedDocument] = []

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()

            if not chunk:
                continue

            if len(chunk) < settings.CHUNK_MIN_SIZE:
                continue

            quality  = _quality_score(chunk)
            subtype  = _detect_subtype(chunk)

            doc = IngestedDocument(
                text=chunk,
                modality="text",
                subtype=subtype,
                source_type="file",
                source=source_name,
                chunk_id=i,
                structure={
                    "doc_id":          doc_id,
                    "session_id":      session_id,
                    "file_hash":       file_hash,
                    "source_path":     source_path,
                    "chunk_index":     i,
                    "total_chunks":    total_chunks,
                    "chunk_length":    len(chunk),
                    "language":        language,
                    "content_type":    "text_chunk",
                    "ingestion_time":  time.time(),
                },
                extra_metadata={
                    "modality_weight":    1.0,
                    "importance_score":   quality,
                    "data_quality_score": quality,
                },
            ).finalize()

            documents.append(doc)

        if not documents:
            raise ValueError("NO_VALID_DOCUMENTS_AFTER_FILTERING")

        avg_chunk_length = round(sum(len(d.text) for d in documents) / len(documents), 1)
        latency          = round(time.time() - start, 2)

        logger.info(
            event="text_ingest_success",
            file=path.name,
            docs=len(documents),
            total_chunks=total_chunks,
            avg_chunk_length=avg_chunk_length,
            line_count=line_count,
            word_count=word_count,
            language=language,
            latency=latency,
            session_id=session_id,
        )

        return documents

    except Exception as e:
        logger.error(
            event="text_ingest_failed",
            file=path.name,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 2),
        )
        raise