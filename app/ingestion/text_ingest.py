import hashlib
import time
import uuid
from pathlib import Path
from typing import List

import chardet

from app.core.config import settings
from app.chunking.chunker import chunk_text
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  HASH 
def _generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


#  ENCODING 
def _detect_encoding(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(10000)

    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def _load_text(file_path: Path) -> str:
    try:
        encoding = _detect_encoding(file_path)

        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            text = f.read()

        return text

    except Exception:
        logger.warning(event="encoding_fallback_latin1")

        with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


#  CLEAN 
def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


#  QUALITY 
def _compute_quality_score(text: str) -> float:
    length = len(text)

    if length < 50:
        return 0.2

    if length < 200:
        return 0.5

    return 1.0


#  MAIN 
def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} NOT FOUND")

    start = time.time()

    try:
        logger.info(event="text_ingest_start", file=str(path), session_id=session_id)

        # FILE SIZE VALIDATION
        size_mb = path.stat().st_size / (1024 * 1024)

        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise ValueError(f"FILE_TOO_LARGE_{size_mb:.2f}MB")

        # LOAD TEXT
        raw_text = _load_text(path)
        text = _normalize_text(raw_text)

        if not text:
            raise ValueError("EMPTY_TEXT")

        # CRITICAL FIX: NO TRUNCATION HERE
        if len(text) < 50:
            raise ValueError("TEXT_TOO_SHORT")

        # IDS
        file_hash = _generate_file_hash(file_path)
        doc_id = str(uuid.uuid4())

        # CHUNKING
        chunks = chunk_text(text)

        if not chunks:
            raise ValueError("NO_CHUNKS")

        # LIMIT
        if len(chunks) > settings.MAX_CHUNKS:
            logger.warning(
                event="chunk_limit_applied",
                original=len(chunks),
                limited=settings.MAX_CHUNKS
            )
            chunks = chunks[:settings.MAX_CHUNKS]

        source_name = path.name
        source_path = str(path.resolve())

        documents: List[IngestedDocument] = []

        for i, chunk in enumerate(chunks):

            chunk = chunk.strip()
            if not chunk:
                continue

            quality_score = _compute_quality_score(chunk)

            doc = IngestedDocument(
                text=chunk,
                modality="text",
                subtype="paragraph",
                source_type="file",
                source=source_name,
                chunk_id=i,

                structure={
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "file_hash": file_hash,
                    "source_path": source_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_length": len(chunk),
                    "content_type": "text_chunk",
                    "ingestion_time": time.time(),
                },

                extra_metadata={
                    "modality_weight": 1.0,
                    "importance_score": quality_score,
                    "data_quality_score": quality_score,
                },
            ).finalize()

            documents.append(doc)

        if not documents:
            raise ValueError("NO_VALID_DOCUMENTS")

        latency = round(time.time() - start, 2)

        logger.info(
            event="text_ingest_success",
            session_id=session_id,
            docs=len(documents),
            latency=latency
        )

        return documents

    except Exception as e:
        logger.error(
            event="text_ingest_failed",
            session_id=session_id,
            file=file_path,
            error=str(e)
        )
        raise