import hashlib
import time
import uuid
from pathlib import Path
from typing import List

from app.core.config import settings
from app.chunking.chunker import chunk_text
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


logger = get_logger(__name__)


# GENERATE FILE HASH FOR DEDUPLICATION
def _generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# LOAD TEXT WITH SAFE ENCODING HANDLING
def _load_text(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except UnicodeDecodeError:
        logger.warning("[TextIngest] ENCODING FALLBACK TO LATIN-1")

        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


# MAIN INGEST FUNCTION
def ingest(file_path: str, session_id: str = "default") -> List[IngestedDocument]:

    # VALIDATE SESSION ID
    if not session_id:
        raise ValueError("SESSION_ID REQUIRED")

    path = Path(file_path)

    # VALIDATE FILE EXISTENCE
    if not path.exists():
        raise FileNotFoundError(f"{file_path} NOT FOUND")

    start = time.time()

    try:
        logger.info("[TextIngest][START] session_id=%s | file=%s", session_id, file_path)

        # VALIDATE FILE SIZE
        size_mb = path.stat().st_size / (1024 * 1024)

        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise ValueError(f"FILE TOO LARGE: {size_mb:.2f}MB")

        # LOAD FILE CONTENT
        text = _load_text(path)

        # VALIDATE EMPTY CONTENT
        if not text or not text.strip():
            raise ValueError("EMPTY TEXT FILE")

        # GLOBAL TEXT TRUNCATION (SAFETY LIMIT)
        if len(text) > settings.MAX_PROMPT_CHARS:
            logger.warning("[TextIngest] TEXT TRUNCATED DUE TO SIZE LIMIT")
            text = text[:settings.MAX_PROMPT_CHARS]

        # GENERATE IDENTIFIERS
        file_hash = _generate_file_hash(file_path)
        doc_id = str(uuid.uuid4())

        # APPLY CHUNKING
        chunks = chunk_text(text)

        # VALIDATE CHUNKS
        if not chunks:
            raise ValueError("NO CHUNKS GENERATED")

        # ENFORCE MAX CHUNK LIMIT
        if len(chunks) > settings.MAX_CHUNKS:
            logger.warning(
                "[TextIngest] CHUNK LIMIT APPLIED %s -> %s",
                len(chunks),
                settings.MAX_CHUNKS
            )
            chunks = chunks[:settings.MAX_CHUNKS]

        source_name = path.name
        source_path = str(path.resolve())

        documents: List[IngestedDocument] = []

        # BUILD DOCUMENT OBJECTS
        for i, chunk in enumerate(chunks):

            # SKIP EMPTY CHUNKS
            if not chunk.strip():
                continue

            doc = IngestedDocument(
                text=chunk,
                modality="text",
                subtype="paragraph",
                source_type="file",
                source=source_name,
                chunk_id=i,

                # STRUCTURED METADATA
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

                # EXTRA METADATA FOR RETRIEVAL SCORING
                extra_metadata={
                    "modality_weight": 1.0,
                    "importance_score": 1.0,
                },
            ).finalize()

            documents.append(doc)

        # FINAL VALIDATION
        if not documents:
            raise ValueError("NO VALID DOCUMENTS CREATED")

        latency = round(time.time() - start, 2)

        logger.info(
            "[TextIngest][SUCCESS] session_id=%s | docs=%s | latency=%ss",
            session_id,
            len(documents),
            latency
        )

        return documents

    except Exception as e:
        logger.error(
            "[TextIngest][FAILED] session_id=%s | file=%s | error=%s",
            session_id,
            file_path,
            str(e)
        )
        raise