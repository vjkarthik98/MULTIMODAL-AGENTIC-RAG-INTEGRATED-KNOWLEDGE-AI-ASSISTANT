import hashlib
import os
import time
import uuid

from app.ingestion.chunking import DEFAULT_CHUNK_OVERLAP, chunk_text
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 5
MAX_CHUNKS = 1000


def _get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def _generate_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as file_handle:
        return hashlib.md5(file_handle.read()).hexdigest()


def _find_chunk_offset(text: str, chunk: str, search_start: int) -> tuple[int, int]:
    chunk_start = text.find(chunk, max(0, search_start))
    if chunk_start < 0:
        chunk_start = max(0, search_start)
    chunk_end = min(chunk_start + len(chunk), len(text))
    return chunk_start, chunk_end


def ingest(file_path: str, session_id: str = "default") -> list[IngestedDocument]:
    start_time = time.time()

    if not session_id:
        raise ValueError("session_id is required")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info("[TextIngest][START] session_id=%s | file=%s", session_id, file_path)

        file_size = _get_file_size_mb(file_path)
        if file_size > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large ({file_size:.2f} MB). Limit={MAX_FILE_SIZE_MB}MB"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                text = file_handle.read()
        except UnicodeDecodeError:
            logger.warning("[TextIngest][ENCODING_FALLBACK] session_id=%s", session_id)
            with open(file_path, "r", encoding="latin-1") as file_handle:
                text = file_handle.read()

        if not text.strip():
            raise ValueError("Empty text file")

        file_hash = _generate_file_hash(file_path)
        doc_id = str(uuid.uuid4())

        try:
            chunks = chunk_text(text, max_chunks=MAX_CHUNKS)
        except Exception as exc:
            logger.error("[TextIngest][CHUNK_FAIL] session_id=%s | error=%s", session_id, exc)
            chunks = [text.strip()]

        source_name = os.path.basename(file_path)
        source_path = os.path.abspath(file_path)
        total_chunks = len(chunks)
        search_start = 0
        documents: list[IngestedDocument] = []

        for index, chunk in enumerate(chunks):
            chunk_start, chunk_end = _find_chunk_offset(text, chunk, search_start)
            line_start = text[:chunk_start].count("\n") + 1
            line_end = text[:chunk_end].count("\n") + 1

            documents.append(
                IngestedDocument(
                    text=chunk,
                    modality="text",
                    subtype="paragraph",
                    source_type="file",
                    source=source_name,
                    chunk_id=index,
                    structure={
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "file_hash": file_hash,
                        "source_path": source_path,
                        "chunk_index": index,
                        "total_chunks": total_chunks,
                        "char_start": chunk_start,
                        "char_end": chunk_end,
                        "line_start": line_start,
                        "line_end": line_end,
                        "chunk_length": len(chunk),
                        "content_type": "text_chunk",
                        "ingestion_time": time.time(),
                    },
                    extra_metadata={
                        "modality_weight": 1.0,
                        "importance_score": 1.0,
                        "text_density": len(chunk) / max(1, chunk.count(" ") + 1),
                    },
                )
            )

            search_start = max(chunk_end - DEFAULT_CHUNK_OVERLAP, chunk_start + 1)

        latency = time.time() - start_time
        logger.info(
            "[TextIngest][SUCCESS] session_id=%s | docs=%s | latency=%.2fs",
            session_id,
            len(documents),
            latency,
        )
        return documents

    except Exception as exc:
        logger.error(
            "[TextIngest][FAILED] session_id=%s | file=%s | error=%s",
            session_id,
            file_path,
            exc,
        )
        raise
