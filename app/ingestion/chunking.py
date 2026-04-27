import time
from typing import List

from app.core.config import settings
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None


logger = get_logger(__name__)


def get_text_splitter():
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Invalid CHUNK_OVERLAP")

    if RecursiveCharacterTextSplitter is None:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _fallback_chunk_text(text: str) -> List[str]:
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_text(text: str) -> List[str]:
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    # Global safety truncation (critical)
    if len(text) > settings.MAX_PROMPT_CHARS:
        logger.warning(
            "[Chunking][TRUNCATE] %s -> %s",
            len(text),
            settings.MAX_PROMPT_CHARS
        )
        text = text[:settings.MAX_PROMPT_CHARS]

    splitter = get_text_splitter()

    chunks = (
        splitter.split_text(text)
        if splitter
        else _fallback_chunk_text(text)
    )

    if not chunks:
        raise ValueError("Chunking failed")

    if len(chunks) > settings.MAX_CHUNKS:
        logger.warning(
            "[Chunking][LIMIT] %s -> %s",
            len(chunks),
            settings.MAX_CHUNKS
        )
        chunks = chunks[:settings.MAX_CHUNKS]

    return chunks


def _single_chunk_document(doc: IngestedDocument, parent_modality: str, content_type=None):
    cloned = doc.clone()

    structure = dict(cloned.structure or {})
    structure.update({
        "chunk_index": structure.get("chunk_index", 0),
        "total_chunks": structure.get("total_chunks", 1),
        "parent_modality": parent_modality,
    })

    if content_type:
        structure["content_type"] = content_type

    cloned.structure = structure
    cloned.chunk_id = structure["chunk_index"]

    return [cloned]


def _chunk_text_document(doc: IngestedDocument) -> List[IngestedDocument]:
    try:
        chunks = chunk_text(doc.text)
        total = len(chunks)

        return [
            doc.clone(
                text=chunk,
                chunk_id=i,
                structure={
                    **(doc.structure or {}),
                    "chunk_index": i,
                    "total_chunks": total,
                    "chunk_length": len(chunk),
                    "parent_modality": doc.modality,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    except Exception as e:
        logger.error("[Chunking][TEXT_FAIL] %s", str(e))
        return [doc]


def _chunk_image_document(doc: IngestedDocument) -> List[IngestedDocument]:
    if doc.subtype == "ocr" and len(doc.text) > settings.CHUNK_SIZE:
        return _chunk_text_document(doc)

    return _single_chunk_document(doc, "image", "semantic_description")


def _chunk_audio_document(doc: IngestedDocument):
    structure = doc.structure or {}

    return _single_chunk_document(
        doc,
        "audio",
        "speech_segment"
    )


def _chunk_video_document(doc: IngestedDocument):
    if doc.subtype == "speech":
        return _single_chunk_document(doc, "video", "video_speech")

    if doc.subtype == "frame":
        return _single_chunk_document(doc, "video", "video_frame")

    return _single_chunk_document(doc, "video")


def chunk_documents(documents: List[IngestedDocument]) -> List[IngestedDocument]:
    if not documents:
        raise ValueError("No documents provided")

    start = time.time()

    handlers = {
        "text": _chunk_text_document,
        "table": _single_chunk_document,
        "image": _chunk_image_document,
        "audio": _chunk_audio_document,
        "video": _chunk_video_document,
    }

    output = []

    for doc in documents:
        handler = handlers.get(doc.modality)

        if not handler:
            logger.warning("[Chunking][UNKNOWN] %s", doc.modality)
            output.append(doc)
            continue

        try:
            if doc.modality == "table":
                output.extend(handler(doc, "table"))
            else:
                output.extend(handler(doc))
        except Exception as e:
            logger.error("[Chunking][FAIL] %s", str(e))
            output.append(doc)

    logger.info(
        "[Chunking][SUCCESS] in=%s | out=%s | latency=%.2fs",
        len(documents),
        len(output),
        time.time() - start
    )

    return output