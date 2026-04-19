import time
from typing import List

from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - optional dependency fallback
    RecursiveCharacterTextSplitter = None


logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
MAX_CHUNKS = 2000


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if RecursiveCharacterTextSplitter is None:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _fallback_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    text_length = len(text)
    separators = ["\n\n", "\n", ". ", " "]

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            search_start = max(start, end - max(chunk_size // 4, 50))
            for separator in separators:
                boundary = text.rfind(separator, search_start, end)
                if boundary > start:
                    end = boundary + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS,
) -> List[str]:
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = (
        splitter.split_text(text)
        if splitter is not None
        else _fallback_chunk_text(text, chunk_size, chunk_overlap)
    )

    if not chunks:
        raise ValueError("Chunking failed: no chunks generated")

    if len(chunks) > max_chunks:
        logger.warning(
            "[Chunking][LIMIT] Truncating chunks from %s to %s",
            len(chunks),
            max_chunks,
        )
        chunks = chunks[:max_chunks]

    logger.info(
        "[Chunking][SUCCESS] chunks=%s | size=%s | overlap=%s",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _single_chunk_document(
    doc: IngestedDocument,
    parent_modality: str,
    content_type: str | None = None,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> List[IngestedDocument]:
    cloned = doc.clone()
    structure = dict(cloned.structure or {})
    index = structure.get("chunk_index", 0)
    total = structure.get("total_chunks", 1)

    if chunk_index is not None:
        index = chunk_index
    if total_chunks is not None:
        total = total_chunks

    structure.update(
        {
            "chunk_index": index,
            "total_chunks": total,
            "parent_modality": parent_modality,
        }
    )

    if content_type:
        structure["content_type"] = content_type

    cloned.structure = structure
    cloned.chunk_id = index
    return [cloned]


def _chunk_text_document(doc: IngestedDocument) -> List[IngestedDocument]:
    try:
        chunks = chunk_text(doc.text)
        chunked_docs: List[IngestedDocument] = []
        total_chunks = len(chunks)

        for index, chunk in enumerate(chunks):
            structure = dict(doc.structure or {})
            structure.update(
                {
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "chunk_length": len(chunk),
                    "parent_modality": doc.modality,
                }
            )

            chunked_doc = doc.clone(
                text=chunk,
                chunk_id=index,
                structure=structure,
            )
            chunked_docs.append(chunked_doc)

        return chunked_docs

    except Exception as exc:
        logger.error("[Chunking][TEXT_FAIL] %s", exc)
        return [doc]


def _chunk_table_document(doc: IngestedDocument) -> List[IngestedDocument]:
    return _single_chunk_document(doc, parent_modality="table")


def _chunk_image_document(doc: IngestedDocument) -> List[IngestedDocument]:
    if doc.subtype == "ocr" and len(doc.text) >= 300:
        try:
            chunks = chunk_text(doc.text)
            total_chunks = len(chunks)
            return [
                doc.clone(
                    text=chunk,
                    chunk_id=index,
                    structure={
                        **dict(doc.structure or {}),
                        "chunk_index": index,
                        "total_chunks": total_chunks,
                        "parent_modality": "image",
                        "content_type": "ocr_chunk",
                    },
                )
                for index, chunk in enumerate(chunks)
            ]
        except Exception as exc:
            logger.error("[Chunking][IMAGE_FAIL] %s", exc)
            return [doc]

    content_type = "ocr_short" if doc.subtype == "ocr" else "semantic_description"
    return _single_chunk_document(doc, parent_modality="image", content_type=content_type)


def _chunk_audio_document(doc: IngestedDocument) -> List[IngestedDocument]:
    structure = dict(doc.structure or {})
    return _single_chunk_document(
        doc,
        parent_modality="audio",
        content_type="speech_segment",
        chunk_index=structure.get("segment_index", 0),
        total_chunks=structure.get("total_segments", 1),
    )


def _chunk_video_document(doc: IngestedDocument) -> List[IngestedDocument]:
    structure = dict(doc.structure or {})

    if doc.subtype == "speech":
        return _single_chunk_document(
            doc,
            parent_modality="video",
            content_type="video_speech",
            chunk_index=structure.get("segment_index", 0),
            total_chunks=structure.get("total_segments", 1),
        )

    if doc.subtype == "frame":
        return _single_chunk_document(
            doc,
            parent_modality="video",
            content_type="video_frame",
            chunk_index=structure.get("frame_index", 0),
            total_chunks=structure.get("total_frames", 1),
        )

    return _single_chunk_document(doc, parent_modality="video")


def chunk_documents(documents: List[IngestedDocument]) -> List[IngestedDocument]:
    if not documents:
        raise ValueError("No documents provided for chunking")

    start_time = time.time()
    handlers = {
        "text": _chunk_text_document,
        "table": _chunk_table_document,
        "image": _chunk_image_document,
        "audio": _chunk_audio_document,
        "video": _chunk_video_document,
    }

    all_chunks: List[IngestedDocument] = []

    for doc in documents:
        handler = handlers.get(doc.modality)
        if handler is None:
            logger.warning("[Chunking][UNKNOWN_MODALITY] %s", doc.modality)
            all_chunks.append(doc)
            continue

        try:
            all_chunks.extend(handler(doc))
        except Exception as exc:
            logger.error("[Chunking][DOC_FAIL] %s", exc)
            all_chunks.append(doc)

    latency = time.time() - start_time
    logger.info(
        "[Chunking][SUCCESS] input_docs=%s | output_chunks=%s | latency=%.2fs",
        len(documents),
        len(all_chunks),
        latency,
    )
    return all_chunks
