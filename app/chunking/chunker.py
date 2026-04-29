import time
from typing import List, Set

from app.core.config import settings
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None


logger = get_logger(__name__)



# TEXT SPLITTER
def get_text_splitter():
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("INVALID CHUNK_OVERLAP")

    if RecursiveCharacterTextSplitter is None:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )



# NORMALIZATION
def _normalize_text(text: str) -> str:
    return " ".join(text.split())



# QUALITY FILTER
def _is_valid_chunk(text: str) -> bool:
    if not text:
        return False
    if len(text.strip()) < 10:
        return False
    return True



# STRUCTURED ROW DETECTION
def _is_structured_row(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    tokens = text.split()

    
    if len(tokens) >= 3:
        if tokens[0].isdigit() and tokens[-1].isdigit():
            return True

    # TOC-style detection
    if "section" in text.lower() and any(char.isdigit() for char in text):
        return True

    return False



# FALLBACK CHUNKING
def _fallback_chunk_text(text: str) -> List[str]:
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if _is_valid_chunk(chunk):
            chunks.append(chunk)

    return chunks



# MAIN CHUNKING FUNCTION
def chunk_text(text: str) -> List[str]:

    if not text or not text.strip():
        raise ValueError("CANNOT CHUNK EMPTY TEXT")

    text = _normalize_text(text)

    # STEP 1: Preserve structured rows (TOC / tables)
    lines = text.split("\n")
    structured_chunks = []

    for line in lines:
        if _is_structured_row(line):
            structured_chunks.append(line.strip())

    # Remove structured rows from main text
    remaining_lines = [l for l in lines if not _is_structured_row(l)]
    remaining_text = "\n".join(remaining_lines)

    # STEP 2: Truncate if needed
    if len(remaining_text) > settings.MAX_PROMPT_CHARS:
        logger.warning(
            "[Chunking][TRUNCATE] %s -> %s",
            len(remaining_text),
            settings.MAX_PROMPT_CHARS
        )
        remaining_text = remaining_text[:settings.MAX_PROMPT_CHARS]

    # STEP 3: Normal chunking
    splitter = get_text_splitter()

    chunks = (
        splitter.split_text(remaining_text)
        if splitter
        else _fallback_chunk_text(remaining_text)
    )

    # STEP 4: Combine structured + normal chunks
    chunks.extend(structured_chunks)

    # STEP 5: Filter invalid chunks
    chunks = [c for c in chunks if _is_valid_chunk(c)]

    if not chunks:
        raise ValueError("CHUNKING FAILED")

    # STEP 6: Limit total chunks
    if len(chunks) > settings.MAX_CHUNKS:
        logger.warning(
            "[Chunking][LIMIT] %s -> %s",
            len(chunks),
            settings.MAX_CHUNKS
        )
        chunks = chunks[:settings.MAX_CHUNKS]

    return chunks



# SINGLE CHUNK HANDLER
def _single_chunk_document(doc: IngestedDocument, parent_modality: str, content_type=None):
    cloned = doc.clone()

    structure = dict(cloned.structure or {})
    structure.update({
        "chunk_index": 0,
        "total_chunks": 1,
        "parent_modality": parent_modality,
    })

    if content_type:
        structure["content_type"] = content_type

    cloned.structure = structure
    cloned.chunk_id = 0

    return [cloned]



# TEXT DOCUMENT CHUNKING
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



# IMAGE CHUNKING
def _chunk_image_document(doc: IngestedDocument):

    if doc.subtype == "ocr" and len(doc.text) > settings.CHUNK_SIZE:
        return _chunk_text_document(doc)

    return _single_chunk_document(doc, "image", "semantic_description")



# AUDIO CHUNKING
def _chunk_audio_document(doc: IngestedDocument):
    return _single_chunk_document(doc, "audio", "speech_segment")



# VIDEO CHUNKING
def _chunk_video_document(doc: IngestedDocument):

    if doc.subtype == "speech":
        return _single_chunk_document(doc, "video", "video_speech")

    if doc.subtype == "frame":
        return _single_chunk_document(doc, "video", "video_frame")

    return _single_chunk_document(doc, "video")



# MAIN ENTRY
def chunk_documents(documents: List[IngestedDocument]) -> List[IngestedDocument]:

    if not documents:
        raise ValueError("NO DOCUMENTS PROVIDED")

    start = time.time()

    handlers = {
        "text": _chunk_text_document,
        "table": _chunk_text_document,
        "image": _chunk_image_document,
        "audio": _chunk_audio_document,
        "video": _chunk_video_document,
    }

    output = []
    seen: Set[str] = set()

    for doc in documents:
        handler = handlers.get(doc.modality)

        if not handler:
            logger.warning("[Chunking][UNKNOWN] %s", doc.modality)
            continue

        try:
            chunks = handler(doc)

            for c in chunks:
                key = c.text[:100]

                # Deduplication
                if key in seen:
                    continue

                seen.add(key)
                output.append(c)

        except Exception as e:
            logger.error("[Chunking][FAIL] %s", str(e))
            continue

    logger.info(
        "[Chunking][SUCCESS] in=%s | out=%s | latency=%.2fs",
        len(documents),
        len(output),
        time.time() - start
    )

    return output