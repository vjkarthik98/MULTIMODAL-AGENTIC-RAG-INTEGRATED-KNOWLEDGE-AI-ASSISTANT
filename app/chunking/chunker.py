import hashlib
import time
from typing import Dict, List, Set

from app.core.config import settings
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

logger = get_logger(__name__)


# SPLITTER

def get_text_splitter() -> object:
    if RecursiveCharacterTextSplitter is None:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


# HASH

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# NORMALIZE

def _normalize(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


# VALID

def _valid(text: str) -> bool:
    return bool(text and len(text.strip()) >= settings.CHUNK_MIN_SIZE)


# TOKEN ESTIMATE

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# STRUCTURED LINE DETECTION

def _is_structured(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    tokens = stripped.split()

    # NUMBERED LIST: "1. Item" or "1) Item"
    if tokens and tokens[0].rstrip(".):").isdigit():
        return True

    # TABLE OF CONTENTS: starts with digit, ends with digit (page number)
    if len(tokens) >= 3 and tokens[0].isdigit() and tokens[-1].isdigit():
        return True

    lower = stripped.lower()

    # SECTION / CHAPTER HEADERS
    if any(lower.startswith(kw) for kw in ("section", "chapter", "appendix", "part ")):
        return True

    # MARKDOWN HEADING
    if stripped.startswith("#"):
        return True

    return False


# FALLBACK SPLITTER (no langchain)

def _fallback(text: str) -> List[str]:
    size    = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    step    = max(size - overlap, 1)

    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + size].strip()
        if _valid(chunk):
            chunks.append(chunk)

    return chunks


# TEXT CHUNKING

def chunk_text(text: str) -> List[str]:

    if not text:
        raise ValueError("EMPTY_TEXT")

    text  = _normalize(text)
    lines = text.split("\n")

    structured = [l for l in lines if _is_structured(l)]
    main_lines = [l for l in lines if not _is_structured(l)]
    main       = "\n".join(main_lines)

    splitter = get_text_splitter()
    chunks   = splitter.split_text(main) if splitter and main.strip() else _fallback(main)

    # APPEND STRUCTURED LINES AS INDIVIDUAL CHUNKS
    chunks.extend(structured)

    chunks = [c.strip() for c in chunks if _valid(c)]

    if not chunks:
        raise ValueError("NO_CHUNKS_PRODUCED")

    if len(chunks) > settings.MAX_CHUNKS:
        chunks = chunks[:settings.MAX_CHUNKS]
        logger.warning(event="chunk_limit_applied", limit=settings.MAX_CHUNKS)

    return chunks


# SINGLE CHUNK WRAPPER

def _single(doc: IngestedDocument, content_type: str = None) -> List[IngestedDocument]:
    s = dict(doc.structure or {})
    s.update({
        "chunk_index":    0,
        "total_chunks":   1,
        "chunk_length":   len(doc.text),
        "parent_modality": doc.modality,
    })

    if content_type:
        s["content_type"] = content_type

    cloned            = doc.clone(structure=s)
    cloned.chunk_id   = 0

    return [cloned]


# TEXT DOC HANDLER

def _text_doc(doc: IngestedDocument) -> List[IngestedDocument]:
    try:
        chunks = chunk_text(doc.text)
        total  = len(chunks)

        return [
            doc.clone(
                text=c,
                chunk_id=i,
                structure={
                    **(doc.structure or {}),
                    "chunk_index":    i,
                    "total_chunks":   total,
                    "chunk_length":   len(c),
                    "token_estimate": _estimate_tokens(c),
                    "parent_modality": doc.modality,
                },
            )
            for i, c in enumerate(chunks)
        ]

    except Exception as e:
        logger.error(event="text_chunk_failed", modality=doc.modality, error=str(e))
        return [doc]


# IMAGE HANDLER

def _image_doc(doc: IngestedDocument) -> List[IngestedDocument]:
    if doc.subtype == "ocr" and len(doc.text) > settings.CHUNK_SIZE:
        return _text_doc(doc)
    return _single(doc, "image_semantic")


# AUDIO HANDLER

def _audio_doc(doc: IngestedDocument) -> List[IngestedDocument]:
    if len(doc.text) > settings.CHUNK_SIZE:
        return _text_doc(doc)
    return _single(doc, "audio_speech_segment")


# VIDEO HANDLER

def _video_doc(doc: IngestedDocument) -> List[IngestedDocument]:
    if doc.subtype == "speech":
        return _single(doc, "video_speech")
    if doc.subtype == "frame":
        return _single(doc, "video_frame")
    if doc.subtype == "ocr":
        return _text_doc(doc)
    return _single(doc)


# MODALITY STATS

def _modality_stats(docs: List[IngestedDocument]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for d in docs:
        stats[d.modality] = stats.get(d.modality, 0) + 1
    return stats


# MAIN

def chunk_documents(docs: List[IngestedDocument]) -> List[IngestedDocument]:

    if not docs:
        raise ValueError("NO_DOCUMENTS_PROVIDED")

    start = time.time()

    handlers = {
        "text":  _text_doc,
        "table": _text_doc,
        "image": _image_doc,
        "audio": _audio_doc,
        "video": _video_doc,
    }

    output: List[IngestedDocument] = []
    seen:   Set[str]               = set()
    skipped = 0

    for doc in docs:
        handler = handlers.get(doc.modality)

        if not handler:
            logger.warning(event="unknown_modality_skipped", modality=doc.modality)
            skipped += 1
            continue

        try:
            chunks = handler(doc)

            for c in chunks:
                h = _hash(c.text)

                if h in seen:
                    continue

                seen.add(h)
                output.append(c)

        except Exception as e:
            logger.error(
                event="chunking_error",
                modality=doc.modality,
                source=doc.source,
                error=str(e),
            )
            skipped += 1

    if not output:
        raise ValueError("NO_CHUNKS_PRODUCED_FROM_DOCUMENTS")

    total_tokens = sum(_estimate_tokens(d.text) for d in output)

    logger.info(
        event="chunking_success",
        input=len(docs),
        output=len(output),
        skipped=skipped,
        total_tokens_est=total_tokens,
        modality_breakdown=_modality_stats(output),
        latency=round(time.time() - start, 2),
    )

    return output