import hashlib
import re
import time
import uuid
from typing import Dict, List, Optional, Set

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


def _token_windows(words: List[str], max_tokens: int, overlap_ratio: float) -> List[str]:
    if not words:
        return []
    max_words = max(1, int(max_tokens * 0.75))
    overlap = max(0, int(max_words * overlap_ratio))
    step = max(1, max_words - overlap)
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)]


def _fingerprint(text: str) -> int:
    bits = settings.SIMHASH_BITS
    vector = [0] * bits
    for token in re.findall(r"\w+", text.lower()):
        value = hash(token)
        for i in range(bits):
            vector[i] += 1 if value & (1 << i) else -1
    fp = 0
    for i, score in enumerate(vector):
        if score > 0:
            fp |= 1 << i
    return fp


def _near_duplicate(left: int, right: int) -> bool:
    similarity = 1.0 - ((left ^ right).bit_count() / max(1, settings.SIMHASH_BITS))
    return similarity >= settings.NEAR_DUPLICATE_THRESHOLD


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
    words = text.split()
    token_chunks = _token_windows(words, settings.CHUNK_MAX_TOKENS, settings.CHUNK_OVERLAP_RATIO)
    return [chunk for chunk in token_chunks if _valid(chunk)]


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
    parent_id = s.get("parent_id") or s.get("doc_id") or str(uuid.uuid4())
    s.update({
        "chunk_index":    0,
        "total_chunks":   1,
        "chunk_length":   len(doc.text),
        "token_estimate": _estimate_tokens(doc.text),
        "parent_id":      parent_id,
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
        parent_id = (doc.structure or {}).get("doc_id") or str(uuid.uuid4())

        return [
            doc.clone(
                text=c,
                chunk_id=i,
                structure={
                    **(doc.structure or {}),
                    "parent_id":       parent_id,
                    "chunk_index":    i,
                    "total_chunks":   total,
                    "chunk_length":   len(c),
                    "token_estimate": _estimate_tokens(c),
                    "overlap_ratio":   settings.CHUNK_OVERLAP_RATIO,
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
    chunks = _single(doc, "image_semantic")
    for chunk in chunks:
        if (doc.structure or {}).get("context_text"):
            chunk.structure["cross_modal_link"] = (doc.structure or {}).get("doc_id")
    return chunks


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
        "text":     _text_doc,
        "pdf":      _text_doc,
        "word":     _text_doc,
        "excel":    _text_doc,
        "document": _text_doc,
        "table":    _text_doc,
        "image":    _image_doc,
        "audio":    _audio_doc,
        "video":    _video_doc,
    }

    output: List[IngestedDocument] = []
    seen:   Set[str]               = set()
    seen_fp: List[int]             = []
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
                fp = _fingerprint(c.text)
                if any(_near_duplicate(fp, existing) for existing in seen_fp):
                    continue

                seen.add(h)
                seen_fp.append(fp)
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


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/chunking/chunker.py -v
# ============================================================

def test_hierarchical_chunks_have_parent_id() -> None:
    doc = IngestedDocument(text="Heading\n\n" + "retrieval metadata chunking " * 80, modality="text").finalize()
    chunks = chunk_documents([doc])
    assert chunks
    assert all("parent_id" in chunk.structure for chunk in chunks)


def test_cross_modal_chunks_linked() -> None:
    doc = IngestedDocument(
        text="A useful image caption for retrieval",
        modality="image",
        subtype="caption",
        structure={"doc_id": "image-parent", "context_text": "nearby text"},
    ).finalize()
    chunks = chunk_documents([doc])
    assert chunks[0].structure.get("cross_modal_link") == "image-parent"


def test_overlap_within_token_budget() -> None:
    chunks = _fallback("word " * 2000)
    assert chunks
    assert all(_estimate_tokens(chunk) <= settings.CHUNK_MAX_TOKENS + 5 for chunk in chunks)


def test_near_duplicate_chunks_skipped() -> None:
    text = "retrieval metadata chunking quality " * 80
    docs = [
        IngestedDocument(text=text, modality="text").finalize(),
        IngestedDocument(text=text, modality="text").finalize(),
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 1
    assert len(chunks) < len(docs) * 3
