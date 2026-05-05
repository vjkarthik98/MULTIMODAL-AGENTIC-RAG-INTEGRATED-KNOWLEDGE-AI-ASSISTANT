import hashlib
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


#  SPLITTER 
def get_text_splitter():
    if RecursiveCharacterTextSplitter is None:
        return None

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


#  HASH 
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


#  NORMALIZE 
def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


#  VALID 
def _valid(text: str) -> bool:
    return bool(text and len(text.strip()) >= 20)


#  STRUCTURE 
def _is_structured(line: str) -> bool:
    tokens = line.strip().split()
    if len(tokens) >= 3 and tokens[0].isdigit() and tokens[-1].isdigit():
        return True
    if "section" in line.lower():
        return True
    return False


#  FALLBACK 
def _fallback(text: str) -> List[str]:
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    step = size - overlap

    return [
        text[i:i + size].strip()
        for i in range(0, len(text), step)
        if _valid(text[i:i + size])
    ]


#  TEXT CHUNK 
def chunk_text(text: str) -> List[str]:

    if not text:
        raise ValueError("EMPTY_TEXT")

    text = _normalize(text)

    lines = text.split("\n")

    structured = [l for l in lines if _is_structured(l)]
    main = "\n".join([l for l in lines if not _is_structured(l)])

    splitter = get_text_splitter()

    chunks = splitter.split_text(main) if splitter else _fallback(main)

    chunks.extend(structured)

    chunks = [c.strip() for c in chunks if _valid(c)]

    if not chunks:
        raise ValueError("NO_CHUNKS")

    if len(chunks) > settings.MAX_CHUNKS:
        chunks = chunks[:settings.MAX_CHUNKS]
        logger.warning(event="chunk_limit_applied")

    return chunks


#  SINGLE 
def _single(doc: IngestedDocument, content_type=None):
    d = doc.clone()

    s = dict(d.structure or {})
    s.update({
        "chunk_index": 0,
        "total_chunks": 1,
        "parent_modality": doc.modality,
    })

    if content_type:
        s["content_type"] = content_type

    d.structure = s
    d.chunk_id = 0

    return [d]


#  TEXT DOC 
def _text_doc(doc: IngestedDocument) -> List[IngestedDocument]:

    try:
        chunks = chunk_text(doc.text)
        total = len(chunks)

        return [
            doc.clone(
                text=c,
                chunk_id=i,
                structure={
                    **(doc.structure or {}),
                    "chunk_index": i,
                    "total_chunks": total,
                    "chunk_length": len(c),
                    "parent_modality": doc.modality,
                },
            )
            for i, c in enumerate(chunks)
        ]

    except Exception as e:
        logger.error(event="text_chunk_failed", error=str(e))
        return [doc]


#  IMAGE 
def _image_doc(doc: IngestedDocument):
    if doc.subtype == "ocr" and len(doc.text) > settings.CHUNK_SIZE:
        return _text_doc(doc)
    return _single(doc, "image_semantic")


#  AUDIO 
def _audio_doc(doc: IngestedDocument):
    return _single(doc, "audio_segment")


#  VIDEO 
def _video_doc(doc: IngestedDocument):
    if doc.subtype == "speech":
        return _single(doc, "video_speech")
    if doc.subtype == "frame":
        return _single(doc, "video_frame")
    return _single(doc)


#  MAIN 
def chunk_documents(docs: List[IngestedDocument]) -> List[IngestedDocument]:

    if not docs:
        raise ValueError("NO_DOCUMENTS")

    start = time.time()

    handlers = {
        "text": _text_doc,
        "table": _text_doc,
        "image": _image_doc,
        "audio": _audio_doc,
        "video": _video_doc,
    }

    output = []
    seen: Set[str] = set()

    for doc in docs:
        handler = handlers.get(doc.modality)

        if not handler:
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
            logger.error(event="chunking_error", error=str(e))

    logger.info(
        event="chunking_success",
        input=len(docs),
        output=len(output),
        latency=round(time.time() - start, 2)
    )

    return output