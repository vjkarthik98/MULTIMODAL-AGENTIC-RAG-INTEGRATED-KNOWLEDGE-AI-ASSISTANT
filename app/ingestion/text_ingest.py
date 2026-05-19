import asyncio
import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chardet
import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.ingestion.schema import IngestedDocument
from app.ingestion.text_repair import (
    detect_error_markers,
    extract_version,
    has_title_mismatch,
    is_placeholder,
    normalize_ocr_noise,
    recover_whitespace,
    repair_text,
    strip_footnotes,
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_ingest_duration = Histogram(
    "text_ingest_duration_seconds",
    "Text ingestion duration",
    ["status"],
)
_ingest_errors = Counter(
    "text_ingest_errors_total",
    "Text ingestion errors by type",
    ["error_type"],
)
_pii_redacted = Counter(
    "pii_entities_redacted_total",
    "PII entities redacted during text ingestion",
    ["entity_type"],
)

# SUPPORTED EXTENSIONS
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".log",
    ".json", ".yaml", ".yml",
}

# BINARY MAGIC BYTES — REJECT IF FOUND AT FILE START
BINARY_MAGIC_SIGNATURES = [
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF8",
    b"%PDF",
    b"PK\x03\x04",
    b"\x1f\x8b",
    b"BM",
    b"ID3",
    b"\x00\x00\x00",
    b"RIFF",
    b"\x7fELF",
]

# SEMAPHORE — CAP CONCURRENT TEXT INGESTION
_semaphore = asyncio.Semaphore(5)


# SHA-256 HASH

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# SIMHASH FOR NEAR-DEDUP

def _simhash(text: str) -> int:
    tokens = text.lower().split()
    v = [0] * 64
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _simhash_distance(h1: int, h2: int) -> int:
    x = h1 ^ h2
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


# BINARY FILE DETECTION VIA MAGIC BYTES

def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        for sig in BINARY_MAGIC_SIGNATURES:
            if header.startswith(sig):
                return True
        # NULL BYTE CHECK — STRONG BINARY INDICATOR
        if b"\x00" in header:
            return True
        return False
    except Exception:
        return False


# BOM DETECTION AND STRIPPING

def _strip_bom(text: str) -> str:
    boms = ["\ufeff", "\ufffe", "\ufeff"]
    for bom in boms:
        if text.startswith(bom):
            return text[len(bom):]
    return text


# NULL BYTE STRIPPING

def _strip_null_bytes(text: str) -> Tuple[str, int]:
    count = text.count("\x00")
    return text.replace("\x00", ""), count


# ENCODING DETECTION

def _detect_encoding(path: Path) -> Tuple[str, float]:
    with open(path, "rb") as f:
        raw = f.read(32768)
    result     = chardet.detect(raw)
    encoding   = result.get("encoding") or "utf-8"
    confidence = float(result.get("confidence") or 0.0)
    return encoding, confidence


# TEXT LOADING WITH ENCODING FALLBACK

def _load_text(path: Path) -> str:
    encoding, confidence = _detect_encoding(path)

    if confidence < 0.7:
        logger.warning(
            "encoding_low_confidence",
            encoding=encoding,
            confidence=round(confidence, 3),
            file=path.name,
        )

    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    except Exception:
        logger.warning("encoding_fallback_latin1", file=path.name)
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


# STREAMING LOAD FOR LARGE FILES — LINE BY LINE

def _load_text_streaming(path: Path, max_bytes: int) -> str:
    encoding, _ = _detect_encoding(path)
    parts: List[str] = []
    total = 0
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                # SPLIT LINES EXCEEDING 100K CHARS AT WORD BOUNDARY
                if len(line) > 100_000:
                    words = line.split(" ")
                    chunk = ""
                    for word in words:
                        if len(chunk) + len(word) > 100_000:
                            parts.append(chunk)
                            total += len(chunk)
                            chunk = word + " "
                        else:
                            chunk += word + " "
                    if chunk:
                        parts.append(chunk)
                        total += len(chunk)
                else:
                    parts.append(line)
                    total += len(line)
                if total >= max_bytes:
                    logger.warning(
                        "streaming_truncated_at_limit",
                        file=path.name,
                        bytes_read=total,
                    )
                    break
    except Exception as exc:
        logger.warning("streaming_load_failed", file=path.name, error=str(exc))
    return "".join(parts)


# UNICODE NORMALIZATION

def _normalize_text(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


# LANGUAGE DETECTION — ENSEMBLE

def _detect_language(text: str) -> Optional[str]:
    sample = text[:3000]
    results: List[str] = []

    # LINGUA
    try:
        from lingua import LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_all_languages().build()
        lang     = detector.detect_language_of(sample)
        if lang:
            results.append(lang.iso_code_639_1.name.lower())
    except Exception:
        pass

    if not results:
        return None

    # RETURN MOST COMMON RESULT
    from collections import Counter as _Counter
    return _Counter(results).most_common(1)[0][0]


# RTL LANGUAGE DETECTION

def _is_rtl(language: Optional[str]) -> bool:
    RTL_LANGS = {"ar", "he", "fa", "ur", "yi", "dv", "ku", "ps"}
    return (language or "").lower() in RTL_LANGS


# PII REDACTION — PRESIDIO

def _redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    if not settings.PII_DETECTION_ENABLED:
        return text, {}

    entity_counts: Dict[str, int] = {}

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        entities = getattr(settings, "PII_ENTITIES", [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "US_SSN", "CREDIT_CARD", "LOCATION", "IP_ADDRESS",
        ])

        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results = analyzer.analyze(text=text, entities=entities, language="en")

        for r in results:
            entity_counts[r.entity_type] = entity_counts.get(r.entity_type, 0) + 1

        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text

    except ImportError:
        logger.warning("presidio_not_installed", hint="pip install presidio-analyzer presidio-anonymizer")
    except Exception as exc:
        logger.warning("pii_redaction_failed", error=str(exc))

    return text, entity_counts


# HEADING DETECTION

def _detect_subtype(chunk: str) -> str:
    lines = [l.strip() for l in chunk.split("\n") if l.strip()]
    if not lines:
        return "paragraph"
    first = lines[0]
    if first.startswith("#"):
        return "heading"
    if len(lines) == 1 and len(first.split()) <= 8 and not first.endswith("."):
        return "heading"
    return "paragraph"


# HEADING HIERARCHY H1-H3

def _extract_heading_level(line: str) -> Optional[int]:
    import re
    match = re.match(r"^(#{1,3})\s", line)
    if match:
        return len(match.group(1))
    return None


# KEYWORD EXTRACTION — YAKE

def _extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(top=max_keywords, stopwords=None)
        keywords     = kw_extractor.extract_keywords(text)
        return [kw for kw, _ in keywords]
    except ImportError:
        pass

    # KEYBERT FALLBACK
    try:
        from keybert import KeyBERT
        kb       = KeyBERT()
        keywords = kb.extract_keywords(text, top_n=max_keywords)
        return [kw for kw, _ in keywords]
    except ImportError:
        pass

    return []


# READABILITY SCORE — FLESCH-KINCAID

def _readability_score(text: str) -> float:
    try:
        import textstat
        return float(textstat.flesch_reading_ease(text))
    except ImportError:
        pass

    # MANUAL APPROXIMATION
    try:
        sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
        words     = max(len(text.split()), 1)
        syllables = sum(
            max(1, sum(1 for ch in w.lower() if ch in "aeiou"))
            for w in text.split()
        )
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return round(max(0.0, min(score, 100.0)), 2)
    except Exception:
        return 0.0


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


# SECTION SPLITTING — DETECT [DOC-NNN] HEADERS
#
# Many synthetic / multi-doc corpora pack several logical documents into one
# file using a header line like "[DOC-001] Some title". We slice the file
# into per-section blocks BEFORE chunking so each chunk carries its own
# section_id / section_title and remains citable down to the section level.
# If no markers are present the splitter falls back to a single (None, None)
# block, which makes the pipeline a no-op for plain txt/md.

_SECTION_HEADER_RE = re.compile(
    r"^[ \t]*\[(DOC-[A-Za-z0-9._\-]+)\][ \t]*(.*)$",
    re.MULTILINE,
)


def _split_sections(text: str) -> List[Tuple[Optional[str], Optional[str], str]]:
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [(None, None, text)]

    sections: List[Tuple[Optional[str], Optional[str], str]] = []

    # PREAMBLE BEFORE FIRST [DOC-NNN] — keep only if non-trivial
    preamble = text[: matches[0].start()].strip()
    if preamble and len(preamble) >= settings.CHUNK_MIN_SIZE:
        sections.append((None, None, preamble))

    for i, m in enumerate(matches):
        section_id    = m.group(1).strip()
        section_title = m.group(2).strip() or None
        body_start    = m.end()
        body_end      = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body          = text[body_start:body_end].strip()
        if not body:
            continue
        sections.append((section_id, section_title, body))

    return sections


# CHUNKING — SEMANTIC BOUNDARY AWARE

def _chunk_text(text: str) -> List[str]:
    # TRY LANGCHAIN SEMANTIC SPLITTER
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n[DOC-", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        chunks = splitter.split_text(text)
        if chunks:
            return [c.strip() for c in chunks if c.strip()]
    except Exception:
        pass

    # FALLBACK — SLIDING WINDOW
    size    = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    step    = max(size - overlap, 1)
    chunks  = []
    for i in range(0, len(text), step):
        chunk = text[i:i + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# MAIN ASYNC INGEST

async def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    ext       = path.suffix.lower()
    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError("EMPTY_FILE")

    if file_size > settings.MAX_FILE_SIZE_TEXT:
        raise ValueError(
            f"FILE_TOO_LARGE: {file_size} bytes exceeds "
            f"{settings.MAX_FILE_SIZE_TEXT} bytes"
        )

    with tracer.start_as_current_span("text_ingest") as span:
        span.set_attribute("file.name", path.name)
        span.set_attribute("file.size", file_size)
        span.set_attribute("session.id", session_id)

        start = time.time()

        async with _semaphore:
            try:
                logger.info(
                    "text_ingest_start",
                    file=path.name,
                    size=file_size,
                    session_id=session_id,
                )

                # BINARY MAGIC BYTE CHECK — REJECT NON-TEXT
                is_bin = await asyncio.get_event_loop().run_in_executor(
                    None, _is_binary, path
                )
                if is_bin:
                    raise ValueError(
                        f"BINARY_FILE_DISGUISED_AS_TEXT: {path.name}"
                    )

                # LOAD — STREAMING FOR LARGE FILES
                stream_threshold = 5 * 1024 * 1024  # 5 MB
                if file_size > stream_threshold:
                    raw_text = await asyncio.get_event_loop().run_in_executor(
                        None, _load_text_streaming, path, settings.MAX_FILE_SIZE_TEXT
                    )
                else:
                    raw_text = await asyncio.get_event_loop().run_in_executor(
                        None, _load_text, path
                    )

                # BOM STRIP
                raw_text = _strip_bom(raw_text)

                # NULL BYTE STRIP
                raw_text, null_count = _strip_null_bytes(raw_text)
                if null_count:
                    logger.warning(
                        "null_bytes_stripped",
                        count=null_count,
                        file=path.name,
                    )

                # FILE-LEVEL REPAIR — mojibake fix + noise-line strip.
                # Cheap, idempotent, no-op on clean text. Per-chunk repairs
                # (OCR, whitespace recovery, footnote strip, placeholder
                # detection, title-mismatch flag, version tagging) run later
                # once chunks are produced.
                raw_text, repair_stats = await asyncio.get_event_loop().run_in_executor(
                    None, repair_text, raw_text
                )
                if repair_stats:
                    logger.info(
                        "text_repair_file_level",
                        file=path.name,
                        **repair_stats,
                    )

                # NORMALIZE
                text = _normalize_text(raw_text)

                if not text or text.isspace():
                    raise ValueError("EMPTY_CONTENT_AFTER_NORMALIZE")

                if len(text) < 50:
                    raise ValueError("TEXT_TOO_SHORT")

                # METADATA
                file_hash   = _hash(text[:10000])
                doc_id      = str(uuid.uuid4())
                source_name = path.name
                source_path = str(path.resolve())
                line_count  = text.count("\n") + 1
                word_count  = len(text.split())

                # LANGUAGE DETECTION
                language = await asyncio.get_event_loop().run_in_executor(
                    None, _detect_language, text
                )
                is_rtl = _is_rtl(language)

                # PII REDACTION
                text, pii_counts = await asyncio.get_event_loop().run_in_executor(
                    None, _redact_pii, text
                )
                for entity_type, count in pii_counts.items():
                    _pii_redacted.labels(entity_type=entity_type).inc(count)

                # SECTION SPLITTING — detect [DOC-NNN] blocks before chunking.
                # Falls back to a single (None, None, text) block for plain files.
                sections = await asyncio.get_event_loop().run_in_executor(
                    None, _split_sections, text
                )

                # CHUNK PER SECTION
                # We preserve (section_id, section_title, chunk_text, section_extras)
                # tuples so the citation layer can render Claude-style numbered
                # references that point back to a specific DOC-NNN.
                chunk_tuples: List[
                    Tuple[Optional[str], Optional[str], str, Dict[str, Any]]
                ] = []
                dropped_sections = 0
                for section_id_val, section_title_val, body in sections:
                    # SKIP EMPTY / PLACEHOLDER SECTIONS
                    if is_placeholder(body):
                        dropped_sections += 1
                        logger.info(
                            "dropped_empty_section",
                            section_id=section_id_val,
                            section_title=section_title_val,
                            file=path.name,
                        )
                        continue

                    # STRIP FOOTNOTES / EDITOR NOTES — preserve as metadata
                    cleaned_body, footnotes = strip_footnotes(body)

                    # ERROR-MARKER DETECTION — strip and preserve as metadata.
                    # The prompt builder consumes `error_markers` and tells the
                    # LLM to treat the chunk's specific claim as suspect.
                    cleaned_body, error_markers = detect_error_markers(cleaned_body)

                    # Surface mislabeled-section markers (e.g. "→ WRONG LABEL")
                    # as a top-level flag too, so callers don't have to parse
                    # the markers string.
                    title_marked_mismatch = any(
                        "wrong label" in m.lower() for m in error_markers
                    )

                    section_extras: Dict[str, Any] = {}
                    if footnotes:
                        section_extras["footnotes"] = footnotes
                    if error_markers:
                        section_extras["error_markers"] = error_markers
                    if title_marked_mismatch:
                        section_extras["title_mismatch"] = True

                    version_info = extract_version(section_id_val, section_title_val)
                    if version_info:
                        section_extras["doc_version"]      = version_info["version"]
                        section_extras["doc_version_kind"] = version_info["kind"]

                    section_chunks = await asyncio.get_event_loop().run_in_executor(
                        None, _chunk_text, cleaned_body
                    )
                    for ch in section_chunks:
                        if ch and ch.strip():
                            chunk_tuples.append(
                                (section_id_val, section_title_val, ch, section_extras)
                            )

                if dropped_sections:
                    logger.info(
                        "dropped_empty_sections_total",
                        count=dropped_sections,
                        file=path.name,
                    )

                if not chunk_tuples:
                    raise ValueError("NO_CHUNKS_PRODUCED")

                if len(chunk_tuples) > settings.MAX_CHUNKS:
                    logger.warning(
                        "chunk_limit_applied",
                        original=len(chunk_tuples),
                        limited=settings.MAX_CHUNKS,
                        file=path.name,
                    )
                    chunk_tuples = chunk_tuples[:settings.MAX_CHUNKS]

                total_chunks = len(chunk_tuples)
                documents:   List[IngestedDocument] = []

                # NEAR-DEDUP VIA SIMHASH
                seen_hashes:    set = set()
                seen_simhashes: List[int] = []

                for i, (section_id_val, section_title_val, chunk, section_extras) in enumerate(chunk_tuples):
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    # PER-CHUNK REPAIRS — only kick in for actually broken chunks
                    chunk_repairs: Dict[str, Any] = {}
                    repaired, ws_fixed = recover_whitespace(chunk)
                    if ws_fixed:
                        chunk = repaired
                        chunk_repairs["whitespace_recovered"] = True
                    repaired, ocr_fixed = normalize_ocr_noise(chunk)
                    if ocr_fixed:
                        chunk = repaired
                        chunk_repairs["ocr_normalized"] = True

                    if len(chunk) < settings.CHUNK_MIN_SIZE:
                        continue

                    # EXACT DEDUP
                    exact_h = _hash(chunk)
                    if exact_h in seen_hashes:
                        continue
                    seen_hashes.add(exact_h)

                    # NEAR DEDUP VIA SIMHASH
                    sh = _simhash(chunk)
                    too_similar = any(
                        _simhash_distance(sh, prev) <= 3
                        for prev in seen_simhashes
                    )
                    if too_similar:
                        continue
                    seen_simhashes.append(sh)

                    quality   = _quality_score(chunk)
                    subtype   = _detect_subtype(chunk)
                    keywords  = _extract_keywords(chunk)
                    fk_score  = _readability_score(chunk)

                    heading_level: Optional[int] = None
                    first_line    = chunk.split("\n")[0]
                    heading_level = _extract_heading_level(first_line)

                    title_mismatch = has_title_mismatch(section_title_val, keywords)

                    chunk_extra_metadata: Dict[str, Any] = {
                        "modality_weight":    1.0,
                        "importance_score":   quality,
                        "data_quality_score": quality,
                    }
                    if section_extras:
                        chunk_extra_metadata.update(section_extras)
                    if chunk_repairs:
                        chunk_extra_metadata.update(chunk_repairs)
                    if title_mismatch:
                        chunk_extra_metadata["title_mismatch"] = True

                    # Fields the retrieval/prompt layer needs must live in
                    # `structure` so they survive the Qdrant payload round-trip
                    # (Qdrant's payload builder reads from structure, not
                    # extra_metadata). Keep them in both for resilience.
                    structure_carry_overs: Dict[str, Any] = {}
                    if section_extras.get("error_markers"):
                        structure_carry_overs["error_markers"] = section_extras["error_markers"]
                    if section_extras.get("doc_version"):
                        structure_carry_overs["doc_version"]      = section_extras["doc_version"]
                        structure_carry_overs["doc_version_kind"] = section_extras.get("doc_version_kind")
                    if section_extras.get("footnotes"):
                        structure_carry_overs["footnotes"] = section_extras["footnotes"]
                    if chunk_extra_metadata.get("title_mismatch"):
                        structure_carry_overs["title_mismatch"] = True

                    structure_payload: Dict[str, Any] = {
                        "doc_id":              doc_id,
                        "session_id":          session_id,
                        "file_hash":           file_hash,
                        "source_path":         source_path,
                        "chunk_index":         i,
                        "total_chunks":        total_chunks,
                        "chunk_length":        len(chunk),
                        "page_number":         None,
                        "total_pages":         None,
                        "section_id":          section_id_val,
                        "section_title":       section_title_val,
                        "ingestion_timestamp": time.time(),
                        "language":            "en",
                        "file_size_bytes":     file_size,
                        "is_rtl":              is_rtl,
                        "heading_level":       heading_level,
                        "readability_score":   fk_score,
                        "tags":                keywords,
                        "pii_redacted":        bool(pii_counts),
                        "content_type":        "text_chunk",
                        "ingestion_time":      time.time(),
                    }
                    structure_payload.update(structure_carry_overs)

                    doc = IngestedDocument(
                        text=chunk,
                        modality="text",
                        subtype=subtype,
                        source_type="file",
                        source=source_name,
                        chunk_id=i,
                        structure=structure_payload,
                        extra_metadata=chunk_extra_metadata,
                    ).finalize()

                    documents.append(doc)

                if not documents:
                    raise ValueError("NO_VALID_DOCUMENTS_AFTER_FILTERING")

                latency = round(time.time() - start, 2)

                _ingest_duration.labels(status="success").observe(latency)

                span.set_attribute("docs.count", len(documents))
                span.set_attribute("language", language or "unknown")
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "text_ingest_success",
                    file=path.name,
                    docs=len(documents),
                    total_chunks=total_chunks,
                    language=language,
                    is_rtl=is_rtl,
                    word_count=word_count,
                    line_count=line_count,
                    latency=latency,
                    session_id=session_id,
                )

                return documents

            except Exception as exc:
                latency    = round(time.time() - start, 2)
                error_type = type(exc).__name__

                _ingest_duration.labels(status="error").observe(latency)
                _ingest_errors.labels(error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "text_ingest_failed",
                    file=path.name,
                    session_id=session_id,
                    error=str(exc),
                    error_type=error_type,
                    latency=latency,
                )
                raise


# SYNC WRAPPER FOR BACKWARD COMPATIBILITY

def ingest_sync(file_path: str, session_id: str) -> List[IngestedDocument]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, ingest(file_path, session_id))
                return future.result()
        return loop.run_until_complete(ingest(file_path, session_id))
    except RuntimeError:
        return asyncio.run(ingest(file_path, session_id))


