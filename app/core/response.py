from __future__ import annotations

import os
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ERROR CODES

class ErrorCode(str, Enum):

    # VALIDATION
    EMPTY_FILE               = "EMPTY_FILE"
    FILE_TOO_LARGE           = "FILE_TOO_LARGE"
    UNSUPPORTED_TYPE         = "UNSUPPORTED_TYPE"
    INVALID_MIME             = "INVALID_MIME"
    CORRUPTED_FILE           = "CORRUPTED_FILE"
    INVALID_DIMENSIONS       = "INVALID_DIMENSIONS"
    IMAGE_TOO_SMALL          = "IMAGE_TOO_SMALL"
    INVALID_AUDIO_DURATION   = "INVALID_AUDIO_DURATION"
    INVALID_SAMPLE_RATE      = "INVALID_SAMPLE_RATE"
    INVALID_WAV_HEADER       = "INVALID_WAV_HEADER"
    INVALID_PDF_STRUCTURE    = "INVALID_PDF_STRUCTURE"
    INVALID_EXCEL_STRUCTURE  = "INVALID_EXCEL_STRUCTURE"
    EMPTY_DOCUMENT           = "EMPTY_DOCUMENT"
    SESSION_ID_REQUIRED      = "SESSION_ID_REQUIRED"
    EMPTY_QUERY              = "EMPTY_QUERY"
    EMPTY_CONTENT            = "EMPTY_CONTENT"
    PASSWORD_PROTECTED       = "PASSWORD_PROTECTED"
    DRM_PROTECTED            = "DRM_PROTECTED"
    DISK_SPACE_INSUFFICIENT  = "DISK_SPACE_INSUFFICIENT"
    DUPLICATE_FILE           = "DUPLICATE_FILE"

    # PROCESSING
    OCR_FAILED               = "OCR_FAILED"
    CAPTION_FAILED           = "CAPTION_FAILED"
    TRANSCRIPTION_FAILED     = "TRANSCRIPTION_FAILED"
    FRAME_EXTRACTION_FAILED  = "FRAME_EXTRACTION_FAILED"
    EMBEDDING_FAILED         = "EMBEDDING_FAILED"
    INGESTION_FAILED         = "INGESTION_FAILED"
    CHUNKING_FAILED          = "CHUNKING_FAILED"
    VECTOR_INSERT_FAILED     = "VECTOR_INSERT_FAILED"
    LIBREOFFICE_FAILED       = "LIBREOFFICE_FAILED"
    DIARIZATION_FAILED       = "DIARIZATION_FAILED"
    SUBTITLE_EXTRACTION_FAILED = "SUBTITLE_EXTRACTION_FAILED"
    PII_REDACTION_FAILED     = "PII_REDACTION_FAILED"
    MALWARE_DETECTED         = "MALWARE_DETECTED"

    # RETRIEVAL
    NO_RESULTS_FOUND         = "NO_RESULTS_FOUND"
    RETRIEVAL_FAILED         = "RETRIEVAL_FAILED"
    RERANK_FAILED            = "RERANK_FAILED"

    # AGENT
    AGENT_TIMEOUT            = "AGENT_TIMEOUT"
    AGENT_FAILED             = "AGENT_FAILED"
    INVALID_PLAN             = "INVALID_PLAN"
    TOOL_NOT_FOUND           = "TOOL_NOT_FOUND"

    # INFRASTRUCTURE
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    REDIS_UNAVAILABLE        = "REDIS_UNAVAILABLE"
    MONGO_UNAVAILABLE        = "MONGO_UNAVAILABLE"
    LLM_UNAVAILABLE          = "LLM_UNAVAILABLE"
    MODEL_TIMEOUT            = "MODEL_TIMEOUT"
    CIRCUIT_BREAKER_OPEN     = "CIRCUIT_BREAKER_OPEN"

    # SECURITY
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    RATE_LIMIT_EXCEEDED       = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED              = "UNAUTHORIZED"
    SSRF_BLOCKED              = "SSRF_BLOCKED"
    PATH_TRAVERSAL_BLOCKED    = "PATH_TRAVERSAL_BLOCKED"

    # GENERIC
    INTERNAL_ERROR           = "INTERNAL_ERROR"
    NOT_IMPLEMENTED          = "NOT_IMPLEMENTED"
    UNKNOWN_ERROR            = "UNKNOWN_ERROR"
    TIMEOUT                  = "TIMEOUT"


# SEVERITY

class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# MODALITY

class Modality(str, Enum):
    TEXT     = "text"
    PDF      = "pdf"
    WORD     = "word"
    EXCEL    = "excel"
    IMAGE    = "image"
    AUDIO    = "audio"
    VIDEO    = "video"
    UNKNOWN  = "unknown"


# PROCESSING STATUS

class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    SUCCESS    = "success"
    FAILED     = "failed"
    SKIPPED    = "skipped"
    DUPLICATE  = "duplicate"


# ERROR DETAIL

class ErrorDetail:

    def __init__(
        self,
        code:     ErrorCode,
        message:  str,
        severity: Severity              = Severity.MEDIUM,
        field:    Optional[str]         = None,
        context:  Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code     = code
        self.message  = message
        self.severity = severity
        self.field    = field
        self.context  = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code":     self.code,
            "message":  self.message,
            "severity": self.severity,
            "field":    self.field,
            "context":  self.context,
        }


# BASE RESPONSE

class BaseResponse:

    def __init__(
        self,
        success:    bool,
        modality:   str                    = Modality.UNKNOWN,
        session_id: str                    = "default",
        latency:    float                  = 0.0,
        metadata:   Optional[Dict[str, Any]] = None,
        trace_id:   Optional[str]          = None,
    ) -> None:
        self.success    = success
        self.modality   = modality
        self.session_id = session_id
        self.latency    = latency
        self.metadata   = metadata or {}
        self.timestamp  = time.time()
        self.trace_id   = trace_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":    self.success,
            "modality":   self.modality,
            "session_id": self.session_id,
            "latency":    self.latency,
            "metadata":   self.metadata,
            "timestamp":  self.timestamp,
            "trace_id":   self.trace_id,
        }


# SUCCESS RESPONSE

class ProcessingResult(BaseResponse):

    def __init__(
        self,
        modality:   str,
        session_id: str,
        latency:    float,
        chunks:     int                    = 0,
        stored:     int                    = 0,
        source:     str                    = "",
        metadata:   Optional[Dict[str, Any]] = None,
        warnings:   Optional[List[str]]    = None,
        trace_id:   Optional[str]          = None,
        file_hash:  Optional[str]          = None,
        status:     str                    = ProcessingStatus.SUCCESS,
    ) -> None:
        super().__init__(
            success=True,
            modality=modality,
            session_id=session_id,
            latency=latency,
            metadata=metadata,
            trace_id=trace_id,
        )
        self.chunks    = chunks
        self.stored    = stored
        self.source    = source
        self.warnings  = warnings or []
        self.file_hash = file_hash
        self.status    = status

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "chunks":    self.chunks,
            "stored":    self.stored,
            "source":    self.source,
            "warnings":  self.warnings,
            "file_hash": self.file_hash,
            "status":    self.status,
        })
        return base


# UNIVERSAL ERROR RESPONSE

class UniversalErrorResponse(BaseResponse):

    def __init__(
        self,
        code:       ErrorCode,
        message:    str,
        modality:   str                    = Modality.UNKNOWN,
        session_id: str                    = "default",
        latency:    float                  = 0.0,
        severity:   Severity               = Severity.MEDIUM,
        field:      Optional[str]          = None,
        context:    Optional[Dict[str, Any]] = None,
        metadata:   Optional[Dict[str, Any]] = None,
        trace_id:   Optional[str]          = None,
    ) -> None:
        super().__init__(
            success=False,
            modality=modality,
            session_id=session_id,
            latency=latency,
            metadata=metadata,
            trace_id=trace_id,
        )
        self.error = ErrorDetail(
            code=code,
            message=message,
            severity=severity,
            field=field,
            context=context,
        )

        logger.warning(
            event="error_response_created",
            code=code,
            severity=severity,
            modality=modality,
            session_id=session_id,
            trace_id=trace_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["error"] = self.error.to_dict()
        return base


# QUERY RESPONSE

# RETRIEVED SOURCE SHAPE — canonical citation record returned to the API
# Fields (Dict[str, Any] to stay style-consistent with the rest of this module):
#   source            str           — basename of the original file (REQUIRED)
#   modality          str           — text|pdf|word|excel|image|audio|video (REQUIRED)
#   chunk_id          str           — unique per-chunk id (REQUIRED)
#   subtype           Optional[str] — page|paragraph|table|caption|ocr|frame|speech|chart
#   page              Optional[int]
#   sheet             Optional[str]
#   timestamp_start   Optional[float]
#   timestamp_end     Optional[float]
#   speaker           Optional[str]
#   asset_path        Optional[str] — for video frames / images
#   score             Optional[float]
#   doc_id            Optional[str]
#   parent_modality   Optional[str] — set when an image came from a word/excel parent
#   parent_page       Optional[int]
#   parent_sheet      Optional[str]
#   cite_key          str           — bracket tag the LLM is told to emit, e.g. [foo.pdf p.4]
#   snippet           str           — first ~240 chars of the chunk text


class QueryResponse(BaseResponse):

    def __init__(
        self,
        answer:       str,
        session_id:   str,
        latency:      float,
        confidence:   float                  = 0.5,
        sources_used: int                    = 0,
        decision:     str                    = "unknown",
        source:       str                    = "agent",
        metadata:     Optional[Dict[str, Any]] = None,
        trace_id:     Optional[str]          = None,
        cache_hit:    bool                   = False,
        sources:      Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            success=True,
            modality=Modality.TEXT,
            session_id=session_id,
            latency=latency,
            metadata=metadata,
            trace_id=trace_id,
        )
        self.answer       = answer
        self.confidence   = max(0.0, min(float(confidence), 1.0))
        self.sources      = list(sources) if sources else []
        self.sources_used = sources_used if sources_used else len(self.sources)
        self.decision     = decision
        self.source       = source
        self.cache_hit    = cache_hit

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "answer":       self.answer,
            "confidence":   self.confidence,
            "sources_used": self.sources_used,
            "sources":      self.sources,
            "decision":     self.decision,
            "source":       self.source,
            "cache_hit":    self.cache_hit,
        })
        return base


# VALIDATION RESULT

class ValidationResult:

    def __init__(
        self,
        valid:     bool,
        modality:  str               = Modality.UNKNOWN,
        file_size: int               = 0,
        mime_type: str               = "",
        errors:    Optional[List[ErrorDetail]] = None,
        warnings:  Optional[List[str]]         = None,
        latency:   float             = 0.0,
        trace_id:  Optional[str]     = None,
    ) -> None:
        self.valid     = valid
        self.modality  = modality
        self.file_size = file_size
        self.mime_type = mime_type
        self.errors    = errors or []
        self.warnings  = warnings or []
        self.latency   = latency
        self.trace_id  = trace_id

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(
        self,
        code:     ErrorCode,
        message:  str,
        severity: Severity          = Severity.HIGH,
        field:    Optional[str]     = None,
    ) -> None:
        self.errors.append(
            ErrorDetail(code=code, message=message, severity=severity, field=field)
        )
        self.valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid":     self.valid,
            "modality":  self.modality,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "errors":    [e.to_dict() for e in self.errors],
            "warnings":  self.warnings,
            "latency":   self.latency,
            "trace_id":  self.trace_id,
        }


# STREAMING CHUNK RESPONSE — SECTION 4.6

class StreamChunk:

    def __init__(
        self,
        token:      str,
        session_id: str               = "default",
        trace_id:   Optional[str]     = None,
        done:       bool              = False,
    ) -> None:
        self.token      = token
        self.session_id = session_id
        self.trace_id   = trace_id
        self.done       = done
        self.timestamp  = time.time()

    def to_sse(self) -> str:
        if self.done:
            return "data: [DONE]\n\n"
        return f"data: {self.token}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token":      self.token,
            "session_id": self.session_id,
            "trace_id":   self.trace_id,
            "done":       self.done,
            "timestamp":  self.timestamp,
        }


# INGESTION PROGRESS EVENT — SECTION 4.6

class IngestionProgressEvent:

    def __init__(
        self,
        file_name:  str,
        stage:      str,
        status:     str,
        session_id: str               = "default",
        trace_id:   Optional[str]     = None,
        details:    Optional[Dict[str, Any]] = None,
    ) -> None:
        self.file_name  = file_name
        self.stage      = stage
        self.status     = status
        self.session_id = session_id
        self.trace_id   = trace_id
        self.details    = details or {}
        self.timestamp  = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name":  self.file_name,
            "stage":      self.stage,
            "status":     self.status,
            "session_id": self.session_id,
            "trace_id":   self.trace_id,
            "details":    self.details,
            "timestamp":  self.timestamp,
        }


# HEALTH RESPONSE

class HealthResponse:

    def __init__(
        self,
        status:   str,
        service:  str,
        version:  str,
        models:   Optional[Dict[str, Any]] = None,
        infra:    Optional[Dict[str, Any]] = None,
        latency:  float                    = 0.0,
    ) -> None:
        self.status    = status
        self.service   = service
        self.version   = version
        self.models    = models or {}
        self.infra     = infra or {}
        self.latency   = latency
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status":    self.status,
            "service":   self.service,
            "version":   self.version,
            "models":    self.models,
            "infra":     self.infra,
            "latency":   self.latency,
            "timestamp": self.timestamp,
        }


# GDPR PURGE RESPONSE — SECTION 5

class GdprPurgeResponse:

    def __init__(
        self,
        user_id:   str,
        redis:     bool              = False,
        mongo:     bool              = False,
        qdrant:    bool              = False,
        errors:    Optional[List[str]] = None,
        trace_id:  Optional[str]    = None,
    ) -> None:
        self.user_id   = user_id
        self.redis     = redis
        self.mongo     = mongo
        self.qdrant    = qdrant
        self.errors    = errors or []
        self.trace_id  = trace_id
        self.purged_at = time.time()
        self.success   = len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id":   self.user_id,
            "success":   self.success,
            "redis":     self.redis,
            "mongo":     self.mongo,
            "qdrant":    self.qdrant,
            "errors":    self.errors,
            "trace_id":  self.trace_id,
            "purged_at": self.purged_at,
        }


# FACTORY HELPERS

def ok(
    modality:   str,
    session_id: str,
    latency:    float,
    chunks:     int                    = 0,
    stored:     int                    = 0,
    source:     str                    = "",
    metadata:   Optional[Dict[str, Any]] = None,
    warnings:   Optional[List[str]]    = None,
    trace_id:   Optional[str]          = None,
    file_hash:  Optional[str]          = None,
) -> ProcessingResult:
    return ProcessingResult(
        modality=modality,
        session_id=session_id,
        latency=latency,
        chunks=chunks,
        stored=stored,
        source=source,
        metadata=metadata,
        warnings=warnings,
        trace_id=trace_id,
        file_hash=file_hash,
    )


def err(
    code:       ErrorCode,
    message:    str,
    modality:   str                    = Modality.UNKNOWN,
    session_id: str                    = "default",
    latency:    float                  = 0.0,
    severity:   Severity               = Severity.MEDIUM,
    field:      Optional[str]          = None,
    context:    Optional[Dict[str, Any]] = None,
    trace_id:   Optional[str]          = None,
) -> UniversalErrorResponse:
    return UniversalErrorResponse(
        code=code,
        message=message,
        modality=modality,
        session_id=session_id,
        latency=latency,
        severity=severity,
        field=field,
        context=context,
        trace_id=trace_id,
    )


def validation_ok(
    modality:  str,
    file_size: int,
    mime_type: str,
    latency:   float                  = 0.0,
    warnings:  Optional[List[str]]    = None,
    trace_id:  Optional[str]          = None,
) -> ValidationResult:
    return ValidationResult(
        valid=True,
        modality=modality,
        file_size=file_size,
        mime_type=mime_type,
        latency=latency,
        warnings=warnings,
        trace_id=trace_id,
    )


def validation_err(
    code:      ErrorCode,
    message:   str,
    modality:  str               = Modality.UNKNOWN,
    file_size: int               = 0,
    mime_type: str               = "",
    latency:   float             = 0.0,
    field:     Optional[str]     = None,
    trace_id:  Optional[str]     = None,
) -> ValidationResult:
    result = ValidationResult(
        valid=False,
        modality=modality,
        file_size=file_size,
        mime_type=mime_type,
        latency=latency,
        trace_id=trace_id,
    )
    result.add_error(code=code, message=message, field=field)
    return result


# RETRIEVED SOURCE BUILDER

_CITE_KEY_RE = re.compile(r"[^A-Za-z0-9._\- =:]+")


def _format_ts(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        s = float(seconds)
    except Exception:
        return None
    if s < 0:
        return None
    m, sec = divmod(int(round(s)), 60)
    h, m   = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _safe_basename(path: Any) -> str:
    if not path:
        return "unknown"
    try:
        return os.path.basename(str(path)) or "unknown"
    except Exception:
        return "unknown"


def _make_cite_key(
    source:          str,
    page:            Optional[int]   = None,
    sheet:           Optional[str]   = None,
    timestamp_start: Optional[float] = None,
    section_id:      Optional[str]   = None,
) -> str:
    """Build the bracket tag the LLM must emit verbatim. Stable & sanitized.

    TXT/transcript sources deliberately fall through to the bare "[source]"
    tag: the small GGUF model mangles a longer "[source — SPEAKER]" tag —
    dropping the brackets, swapping the separator, and dumping a trailing run
    of every chunk header it saw — which then leaks into the visible answer.
    Speaker attribution for TXT is carried on the source chip
    (RetrievedSource.speaker), never in the inline tag the model echoes.
    """
    src = _safe_basename(source)
    src = _CITE_KEY_RE.sub("_", src)
    if section_id:
        sec_clean = _CITE_KEY_RE.sub("_", str(section_id))[:40]
        return f"[{src} {sec_clean}]"
    if page is not None:
        try:
            return f"[{src} p.{int(page)}]"
        except Exception:
            pass
    if sheet:
        sheet_clean = _CITE_KEY_RE.sub("_", str(sheet))[:40]
        return f"[{src} sheet={sheet_clean}]"
    ts = _format_ts(timestamp_start)
    if ts:
        return f"[{src} t={ts}]"
    return f"[{src}]"


def build_retrieved_source(
    doc:       Dict[str, Any],
    rank:      int = 0,
    snippet_chars: int = 240,
) -> Dict[str, Any]:
    """Map one retrieved doc (text + metadata + score) to the canonical RetrievedSource shape."""
    meta = doc.get("metadata") or {}
    text = doc.get("text") or ""

    raw_source     = meta.get("source") or meta.get("file_path") or meta.get("doc_id") or ""
    source_name    = _safe_basename(raw_source)
    modality       = str(meta.get("modality") or "text")
    subtype        = meta.get("subtype")
    page           = meta.get("page")
    sheet          = meta.get("sheet")
    ts_start       = meta.get("timestamp_start")
    ts_end         = meta.get("timestamp_end")
    speaker        = meta.get("speaker")
    asset_path     = meta.get("asset_path")
    chunk_id       = meta.get("chunk_id") or f"r{rank}"
    doc_id         = meta.get("doc_id")
    parent_modality = meta.get("parent_modality")
    parent_page    = meta.get("parent_page")
    parent_sheet   = meta.get("parent_sheet")
    section_id     = meta.get("section_id")
    section_title  = meta.get("section_title")

    snippet = " ".join(str(text).split())[:snippet_chars]

    cite_key = _make_cite_key(
        source=source_name,
        page=page if isinstance(page, int) else None,
        sheet=sheet,
        timestamp_start=ts_start,
        section_id=section_id if isinstance(section_id, str) and section_id else None,
    )

    return {
        "source":          source_name,
        "modality":        modality,
        "subtype":         subtype,
        "page":            page if isinstance(page, int) else None,
        "sheet":           sheet,
        "timestamp_start": float(ts_start) if isinstance(ts_start, (int, float)) else None,
        "timestamp_end":   float(ts_end)   if isinstance(ts_end,   (int, float)) else None,
        "speaker":         speaker,
        "asset_path":      asset_path,
        "score":           float(doc.get("score")) if isinstance(doc.get("score"), (int, float)) else None,
        "chunk_id":        str(chunk_id),
        "doc_id":          str(doc_id) if doc_id else None,
        "parent_modality": parent_modality,
        "parent_page":     parent_page if isinstance(parent_page, int) else None,
        "parent_sheet":    parent_sheet,
        "section_id":      section_id if isinstance(section_id, str) and section_id else None,
        "section_title":   section_title if isinstance(section_title, str) and section_title else None,
        "cite_key":        cite_key,
        "snippet":         snippet,
    }


def extract_cited_indices(text: str) -> set:
    """Return the set of 1-based source indices the LLM cited in its answer.

    Matches patterns like [1], [2], [1,2], [1, 3], [1,2,3].
    Returns an empty set when no numeric citations are found (fallback: show all sources).
    """
    indices: set = set()
    for m in re.finditer(r'\[(\d+(?:\s*,\s*\d+)*)\]', text or ""):
        for part in m.group(1).split(","):
            try:
                indices.add(int(part.strip()))
            except ValueError:
                pass
    return indices


# Structured locator/marker tokens that may leak from ingestion, the reranker
# context builder, or a model that echoes a chunk header verbatim. These must
# NOT appear in the user-facing answer — locators are surfaced on source chips.
_STRUCT_MARKER_RE = re.compile(
    r'\s*\[\s*(?:'
    r'sheet|page|pg|pages?|hyperlink|src|spk|t|para|paragraph|rows?|'
    r'section|sec|figure|fig|table|caption|slide|frame|timestamp|ts|'
    r'error_markers'
    r')\b[^\]]*\]',
    re.IGNORECASE,
)
# Bare filename citation, e.g. [report.pdf], [gdp.jpg] — strip so no provenance
# string leaks into prose. Requires a dot + 2-4 char extension to avoid eating
# ordinary bracketed words.
# Matches "[<name>.<ext>]" AND "[<name>.<ext> <locator>]" where the locator is a
# timestamp/page appended after the extension, e.g. the reasoning engine's audio
# citation "[FOMC Press Conference September 18_ 2024.mp3 t=00:07]" or a page
# form "[report.pdf p.6]". Requires a dotted file extension inside the bracket,
# so ordinary bracketed prose is never touched.
_CITE_FILE_EXT = (
    r'(?:mp3|mp4|wav|m4a|flac|ogg|aac|opus|pdf|txt|md|docx?|xlsx?|xls|csv|pptx?|'
    r'jpe?g|png|gif|webp|bmp|tiff?|heic|svg|mov|avi|mkv|webm)'
)
# Matches a citation bracket whose content contains a KNOWN file extension —
# "[report.pdf]", "[report.pdf p.6]", or the reasoning engine's audio form
# "[FOMC Press Conference September 18_ 2024.mp3 t=00:07]". Using an explicit
# extension list (not "\.\w+") avoids stripping legitimate bracketed prose that
# merely contains a decimal, e.g. "[see section 3.14 for details]".
_FILENAME_CITATION_RE = re.compile(
    r'\s*\[[^\]\n]*?\.' + _CITE_FILE_EXT + r'(?:\s+[^\]\n]*)?\s*\]',
    re.IGNORECASE,
)
# TXT speaker cite_key echoed verbatim, e.g. [fomc_dec2024.txt — CHAIR POWELL].
# _FILENAME_CITATION_RE above only matches when the extension is immediately
# followed by "]"; the speaker-locator format (_make_cite_key's text/txt
# branch in this module) appends "— SPEAKER" after the extension, so it needs
# its own pattern. Requires both a dotted extension AND an em/en-dash inside
# the same bracket so ordinary bracketed prose is never touched.
_SPEAKER_CITATION_RE = re.compile(r'\s*\[[^\]\n]*?\.[A-Za-z0-9]{2,4}\s*[—–]\s*[^\]\n]*\]')
# TXT citation/transcript DUMP — the trailing garbage the small GGUF model
# produces on plain-text/transcript sources. Because every chunk of a single
# .txt file carries the identical "[source.txt]" tag, the "append a tag after
# each sentence" instruction makes the model spray the same tag, then it
# degrades into a run of bare filenames, "filename — SPEAKER" headers, raw
# "SPEAKER: quote" transcript lines, or leaked "Answer Tags:/Sources Used:"
# fields — always at the END of the answer. One tested tail-matcher handles
# every observed variant, separator-agnostic (space/period/colon/comma/dash,
# bracketed or bare). A code-side guard (_TXT_DUMP_HAS_CITE_RE) ensures the
# matched tail actually contains a filename or speaker token, so ordinary
# trailing prose/punctuation is never stripped.
_TXT_DUMP_FN     = r'[\w.\-]+\.(?:txt|md|rst|csv|log)'
_TXT_DUMP_FN_RE  = re.compile(_TXT_DUMP_FN)
_TXT_DUMP_NAME   = r"[A-Z][A-Za-z.'&-]*(?:[ ,]+[A-Z][A-Za-z.'&-]*){0,5}"
_TXT_DUMP_FN_TOK = r'[\(\[]?\s*' + _TXT_DUMP_FN + r'\s*[\)\]]?(?:\s*[—–\-]\s*' + _TXT_DUMP_NAME + r')?'
# The speaker label must end in a COLON (the transcript turn format,
# "CHAIR POWELL:"), never a bare period — a Title-Case phrase ending in a
# period ("...under the Federal Reserve Act.") is ordinary prose and must not
# be mistaken for a dumped speaker turn.
_TXT_DUMP_SPK_TOK = (
    r"[A-Z][A-Za-z.'&-]+(?:[ ][A-Z][A-Za-z.'&-]+){0,4}\s*:\s*"
    r'(?:"[^"]*"|“[^”]*”|[^\n]*?)'
)
_TXT_DUMP_FIELD_TOK = r'(?:Answer Tags?|Sources? Used|Tags?|Sources?|References?)\s*:\s*[^\n]*?'
# Bracketed numeric reference list, e.g. "[References: 1, 2, 3, 4, 5]".
_TXT_DUMP_BRACKET_REFLIST_RE = re.compile(
    r'\s*\[\s*References?\s*:\s*[\d,\s]+\]', re.IGNORECASE
)
_TXT_DUMP_SEP       = r'[\s.:;,()\[\]—–\-]*'
_TXT_DUMP_UNIT = (
    r'(?:' + _TXT_DUMP_FN_TOK + r'|' + _TXT_DUMP_SPK_TOK + r'|'
    + _TXT_DUMP_FIELD_TOK + r'|\(incomplete\))'
)
_TXT_DUMP_TAIL_RE = re.compile(
    _TXT_DUMP_SEP + r'(?:' + _TXT_DUMP_UNIT + _TXT_DUMP_SEP + r')+$'
)
_TXT_DUMP_HAS_CITE_RE = re.compile(
    _TXT_DUMP_FN + r"|[A-Z][A-Za-z.'&-]+[ ][A-Z][A-Za-z.'&-]+\s*:"
)
# Fabricated web citation — a trailing "Author, \"Title,\" Publication, Date,
# <https://...>" bibliography the model invents. A local plain-text/transcript
# answer never legitimately contains a URL, and an angle-bracketed URL / the
# Presidio "<URL>" scrub placeholder is always an artifact, so anything from
# the sentence that introduces one onward is dropped.
_TXT_DUMP_ANGLE_URL_RE = re.compile(r'<+\s*(?:https?://|www\.|URL\b)[^>]*>*', re.IGNORECASE)
# Trailing/inline parenthetical source attribution the model invents, e.g.
# "(Chair Powell, FOMC Press Conference)" or "(FOMC Transcript)" — a
# citation-shaped aside, not prose. Distinguished from a legitimate
# parenthetical (which contains ordinary lowercase words, e.g. "(a quarter
# point cut)") by consisting ENTIRELY of Title-Case words/names.
_TXT_DUMP_PAREN_SRC_RE = re.compile(
    r'\s*\((?:[A-Z][A-Za-z.\'&-]*(?:[ ,]+[A-Z][A-Za-z.\'&-]*){0,4})\)'
)


def _strip_txt_citation_dump(text: str) -> str:
    """Remove a trailing TXT citation/transcript dump; leave prose untouched."""
    if not text:
        return text
    # GENERAL BACKSTOP: the model has invented several different phrasings
    # trailing a repeated filename ("file — SPEAKER", "file - generic phrase",
    # bare "file file file", ...). Rather than enumerate every phrase shape,
    # use the one invariant that is always true for a single-file KB: a real
    # answer never says the filename itself, let alone twice. Two or more
    # mentions of the source filename is always a dump — cut from the first
    # one, whatever text follows each mention.
    fn_hits = list(_TXT_DUMP_FN_RE.finditer(text))
    if len(fn_hits) >= 2:
        cut = text[:fn_hits[0].start()].rstrip()
        if cut:
            text = cut
    m = _TXT_DUMP_TAIL_RE.search(text)
    if m and _TXT_DUMP_HAS_CITE_RE.search(m.group(0)):
        cut = text[:m.start()].rstrip()
        # If stripping would delete essentially the whole answer, the "prose"
        # was itself a dump — keep the original so we never return empty.
        if cut:
            text = cut
    text = _TXT_DUMP_BRACKET_REFLIST_RE.sub('', text)
    text = _TXT_DUMP_PAREN_SRC_RE.sub('', text)
    um = _TXT_DUMP_ANGLE_URL_RE.search(text)
    if um:
        boundary = text.rfind('. ', 0, um.start())
        cut = text[:boundary + 1].rstrip() if boundary != -1 else text[:um.start()].rstrip()
        if cut:
            text = cut
    return text.rstrip()
# Filename/doc-id stem citation with no clean extension, e.g. [aapl_def14a_2023]
# or the PII-mangled [aapl_def14a_<URL>cx] (the scrubber ate the ".docx"). Any
# bracketed token that has no spaces and contains an underscore or a <PLACEHOLDER>
# is an identifier, never prose — strip it.
_STEM_CITATION_RE = re.compile(r'\s*\[[^\]\s]*(?:_[^\]\s]*|<[A-Za-z_]{2,20}>[^\]\s]*)\]')
# DOCX section-number citation attempts the model invents mid-sentence despite
# being told not to — [§4.1], [4.1], [5.1.1], [1: 2.1]. The final citation is
# appended once at the end of the answer instead (see rag_pipeline
# _attach_section_citations), so any of these shapes found in the body is
# leftover noise, never intentional prose. The first alternative requires a
# decimal point so a legitimate bare "[1]" numeric citation isn't touched.
_SECTION_CITATION_RE = re.compile(
    r'\s*\[\s*§?\s*\d+(?:\.\d+)+\s*\]'
    r'|\s*\[\s*§?\s*\d+\s*:\s*\d+(?:\.\d+)*\s*\]'
)


def strip_inline_citations(text: str) -> str:
    """Remove every inline citation/marker so the answer prose carries no
    references or filename leakage. Handles numeric [1]/[2,3], structured
    markers ([Sheet: ...], [PG:3], [HYPERLINK ...], [T:12.0s], ...) and bare
    filename citations ([report.pdf]) — without leaving double spaces or a
    stray space before punctuation (the "net sales of  net sales" bug).
    """
    if not text:
        return ""
    cleaned = re.sub(r'\s*\[\d+(?:\s*,\s*\d+)*\]', '', text)
    cleaned = _STRUCT_MARKER_RE.sub('', cleaned)
    cleaned = _SPEAKER_CITATION_RE.sub('', cleaned)
    cleaned = _strip_txt_citation_dump(cleaned)
    cleaned = _FILENAME_CITATION_RE.sub('', cleaned)
    cleaned = _STEM_CITATION_RE.sub('', cleaned)
    cleaned = _SECTION_CITATION_RE.sub('', cleaned)
    # Collapse whitespace and repair punctuation spacing left by removals.
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = re.sub(r'\s+([,.;:!?])', r'\1', cleaned)
    # Orphaned separators left where a comma-separated citation list was removed,
    # e.g. "...point. [tag1], [tag2]" → "...point.," or "...point., ,". Drop a
    # comma/semicolon that now trails a sentence terminator or another comma.
    cleaned = re.sub(r'([.!?])\s*[,;](?:\s*[,;])*', r'\1', cleaned)
    cleaned = re.sub(r',\s*(?=[,;])', '', cleaned)
    cleaned = re.sub(r'[,;]\s*([.!?])', r'\1', cleaned)
    cleaned = re.sub(r'\(\s*\)', '', cleaned)            # empty parens
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    # An orphaned "Sources:"/"Tags:" label left after the [n] tokens it referenced
    # were removed (e.g. "Sources: [1],[2],[3]" → "Sources:,,," → ""). Drop the
    # bare label and its leftover separators wherever it now trails the text.
    cleaned = re.sub(r'\s*\b(?:Sources?|Tags?)\s*:\s*[,;\s]*$', '', cleaned,
                     flags=re.IGNORECASE)
    # A sentence-final colon left behind when _TXT_SOURCE_CITE_DUMP_RE removed
    # everything after it (the colon introduced the now-deleted dump) — collapse
    # ".:"/" :" at end of string down to the sentence's own terminal period.
    cleaned = re.sub(r'\s*:\s*$', '', cleaned)
    # A trailing comma/semicolon left where a citation list ended the text
    # ("...the 50bp cut, [tag], [tag]" → "...the 50bp cut,"). Never valid at the
    # end of an answer — drop it (and add back a period if the text now ends bare).
    cleaned = re.sub(r'[\s,;]+$', '', cleaned)
    return cleaned.strip()


def build_sources(
    docs:          List[Dict[str, Any]],
    snippet_chars: int = 240,
) -> List[Dict[str, Any]]:
    """Deterministic list of RetrievedSource records from final retrieved docs."""
    out: List[Dict[str, Any]] = []
    seen_cite: Dict[str, int] = {}
    for rank, d in enumerate(docs or []):
        rec = build_retrieved_source(d, rank=rank, snippet_chars=snippet_chars)
        key = rec["cite_key"]
        # Keep the first occurrence (highest-ranked) per cite_key; later dupes are dropped
        # so the citation tag is a closed set.
        if key in seen_cite:
            continue
        seen_cite[key] = rank
        out.append(rec)
    return out


def query_ok(
    answer:       str,
    session_id:   str,
    latency:      float,
    confidence:   float                  = 0.5,
    sources_used: int                    = 0,
    decision:     str                    = "rag",
    source:       str                    = "agent",
    metadata:     Optional[Dict[str, Any]] = None,
    trace_id:     Optional[str]          = None,
    cache_hit:    bool                   = False,
    sources:      Optional[List[Dict[str, Any]]] = None,
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        session_id=session_id,
        latency=latency,
        confidence=confidence,
        sources_used=sources_used,
        decision=decision,
        source=source,
        metadata=metadata,
        trace_id=trace_id,
        cache_hit=cache_hit,
        sources=sources,
    )