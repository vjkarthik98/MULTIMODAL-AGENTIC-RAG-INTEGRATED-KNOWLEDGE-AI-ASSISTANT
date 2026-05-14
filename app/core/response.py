from __future__ import annotations

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
        self.sources_used = sources_used
        self.decision     = decision
        self.source       = source
        self.cache_hit    = cache_hit

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "answer":       self.answer,
            "confidence":   self.confidence,
            "sources_used": self.sources_used,
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
    )