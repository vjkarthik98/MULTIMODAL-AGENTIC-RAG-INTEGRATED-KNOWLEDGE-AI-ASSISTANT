import time
from enum import Enum
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ERROR CODES

class ErrorCode(str, Enum):

    # VALIDATION
    EMPTY_FILE              = "EMPTY_FILE"
    FILE_TOO_LARGE          = "FILE_TOO_LARGE"
    UNSUPPORTED_TYPE        = "UNSUPPORTED_TYPE"
    INVALID_MIME            = "INVALID_MIME"
    CORRUPTED_FILE          = "CORRUPTED_FILE"
    INVALID_DIMENSIONS      = "INVALID_DIMENSIONS"
    IMAGE_TOO_SMALL         = "IMAGE_TOO_SMALL"
    INVALID_AUDIO_DURATION  = "INVALID_AUDIO_DURATION"
    INVALID_SAMPLE_RATE     = "INVALID_SAMPLE_RATE"
    INVALID_WAV_HEADER      = "INVALID_WAV_HEADER"
    INVALID_PDF_STRUCTURE   = "INVALID_PDF_STRUCTURE"
    INVALID_EXCEL_STRUCTURE = "INVALID_EXCEL_STRUCTURE"
    EMPTY_DOCUMENT          = "EMPTY_DOCUMENT"
    SESSION_ID_REQUIRED     = "SESSION_ID_REQUIRED"
    EMPTY_QUERY             = "EMPTY_QUERY"

    # PROCESSING
    OCR_FAILED              = "OCR_FAILED"
    CAPTION_FAILED          = "CAPTION_FAILED"
    TRANSCRIPTION_FAILED    = "TRANSCRIPTION_FAILED"
    FRAME_EXTRACTION_FAILED = "FRAME_EXTRACTION_FAILED"
    EMBEDDING_FAILED        = "EMBEDDING_FAILED"
    INGESTION_FAILED        = "INGESTION_FAILED"
    CHUNKING_FAILED         = "CHUNKING_FAILED"
    VECTOR_INSERT_FAILED    = "VECTOR_INSERT_FAILED"

    # RETRIEVAL
    NO_RESULTS_FOUND        = "NO_RESULTS_FOUND"
    RETRIEVAL_FAILED        = "RETRIEVAL_FAILED"
    RERANK_FAILED           = "RERANK_FAILED"

    # AGENT
    AGENT_TIMEOUT           = "AGENT_TIMEOUT"
    AGENT_FAILED            = "AGENT_FAILED"
    INVALID_PLAN            = "INVALID_PLAN"
    TOOL_NOT_FOUND          = "TOOL_NOT_FOUND"

    # INFRASTRUCTURE
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    REDIS_UNAVAILABLE        = "REDIS_UNAVAILABLE"
    MONGO_UNAVAILABLE        = "MONGO_UNAVAILABLE"
    LLM_UNAVAILABLE          = "LLM_UNAVAILABLE"
    MODEL_TIMEOUT            = "MODEL_TIMEOUT"

    # GENERIC
    INTERNAL_ERROR          = "INTERNAL_ERROR"
    NOT_IMPLEMENTED         = "NOT_IMPLEMENTED"
    UNKNOWN_ERROR           = "UNKNOWN_ERROR"


# SEVERITY

class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# MODALITY

class Modality(str, Enum):
    TEXT     = "text"
    IMAGE    = "image"
    AUDIO    = "audio"
    VIDEO    = "video"
    PDF      = "pdf"
    DOCX     = "docx"
    XLSX     = "xlsx"
    UNKNOWN  = "unknown"


# BASE RESPONSE

class BaseResponse:

    def __init__(
        self,
        success: bool,
        modality: str = Modality.UNKNOWN,
        session_id: str = "default",
        latency: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.success    = success
        self.modality   = modality
        self.session_id = session_id
        self.latency    = latency
        self.metadata   = metadata or {}
        self.timestamp  = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":    self.success,
            "modality":   self.modality,
            "session_id": self.session_id,
            "latency":    self.latency,
            "metadata":   self.metadata,
            "timestamp":  self.timestamp,
        }


# SUCCESS RESPONSE

class ProcessingResult(BaseResponse):

    def __init__(
        self,
        modality: str,
        session_id: str,
        latency: float,
        chunks: int = 0,
        stored: int = 0,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            success=True,
            modality=modality,
            session_id=session_id,
            latency=latency,
            metadata=metadata,
        )
        self.chunks   = chunks
        self.stored   = stored
        self.source   = source
        self.warnings = warnings or []

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "chunks":   self.chunks,
            "stored":   self.stored,
            "source":   self.source,
            "warnings": self.warnings,
        })
        return base


# ERROR RESPONSE

class ErrorDetail:

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        severity: Severity = Severity.MEDIUM,
        field: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
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


class UniversalErrorResponse(BaseResponse):

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        modality: str = Modality.UNKNOWN,
        session_id: str = "default",
        latency: float = 0.0,
        severity: Severity = Severity.MEDIUM,
        field: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            success=False,
            modality=modality,
            session_id=session_id,
            latency=latency,
            metadata=metadata,
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
        )

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["error"] = self.error.to_dict()
        return base


# QUERY RESPONSE

class QueryResponse(BaseResponse):

    def __init__(
        self,
        answer: str,
        session_id: str,
        latency: float,
        confidence: float = 0.5,
        sources_used: int = 0,
        decision: str = "unknown",
        source: str = "agent",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            success=True,
            modality=Modality.TEXT,
            session_id=session_id,
            latency=latency,
            metadata=metadata,
        )
        self.answer      = answer
        self.confidence  = max(0.0, min(confidence, 1.0))
        self.sources_used = sources_used
        self.decision    = decision
        self.source      = source

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "answer":       self.answer,
            "confidence":   self.confidence,
            "sources_used": self.sources_used,
            "decision":     self.decision,
            "source":       self.source,
        })
        return base


# VALIDATION RESPONSE

class ValidationResult:

    def __init__(
        self,
        valid: bool,
        modality: str = Modality.UNKNOWN,
        file_size: int = 0,
        mime_type: str = "",
        errors: Optional[List[ErrorDetail]] = None,
        warnings: Optional[List[str]] = None,
        latency: float = 0.0,
    ) -> None:
        self.valid     = valid
        self.modality  = modality
        self.file_size = file_size
        self.mime_type = mime_type
        self.errors    = errors or []
        self.warnings  = warnings or []
        self.latency   = latency

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(self, code: ErrorCode, message: str, severity: Severity = Severity.HIGH, field: Optional[str] = None) -> None:
        self.errors.append(ErrorDetail(code=code, message=message, severity=severity, field=field))
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
        }


# FACTORIES

def ok(
    modality: str,
    session_id: str,
    latency: float,
    chunks: int = 0,
    stored: int = 0,
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
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
    )


def err(
    code: ErrorCode,
    message: str,
    modality: str = Modality.UNKNOWN,
    session_id: str = "default",
    latency: float = 0.0,
    severity: Severity = Severity.MEDIUM,
    field: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
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
    )


def validation_ok(
    modality: str,
    file_size: int,
    mime_type: str,
    latency: float = 0.0,
    warnings: Optional[List[str]] = None,
) -> ValidationResult:
    return ValidationResult(
        valid=True,
        modality=modality,
        file_size=file_size,
        mime_type=mime_type,
        latency=latency,
        warnings=warnings,
    )


def validation_err(
    code: ErrorCode,
    message: str,
    modality: str = Modality.UNKNOWN,
    file_size: int = 0,
    mime_type: str = "",
    latency: float = 0.0,
    field: Optional[str] = None,
) -> ValidationResult:
    result = ValidationResult(
        valid=False,
        modality=modality,
        file_size=file_size,
        mime_type=mime_type,
        latency=latency,
    )
    result.add_error(code=code, message=message, field=field)
    return result