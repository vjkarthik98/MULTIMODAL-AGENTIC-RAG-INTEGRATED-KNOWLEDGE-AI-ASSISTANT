from __future__ import annotations

import hashlib
import time
import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from copy import deepcopy

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# RAW EXTRACT — lightweight intermediary between ingestor and chunker (Phase 1)
# Each ingestor returns List[RawExtract]; chunkers consume them and produce List[IngestedDocument].

@dataclass
class RawExtract:
    text: str                                # raw extracted text (prose, row, transcript, caption)
    extract_type: str                        # "prose"|"table_row"|"heading"|"footnote"|"header_footer"
                                             # "image_raw"|"image_region"|"scanned_page"
                                             # "audio_raw"|"video_raw"|"segment"|"frame"
                                             # "named_range"|"unit_header"|"chart_image"
                                             # "comment"|"speaker_turn"
    page: Optional[int] = None              # PDF page number (1-indexed)
    sheet: Optional[str] = None             # Excel sheet name
    bbox: Optional[Tuple[float, float, float, float]] = None   # (x0, y0, x1, y1)
    timestamp_start: Optional[float] = None # audio/video seconds
    timestamp_end: Optional[float] = None
    font_size: Optional[float] = None       # PDF font size in points
    is_bold: bool = False                   # PDF/DOCX bold flag
    speaker_label: Optional[str] = None     # audio/video speaker ID
    raw_source_ref: str = ""                # e.g. "page_12", "sheet_Income_Statement", "0:28:30"
    raw_bytes: Optional[bytes] = None       # image/audio/video binary content
    style_name: Optional[str] = None        # DOCX paragraph style name
    extra: Dict[str, Any] = dc_field(default_factory=dict)  # modality-specific overflow


# MODALITY ENUM

class Modality(str, Enum):
    TEXT  = "text"
    PDF   = "pdf"
    WORD  = "word"
    EXCEL = "excel"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


# PROCESSING STATUS ENUM

class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    SUCCESS    = "success"
    FAILED     = "failed"
    SKIPPED    = "skipped"
    DUPLICATE  = "duplicate"


# EMBEDDING SPACE ENUM

class EmbeddingSpace(str, Enum):
    TEXT   = "text"
    VISION = "vision"


# ALLOWED MODALITIES AND SUBTYPES

ALLOWED_MODALITIES = {
    "text", "table", "pdf",
    # extension-based names used by per-modality chunkers
    "txt",                       # txt_chunker
    "image", "jpg", "jpeg", "png",  # image_chunker
    "audio", "mp3", "wav",       # audio_chunker
    "video", "mp4",              # video_chunker
    "word", "docx",              # docx_chunker
    "excel", "xlsx",             # xlsx_chunker
}

_TXT_SUBTYPES   = {"paragraph", "heading", "section", "list", "speaker_turn", "unknown"}
_IMAGE_SUBTYPES = {"caption", "ocr", "image_frame", "unknown"}
_AUDIO_SUBTYPES = {"speech", "unknown"}
_VIDEO_SUBTYPES = {"speech", "frame", "ocr", "transcript_frame", "unknown"}
_WORD_SUBTYPES  = {"paragraph", "table", "list", "heading", "annotation", "definition", "unknown"}
_EXCEL_SUBTYPES = {"table_row_group", "assumptions", "named_ranges", "chart_caption", "unknown"}

ALLOWED_SUBTYPES: Dict[str, set] = {
    "text":  {"paragraph", "heading", "page", "chunk", "speaker_turn", "section", "list", "unknown"},
    "table": {"structured", "unknown"},
    "pdf":   {"paragraph", "table", "footnote", "figure_caption", "list", "heading", "ocr", "unknown"},
    # extension-based aliases share subtype sets with their semantic counterparts
    "txt":  _TXT_SUBTYPES,
    "image": _IMAGE_SUBTYPES,
    "jpg":   _IMAGE_SUBTYPES,
    "jpeg":  _IMAGE_SUBTYPES,
    "png":   _IMAGE_SUBTYPES,
    "audio": _AUDIO_SUBTYPES,
    "mp3":   _AUDIO_SUBTYPES,
    "wav":   _AUDIO_SUBTYPES,
    "video": _VIDEO_SUBTYPES,
    "mp4":   _VIDEO_SUBTYPES,
    "word":  _WORD_SUBTYPES,
    "docx":  _WORD_SUBTYPES,
    "excel": _EXCEL_SUBTYPES,
    "xlsx":  _EXCEL_SUBTYPES,
}

ALLOWED_EMBEDDING_SPACES = {"text", "vision"}

MODALITY_QUALITY_FLOORS: Dict[str, float] = {
    "text":  0.1,
    "table": 0.1,
    "pdf":   0.1,
    "txt":   0.1,
    "image": 0.0,
    "jpg":   0.0,
    "jpeg":  0.0,
    "png":   0.0,
    "audio": 0.0,
    "mp3":   0.0,
    "wav":   0.0,
    "video": 0.0,
    "mp4":   0.0,
    "word":  0.1,
    "docx":  0.1,
    "excel": 0.1,
    "xlsx":  0.1,
}


# UNIVERSAL METADATA 

class UniversalMetadata(BaseModel):
    # UNIQUE INGESTION EVENT ID
    file_id: UUID = Field(default_factory=uuid4)

    # SOURCE
    source_path: str = Field(default="")
    modality: str    = Field(default="text")

    # MIME — MAGIC-BYTE DETECTED; NEVER TRUST EXTENSION
    mime_type: str = Field(default="application/octet-stream")

    # SIZE AND CHECKSUM
    file_size_bytes: int   = Field(default=0, ge=0)
    checksum_sha256: str   = Field(default="")

    # TIMESTAMPS
    created_at: Optional[float]  = Field(default=None)
    modified_at: Optional[float] = Field(default=None)
    ingested_at: float           = Field(default_factory=time.time)

    # CONTENT PROPERTIES
    language: str    = Field(default="unknown")
    encoding: str    = Field(default="utf-8")

    # DOCUMENT STRUCTURE
    page_count: Optional[int]    = Field(default=None, ge=0)
    duration_s: Optional[float]  = Field(default=None, ge=0.0)

    # PROCESSING RESULTS
    chunk_count: int             = Field(default=0, ge=0)
    embedding_model: str         = Field(default="")
    extraction_quality: float    = Field(default=1.0, ge=0.0)

    # NON-FATAL ERRORS DURING PROCESSING
    error_log: List[str] = Field(default_factory=list)

    # AUTO-GENERATED SEMANTIC TAGS (KEYBERT/YAKE)
    tags: List[str] = Field(default_factory=list)

    # MODALITY-SPECIFIC EXTRAS
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

    # PII REDACTION FLAG
    pii_redacted: bool          = Field(default=False)
    pii_entities_found: int     = Field(default=0)

    # DEDUP FLAG
    is_duplicate: bool          = Field(default=False)
    duplicate_of: Optional[str] = Field(default=None)

    # SECURITY FLAGS
    malware_scanned: bool       = Field(default=False)
    malware_clean: bool         = Field(default=True)
    password_protected: bool    = Field(default=False)
    has_javascript: bool        = Field(default=False)
    has_macros: bool            = Field(default=False)
    has_signatures: bool        = Field(default=False)

    # PROCESSING STATUS
    status: str = Field(default=ProcessingStatus.PENDING)

    model_config = {"use_enum_values": True}

    @field_validator("checksum_sha256")
    @classmethod
    def validate_checksum(cls, v: str) -> str:
        if v and len(v) != 64:
            raise ValueError(f"SHA-256 checksum must be 64 hex chars, got {len(v)}")
        return v.lower() if v else v

    @field_validator("extraction_quality")
    @classmethod
    def clamp_quality(cls, v: float) -> float:
        return max(0.0, min(float(v), 1.0))

    def add_error(self, msg: str) -> None:
        self.error_log.append(msg)
        logger.warning(event="metadata_error_logged", error_msg=msg, file_id=str(self.file_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id":            str(self.file_id),
            "source_path":        self.source_path,
            "modality":           self.modality,
            "mime_type":          self.mime_type,
            "file_size_bytes":    self.file_size_bytes,
            "checksum_sha256":    self.checksum_sha256,
            "created_at":         self.created_at,
            "modified_at":        self.modified_at,
            "ingested_at":        self.ingested_at,
            "language":           self.language,
            "encoding":           self.encoding,
            "page_count":         self.page_count,
            "duration_s":         self.duration_s,
            "chunk_count":        self.chunk_count,
            "embedding_model":    self.embedding_model,
            "extraction_quality": self.extraction_quality,
            "error_log":          self.error_log,
            "tags":               self.tags,
            "custom_fields":      self.custom_fields,
            "pii_redacted":       self.pii_redacted,
            "pii_entities_found": self.pii_entities_found,
            "is_duplicate":       self.is_duplicate,
            "duplicate_of":       self.duplicate_of,
            "malware_scanned":    self.malware_scanned,
            "malware_clean":      self.malware_clean,
            "password_protected": self.password_protected,
            "has_javascript":     self.has_javascript,
            "has_macros":         self.has_macros,
            "has_signatures":     self.has_signatures,
            "status":             self.status,
        }


# INGESTED DOCUMENT SCHEMA

class IngestedDocument(BaseModel):

    # CORE FIELDS
    text:        str
    modality:    str
    subtype:     Optional[str] = None
    source_type: str           = "file"
    source:      Optional[str] = None
    page:        Optional[int] = None
    chunk_id:    Optional[int] = None

    # STRUCTURED FIELDS
    structure:      Dict[str, Any]        = Field(default_factory=dict)
    extra_metadata: Dict[str, Any]        = Field(default_factory=dict)
    embedding:      Optional[List[float]] = None
    embedding_alt:  Optional[List[float]] = None   # secondary embedding (e.g. xlsx markdown repr)

    # UNIVERSAL METADATA ATTACHED AT INGEST TIME
    universal_metadata: Optional[UniversalMetadata] = None

    # NORMALIZE

    def normalize(self) -> "IngestedDocument":
        # NFC UNICODE NORMALIZATION — SECTION 2.3
        self.text = unicodedata.normalize("NFC", (self.text or "").strip())

        if not self.text:
            raise ValueError("EMPTY_TEXT")

        # STRIP NULL BYTES — SECTION 2.3
        if settings.STRIP_NULL_BYTES and "\x00" in self.text:
            null_count = self.text.count("\x00")
            self.text = self.text.replace("\x00", "")
            logger.warning(event="null_bytes_stripped", count=null_count)

        self.modality    = (self.modality or "").strip().lower()
        self.source_type = (self.source_type or "file").strip().lower()

        if self.subtype:
            self.subtype = self.subtype.strip().lower()

        if self.source:
            self.source = self.source.strip()

        if self.structure is None:
            self.structure = {}

        if self.extra_metadata is None:
            self.extra_metadata = {}

        return self

    # VALIDATE

    def validate(self) -> "IngestedDocument":

        # TEXT LENGTH
        if len(self.text) < 3:
            raise ValueError("TEXT_TOO_SHORT")

        # MODALITY
        if self.modality not in ALLOWED_MODALITIES:
            raise ValueError(f"INVALID_MODALITY_{self.modality}")

        # SUBTYPE
        if self.subtype:
            allowed_sub = ALLOWED_SUBTYPES.get(self.modality, set())
            if self.subtype not in allowed_sub:
                logger.warning(
                    event="unknown_subtype",
                    modality=self.modality,
                    subtype=self.subtype,
                )
                self.subtype = "unknown"

        # PAGE
        if self.page is not None and self.page < 1:
            raise ValueError("INVALID_PAGE")

        # CHUNK ID
        if self.chunk_id is not None and self.chunk_id < 0:
            raise ValueError("INVALID_CHUNK_ID")

        # STRUCTURE
        if not isinstance(self.structure, dict):
            raise ValueError("INVALID_STRUCTURE")

        self.structure.setdefault("doc_id",          str(uuid4()))
        self.structure.setdefault("session_id",      "default")
        self.structure.setdefault("content_type",    "unknown")
        self.structure.setdefault("embedding_space", "text")

        if self.structure["embedding_space"] not in ALLOWED_EMBEDDING_SPACES:
            raise ValueError("INVALID_EMBEDDING_SPACE")

        # EXTRA METADATA
        if not isinstance(self.extra_metadata, dict):
            raise ValueError("INVALID_EXTRA_METADATA")

        floor = MODALITY_QUALITY_FLOORS.get(self.modality, 0.0)

        self.extra_metadata.setdefault("importance_score",    1.0)
        self.extra_metadata.setdefault("modality_weight",     1.0)
        self.extra_metadata.setdefault("data_quality_score",  1.0)

        # CLAMP SCORES
        for key in ("importance_score", "modality_weight", "data_quality_score"):
            try:
                val = float(self.extra_metadata[key])
                self.extra_metadata[key] = max(floor, min(val, 1.0))
            except (TypeError, ValueError):
                self.extra_metadata[key] = 1.0

        # EMBEDDING VALIDATION
        if self.embedding is not None:
            if not isinstance(self.embedding, list):
                raise ValueError("INVALID_EMBEDDING")

            dim = len(self.embedding)
            if dim not in (settings.TEXT_EMBEDDING_DIM, settings.VISION_EMBEDDING_DIM):
                raise ValueError(
                    f"INVALID_EMBEDDING_DIM: got {dim}, "
                    f"expected {settings.TEXT_EMBEDDING_DIM} or {settings.VISION_EMBEDDING_DIM}"
                )

        return self

    # FINALIZE

    def finalize(self) -> "IngestedDocument":
        return self.normalize().validate()

    # CLONE

    def clone(self, **updates: Any) -> "IngestedDocument":
        data = deepcopy(self.to_dict())

        data.update(updates)

        data.setdefault("structure", {})

        data["structure"]["doc_id"] = str(uuid4())

        # UNIVERSAL METADATA NOT CLONED INTO DICT — PASS SEPARATELY
        data.pop("universal_metadata", None)
        return IngestedDocument(**data).finalize()

    # QUALITY HELPERS

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        score = self.extra_metadata.get("data_quality_score", 1.0)
        return float(score) >= threshold

    def is_embeddable(self) -> bool:
        return (
            self.embedding is not None
            and isinstance(self.embedding, list)
            and len(self.embedding) in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM,
            )
        )

    def embedding_space(self) -> str:
        return self.structure.get("embedding_space", "text")

    def doc_id(self) -> str:
        return self.structure.get("doc_id", "")

    def session_id(self) -> str:
        return self.structure.get("session_id", "default")

    # SHA-256 CONTENT HASH FOR DEDUP — SECTION 2.3

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    # SERIALIZE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text":               self.text,
            "modality":           self.modality,
            "subtype":            self.subtype,
            "source_type":        self.source_type,
            "source":             self.source,
            "page":               self.page,
            "chunk_id":           self.chunk_id,
            "structure":          self.structure,
            "extra_metadata":     self.extra_metadata,
            "embedding":          self.embedding,
            "universal_metadata": self.universal_metadata.to_dict() if self.universal_metadata else None,
        }

    # SUMMARY

    def summary(self) -> Dict[str, Any]:
        return {
            "modality":        self.modality,
            "subtype":         self.subtype,
            "source":          self.source,
            "chunk_id":        self.chunk_id,
            "doc_id":          self.doc_id(),
            "session_id":      self.session_id(),
            "embedding_space": self.embedding_space(),
            "quality":         self.extra_metadata.get("data_quality_score", 1.0),
            "has_embedding":   self.is_embeddable(),
            "text_length":     len(self.text),
            "content_hash":    self.content_hash(),
        }

def normalize_text(text: str) -> str:

    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Strip BOM
    if settings.STRIP_BOM:
        text = text.replace("\ufeff", "")

    # Strip null bytes
    if settings.STRIP_NULL_BYTES:
        text = text.replace("\x00", "")

    return text.strip()

def redact_pii(text: str) -> str:

    if not text:
        return ""

    if not settings.PII_DETECTION_ENABLED:
        return text

    # EMAILS
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[REDACTED_EMAIL]",
        text,
    )

    # PHONE NUMBERS
    text = re.sub(
        r"\b\d{10,15}\b",
        "[REDACTED_PHONE]",
        text,
    )

    # CREDIT CARD LIKE NUMBERS
    text = re.sub(
        r"\b(?:\d[ -]*?){13,16}\b",
        "[REDACTED_CARD]",
        text,
    )

    return text

def sanitize_prompt_injection(text: str) -> str:
  
    if not text:
        return ""

    suspicious_patterns = [
        r"ignore\s+previous\s+instructions",
        r"system\s*:",
        r"assistant\s*:",
        r"developer\s*:",
        r"tool\s*:",
        r"<\s*system\s*>",
        r"<\s*/system\s*>",
        r"BEGIN\s+PROMPT",
        r"END\s+PROMPT",
    ]

    sanitized = text

    for pattern in suspicious_patterns:
        sanitized = re.sub(
            pattern,
            "[FILTERED]",
            sanitized,
            flags=re.IGNORECASE,
        )

    return sanitized
    
def build_universal_metadata(
    source_path: str = "",
    modality: str = "text",
    mime_type: str = "application/octet-stream",
    file_size_bytes: int = 0,
    checksum_sha256: str = "",
    **kwargs,
) -> UniversalMetadata:


    return UniversalMetadata(
        source_path=source_path,
        modality=modality,
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
        checksum_sha256=checksum_sha256,
        **kwargs,
    )

def sha256_file(file_path: str) -> str:

    h = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)

    return h.hexdigest()

from app.core.response import ProcessingResult  # noqa: F401  (re-exported for ingestion callers)

# ERROR DETAIL

class ErrorDetail(BaseModel):
    code:     str
    message:  str
    severity: str                    = "medium"
    field:    Optional[str]          = None
    context:  Dict[str, Any]         = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code":     self.code,
            "message":  self.message,
            "severity": self.severity,
            "field":    self.field,
            "context":  self.context,
        }


# VALIDATION RESULT

class ValidationResult(BaseModel):
    valid:     bool
    modality:  str   = "unknown"
    file_size: int   = 0
    mime_type: str   = ""
    errors:    List[ErrorDetail] = Field(default_factory=list)
    warnings:  List[str]         = Field(default_factory=list)
    latency:   float             = 0.0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(
        self,
        code: str,
        message: str,
        severity: str = "high",
        field: Optional[str] = None,
    ) -> None:
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


# QUERY RESPONSE

class QueryResponse(BaseModel):
    success:      bool  = True
    answer:       str   = ""
    confidence:   float = Field(default=0.5, ge=0.0, le=1.0)
    sources_used: int   = 0
    decision:     str   = "unknown"
    source:       str   = "agent"
    session_id:   str   = "default"
    latency:      float = 0.0
    metadata:     Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":      self.success,
            "answer":       self.answer,
            "confidence":   self.confidence,
            "sources_used": self.sources_used,
            "decision":     self.decision,
            "source":       self.source,
            "session_id":   self.session_id,
            "latency":      self.latency,
            "metadata":     self.metadata,
        }


# FACTORY HELPERS

def ok(
    modality: str,
    session_id: str,
    latency: float,
    chunks: int = 0,
    stored: int = 0,
    source: str = "",
    metadata: Optional[UniversalMetadata] = None,
    warnings: Optional[List[str]] = None,
) -> ProcessingResult:
    return ProcessingResult(
        success=True,
        modality=modality,
        session_id=session_id,
        latency=latency,
        chunks=chunks,
        stored=stored,
        source=source,
        metadata=metadata,
        warnings=warnings or [],
    )


def err(
    code: str,
    message: str,
    modality: str = "unknown",
    session_id: str = "default",
    latency: float = 0.0,
    severity: str = "medium",
    field: Optional[str] = None,
) -> ProcessingResult:
    result = ProcessingResult(
        success=False,
        modality=modality,
        session_id=session_id,
        latency=latency,
    )
    result.errors.append(
        ErrorDetail(code=code, message=message, severity=severity, field=field)
    )
    return result


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
        warnings=warnings or [],
    )


def validation_err(
    code: str,
    message: str,
    modality: str = "unknown",
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






# CUSTOM EXCEPTIONS

class EmptyFileError(ValueError):
    pass

class FileTooLargeError(ValueError):
    pass

class UnsupportedMimeError(ValueError):
    pass

class CorruptFileError(ValueError):
    pass

class DuplicateFileError(ValueError):
    pass

class PasswordProtectedError(ValueError):
    pass

class EmptyContentError(ValueError):
    pass

class MalwareDetectedError(RuntimeError):
    pass

class DiskSpaceError(RuntimeError):
    pass


