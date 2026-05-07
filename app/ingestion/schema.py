from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# CONSTANTS

ALLOWED_MODALITIES = {"text", "table", "image", "audio", "video"}
ALLOWED_SUBTYPES = {
    "text":  {"paragraph", "heading", "page", "chunk", "unknown"},
    "table": {"structured", "unknown"},
    "image": {"caption", "ocr", "unknown"},
    "audio": {"speech", "unknown"},
    "video": {"speech", "frame", "ocr", "unknown"},
}
ALLOWED_EMBEDDING_SPACES = {"text", "vision"}

MODALITY_QUALITY_FLOORS: Dict[str, float] = {
    "text":  0.1,
    "table": 0.1,
    "image": 0.0,
    "audio": 0.0,
    "video": 0.0,
}


# SCHEMA

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
    structure:      Dict[str, Any] = Field(default_factory=dict)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding:      Optional[List[float]] = None

    # NORMALIZE

    def normalize(self) -> "IngestedDocument":

        self.text = (self.text or "").strip()

        if not self.text:
            raise ValueError("EMPTY_TEXT")

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

        # TEXT
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

        # EMBEDDING
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
        data = self.to_dict()
        data.update(updates)
        return IngestedDocument(**data).finalize()

    # QUALITY CHECK

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

    # SERIALIZE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text":           self.text,
            "modality":       self.modality,
            "subtype":        self.subtype,
            "source_type":    self.source_type,
            "source":         self.source,
            "page":           self.page,
            "chunk_id":       self.chunk_id,
            "structure":      self.structure,
            "extra_metadata": self.extra_metadata,
            "embedding":      self.embedding,
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
        }