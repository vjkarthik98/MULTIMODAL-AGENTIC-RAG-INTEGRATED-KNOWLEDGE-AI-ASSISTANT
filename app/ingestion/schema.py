from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

try:  # pragma: no branch - compatibility shim
    from pydantic import field_validator

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator

    PYDANTIC_V2 = False


ALLOWED_MODALITIES = {"text", "table", "image", "audio", "video"}


def _validate_text_value(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError("text cannot be empty")
    return cleaned


def _validate_modality_value(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized not in ALLOWED_MODALITIES:
        raise ValueError(f"Invalid modality: {value}")
    return normalized


def _validate_source_type_value(value: Optional[str]) -> str:
    return (value or "file").strip() or "file"


def _validate_structure_value(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if value is None:
        structure: Dict[str, Any] = {}
    elif not isinstance(value, dict):
        raise ValueError("structure must be a dictionary")
    else:
        structure = dict(value)

    structure.setdefault("doc_id", str(uuid4()))
    structure.setdefault("session_id", "default")
    return structure


def _validate_extra_metadata_value(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("extra_metadata must be a dictionary")
    return dict(value)


class IngestedDocument(BaseModel):
    text: str = Field(..., min_length=1)

    modality: str
    subtype: Optional[str] = None

    source_type: str = "file"
    source: Optional[str] = None

    page: Optional[int] = Field(default=None, ge=1)
    chunk_id: Optional[int] = Field(default=None, ge=0)

    structure: Dict[str, Any] = Field(default_factory=dict)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    embedding: Optional[List[float]] = None

    if PYDANTIC_V2:

        @field_validator("text")
        @classmethod
        def validate_text(cls, value: str) -> str:
            return _validate_text_value(value)

        @field_validator("modality")
        @classmethod
        def validate_modality(cls, value: str) -> str:
            return _validate_modality_value(value)

        @field_validator("source_type", mode="before")
        @classmethod
        def validate_source_type(cls, value: Optional[str]) -> str:
            return _validate_source_type_value(value)

        @field_validator("structure", mode="before")
        @classmethod
        def validate_structure(cls, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return _validate_structure_value(value)

        @field_validator("extra_metadata", mode="before")
        @classmethod
        def validate_extra_metadata(cls, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return _validate_extra_metadata_value(value)

    else:

        @validator("text")
        def validate_text(cls, value: str) -> str:
            return _validate_text_value(value)

        @validator("modality")
        def validate_modality(cls, value: str) -> str:
            return _validate_modality_value(value)

        @validator("source_type", pre=True, always=True)
        def validate_source_type(cls, value: Optional[str]) -> str:
            return _validate_source_type_value(value)

        @validator("structure", pre=True, always=True)
        def validate_structure(cls, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return _validate_structure_value(value)

        @validator("extra_metadata", pre=True, always=True)
        def validate_extra_metadata(cls, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return _validate_extra_metadata_value(value)

    def clone(self, **updates: Any) -> "IngestedDocument":
        if hasattr(self, "model_copy"):
            return self.model_copy(update=updates, deep=True)
        return self.copy(update=updates, deep=True)

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    def summary(self) -> Dict[str, Any]:
        structure = self.structure or {}
        return {
            "modality": self.modality,
            "subtype": self.subtype,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "doc_id": structure.get("doc_id"),
            "session_id": structure.get("session_id"),
        }
