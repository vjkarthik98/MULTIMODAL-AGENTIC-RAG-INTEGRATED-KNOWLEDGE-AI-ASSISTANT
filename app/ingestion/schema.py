from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from app.core.config import settings


ALLOWED_MODALITIES = {"text", "table", "image", "audio", "video"}
ALLOWED_EMBEDDING_SPACES = {"text", "vision"}


class IngestedDocument(BaseModel):

    #  CORE 
    text: str
    modality: str
    subtype: Optional[str] = None

    source_type: str = "file"
    source: Optional[str] = None

    page: Optional[int] = None
    chunk_id: Optional[int] = None

    #  STRUCTURE 
    structure: Dict[str, Any] = Field(default_factory=dict)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    embedding: Optional[List[float]] = None

    #  NORMALIZATION 
    def normalize(self) -> "IngestedDocument":

        self.text = (self.text or "").strip()

        # STRICT: NO BLIND TRUNCATION (critical fix)
        if not self.text:
            raise ValueError("EMPTY_TEXT")

        self.modality = (self.modality or "").strip().lower()

        if self.subtype:
            self.subtype = self.subtype.strip().lower()

        self.source_type = (self.source_type or "file").strip().lower()

        if self.structure is None:
            self.structure = {}

        if self.extra_metadata is None:
            self.extra_metadata = {}

        return self

    #  VALIDATION 
    def validate(self) -> "IngestedDocument":

        # TEXT VALIDATION
        if len(self.text) < 3:
            raise ValueError("TEXT_TOO_SHORT")

        # MODALITY
        if self.modality not in ALLOWED_MODALITIES:
            raise ValueError(f"INVALID_MODALITY_{self.modality}")

        # PAGE
        if self.page is not None and self.page < 1:
            raise ValueError("INVALID_PAGE")

        # CHUNK
        if self.chunk_id is not None and self.chunk_id < 0:
            raise ValueError("INVALID_CHUNK_ID")

        # STRUCTURE
        if not isinstance(self.structure, dict):
            raise ValueError("INVALID_STRUCTURE")

        # REQUIRED FIELDS
        self.structure.setdefault("doc_id", str(uuid4()))
        self.structure.setdefault("session_id", "default")

        # STANDARD FIELDS
        self.structure.setdefault("content_type", "unknown")
        self.structure.setdefault("embedding_space", "text")

        # VALIDATE EMBEDDING SPACE
        if self.structure["embedding_space"] not in ALLOWED_EMBEDDING_SPACES:
            raise ValueError("INVALID_EMBEDDING_SPACE")

        # EXTRA METADATA
        if not isinstance(self.extra_metadata, dict):
            raise ValueError("INVALID_EXTRA_METADATA")

        self.extra_metadata.setdefault("importance_score", 1.0)
        self.extra_metadata.setdefault("modality_weight", 1.0)
        self.extra_metadata.setdefault("data_quality_score", 1.0)

        # EMBEDDING
        if self.embedding is not None:
            if not isinstance(self.embedding, list):
                raise ValueError("INVALID_EMBEDDING")

            dim = len(self.embedding)

            if dim not in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM,
            ):
                raise ValueError("INVALID_EMBEDDING_DIM")

        return self

    #  FINALIZE 
    def finalize(self) -> "IngestedDocument":
        return self.normalize().validate()

    #  CLONE 
    def clone(self, **updates: Any) -> "IngestedDocument":
        data = self.to_dict()
        data.update(updates)
        return IngestedDocument(**data).finalize()

    #  SERIALIZE 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "modality": self.modality,
            "subtype": self.subtype,
            "source_type": self.source_type,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "structure": self.structure,
            "extra_metadata": self.extra_metadata,
            "embedding": self.embedding,
        }

    #  SUMMARY 
    def summary(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "subtype": self.subtype,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "doc_id": self.structure.get("doc_id"),
            "session_id": self.structure.get("session_id"),
            "embedding_space": self.structure.get("embedding_space"),
        }