from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.config import settings


ALLOWED_MODALITIES = {"text", "table", "image", "audio", "video"}


class IngestedDocument(BaseModel):

    # CORE FIELDS
    text: str
    modality: str
    subtype: Optional[str] = None

    source_type: str = "file"
    source: Optional[str] = None

    page: Optional[int] = None
    chunk_id: Optional[int] = None

    # SAFE DEFAULTS (CRITICAL FIX)
    structure: Dict[str, Any] = Field(default_factory=dict)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    embedding: Optional[List[float]] = None

    #  NORMALIZATION 
    def normalize(self):

        # CLEAN TEXT
        self.text = (self.text or "").strip()

        if len(self.text) > settings.MAX_PROMPT_CHARS:
            self.text = self.text[:settings.MAX_PROMPT_CHARS]

        # NORMALIZE MODALITY
        self.modality = (self.modality or "").strip().lower()

        if self.subtype:
            self.subtype = self.subtype.strip().lower()

        self.source_type = (self.source_type or "file").strip()

        # ENSURE DICTS
        if self.structure is None:
            self.structure = {}

        if self.extra_metadata is None:
            self.extra_metadata = {}

        return self

    #  VALIDATION 
    def validate(self):

        # TEXT VALIDATION
        if not self.text or len(self.text.strip()) < 3:
            raise ValueError("TEXT TOO SHORT OR EMPTY")

        # MODALITY VALIDATION
        if self.modality not in ALLOWED_MODALITIES:
            raise ValueError(f"INVALID MODALITY: {self.modality}")

        # PAGE VALIDATION
        if self.page is not None and self.page < 1:
            raise ValueError("PAGE MUST BE >= 1")

        # CHUNK VALIDATION
        if self.chunk_id is not None and self.chunk_id < 0:
            raise ValueError("CHUNK_ID MUST BE >= 0")

        # STRUCTURE VALIDATION
        if not isinstance(self.structure, dict):
            raise ValueError("STRUCTURE MUST BE DICT")

        # REQUIRED STRUCTURE FIELDS
        self.structure.setdefault("doc_id", str(uuid4()))
        self.structure.setdefault("session_id", "default")

        # STANDARDIZE OPTIONAL FIELDS
        self.structure.setdefault("content_type", "unknown")
        self.structure.setdefault("embedding_space", "text")

        # METADATA VALIDATION
        if not isinstance(self.extra_metadata, dict):
            raise ValueError("EXTRA_METADATA MUST BE DICT")

        # EMBEDDING VALIDATION
        if self.embedding is not None:

            if not isinstance(self.embedding, list):
                raise ValueError("EMBEDDING MUST BE LIST")

            if len(self.embedding) not in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM,
            ):
                raise ValueError("INVALID EMBEDDING DIMENSION")

        return self

    #  FINALIZE 
    def finalize(self):
        self.normalize()
        self.validate()
        return self

    #  CLONE 
    def clone(self, **updates: Any) -> "IngestedDocument":
        data = self.to_dict()
        data.update(updates)
        return IngestedDocument(**data).finalize()

    #  SERIALIZATION 
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
        }