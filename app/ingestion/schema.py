from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel

from app.core.config import settings


ALLOWED_MODALITIES = {"text", "table", "image", "audio", "video"}


class IngestedDocument(BaseModel):
    text: str
    modality: str
    subtype: Optional[str] = None

    source_type: str = "file"
    source: Optional[str] = None

    page: Optional[int] = None
    chunk_id: Optional[int] = None

    structure: Dict[str, Any] = {}
    extra_metadata: Dict[str, Any] = {}

    embedding: Optional[List[float]] = None

    # NORMALIZATION 

    def normalize(self):
        self.text = (self.text or "").strip()

        if len(self.text) > settings.MAX_PROMPT_CHARS:
            self.text = self.text[:settings.MAX_PROMPT_CHARS]

        self.modality = (self.modality or "").strip().lower()

        if self.subtype:
            self.subtype = self.subtype.strip().lower()

        self.source_type = (self.source_type or "file").strip()

        if self.structure is None:
            self.structure = {}

        if self.extra_metadata is None:
            self.extra_metadata = {}

        return self

    # VALIDATION 

    def validate(self):
        if not self.text:
            raise ValueError("text cannot be empty")

        if self.modality not in ALLOWED_MODALITIES:
            raise ValueError(f"Invalid modality: {self.modality}")

        if self.page is not None and self.page < 1:
            raise ValueError("page must be >= 1")

        if self.chunk_id is not None and self.chunk_id < 0:
            raise ValueError("chunk_id must be >= 0")

        # structure defaults
        if not isinstance(self.structure, dict):
            raise ValueError("structure must be dict")

        self.structure.setdefault("doc_id", str(uuid4()))
        self.structure.setdefault("session_id", "default")

        # metadata validation
        if not isinstance(self.extra_metadata, dict):
            raise ValueError("extra_metadata must be dict")

        # embedding validation
        if self.embedding is not None:
            if not isinstance(self.embedding, list):
                raise ValueError("embedding must be list")

            if len(self.embedding) not in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM,
            ):
                # soft warning (not crash)
                pass

        return self

    # FINALIZE 

    def finalize(self):
        self.normalize()
        self.validate()
        return self

    # UTILITIES 

    def clone(self, **updates: Any) -> "IngestedDocument":
        data = self.to_dict()
        data.update(updates)
        return IngestedDocument(**data).finalize()

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

    def summary(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "subtype": self.subtype,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "doc_id": self.structure.get("doc_id"),
            "session_id": self.structure.get("session_id"),
        }