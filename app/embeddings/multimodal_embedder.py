import hashlib
import time
from typing import List, Tuple, Dict
from pathlib import Path

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalEmbedder:

    def __init__(self, text_embedder, image_embedder):
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder

        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.max_docs = getattr(settings, "MAX_PARALLEL_REQUESTS", 100)

    #  HASH 
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    #  PATH 
    def _resolve_asset_path(self, doc) -> str:
        structure = getattr(doc, "structure", {}) or {}

        path = structure.get("asset_path") or structure.get("frame_path")
        if not path:
            return None

        p = Path(path)
        if not p.exists():
            logger.warning(event="asset_missing", path=str(path))
            return None

        return str(p)

    #  VALID 
    def _valid(self, emb, dim: int) -> bool:
        return isinstance(emb, list) and len(emb) == dim

    #  ROUTE 
    def _route(self, docs):

        text_docs = []
        vision_docs = []

        for doc in docs:
            try:
                m = getattr(doc, "modality", "")
                st = getattr(doc, "subtype", "")

                if m in {"text", "table", "audio"}:
                    text_docs.append(doc)

                elif m == "image":
                    text_docs.append(doc)  # caption + OCR
                    if st == "caption":
                        vision_docs.append(doc)

                elif m == "video":
                    if st == "speech":
                        text_docs.append(doc)
                    elif st == "frame":
                        vision_docs.append(doc)

            except Exception as e:
                logger.warning(event="route_failed", error=str(e))

        return text_docs, vision_docs

    #  MAIN 
    def embed_documents(
        self,
        documents: List,
        session_id: str
    ) -> Tuple[List, List]:

        if not documents:
            return [], []

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        start = time.time()

        docs = documents[:self.max_docs]

        # stable dedup
        seen: Dict[str, bool] = {}
        unique_docs = []

        for d in docs:
            h = self._hash(getattr(d, "text", ""))
            if h in seen:
                continue
            seen[h] = True
            unique_docs.append(d)

        text_docs, vision_docs = self._route(unique_docs)

        #  TEXT 
        embedded_text = []
        if text_docs:
            try:
                embedded_text = self.text_embedder.embed_documents(
                    text_docs,
                    session_id=session_id
                )
            except Exception as e:
                logger.error(event="text_embed_failed", error=str(e))

        #  VISION 
        embedded_vision = []

        for i in range(0, len(vision_docs), self.batch_size):
            batch = vision_docs[i:i + self.batch_size]

            paths = []
            valid_docs = []

            for d in batch:
                p = self._resolve_asset_path(d)
                if not p:
                    continue

                paths.append(p)
                valid_docs.append(d)

            if not paths:
                continue

            try:
                embs = self.image_embedder.embed_batch(paths)

                for doc, emb in zip(valid_docs, embs):

                    if not self._valid(emb, settings.VISION_EMBEDDING_DIM):
                        continue

                    doc.embedding = emb

                    s = dict(doc.structure or {})
                    s["embedding_space"] = "vision"
                    doc.structure = s

                    embedded_vision.append(doc)

            except Exception as e:
                logger.error(event="vision_embed_failed", error=str(e))

        logger.info(
            event="multimodal_embed_success",
            text=len(embedded_text),
            vision=len(embedded_vision),
            latency=round(time.time() - start, 3)
        )

        return embedded_text, embedded_vision