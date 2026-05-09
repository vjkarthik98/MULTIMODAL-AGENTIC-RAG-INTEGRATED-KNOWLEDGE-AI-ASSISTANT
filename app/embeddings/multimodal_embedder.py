import hashlib
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalEmbedder:

    def __init__(self, text_embedder, image_embedder) -> None:
        self.text_embedder  = text_embedder
        self.image_embedder = image_embedder
        self.batch_size     = settings.EMBEDDING_BATCH_SIZE
        self.max_docs       = settings.INGESTION_BATCH_SIZE * 10

    # HASH

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _hash_path(self, path: str) -> str:
        return hashlib.sha256(path.encode("utf-8")).hexdigest()

    # ASSET PATH RESOLUTION

    def _resolve_asset_path(self, doc) -> Optional[str]:
        structure = getattr(doc, "structure", {}) or {}

        for key in ("asset_path", "frame_path", "source_path"):
            path = structure.get(key)
            if not path:
                continue

            p = Path(path)
            if p.exists():
                return str(p)

            logger.warning(
                event="asset_path_not_found",
                key=key,
                path=str(path),
            )

        return None

    # EMBEDDING VALIDATION

    def _valid_embedding(self, emb, dim: int) -> bool:
        if not isinstance(emb, list):
            return False
        if len(emb) != dim:
            return False
        if any(math.isnan(v) or math.isinf(v) for v in emb):
            return False
        return True

    # ROUTING

    def _route(self, docs) -> Tuple[List, List]:
        text_docs:   List = []
        vision_docs: List = []

        for doc in docs:
            try:
                m  = getattr(doc, "modality", "")
                st = getattr(doc, "subtype", "")

                if m in {"text", "table"}:
                    text_docs.append(doc)

                elif m == "audio":
                    text_docs.append(doc)

                elif m == "image":
                    text_docs.append(doc)
                    if st == "caption":
                        vision_docs.append(doc)

                elif m == "video":
                    if st == "speech":
                        text_docs.append(doc)
                    elif st == "frame":
                        vision_docs.append(doc)
                    elif st == "ocr":
                        text_docs.append(doc)

            except Exception as e:
                logger.warning(event="route_failed", error=str(e))

        return text_docs, vision_docs

    # MAIN

    def embed_documents(
        self,
        documents: List,
        session_id: str,
    ) -> Tuple[List, List]:

        if not documents:
            return [], []

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        start = time.time()

        # CAP AND DEDUP
        docs = documents[:self.max_docs]

        seen_text: Dict[str, bool] = {}
        unique_docs: List          = []

        for d in docs:
            h = self._hash_text(getattr(d, "text", ""))
            if h in seen_text:
                continue
            seen_text[h] = True
            unique_docs.append(d)

        text_docs, vision_docs = self._route(unique_docs)

        # TEXT EMBEDDING
        embedded_text: List = []
        failed_text         = 0

        if text_docs:
            try:
                embedded_text = self.text_embedder.embed_documents(
                    text_docs,
                    session_id=session_id,
                )
            except Exception as e:
                failed_text = len(text_docs)
                logger.error(
                    event="text_embed_failed",
                    count=len(text_docs),
                    error=str(e),
                    session_id=session_id,
                )

        # VISION EMBEDDING
        embedded_vision: List          = []
        seen_paths: Dict[str, bool]    = {}
        failed_vision                   = 0

        for i in range(0, len(vision_docs), self.batch_size):
            batch      = vision_docs[i:i + self.batch_size]
            paths:     List = []
            valid_docs: List = []

            for d in batch:
                p = self._resolve_asset_path(d)
                if not p:
                    continue

                ph = self._hash_path(p)
                if ph in seen_paths:
                    continue
                seen_paths[ph] = True

                paths.append(p)
                valid_docs.append(d)

            if not paths:
                continue

            try:
                embs = self.image_embedder.embed_batch(paths, session_id=session_id)

                for doc, emb in zip(valid_docs, embs):
                    if not self._valid_embedding(emb, settings.VISION_EMBEDDING_DIM):
                        failed_vision += 1
                        continue

                    doc.embedding = emb

                    s                    = dict(doc.structure or {})
                    s["embedding_space"] = "vision"
                    doc.structure        = s

                    embedded_vision.append(doc)

            except Exception as e:
                failed_vision += len(valid_docs)
                logger.error(
                    event="vision_embed_failed",
                    batch_start=i,
                    error=str(e),
                    session_id=session_id,
                )

        total_latency = round(time.time() - start, 3)

        logger.info(
            event="multimodal_embed_success",
            text_embedded=len(embedded_text),
            vision_embedded=len(embedded_vision),
            failed_text=failed_text,
            failed_vision=failed_vision,
            total_input=len(unique_docs),
            latency=total_latency,
            session_id=session_id,
        )

        return embedded_text, embedded_vision