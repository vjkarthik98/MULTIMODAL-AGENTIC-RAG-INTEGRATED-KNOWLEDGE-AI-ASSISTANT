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

    # RESOLVE IMAGE/FRAME PATH
    def _resolve_asset_path(self, document) -> str:
        structure = getattr(document, "structure", {}) or {}

        path = structure.get("asset_path") or structure.get("frame_path")

        if not path:
            return None

        p = Path(path)

        if not p.exists():
            logger.warning("[Embedder] FILE NOT FOUND: %s", path)
            return None

        return str(p)

    # VALIDATE EMBEDDING
    def _is_valid_embedding(self, emb, expected_dim):
        return isinstance(emb, list) and len(emb) == expected_dim

    # ROUTING LOGIC (STRICT)
    def _route_documents(self, documents):

        text_docs = []
        vision_docs = []

        for doc in documents:
            try:
                modality = getattr(doc, "modality", "")
                subtype = getattr(doc, "subtype", "")

                # TEXT SPACE
                if modality in {"text", "table", "audio"}:
                    text_docs.append(doc)

                # IMAGE
                elif modality == "image":
                    text_docs.append(doc)  # caption + OCR → text space
                    if subtype == "caption":
                        vision_docs.append(doc)

                # VIDEO
                elif modality == "video":
                    if subtype == "speech":
                        text_docs.append(doc)
                    elif subtype == "frame":
                        vision_docs.append(doc)

            except Exception as e:
                logger.warning("[MultimodalEmbedder][ROUTE_FAIL] %s", str(e))

        return text_docs, vision_docs

    # MAIN EMBEDDING FUNCTION
    def embed_documents(
        self,
        documents: List,
        session_id: str = "default"
    ) -> Tuple[List, List]:

        if not documents:
            return [], []

        if not session_id:
            raise ValueError("SESSION_ID REQUIRED")

        start_time = time.time()

        # GLOBAL LIMIT
        documents = documents[:self.max_docs]

        # DEDUPLICATION
        seen: Dict[str, int] = {}
        unique_docs = []

        for doc in documents:
            key = getattr(doc, "text", "")[:100]

            if key in seen:
                continue

            seen[key] = 1
            unique_docs.append(doc)

        # ROUTE DOCUMENTS
        text_documents, vision_documents = self._route_documents(unique_docs)

        #  TEXT EMBEDDING 
        embedded_text_documents = []

        if text_documents:
            try:
                embedded_text_documents = self.text_embedder.embed_documents(
                    text_documents,
                    session_id=session_id
                )
            except Exception as e:
                logger.error("[MultimodalEmbedder][TEXT_FAIL] %s", str(e))

        #  VISION EMBEDDING 
        embedded_vision_documents = []

        for i in range(0, len(vision_documents), self.batch_size):

            batch = vision_documents[i:i + self.batch_size]

            paths = []
            valid_docs = []

            for doc in batch:
                try:
                    path = self._resolve_asset_path(doc)

                    if not path:
                        continue

                    paths.append(path)
                    valid_docs.append(doc)

                except Exception as e:
                    logger.warning("[MultimodalEmbedder][PATH_FAIL] %s", str(e))

            if not paths:
                continue

            try:
                embeddings = self.image_embedder.embed_batch(paths)

                for doc, emb in zip(valid_docs, embeddings):

                    if not self._is_valid_embedding(emb, settings.VISION_EMBEDDING_DIM):
                        continue

                    doc.embedding = emb

                    structure = dict(getattr(doc, "structure", {}) or {})
                    structure["embedding_space"] = "vision"
                    doc.structure = structure

                    embedded_vision_documents.append(doc)

            except Exception as e:
                logger.error("[MultimodalEmbedder][VISION_FAIL] %s", str(e))
                continue

        latency = round(time.time() - start_time, 2)

        logger.info(
            "[MultimodalEmbedder][SUCCESS] text=%s | vision=%s | latency=%ss",
            len(embedded_text_documents),
            len(embedded_vision_documents),
            latency
        )

        return embedded_text_documents, embedded_vision_documents