import time
from typing import List, Tuple

from app.core.config import settings
from app.core.model_loader import model_loader
from app.utils.logger import get_logger
from pathlib import Path


logger = get_logger(__name__)


class MultimodalEmbedder:
    def __init__(self):
        # Use ModelLoader 
        self.text_embedder = model_loader.get_embedder()
        self.image_embedder = model_loader.get_image_embedder()

        self.batch_size = settings.EMBEDDING_BATCH_SIZE

    def _resolve_asset_path(self, document) -> str:
        structure = getattr(document, "structure", {}) or {}

        path = (
            structure.get("asset_path")
            or structure.get("frame_path")
        )

        # STRICT VALIDATION 
        if not path:
            return None
        
        
        p = Path(path)

        if not p.exists():
            logger.warning("[Embedder] missing file: %s", path)
            return None
        
        return str(p)


    def embed_documents(
        self,
        documents: List,
        session_id: str = "default"
    ) -> Tuple[List, List]:

        if not documents:
            return [], []

        if not session_id:
            raise ValueError("session_id required")

        start_time = time.time()

        text_documents = []
        vision_documents = []

        # ROUTING
        for doc in documents:
            try:
                modality = getattr(doc, "modality", "text")
                subtype = getattr(doc, "subtype", "")

                if modality in {"text", "table", "audio"}:
                    text_documents.append(doc)

                elif modality == "image":
                    text_documents.append(doc)
                    if subtype == "caption":
                        vision_documents.append(doc)

                elif modality == "video":
                    if subtype == "speech":
                        text_documents.append(doc)
                    elif subtype == "frame":
                        vision_documents.append(doc)

            except Exception as e:
                logger.warning("[MultimodalEmbedder][ROUTING_FAIL] %s", str(e))

        # TEXT EMBEDDING
        embedded_text_documents = []
        if text_documents:
            try:
                embedded_text_documents = self.text_embedder.embed_documents(
                    text_documents,
                    session_id=session_id
                )
            except Exception as e:
                logger.error("[MultimodalEmbedder][TEXT_FAIL] %s", str(e))

        # VISION EMBEDDING (BATCHED)
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
                    doc.embedding = emb

                    structure = dict(getattr(doc, "structure", {}) or {})
                    structure["embedding_space"] = "vision"
                    doc.structure = structure

                    if len(emb) != settings.VISION_EMBEDDING_DIM:
                        logger.warning(
                            "[MultimodalEmbedder] dim mismatch | expected=%s got=%s",
                            settings.VISION_EMBEDDING_DIM,
                            len(emb)
                        )

                    embedded_vision_documents.append(doc)

            except Exception as e:
                logger.error("[MultimodalEmbedder][VISION_FAIL] %s", str(e))

        latency = round(time.time() - start_time, 2)

        logger.info(
            "[MultimodalEmbedder][SUCCESS] text=%s | vision=%s | latency=%ss",
            len(embedded_text_documents),
            len(embedded_vision_documents),
            latency
        )

        return embedded_text_documents, embedded_vision_documents