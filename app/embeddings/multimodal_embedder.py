import time

from app.embeddings.image_embedder import ImageEmbedder
from app.embeddings.text_embedder import TextEmbedder
from app.utils.logger import get_logger


logger = get_logger(__name__)


class MultimodalEmbedder:
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()

    def _resolve_asset_path(self, document) -> str | None:
        structure = document.structure or {}
        return (
            structure.get("asset_path")
            or structure.get("frame_path")
            or structure.get("source_path")
            or document.source
        )

    def embed_documents(self, documents, session_id: str = "default"):
        if not session_id:
            raise ValueError("session_id required")

        start_time = time.time()
        text_documents = []
        vision_documents = []

        for document in documents:
            try:
                if document.modality in {"text", "table", "audio"}:
                    text_documents.append(document)
                elif document.modality == "image":
                    text_documents.append(document)
                    if document.subtype == "caption":
                        vision_documents.append(document)
                elif document.modality == "video":
                    if document.subtype == "speech":
                        text_documents.append(document)
                    elif document.subtype == "frame":
                        vision_documents.append(document)
            except Exception as exc:
                logger.error("[MultimodalEmbedder][ROUTER_FAIL] error=%s", exc)

        embedded_text_documents = (
            self.text_embedder.embed_documents(text_documents, session_id=session_id)
            if text_documents
            else []
        )

        embedded_vision_documents = []
        for document in vision_documents:
            try:
                image_path = self._resolve_asset_path(document)
                if not image_path:
                    logger.warning("[MultimodalEmbedder][VISION_SKIP] missing image path")
                    continue

                document.embedding = self.image_embedder.embed(image_path)
                structure = dict(document.structure or {})
                structure["embedding_space"] = "vision"
                document.structure = structure
                embedded_vision_documents.append(document)

            except Exception as exc:
                logger.error("[MultimodalEmbedder][VISION_FAIL] error=%s", exc)

        latency = time.time() - start_time
        logger.info(
            "[MultimodalEmbedder][SUCCESS] text_docs=%s | vision_docs=%s | latency=%.2fs",
            len(embedded_text_documents),
            len(embedded_vision_documents),
            latency,
        )

        return embedded_text_documents, embedded_vision_documents
