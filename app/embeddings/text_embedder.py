import time

from app.core.model_loader import model_loader
from app.utils.logger import get_logger


logger = get_logger(__name__)

MAX_BATCH_SIZE = 64
MAX_TEXT_LENGTH = 5000


class TextEmbedder:
    def __init__(self):
        self.model = model_loader.get_text_embedding_model()

    def _sanitize(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Empty text for embedding")

        if len(cleaned) > MAX_TEXT_LENGTH:
            logger.warning(
                "[TextEmbedder][TRUNCATE] length=%s -> %s",
                len(cleaned),
                MAX_TEXT_LENGTH,
            )
            cleaned = cleaned[:MAX_TEXT_LENGTH].strip()
        return cleaned

    def _build_prefix(self, document) -> str:
        structure = document.structure or {}

        if document.modality == "table":
            return "Table data: "

        if document.modality == "image":
            return "Extracted text from image: " if document.subtype == "ocr" else "Image description: "

        if document.modality == "audio":
            start = structure.get("start_time")
            end = structure.get("end_time")
            if start is not None and end is not None:
                return f"Spoken content from {start}s to {end}s: "
            return "Spoken audio content: "

        if document.modality == "video":
            if document.subtype == "speech":
                start = structure.get("start_time")
                end = structure.get("end_time")
                if start is not None and end is not None:
                    return f"Video speech from {start}s to {end}s: "
                return "Video spoken content: "

            if document.subtype == "frame":
                timestamp = structure.get("timestamp")
                if timestamp is not None:
                    return f"Visual scene at {timestamp}s: "
                return "Video visual content: "

        if document.modality == "text" and document.subtype == "heading":
            return "Section heading: "

        return ""

    def _enrich_text(self, document, clean_text: str) -> str:
        context = []
        if document.source_type:
            context.append(f"[{document.source_type.upper()}]")
        if document.modality:
            context.append(f"[{document.modality.upper()}]")
        if document.page:
            context.append(f"[Page {document.page}]")

        prefix = self._build_prefix(document)
        return "".join(context) + prefix + clean_text

    def _embedding_to_list(self, embedding):
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def embed_text(self, text: str, session_id: str = "default"):
        start_time = time.time()
        if not session_id:
            raise ValueError("session_id required")

        try:
            clean_text = self._sanitize(text)
            embedding = self.model.encode(
                clean_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            latency = time.time() - start_time
            logger.info("[TextEmbedder][SINGLE] session_id=%s | latency=%.2fs", session_id, latency)
            return self._embedding_to_list(embedding)
        except Exception as exc:
            logger.error("[TextEmbedder][FAILED] session_id=%s | error=%s", session_id, exc)
            raise

    def embed_documents(self, documents, session_id: str = "default"):
        start_time = time.time()
        if not session_id:
            raise ValueError("session_id required")
        if not documents:
            return []

        try:
            valid_documents = []
            texts = []

            for document in documents:
                try:
                    clean_text = self._sanitize(document.text)
                except ValueError:
                    continue

                texts.append(self._enrich_text(document, clean_text))
                valid_documents.append(document)

            if len(valid_documents) > MAX_BATCH_SIZE:
                logger.warning(
                    "[TextEmbedder][BATCH_LIMIT] reducing %s -> %s",
                    len(valid_documents),
                    MAX_BATCH_SIZE,
                )
                valid_documents = valid_documents[:MAX_BATCH_SIZE]
                texts = texts[:MAX_BATCH_SIZE]

            if not valid_documents:
                logger.warning("[TextEmbedder][EMPTY] session_id=%s | No valid documents", session_id)
                return []

            embeddings = self.model.encode(
                texts,
                batch_size=min(32, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            for document, embedding in zip(valid_documents, embeddings):
                document.embedding = self._embedding_to_list(embedding)
                structure = dict(document.structure or {})
                structure["embedding_space"] = "text"
                document.structure = structure

            latency = time.time() - start_time
            logger.info(
                "[TextEmbedder][BATCH] session_id=%s | docs=%s | latency=%.2fs",
                session_id,
                len(valid_documents),
                latency,
            )
            return valid_documents

        except Exception as exc:
            logger.error("[TextEmbedder][FAILED] session_id=%s | error=%s", session_id, exc)
            raise

    def embed_query(self, query: str, session_id: str = "default"):
        return self.embed_text(query, session_id=session_id)
