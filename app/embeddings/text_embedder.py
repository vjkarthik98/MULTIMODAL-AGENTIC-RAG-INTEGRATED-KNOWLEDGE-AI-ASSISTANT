import time
from typing import List

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    def __init__(self, model_name: str, batch_size: int, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

        self.max_text_length = settings.MAX_PROMPT_CHARS
        self.expected_dim = settings.TEXT_EMBEDDING_DIM

        logger.info(
            "[TextEmbedder] initialized | model=%s | device=%s",
            model_name,
            device
        )

    def _sanitize(self, text: str) -> str:
        cleaned = (text or "").strip()

        if not cleaned:
            raise ValueError("Empty text for embedding")

        if len(cleaned) > self.max_text_length:
            logger.warning(
                "[TextEmbedder][TRUNCATE] %s -> %s",
                len(cleaned),
                self.max_text_length
            )
            cleaned = cleaned[:self.max_text_length]

        return cleaned

    def _build_prefix(self, document) -> str:
        structure = getattr(document, "structure", {}) or {}

        modality = getattr(document, "modality", "")
        subtype = getattr(document, "subtype", "")

        if modality == "table":
            return "Table data: "

        if modality == "image":
            return "Extracted text from image: " if subtype == "ocr" else "Image description: "

        if modality == "audio":
            start = structure.get("start_time")
            end = structure.get("end_time")
            if start is not None and end is not None:
                return f"Audio {start}-{end}s: "
            return "Audio content: "

        if modality == "video":
            if subtype == "speech":
                return "Video speech: "
            if subtype == "frame":
                return "Video frame: "

        if modality == "text" and subtype == "heading":
            return "Heading: "

        return ""

    def _enrich_text(self, document, clean_text: str) -> str:
        context = []

        if getattr(document, "source_type", None):
            context.append(f"[{document.source_type.upper()}]")

        if getattr(document, "modality", None):
            context.append(f"[{document.modality.upper()}]")

        if getattr(document, "page", None):
            context.append(f"[Page {document.page}]")

        prefix = self._build_prefix(document)

        return " ".join(context) + " " + prefix + clean_text

    def _validate_embedding(self, emb: List[float]):
        if len(emb) != self.expected_dim:
            logger.warning(
                "[TextEmbedder] dim mismatch | expected=%s got=%s",
                self.expected_dim,
                len(emb)
            )

    def embed_text(self, text: str, session_id: str = "default") -> List[float]:
        if not session_id:
            raise ValueError("session_id required")

        start = time.time()

        try:
            clean_text = self._sanitize(text)

            emb = self.model.encode(
                clean_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            emb_list = emb.tolist()

            self._validate_embedding(emb_list)

            logger.info(
                "[TextEmbedder][SINGLE] session_id=%s | latency=%.2fs",
                session_id,
                time.time() - start
            )

            return emb_list

        except Exception as e:
            logger.error(
                "[TextEmbedder][FAILED] session_id=%s | error=%s",
                session_id,
                str(e)
            )
            raise

    def embed_documents(self, documents, session_id: str = "default"):
        if not session_id:
            raise ValueError("session_id required")

        if not documents:
            return []

        start = time.time()

        valid_docs = []
        texts = []

        for doc in documents:
            try:
                clean = self._sanitize(getattr(doc, "text", ""))
                enriched = self._enrich_text(doc, clean)

                texts.append(enriched)
                valid_docs.append(doc)

            except Exception as e:
                logger.warning("[TextEmbedder][SKIP] %s", str(e))
                continue

        if not valid_docs:
            return []

        # Config-based batching
        batch_size = min(self.batch_size, len(texts))

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            for doc, emb in zip(valid_docs, embeddings):
                emb_list = emb.tolist()

                self._validate_embedding(emb_list)

                doc.embedding = emb_list

                structure = dict(getattr(doc, "structure", {}) or {})
                structure["embedding_space"] = "text"
                doc.structure = structure

            logger.info(
                "[TextEmbedder][BATCH] session_id=%s | docs=%s | latency=%.2fs",
                session_id,
                len(valid_docs),
                time.time() - start
            )

            return valid_docs

        except Exception as e:
            logger.error(
                "[TextEmbedder][FAILED] session_id=%s | error=%s",
                session_id,
                str(e)
            )
            raise

    def embed_query(self, query: str, session_id: str = "default"):
        return self.embed_text(query, session_id)