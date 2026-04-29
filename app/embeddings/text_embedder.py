import time
from typing import List, Dict

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

    # SANITIZE TEXT
    def _sanitize(self, text: str) -> str:
        cleaned = (text or "").strip()

        if not cleaned:
            return None

        if len(cleaned) > self.max_text_length:
            logger.warning(
                "[TextEmbedder][TRUNCATE] %s -> %s",
                len(cleaned),
                self.max_text_length
            )
            cleaned = cleaned[:self.max_text_length]

        return cleaned

    # PREFIX BASED ON MODALITY
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

    # ENRICH TEXT SAFELY
    def _enrich_text(self, document, clean_text: str) -> str:

        context = []

        if getattr(document, "source_type", None):
            context.append(f"[{document.source_type.upper()}]")

        if getattr(document, "modality", None):
            context.append(f"[{document.modality.upper()}]")

        if getattr(document, "page", None):
            context.append(f"[Page {document.page}]")

        prefix = self._build_prefix(document)

        enriched = " ".join(context) + " " + prefix + clean_text

        # FINAL SAFETY TRUNCATION
        if len(enriched) > self.max_text_length:
            enriched = enriched[:self.max_text_length]

        return enriched

    # STRICT EMBEDDING VALIDATION
    def _validate_embedding(self, emb: List[float]):

        if not emb or not isinstance(emb, list):
            return False

        if len(emb) != self.expected_dim:
            logger.warning(
                "[TextEmbedder] DIM MISMATCH expected=%s got=%s",
                self.expected_dim,
                len(emb)
            )
            return False

        return True

    # SINGLE EMBEDDING
    def embed_text(self, text: str, session_id: str = "default") -> List[float]:

        if not session_id:
            raise ValueError("session_id required")

        start = time.time()

        try:
            clean_text = self._sanitize(text)
            if not clean_text:
                raise ValueError("EMPTY TEXT")

            emb = self.model.encode(
                clean_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            emb_list = emb.tolist()

            if not self._validate_embedding(emb_list):
                raise ValueError("INVALID EMBEDDING")

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

    # BATCH EMBEDDING (PRODUCTION SAFE)
    def embed_documents(self, documents, session_id: str = "default"):

        if not session_id:
            raise ValueError("session_id required")

        if not documents:
            return []

        start = time.time()

        valid_docs = []
        texts = []
        seen: Dict[str, int] = {}

        # PREPARE INPUT
        for doc in documents:
            try:
                clean = self._sanitize(getattr(doc, "text", ""))

                if not clean:
                    continue

                enriched = self._enrich_text(doc, clean)

                key = enriched[:100]

                # DEDUPLICATION
                if key in seen:
                    continue

                seen[key] = 1

                texts.append(enriched)
                valid_docs.append(doc)

            except Exception as e:
                logger.warning("[TextEmbedder][SKIP] %s", str(e))
                continue

        if not valid_docs:
            return []

        # LIMIT SAFETY
        max_batch = getattr(settings, "MAX_PARALLEL_REQUESTS", 100)
        texts = texts[:max_batch]
        valid_docs = valid_docs[:max_batch]

        embedded_docs = []

        # SAFE BATCH PROCESSING
        for i in range(0, len(texts), self.batch_size):

            batch_texts = texts[i:i + self.batch_size]
            batch_docs = valid_docs[i:i + self.batch_size]

            try:
                embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                for doc, emb in zip(batch_docs, embeddings):

                    emb_list = emb.tolist()

                    if not self._validate_embedding(emb_list):
                        continue

                    doc.embedding = emb_list

                    structure = dict(getattr(doc, "structure", {}) or {})
                    structure["embedding_space"] = "text"
                    doc.structure = structure

                    embedded_docs.append(doc)

            except Exception as e:
                logger.error("[TextEmbedder][BATCH_FAIL] %s", str(e))
                continue

        latency = round(time.time() - start, 2)

        logger.info(
            "[TextEmbedder][SUCCESS] session_id=%s | embedded=%s | latency=%.2fs",
            session_id,
            len(embedded_docs),
            latency
        )

        return embedded_docs

    # QUERY EMBEDDING
    def embed_query(self, query: str, session_id: str = "default"):
        return self.embed_text(query, session_id)