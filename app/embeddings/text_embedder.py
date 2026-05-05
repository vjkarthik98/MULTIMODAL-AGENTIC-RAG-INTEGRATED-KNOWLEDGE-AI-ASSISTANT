import hashlib
import time
from typing import List, Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextEmbedder:

    def __init__(self, model_name: str, batch_size: int, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

        self.expected_dim = settings.TEXT_EMBEDDING_DIM
        self.max_text_length = settings.MAX_PROMPT_CHARS

        logger.info(
            event="text_embedder_initialized",
            model=model_name,
            device=device
        )

    #  HASH 
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    #  SANITIZE 
    def _sanitize(self, text: str) -> Optional[str]:
        text = (text or "").strip()

        if not text:
            return None

        # SOFT LIMIT (no hard truncation loss)
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]

        return text

    #  PREFIX 
    def _prefix(self, doc) -> str:
        s = getattr(doc, "structure", {}) or {}
        m = getattr(doc, "modality", "")
        st = getattr(doc, "subtype", "")

        if m == "table":
            return "Table: "
        if m == "image":
            return "OCR: " if st == "ocr" else "Image: "
        if m == "audio":
            return f"Audio {s.get('timestamp_start')}s: "
        if m == "video":
            return "Video speech: " if st == "speech" else "Video frame: "
        if m == "text" and st == "heading":
            return "Heading: "

        return ""

    #  ENRICH 
    def _enrich(self, doc, text: str) -> str:

        context = []

        if doc.source_type:
            context.append(f"[{doc.source_type.upper()}]")

        if doc.modality:
            context.append(f"[{doc.modality.upper()}]")

        if doc.page:
            context.append(f"[Page {doc.page}]")

        enriched = " ".join(context) + " " + self._prefix(doc) + text

        return enriched.strip()[:self.max_text_length]

    #  VALIDATE 
    def _valid_embedding(self, emb: List[float]) -> bool:
        return isinstance(emb, list) and len(emb) == self.expected_dim

    #  SINGLE 
    def embed_text(self, text: str, session_id: str) -> List[float]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        start = time.time()

        clean = self._sanitize(text)
        if not clean:
            raise ValueError("EMPTY_TEXT")

        emb = self.model.encode(
            clean,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        if not self._valid_embedding(emb):
            raise ValueError("INVALID_EMBEDDING")

        logger.info(
            event="embed_single",
            latency=round(time.time() - start, 3)
        )

        return emb

    #  BATCH 
    def embed_documents(self, documents, session_id: str):

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        if not documents:
            return []

        start = time.time()

        texts = []
        valid_docs = []
        seen: Dict[str, bool] = {}

        for doc in documents:
            try:
                clean = self._sanitize(getattr(doc, "text", ""))
                if not clean:
                    continue

                enriched = self._enrich(doc, clean)

                h = self._hash(enriched)
                if h in seen:
                    continue

                seen[h] = True

                texts.append(enriched)
                valid_docs.append(doc)

            except Exception as e:
                logger.warning(event="embed_skip", error=str(e))

        if not valid_docs:
            return []

        # limit safety
        limit = getattr(settings, "MAX_PARALLEL_REQUESTS", 100)
        texts = texts[:limit]
        valid_docs = valid_docs[:limit]

        results = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_docs = valid_docs[i:i + self.batch_size]

            try:
                embs = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                for doc, emb in zip(batch_docs, embs):
                    emb_list = emb.tolist()

                    if not self._valid_embedding(emb_list):
                        continue

                    doc.embedding = emb_list

                    structure = dict(doc.structure or {})
                    structure["embedding_space"] = "text"
                    doc.structure = structure

                    results.append(doc)

            except Exception as e:
                logger.error(event="embed_batch_failed", error=str(e))

        logger.info(
            event="embed_success",
            embedded=len(results),
            latency=round(time.time() - start, 3)
        )

        return results

    #  QUERY 
    def embed_query(self, query: str, session_id: str):
        return self.embed_text(query, session_id)