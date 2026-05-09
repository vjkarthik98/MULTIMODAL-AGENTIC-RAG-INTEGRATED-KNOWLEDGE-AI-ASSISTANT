import hashlib
import time
from functools import lru_cache
from typing import Dict, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextEmbedder:

    def __init__(self, model_name: str, batch_size: int, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model      = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.device     = device
        self.model_name = model_name

        self.expected_dim    = settings.TEXT_EMBEDDING_DIM
        self.max_text_length = settings.MAX_PROMPT_CHARS

        logger.info(
            event="text_embedder_initialized",
            model=model_name,
            device=device,
            dim=self.expected_dim,
        )

    # HASH

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # SANITIZE

    def _sanitize(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return None
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        return text

    # MODALITY PREFIX

    def _prefix(self, doc) -> str:
        s  = getattr(doc, "structure", {}) or {}
        m  = getattr(doc, "modality", "")
        st = getattr(doc, "subtype", "")

        if m == "table":
            return "Table: "
        if m == "image":
            return "OCR: " if st == "ocr" else "Image: "
        if m == "audio":
            ts = s.get("timestamp_start", "")
            return f"Audio {ts}s: " if ts else "Audio: "
        if m == "video":
            if st == "speech":
                return "Video speech: "
            if st == "frame":
                return "Video frame: "
            if st == "ocr":
                return "Video OCR: "
        if m == "text" and st == "heading":
            return "Heading: "

        return ""

    # CONTEXT ENRICHMENT

    def _enrich(self, doc, text: str) -> str:
        context = []

        if getattr(doc, "source_type", None):
            context.append(f"[{doc.source_type.upper()}]")

        if getattr(doc, "modality", None):
            context.append(f"[{doc.modality.upper()}]")

        if getattr(doc, "page", None):
            context.append(f"[Page {doc.page}]")

        if getattr(doc, "source", None):
            context.append(f"[{doc.source}]")

        enriched = " ".join(context) + " " + self._prefix(doc) + text
        return enriched.strip()[:self.max_text_length]

    # EMBEDDING VALIDATION

    def _valid_embedding(self, emb: List[float]) -> bool:
        if not isinstance(emb, list):
            return False
        if len(emb) != self.expected_dim:
            return False
        import math
        if any(math.isnan(v) or math.isinf(v) for v in emb):
            return False
        return True

    # SINGLE TEXT EMBED

    def embed_text(self, text: str, session_id: str = "default") -> List[float]:

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        start = time.time()
        clean = self._sanitize(text)

        if not clean:
            raise ValueError("EMPTY_TEXT")

        emb = self.model.encode(
            clean,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        if not self._valid_embedding(emb):
            raise ValueError("INVALID_EMBEDDING_SINGLE")

        logger.debug(
            event="embed_single_success",
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )

        return emb

    # RAW STRING LIST EMBED

    def embed_texts(self, texts: List[str], session_id: str = "default") -> List[List[float]]:

        if not texts:
            return []

        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = [self._sanitize(t) for t in texts[i:i + self.batch_size]]
            batch = [t for t in batch if t]

            if not batch:
                continue

            try:
                embs = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for emb in embs:
                    emb_list = emb.tolist()
                    if self._valid_embedding(emb_list):
                        results.append(emb_list)

            except Exception as e:
                logger.error(event="embed_texts_batch_failed", error=str(e), session_id=session_id)

        return results

    # DOCUMENT BATCH EMBED

    def embed_documents(self, documents, session_id: str = "default"):

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        if not documents:
            return []

        start      = time.time()
        texts:     List[str] = []
        valid_docs           = []
        seen: Dict[str, bool] = {}

        for doc in documents:
            try:
                clean = self._sanitize(getattr(doc, "text", ""))
                if not clean:
                    continue

                enriched = self._enrich(doc, clean)
                h        = self._hash(enriched)

                if h in seen:
                    continue

                seen[h] = True
                texts.append(enriched)
                valid_docs.append(doc)

            except Exception as e:
                logger.warning(event="embed_doc_skip", error=str(e), session_id=session_id)

        if not valid_docs:
            return []

        # SAFETY CAP
        cap        = settings.INGESTION_BATCH_SIZE * 10
        texts      = texts[:cap]
        valid_docs = valid_docs[:cap]

        results  = []
        t_target = settings.LATENCY_TARGET_EMBED_BATCH_MS / 1000.0

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_docs  = valid_docs[i:i + self.batch_size]

            t_batch = time.time()

            try:
                embs = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                batch_latency = time.time() - t_batch

                if batch_latency > t_target:
                    logger.warning(
                        event="embed_batch_latency_exceeded",
                        latency=round(batch_latency, 3),
                        target=t_target,
                        batch_size=len(batch_texts),
                        session_id=session_id,
                    )

                for doc, emb in zip(batch_docs, embs):
                    emb_list = emb.tolist()

                    if not self._valid_embedding(emb_list):
                        continue

                    doc.embedding = emb_list

                    structure                    = dict(doc.structure or {})
                    structure["embedding_space"] = "text"
                    doc.structure                = structure

                    results.append(doc)

            except Exception as e:
                logger.error(
                    event="embed_batch_failed",
                    batch_start=i,
                    error=str(e),
                    session_id=session_id,
                )

        total_latency = round(time.time() - start, 3)
        throughput    = round(len(results) / max(total_latency, 1e-6), 1)

        logger.info(
            event="embed_documents_success",
            embedded=len(results),
            total=len(valid_docs),
            throughput_per_sec=throughput,
            latency=total_latency,
            session_id=session_id,
        )

        return results

    # QUERY EMBED

    def embed_query(self, query: str, session_id: str = "default") -> List[float]:
        return self.embed_text(query, session_id)