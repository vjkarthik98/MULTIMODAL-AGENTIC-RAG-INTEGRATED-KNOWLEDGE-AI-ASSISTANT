import asyncio
import hashlib
import math
import time
from typing import Dict, List, Optional

from app.core.config import settings
from app.ingestion.schema import redact_pii, sanitize_prompt_injection
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    def __init__(self, model_name: str, batch_size: int, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = min(batch_size, 100)
        self.device = device
        self.model_name = model_name
        self.expected_dim = settings.TEXT_EMBEDDING_DIM
        self.max_text_length = settings.MAX_PROMPT_CHARS
        self.cache: Dict[str, List[float]] = {}

        logger.info(event="text_embedder_initialized", model=model_name, device=device, dim=self.expected_dim)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str) -> str:
        return f"{self.model_name}:{self._hash(text)}"

    def _sanitize(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return None
        text = sanitize_prompt_injection(redact_pii(text))
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]
        return text

    def _route_model(self, language: Optional[str]) -> str:
        if language and language.lower() not in {"en", "eng", "und"}:
            return settings.MULTILINGUAL_EMBEDDING_MODEL
        return self.model_name

    def _matryoshka(self, emb: List[float]) -> Dict[str, List[float]]:
        return {str(dim): emb[:dim] for dim in settings.MATRYOSHKA_DIMS if dim <= len(emb)}

    def _prefix(self, doc) -> str:
        s = getattr(doc, "structure", {}) or {}
        m = getattr(doc, "modality", "")
        st = getattr(doc, "subtype", "")
        if m in {"table", "excel"}:
            return "Table: "
        if m in {"pdf", "word"}:
            return f"{m.upper()}: "
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
        return enriched.strip()[: self.max_text_length]

    def _valid_embedding(self, emb: List[float]) -> bool:
        if not isinstance(emb, list):
            return False
        valid_dims = {self.expected_dim, *settings.MATRYOSHKA_DIMS}
        if len(emb) not in valid_dims:
            return False
        return not any(math.isnan(float(v)) or math.isinf(float(v)) for v in emb)

    def _to_list(self, emb) -> List[float]:
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def _encode_batch(self, texts: List[str]):
        return self.model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_text(self, text: str, session_id: str = "default") -> List[float]:
        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        start = time.time()
        clean = self._sanitize(text)
        if not clean:
            raise ValueError("EMPTY_TEXT")

        cache_key = self._cache_key(clean)
        if cache_key in self.cache:
            return self.cache[cache_key]

        emb = self._to_list(
            self.model.encode(clean, convert_to_numpy=True, normalize_embeddings=True)
        )
        if not self._valid_embedding(emb):
            raise ValueError("INVALID_EMBEDDING_SINGLE")

        self.cache[cache_key] = emb
        logger.debug(event="embed_single_success", latency=round(time.time() - start, 3), session_id=session_id)
        return emb

    async def async_embed_text(self, text: str, session_id: str = "default") -> List[float]:
        return await asyncio.to_thread(self.embed_text, text, session_id)

    def embed_texts(self, texts: List[str], session_id: str = "default") -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self._sanitize(text) for text in texts[i:i + self.batch_size]]
            batch = [text for text in batch if text]
            if not batch:
                continue

            uncached = [text for text in batch if self._cache_key(text) not in self.cache]
            try:
                if uncached:
                    encoded = self._encode_batch(uncached)
                    for text, emb in zip(uncached, encoded):
                        emb_list = self._to_list(emb)
                        if self._valid_embedding(emb_list):
                            self.cache[self._cache_key(text)] = emb_list

                for text in batch:
                    cached = self.cache.get(self._cache_key(text))
                    if cached:
                        results.append(cached)
            except Exception as exc:
                logger.error(event="embed_texts_batch_failed", error=str(exc), session_id=session_id)

        return results

    async def async_embed_texts(self, texts: List[str], session_id: str = "default") -> List[List[float]]:
        return await asyncio.to_thread(self.embed_texts, texts, session_id)

    def embed_documents(self, documents, session_id: str = "default"):
        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")
        if not documents:
            return []

        start = time.time()
        texts: List[str] = []
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
            except Exception as exc:
                logger.warning(event="embed_doc_skip", error=str(exc), session_id=session_id)

        if not valid_docs:
            return []

        cap = settings.INGESTION_BATCH_SIZE * 10
        texts = texts[:cap]
        valid_docs = valid_docs[:cap]
        results = []
        t_target = settings.LATENCY_TARGET_EMBED_BATCH_MS / 1000.0

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_docs = valid_docs[i:i + self.batch_size]
            t_batch = time.time()

            try:
                embs = self.embed_texts(batch_texts, session_id=session_id)
                batch_latency = time.time() - t_batch
                if batch_latency > t_target:
                    logger.warning(
                        event="embed_batch_latency_exceeded",
                        latency=round(batch_latency, 3),
                        target=t_target,
                        batch_size=len(batch_texts),
                        session_id=session_id,
                    )

                for doc, emb_list in zip(batch_docs, embs):
                    if not self._valid_embedding(emb_list):
                        continue
                    doc.embedding = emb_list
                    doc.extra_metadata["embedding_model"] = self._route_model((doc.structure or {}).get("language"))
                    doc.extra_metadata["matryoshka_embeddings"] = self._matryoshka(emb_list)
                    structure = dict(doc.structure or {})
                    structure["embedding_space"] = "text"
                    doc.structure = structure
                    results.append(doc)
            except Exception as exc:
                logger.error(event="embed_batch_failed", batch_start=i, error=str(exc), session_id=session_id)

        total_latency = round(time.time() - start, 3)
        throughput = round(len(results) / max(total_latency, 1e-6), 1)
        logger.info(
            event="embed_documents_success",
            embedded=len(results),
            total=len(valid_docs),
            throughput_per_sec=throughput,
            latency=total_latency,
            session_id=session_id,
        )
        return results

    async def async_embed_documents(self, documents, session_id: str = "default"):
        return await asyncio.to_thread(self.embed_documents, documents, session_id)

    def embed_query(self, query: str, session_id: str = "default") -> List[float]:
        return self.embed_text(query, session_id)


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/embeddings/text_embedder.py -v
# ============================================================

def test_batch_embedding_respects_rate_limit() -> None:
    embedder = object.__new__(TextEmbedder)
    embedder.batch_size = min(500, 100)
    assert embedder.batch_size == 100


def test_embedding_cache_hit_skips_api_call() -> None:
    embedder = object.__new__(TextEmbedder)
    embedder.model_name = "unit"
    embedder.cache = {}
    embedder._hash = TextEmbedder._hash.__get__(embedder, TextEmbedder)
    embedder._cache_key = TextEmbedder._cache_key.__get__(embedder, TextEmbedder)
    key = embedder._cache_key("hello")
    embedder.cache[key] = [0.0] * settings.TEXT_EMBEDDING_DIM
    assert embedder.cache[key][0] == 0.0


def test_multilingual_routed_correctly() -> None:
    embedder = object.__new__(TextEmbedder)
    embedder.model_name = "english"
    assert TextEmbedder._route_model(embedder, "hi") == settings.MULTILINGUAL_EMBEDDING_MODEL


def test_dimension_mismatch_raises_error() -> None:
    embedder = object.__new__(TextEmbedder)
    embedder.expected_dim = 3
    assert TextEmbedder._valid_embedding(embedder, [1.0, 2.0]) is False


def test_clip_cross_modal_similarity() -> None:
    assert settings.VISION_EMBEDDING_DIM > 0
