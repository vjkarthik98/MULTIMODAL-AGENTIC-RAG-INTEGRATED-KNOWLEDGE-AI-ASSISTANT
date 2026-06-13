from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = get_logger(__name__)


# PROMETHEUS METRICS

def _make_metrics():
    if not settings.PROMETHEUS_ENABLED:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def labels(self, **kw): return self
        n = _Noop()
        return n, n, n, n
    try:
        from prometheus_client import Counter, Histogram
        embed_latency = Histogram(
            "image_embedding_latency_seconds",
            "Image embedding batch latency",
            ["batch_size"],
        )
        embed_errors = Counter(
            "image_embedding_errors_total",
            "Image embedding failures",
            ["reason"],
        )
        embed_total = Counter(
            "image_embeddings_produced_total",
            "Total image embeddings produced",
        )
        cache_hits = Counter(
            "image_embedding_cache_hits_total",
            "Redis embedding cache hits",
        )
        return embed_latency, embed_errors, embed_total, cache_hits
    except Exception:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def labels(self, **kw): return self
        n = _Noop()
        return n, n, n, n


_EMBED_LATENCY, _EMBED_ERRORS, _EMBED_TOTAL, _CACHE_HITS = _make_metrics()


# SEMAPHORE

_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)
    return _SEMAPHORE


# CUSTOM EXCEPTIONS

class ImageEmbedderError(Exception):
    """Base exception for image embedding errors."""


class ImageLoadError(ImageEmbedderError):
    """Raised when an image cannot be loaded."""


class DimensionMismatchError(ImageEmbedderError):
    """Raised when embedding dimension does not match expected."""


class NoValidImagesError(ImageEmbedderError):
    """Raised when all images fail validation."""


# EMBEDDING RESULT MODEL

class ImageEmbeddingResult:
    """Structured result for a single image embedding."""

    def __init__(
        self,
        path: str,
        embedding: List[float],
        embedding_dim: int,
        model_name: str,
        checksum_sha256: str,
        width: int,
        height: int,
        mode: str,
        embedding_id: str,
        latency_ms: float,
        cache_hit: bool,
        session_id: str,
    ) -> None:
        self.path             = path
        self.embedding        = embedding
        self.embedding_dim    = embedding_dim
        self.model_name       = model_name
        self.checksum_sha256  = checksum_sha256
        self.width            = width
        self.height           = height
        self.mode             = mode
        self.embedding_id     = embedding_id
        self.latency_ms       = latency_ms
        self.cache_hit        = cache_hit
        self.session_id       = session_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path":            self.path,
            "embedding":       self.embedding,
            "embedding_dim":   self.embedding_dim,
            "model_name":      self.model_name,
            "checksum_sha256": self.checksum_sha256,
            "width":           self.width,
            "height":          self.height,
            "mode":            self.mode,
            "embedding_id":    self.embedding_id,
            "latency_ms":      self.latency_ms,
            "cache_hit":       self.cache_hit,
            "session_id":      self.session_id,
        }


# FILE HASH

def _sha256_path(path: str) -> str:
    """SHA-256 hash of the file content for dedup and cache keying."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception:
        h.update(path.encode("utf-8"))
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# EMBEDDING VALIDATION

def _valid_embedding(emb: List[float], expected_dim: int) -> bool:
    if not isinstance(emb, list):
        return False
    if len(emb) != expected_dim:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in emb):
        return False
    return True


# REDIS CACHE HELPERS

def _cache_key(checksum: str) -> str:
    return f"img_emb:{checksum}"


def _cache_get(checksum: str) -> Optional[List[float]]:
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem is None:
            return None
        raw = mem.cache_get(_cache_key(checksum))
        if raw and isinstance(raw, list):
            return raw
    except Exception:
        pass
    return None


def _cache_set(checksum: str, embedding: List[float]) -> None:
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem is None:
            return
        mem.cache_set(
            _cache_key(checksum),
            embedding,
            ttl=settings.REDIS_EMBEDDING_CACHE_TTL,
        )
    except Exception:
        pass


# IMAGE LOADING AND VALIDATION

def _load_image(
    path: str,
    max_dim: int,
    session_id: str,
) -> Tuple[Optional["Image.Image"], int, int, str]:
    """
    Load, validate, orient, convert, and resize an image.

    Returns: (pil_image, width, height, mode) or (None, 0, 0, '') on failure.
    """
    if not PIL_AVAILABLE:
        raise ImportError("PILLOW_REQUIRED_FOR_IMAGE_LOADING")

    p = Path(path)
    if not p.exists():
        logger.warning(event="image_path_not_found", path=path, session_id=session_id)
        _EMBED_ERRORS.labels(reason="not_found").inc()
        return None, 0, 0, ""

    file_size = p.stat().st_size
    if file_size == 0:
        logger.warning(event="image_empty_file", path=path, session_id=session_id)
        _EMBED_ERRORS.labels(reason="empty_file").inc()
        return None, 0, 0, ""

    if file_size > settings.MAX_FILE_SIZE_IMAGE:
        logger.warning(
            event="image_too_large_skipped",
            path=p.name,
            size=file_size,
            limit=settings.MAX_FILE_SIZE_IMAGE,
            session_id=session_id,
        )
        _EMBED_ERRORS.labels(reason="too_large").inc()
        return None, 0, 0, ""

    try:
        with Image.open(path) as raw:
            # EXIF AUTO-ROTATION
            img = ImageOps.exif_transpose(raw)

            # ZERO-DIMENSION GUARD
            w, h = img.size
            if w == 0 or h == 0:
                logger.warning(event="image_zero_dimension", path=p.name, session_id=session_id)
                _EMBED_ERRORS.labels(reason="zero_dimension").inc()
                return None, 0, 0, ""

            # MINIMUM DIMENSION GUARD
            if w < 32 or h < 32:
                logger.warning(
                    event="image_too_small",
                    path=p.name,
                    width=w,
                    height=h,
                    session_id=session_id,
                )
                _EMBED_ERRORS.labels(reason="too_small").inc()
                return None, 0, 0, ""

            # MEGAPIXEL GUARD — RESIZE BEFORE CONVERSION
            mp = (w * h) / 1_000_000
            if mp > settings.MAX_IMAGE_SIZE_MP:
                scale = math.sqrt(settings.MAX_IMAGE_SIZE_MP / mp)
                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)
                img   = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info(
                    event="image_resized_mp_limit",
                    path=p.name,
                    original_mp=round(mp, 1),
                    session_id=session_id,
                )
                w, h = new_w, new_h

            original_mode = img.mode

            # ALPHA CHANNEL — FLATTEN TO WHITE BACKGROUND
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # MAX DIM RESIZE
            if max(w, h) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
                w, h = img.size

            return img.copy(), w, h, original_mode

    except UnidentifiedImageError:
        logger.warning(event="image_unidentified", path=p.name, session_id=session_id)
        _EMBED_ERRORS.labels(reason="unidentified").inc()
        return None, 0, 0, ""

    except Exception as exc:
        logger.warning(event="image_load_failed", path=p.name, error=str(exc), session_id=session_id)
        _EMBED_ERRORS.labels(reason="load_exception").inc()
        return None, 0, 0, ""


# IMAGE EMBEDDER CLASS

class ImageEmbedder:
    """
    SigLIP-based image embedder.

    Responsibilities:
      - Load and validate images (size, dimension, mode).
      - Redis embedding cache keyed by SHA-256 file hash.
      - Batch SigLIP inference with L2 normalisation.
      - Dimension consistency enforcement.
      - Full Phase 24 metadata on every result.
    """

    def __init__(self, model, processor, device: str) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("TORCH_REQUIRED_FOR_IMAGE_EMBEDDER")

        self.model        = model
        self.processor    = processor
        self.device       = device
        self.model_name   = settings.SIGLIP_MODEL
        self.expected_dim = settings.VISION_EMBEDDING_DIM
        self.batch_size   = settings.EMBEDDING_BATCH_SIZE
        self.max_dim      = settings.MAX_IMAGE_DIM

        logger.info(
            event="image_embedder_initialized",
            device=device,
            model=self.model_name,
            expected_dim=self.expected_dim,
            batch_size=self.batch_size,
        )

    # SINGLE IMAGE EMBED

    def embed(
        self,
        image_path: str,
        session_id: str = "default",
    ) -> List[float]:
        """Embed a single image. Returns embedding vector."""
        results = self.embed_batch([image_path], session_id=session_id)
        if not results:
            raise NoValidImagesError(f"NO_EMBEDDING_PRODUCED for {image_path}")
        return results[0].embedding

    # BATCH IMAGE EMBED — PRIMARY METHOD

    def embed_batch(
        self,
        image_paths: List[str],
        session_id: str = "default",
    ) -> List[ImageEmbeddingResult]:
        """
        Embed a batch of image paths.

        Steps:
          1. Validate + load each image.
          2. Check Redis cache per image (SHA-256 keyed).
          3. Batch SigLIP inference for cache-miss images.
          4. L2 normalise all vectors.
          5. Validate embedding dimensions.
          6. Cache results in Redis.
          7. Return ImageEmbeddingResult list.
        """
        if not image_paths:
            return []

        if not session_id:
            raise ValueError("SESSION_ID_REQUIRED")

        start_total = time.time()

        # DEDUP INPUT PATHS
        seen_paths: Dict[str, bool] = {}
        unique_paths: List[str]     = []
        for p in image_paths:
            if str(p) not in seen_paths:
                seen_paths[str(p)] = True
                unique_paths.append(str(p))

        # LOAD + VALIDATE + CACHE CHECK
        loaded_images: List[Image.Image]          = []
        loaded_meta:   List[Dict[str, Any]]       = []
        cached_results: List[ImageEmbeddingResult] = []

        for path in unique_paths:
            checksum = _sha256_path(path)

            # CACHE HIT
            cached_emb = _cache_get(checksum)
            if cached_emb and _valid_embedding(cached_emb, self.expected_dim):
                _CACHE_HITS.inc()
                p = Path(path)
                cached_results.append(ImageEmbeddingResult(
                    path            = path,
                    embedding       = cached_emb,
                    embedding_dim   = self.expected_dim,
                    model_name      = self.model_name,
                    checksum_sha256 = checksum,
                    width           = 0,
                    height          = 0,
                    mode            = "cached",
                    embedding_id    = str(uuid.uuid4()),
                    latency_ms      = 0.0,
                    cache_hit       = True,
                    session_id      = session_id,
                ))
                continue

            # LOAD IMAGE
            img, w, h, mode = _load_image(path, self.max_dim, session_id)
            if img is None:
                continue

            loaded_images.append(img)
            loaded_meta.append({
                "path":     path,
                "checksum": checksum,
                "width":    w,
                "height":   h,
                "mode":     mode,
            })

        if not loaded_images and not cached_results:
            raise NoValidImagesError(
                f"NO_VALID_IMAGES from {len(unique_paths)} input paths"
            )

        # BATCH INFERENCE
        fresh_results: List[ImageEmbeddingResult] = []

        for i in range(0, len(loaded_images), self.batch_size):
            batch_imgs = loaded_images[i:i + self.batch_size]
            batch_meta = loaded_meta[i:i + self.batch_size]
            t_batch    = time.time()

            try:
                inputs = self.processor(
                    images=batch_imgs,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                    # SigLIP returns BaseModelOutputWithPooling — use pooler_output.
                    if hasattr(features, "pooler_output"):
                        features = features.pooler_output
                    features = F.normalize(features.float(), p=2, dim=-1)

                embeddings    = features.detach().cpu().numpy().tolist()
                batch_latency = round((time.time() - t_batch) * 1000, 1)

                # LATENCY WARNING
                target_ms = settings.LATENCY_TARGET_IMAGE_MS
                if batch_latency > target_ms:
                    logger.warning(
                        event="image_embed_batch_latency_exceeded",
                        latency_ms=batch_latency,
                        target_ms=target_ms,
                        batch_size=len(batch_imgs),
                        session_id=session_id,
                    )

                _EMBED_LATENCY.labels(batch_size=str(len(batch_imgs))).observe(
                    batch_latency / 1000.0
                )

                for emb, meta in zip(embeddings, batch_meta):
                    emb_list = emb if isinstance(emb, list) else list(emb)

                    # DIMENSION CONSISTENCY CHECK
                    if len(emb_list) != self.expected_dim:
                        logger.error(
                            event="image_embed_dim_mismatch",
                            got=len(emb_list),
                            expected=self.expected_dim,
                            path=meta["path"],
                            session_id=session_id,
                        )
                        _EMBED_ERRORS.labels(reason="dim_mismatch").inc()
                        raise DimensionMismatchError(
                            f"DIMENSION_MISMATCH: got {len(emb_list)}, "
                            f"expected {self.expected_dim}"
                        )

                    # NaN / Inf GUARD
                    if not _valid_embedding(emb_list, self.expected_dim):
                        logger.warning(
                            event="image_embed_invalid_values",
                            path=meta["path"],
                            session_id=session_id,
                        )
                        _EMBED_ERRORS.labels(reason="invalid_values").inc()
                        continue

                    # CACHE STORE
                    _cache_set(meta["checksum"], emb_list)
                    _EMBED_TOTAL.inc()

                    fresh_results.append(ImageEmbeddingResult(
                        path            = meta["path"],
                        embedding       = emb_list,
                        embedding_dim   = self.expected_dim,
                        model_name      = self.model_name,
                        checksum_sha256 = meta["checksum"],
                        width           = meta["width"],
                        height          = meta["height"],
                        mode            = meta["mode"],
                        embedding_id    = str(uuid.uuid4()),
                        latency_ms      = batch_latency,
                        cache_hit       = False,
                        session_id      = session_id,
                    ))

            except DimensionMismatchError:
                raise

            except Exception as exc:
                logger.error(
                    event="image_embed_batch_failed",
                    batch_start=i,
                    error=str(exc),
                    session_id=session_id,
                )
                _EMBED_ERRORS.labels(reason="batch_exception").inc()

        all_results = cached_results + fresh_results

        if not all_results:
            raise NoValidImagesError("NO_IMAGE_EMBEDDINGS_PRODUCED")

        total_latency = round(time.time() - start_total, 3)
        throughput    = round(len(all_results) / max(total_latency, 1e-6), 1)

        logger.info(
            event="image_embed_batch_success",
            total=len(all_results),
            cached=len(cached_results),
            fresh=len(fresh_results),
            throughput_per_sec=throughput,
            total_latency_sec=total_latency,
            session_id=session_id,
        )

        return all_results

    # ASYNC WRAPPER

    async def embed_async(
        self,
        image_path: str,
        session_id: str = "default",
    ) -> List[float]:
        async with _get_semaphore():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.embed(image_path, session_id),
            )

    async def embed_batch_async(
        self,
        image_paths: List[str],
        session_id: str = "default",
    ) -> List[ImageEmbeddingResult]:
        async with _get_semaphore():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.embed_batch(image_paths, session_id),
            )

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "model_loaded":    self.model is not None,
            "processor_loaded": self.processor is not None,
            "device":          self.device,
            "model_name":      self.model_name,
            "expected_dim":    self.expected_dim,
            "batch_size":      self.batch_size,
            "torch_available": TORCH_AVAILABLE,
            "pil_available":   PIL_AVAILABLE,
        }


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE DOCUMENT EMBEDDER (Phase 3)
# Extends BaseEmbedder to embed IngestedDocument objects produced by ImageChunker.
# Each doc's text = "{image_type}: {caption}\n{ocr_text}" — embed with BGE so
# image chunks are retrievable by finance text queries.
# The existing ImageEmbedder above handles raw image file paths via SigLIP and
# remains the vision-collection path.
# ══════════════════════════════════════════════════════════════════════════════

from app.embeddings.base_embedder import BaseEmbedder as _BaseEmbedder  # noqa: E402


import re as _re
_NUMBERS_ONLY_RE = _re.compile(r'[$€£¥₹]?[\d,]+\.?\d*\s*(?:[BMKTbmkt]n?|%|bps|x)?\b')


class ImageDocEmbedder(_BaseEmbedder):
    """BGE text embedder for image IngestedDocument objects.

    Enrichment:
      [{image_type}] {caption}. {ocr_text} {extracted_numbers}

    Phase 2.5 dual embedding:
      doc.embedding     = full caption (semantic retrieval)
      doc.embedding_alt = numbers-only text (exact numeric retrieval)

    Keeps image chunks in the text collection so they surface for text queries.
    SigLIP vision-space embedding (vision collection) is handled separately by
    ImageEmbedder.embed_batch() called from ingestion_pipeline.
    """

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        from app.core.config import settings as _s
        s          = getattr(doc, "structure", {}) or {}
        image_type = (s.get("image_type") or "").replace("_", " ").strip()
        caption    = (s.get("caption")    or "").strip()
        ocr_text   = (s.get("ocr_text")   or "").strip()

        nums: list = s.get("extracted_numbers") or []
        num_str    = " ".join(str(n) for n in nums[:10])

        # Build from structure fields (richer than cleaned_text which is the
        # combined caption+ocr string already, so use it as fallback)
        if image_type and caption:
            base = f"[{image_type}] {caption}"
            if ocr_text:
                base += f". {ocr_text}"
        else:
            base = cleaned_text

        if num_str:
            base += f" {num_str}"

        return base[:_s.MAX_PROMPT_CHARS]

    def _build_numbers_only_text(self, doc: Any) -> str:
        """Build numbers-only embed text for embedding_alt (Phase 2.5)."""
        from app.core.config import settings as _s
        s        = getattr(doc, "structure", {}) or {}
        ocr_text = (s.get("ocr_text") or "").strip()
        caption  = (s.get("caption")  or "").strip()
        combined = f"{ocr_text} {caption}".strip()

        # Extract all numeric tokens (dollar amounts, percentages, bps, etc.)
        nums = _NUMBERS_ONLY_RE.findall(combined)
        if not nums:
            # Fall back to extracted_numbers list from structure
            nums = [str(n) for n in (s.get("extracted_numbers") or [])]

        return " ".join(nums)[:_s.MAX_PROMPT_CHARS] if nums else ""

    def embed_documents(self, docs: List[Any], session_id: str = "default") -> List[Any]:
        """Embed image docs with dual vectors: primary (full caption) + alt (numbers-only).

        Calls the parent embed_documents for the primary vector, then computes
        embedding_alt via a second BGE pass over numbers-only text.
        """
        results = super().embed_documents(docs, session_id=session_id)
        if not results:
            return results

        # Phase 2.5: compute numbers-only secondary embedding for each result
        embedder   = self._get_model()
        alt_texts  = []
        alt_docs   = []
        for doc in results:
            alt_text = self._build_numbers_only_text(doc)
            if alt_text:
                alt_texts.append(alt_text)
                alt_docs.append(doc)

        if alt_texts:
            from app.embeddings.base_embedder import valid_embedding
            for i in range(0, len(alt_texts), embedder.batch_size):
                batch_texts = alt_texts[i:i + embedder.batch_size]
                batch_docs  = alt_docs[i:i + embedder.batch_size]
                try:
                    embs = embedder._encode_with_retry(embedder.model, batch_texts)
                    for doc, emb in zip(batch_docs, embs):
                        if valid_embedding(emb, embedder.expected_dim):
                            doc.embedding_alt = emb
                            struct = dict(getattr(doc, "structure", {}) or {})
                            struct["has_numeric_alt_embedding"] = True
                            doc.structure = struct
                except Exception as exc:
                    logger.debug(event="image_alt_embed_skip", error=str(exc))

        return results
