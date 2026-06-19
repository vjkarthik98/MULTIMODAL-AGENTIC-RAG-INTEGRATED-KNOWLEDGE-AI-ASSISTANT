"""
Per-modality model registry.

Maps each upload modality (text, document, image, audio, video) to the
minimal set of models that must be warm before its handler runs. Same for
the query path. The registry calls `model_loader.get_*` lazily — nothing
heavy is touched until an ingest/query path actually needs it.

This is what makes "upload a .txt file" stop pulling SigLIP/BLIP/Whisper
into memory.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, Dict, Iterable, List, Set, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Per-modality model lists. Names map to ModelLoader.get_<name>() / aliases.
# Keep these short: only the models actually called during that modality's
# ingest path. Query path is separate.
MODALITY_MODELS: Dict[str, Tuple[str, ...]] = {
    "text":     ("text_embedder",),
    "txt":      ("text_embedder",),
    "document": ("text_embedder",),
    "pdf":      ("text_embedder", "trocr"),
    "word":     ("text_embedder",),
    "excel":    ("text_embedder",),
    "image":    ("text_embedder", "blip", "qwen2_vl", "trocr", "siglip", "image_embedder", "siglip_text_embedder"),
    "audio":    ("text_embedder", "whisper", "diarizer", "ner"),
    "video":    ("text_embedder", "qwen2_vl", "trocr", "siglip", "image_embedder",
                 "siglip_text_embedder", "whisper", "diarizer", "ner"),
}

# Models the query path needs (no upload).
QUERY_MODELS:        Tuple[str, ...] = ("text_embedder", "llm", "reranker")
QUERY_VISION_MODELS: Tuple[str, ...] = ("siglip_text_embedder",)


class ModelRegistry:

    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._loaded:  Set[str] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, settings.THREAD_POOL_SIZE),
            thread_name_prefix="model_registry",
        )

    # PUBLIC: warm exactly the models this modality needs.

    def ensure_for_modality(self, modality: str) -> List[str]:
        names = MODALITY_MODELS.get(modality)
        if not names:
            logger.warning(event="modality_registry_unknown", modality=modality)
            return []
        return self._ensure(names, scope=f"modality:{modality}")

    # PUBLIC: warm the query path. Pass needs_vision=True for cross-modal queries.
    # EVAL_SKIP_LLM_WARMUP=true prevents loading the LLM in eval subprocesses
    # where the server already holds the GPU — avoids double-load VRAM crash.

    def ensure_for_query(self, needs_vision: bool = False) -> List[str]:
        import os
        names = list(QUERY_MODELS)
        if os.getenv("EVAL_SKIP_LLM_WARMUP", "").lower() == "true":
            names = [n for n in names if n != "llm"]
        if needs_vision:
            names += list(QUERY_VISION_MODELS)
        return self._ensure(tuple(names), scope="query")

    # PUBLIC: warm a custom list (used by warmup_at_startup).

    def ensure(self, names: Iterable[str]) -> List[str]:
        return self._ensure(tuple(names), scope="custom")

    def already_loaded(self, name: str) -> bool:
        return name in self._loaded

    # INTERNAL: dispatch in parallel; treat individual failures as warnings
    # so e.g. an audio upload doesn't fail because BLIP failed.

    def _ensure(self, names: Tuple[str, ...], scope: str) -> List[str]:
        # Filter out models already loaded and disabled-by-flag models.
        to_load: List[str] = []
        for n in names:
            if n in self._loaded:
                continue
            if not self._is_enabled(n):
                logger.debug(event="model_disabled_skip", model=n, scope=scope)
                continue
            to_load.append(n)

        if not to_load:
            return []

        start  = time.time()
        loaded: List[str] = []

        if settings.MODEL_PARALLEL_LOAD and len(to_load) > 1:
            futures = {}
            for name in to_load:
                try:
                    futures[name] = self._executor.submit(self._load_one, name)
                except RuntimeError as exc:
                    if "interpreter shutdown" in str(exc) or "shutdown" in str(exc).lower():
                        logger.debug(event="model_load_skipped_shutdown", model=name)
                        return loaded
                    raise
            for name, fut in futures.items():
                try:
                    fut.result(timeout=settings.MODEL_TIMEOUT_SEC)
                    with self._lock:
                        self._loaded.add(name)
                    loaded.append(name)
                except FuturesTimeoutError:
                    logger.error(event="model_warmup_timeout", model=name, scope=scope)
                except Exception as exc:
                    logger.warning(
                        event="model_warmup_failed",
                        model=name,
                        scope=scope,
                        error=str(exc),
                    )
        else:
            for name in to_load:
                try:
                    self._load_one(name)
                    with self._lock:
                        self._loaded.add(name)
                    loaded.append(name)
                except RuntimeError as exc:
                    if "interpreter shutdown" in str(exc) or "shutdown" in str(exc).lower():
                        logger.debug(event="model_load_skipped_shutdown", model=name)
                        return loaded
                    logger.warning(
                        event="model_warmup_failed",
                        model=name,
                        scope=scope,
                        error=str(exc),
                    )
                except Exception as exc:
                    logger.warning(
                        event="model_warmup_failed",
                        model=name,
                        scope=scope,
                        error=str(exc),
                    )

        if loaded:
            logger.info(
                event="models_warmed",
                scope=scope,
                models=loaded,
                latency=round(time.time() - start, 2),
            )
        return loaded

    def _is_enabled(self, name: str) -> bool:
        # Vision models gated by ENABLE_VISION.
        # Audio/diarizer/ner gated by ENABLE_AUDIO.
        if name in ("siglip", "blip", "qwen2_vl", "trocr",
                    "image_embedder", "siglip_text_embedder"):
            return bool(settings.ENABLE_VISION)
        if name in ("whisper", "diarizer", "ner"):
            return bool(settings.ENABLE_AUDIO)
        return True

    def _load_one(self, name: str) -> None:
        # Local import — keeps registry import-light and avoids cycles.
        from app.core.model_loader import model_loader

        loaders: Dict[str, Callable[[], object]] = {
            "llm":                  model_loader.get_llm,
            "text_embedder":        model_loader.get_embedder,
            "siglip":               model_loader.get_siglip,
            "blip":                 model_loader.get_blip,
            "qwen2_vl":             model_loader.get_qwen2_vl,
            "trocr":                model_loader.get_trocr,
            "diarizer":             model_loader.get_diarizer,
            "ner":                  model_loader.get_ner,
            "finbert":              model_loader.get_finbert,
            "image_embedder":       model_loader.get_image_embedder,
            "siglip_text_embedder": model_loader.get_siglip_text_embedder,
            "whisper":              model_loader.get_whisper,
            "reranker":             model_loader.get_reranker,
        }
        fn = loaders.get(name)
        if fn is None:
            raise KeyError(f"UNKNOWN_MODEL: {name}")
        fn()

    def snapshot(self) -> Dict[str, bool]:
        all_known = (
            "llm", "text_embedder", "siglip", "blip", "qwen2_vl",
            "trocr", "diarizer", "ner",
            "image_embedder", "siglip_text_embedder", "whisper", "reranker",
        )
        return {n: n in self._loaded for n in all_known}


# SINGLETON

model_registry = ModelRegistry()
