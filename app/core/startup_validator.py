"""
Startup model manifest validator.

Called from main.py lifespan before serving traffic. When
MODEL_CACHE_REQUIRE_MANIFEST=true, raises RuntimeError listing any models
that are absent from .hf_cache/download_manifest.json so the operator knows
to run `python app/bin/models/download_all_models.py` before restarting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_MODELS: Set[str] = {
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-reranker-large",
    "Salesforce/blip2-opt-2.7b",
    "llava-hf/llava-1.5-7b-hf",
    "openai/whisper-large-v3",
    "pyannote/speaker-diarization-3.1",
    "microsoft/trocr-large-printed",
    "dslim/bert-base-NER",
    "google/siglip-so400m-patch14-384",
}


def validate_model_manifest() -> None:
    """Fail fast if any required model is absent from the download manifest."""
    if not settings.MODEL_CACHE_REQUIRE_MANIFEST:
        return

    manifest_path = Path(settings.HF_HOME) / "download_manifest.json"

    if not manifest_path.exists():
        raise RuntimeError(
            "Model manifest missing — .hf_cache/download_manifest.json not found.\n"
            "Run:  python app/bin/models/download_all_models.py"
        )

    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached: Set[str] = {e["model_id"] for e in entries if "model_id" in e}
    except Exception as exc:
        raise RuntimeError(f"Failed to parse download_manifest.json: {exc}") from exc

    missing = REQUIRED_MODELS - cached
    if missing:
        missing_list = "\n  - ".join(sorted(missing))
        raise RuntimeError(
            f"Required models not cached ({len(missing)} missing):\n  - {missing_list}\n"
            "Run:  python app/bin/models/download_all_models.py"
        )

    logger.info(
        event="model_manifest_ok",
        cached=len(cached),
        required=len(REQUIRED_MODELS),
    )
