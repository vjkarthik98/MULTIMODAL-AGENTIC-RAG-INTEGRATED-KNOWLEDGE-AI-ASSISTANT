"""
Startup model manifest validator.

Called from main.py lifespan before serving traffic. When
MODEL_CACHE_REQUIRE_MANIFEST=true, raises RuntimeError listing any models
that are absent from .hf_cache/download_manifest.json so the operator knows
to run `python app/bin/models/download_all_models.py` before restarting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_MODELS: set[str] = {
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-reranker-large",
    "Salesforce/blip-image-captioning-large",
    settings.QWEN2_VL_MODEL,  # image/video/chart captioning (2B or 7B per config)
    "Systran/faster-whisper-large-v3",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/speaker-diarization-3.1",
    "microsoft/trocr-large-printed",
    "dslim/bert-base-NER",
    "google/siglip-so400m-patch14-384",
    "yiyanghkust/finbert-tone",
}


def validate_auth_config() -> None:
    """Fail fast if AUTH_ENABLED=False is attempted in a production environment."""
    if not settings.AUTH_ENABLED and settings.ENV not in ("development", "test", "dev"):
        raise RuntimeError(
            "AUTH_ENABLED=False is not allowed in production environment "
            f"(ENV={settings.ENV!r}). Set AUTH_ENABLED=true or change ENV to development/test."
        )


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
        cached: set[str] = {e["model_id"] for e in entries if "model_id" in e}
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

    _validate_gguf_checksum(entries)


def _validate_gguf_checksum(manifest_entries: list[dict]) -> None:
    """Fail fast if the on-disk GGUF file no longer matches the SHA-256
    recorded at download time (app/bin/models/download_all_models.py).
    Presence-checking alone (above) only proves a model_id was ever
    downloaded, not that the file is still the same bytes — this is the
    other half of "pinned model file + checksum" (System Design v2.0 §2.2).
    Scoped to the GGUF only (not all 11 models): it's the single highest-
    stakes file (the actual generation model) and the cheapest to verify —
    hashing every multimodal model's full snapshot dir on every boot would
    add real startup latency for comparatively low incremental safety, since
    download_all_models.py already re-verifies all of them on every re-run.
    """
    # More than one GGUF-type entry can exist in the manifest now (the main
    # LLM plus eval-only judges like Prometheus, see
    # app/bin/models/download_all_models.py's GGUF_MODELS) — "the first
    # gguf-type entry" is no longer a safe way to find the main LLM's entry,
    # so match on the filename actually configured as LLM_MODEL_PATH.
    # Manifests written before this field existed have no "gguf_filename" key
    # at all (only one GGUF model existed then) — fall back to that entry for
    # those, so an already-deployed manifest.json isn't treated as invalid.
    expected_name = Path(settings.LLM_MODEL_PATH).name
    gguf_entries = [e for e in manifest_entries if e.get("type") == "gguf"]
    gguf_entry = next((e for e in gguf_entries if e.get("gguf_filename") == expected_name), None)
    if gguf_entry is None:
        gguf_entry = next((e for e in gguf_entries if "gguf_filename" not in e), None)
    if gguf_entry is None:
        return  # no GGUF entry yet — validate_model_manifest()'s presence check already covers this
    expected = gguf_entry.get("sha256")
    if not expected:
        logger.warning(
            event="gguf_checksum_not_recorded",
            hint="Model was downloaded before checksum recording existed. "
            "Re-run download_all_models.py to backfill it.",
        )
        return

    gguf_path = Path(settings.LLM_MODEL_PATH)
    if not gguf_path.exists():
        raise RuntimeError(f"GGUF model file missing on disk: {gguf_path}")

    h = hashlib.sha256()
    with open(gguf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()

    if actual != expected:
        raise RuntimeError(
            f"GGUF_CHECKSUM_MISMATCH: {gguf_path} does not match the SHA-256 recorded "
            f"at last download (expected {expected[:16]}..., got {actual[:16]}...). "
            "Possible corruption or an unexpected file swap — re-download before serving "
            "traffic:  python app/bin/models/download_all_models.py --only gguf"
        )

    logger.info(event="gguf_checksum_ok", sha256=actual[:16])
