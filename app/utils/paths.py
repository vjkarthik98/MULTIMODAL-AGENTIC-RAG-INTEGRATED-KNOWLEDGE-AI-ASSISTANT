from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

# DATA_ROOT is always data/users/ — relative to project root.
# In production (Phase 30) this switches to S3; for now we keep it local.
DATA_ROOT = Path("data/users")

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# CONTEXTVAR — set at pipeline entry by ingestion_pipeline.process_file().
# Every storage helper reads from this so no signature changes are needed
# in 30+ internal helpers across ingestion modules.
_current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)


def set_current_user(user_id: Optional[str]) -> object:
    """Set the active user for this async task. Returns a token for reset."""
    return _current_user_id.set(user_id)


def reset_current_user(token: object) -> None:
    _current_user_id.reset(token)  # type: ignore[arg-type]


def get_current_user() -> Optional[str]:
    return _current_user_id.get()


def _active_user(user_id: Optional[str] = None) -> Optional[str]:
    """Resolution order: explicit arg → contextvar → None (caller decides fallback)."""
    return user_id or _current_user_id.get()


def user_dir(user_id: Optional[str] = None) -> Path:
    uid = _active_user(user_id)
    if not uid:
        # No active user — caller must pass one explicitly.
        raise ValueError("user_dir called without user_id and no contextvar set")
    path = DATA_ROOT / uid
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_knowledge_base_dir(user_id: Optional[str] = None) -> Path:
    """Persistent store for original uploaded files. Survives ingestion cleanup."""
    p = user_dir(user_id) / "knowledge_base"
    p.mkdir(exist_ok=True)
    return p


def user_staging_dir(user_id: Optional[str] = None) -> Path:
    """Temp staging area used during ingestion — cleaned up after pipeline completes."""
    p = user_dir(user_id) / "staging"
    p.mkdir(exist_ok=True)
    return p


def user_documents_dir(user_id: Optional[str] = None) -> Path:
    p = user_dir(user_id) / "documents"
    p.mkdir(exist_ok=True)
    return p


def user_images_dir(user_id: Optional[str] = None) -> Path:
    p = user_dir(user_id) / "images"
    p.mkdir(exist_ok=True)
    return p


def user_temp_dir(user_id: Optional[str] = None) -> Path:
    p = user_dir(user_id) / "temp"
    p.mkdir(exist_ok=True)
    return p


def user_temp_frames_dir(user_id: Optional[str] = None) -> Path:
    p = user_dir(user_id) / "temp_frames"
    p.mkdir(exist_ok=True)
    return p


def user_bm25_path(user_id: Optional[str] = None, modality: str = "") -> Path:
    """Per-user (optionally per-modality) BM25 index file."""
    p = user_dir(user_id) / "bm25" if modality else user_dir(user_id) / "bm25_index"
    p.mkdir(exist_ok=True)
    filename = f"{modality}_index.pkl" if modality else "bm25.pkl"
    return p / filename


# ─── .HF_CACHE PATH HELPERS (Phase 8) ─────────────────────────────────────────

def _hf_home() -> Path:
    """Resolve HF_HOME from environment or project-relative default."""
    from app.core.config import settings as _s
    return Path(getattr(_s, "HF_HOME", ".hf_cache"))


def hf_model_path(model_id: str) -> Path:
    """Returns .hf_cache/hub/models--{org}--{name}/ for a HuggingFace model."""
    safe_id = model_id.replace("/", "--")
    return _hf_home() / "hub" / f"models--{safe_id}"


def gguf_path(filename: Optional[str] = None) -> Path:
    """Returns path to a GGUF model file inside .hf_cache/gguf/."""
    from app.core.config import settings as _s
    fname = filename or Path(getattr(_s, "LLM_MODEL_PATH", "model.gguf")).name
    return _hf_home() / "gguf" / fname


# ─── RESOLVED PATHS (contextvar-aware, with global-fallback) ──────────────────
# These wrappers make ingestors' call sites uniform: replace
#   settings.TEMP_DIR  →  resolved_temp_dir()
#   settings.PDF_IMAGE_DIR  →  resolved_images_dir()
# etc. They use the active user when present, otherwise fall back to the
# settings paths so non-ingestion callers (eg. tests, scripts) still work.


def resolved_staging_dir() -> Path:
    uid = _current_user_id.get()
    if not uid:
        raise ValueError("resolved_staging_dir called without active user context")
    return user_staging_dir(uid)


def resolved_temp_dir() -> Path:
    uid = _current_user_id.get()
    if not uid:
        raise ValueError("resolved_temp_dir called without active user context")
    return user_temp_dir(uid)


def resolved_images_dir() -> Path:
    uid = _current_user_id.get()
    if not uid:
        raise ValueError("resolved_images_dir called without active user context")
    return user_images_dir(uid)


def resolved_temp_frames_dir() -> Path:
    uid = _current_user_id.get()
    if not uid:
        raise ValueError("resolved_temp_frames_dir called without active user context")
    return user_temp_frames_dir(uid)
