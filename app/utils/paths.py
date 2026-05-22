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


def user_bm25_path(user_id: Optional[str] = None) -> Path:
    p = user_dir(user_id) / "bm25_index"
    p.mkdir(exist_ok=True)
    return p / "bm25.pkl"


# ─── RESOLVED PATHS (contextvar-aware, with global-fallback) ──────────────────
# These wrappers make ingestors' call sites uniform: replace
#   settings.TEMP_DIR  →  resolved_temp_dir()
#   settings.PDF_IMAGE_DIR  →  resolved_images_dir()
# etc. They use the active user when present, otherwise fall back to the
# settings paths so non-ingestion callers (eg. tests, scripts) still work.


def resolved_staging_dir() -> Path:
    uid = _current_user_id.get()
    if uid:
        return user_staging_dir(uid)
    from app.core.config import settings
    settings.UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    return settings.UPLOAD_STAGING_DIR


def resolved_temp_dir() -> Path:
    uid = _current_user_id.get()
    if uid:
        return user_temp_dir(uid)
    from app.core.config import settings
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return settings.TEMP_DIR


def resolved_images_dir() -> Path:
    uid = _current_user_id.get()
    if uid:
        return user_images_dir(uid)
    from app.core.config import settings
    settings.PDF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return settings.PDF_IMAGE_DIR


def resolved_temp_frames_dir() -> Path:
    uid = _current_user_id.get()
    if uid:
        return user_temp_frames_dir(uid)
    from app.core.config import settings
    settings.VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    return settings.VIDEO_FRAMES_DIR
