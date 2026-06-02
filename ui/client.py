"""
ui/client.py — Synchronous httpx client that calls the FastAPI backend.
All UI event handlers are synchronous (Gradio's default), so this uses
httpx in sync mode throughout.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Generous timeout — GGUF inference can take up to ~90s on CPU
_QUERY_TIMEOUT = 180.0
_UPLOAD_TIMEOUT = 120.0
_AUTH_TIMEOUT   = 15.0
_KB_TIMEOUT     = 15.0


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(email: str, password: str) -> Dict[str, Any]:
    """POST /auth/login → {access_token, refresh_token} or {detail}."""
    with httpx.Client(timeout=_AUTH_TIMEOUT) as c:
        r = c.post(f"{BASE_URL}/auth/login", json={"email": email, "password": password})
        return r.json()


def register(email: str, password: str) -> Dict[str, Any]:
    """POST /auth/register → UserPublic or {detail}."""
    with httpx.Client(timeout=_AUTH_TIMEOUT) as c:
        r = c.post(f"{BASE_URL}/auth/register", json={"email": email, "password": password})
        return r.json()


def logout_user(token: str) -> Dict[str, Any]:
    """POST /auth/logout — revokes current token in Redis blacklist."""
    try:
        with httpx.Client(timeout=_AUTH_TIMEOUT) as c:
            r = c.post(f"{BASE_URL}/auth/logout", headers=_auth_headers(token))
            return r.json()
    except Exception:
        return {"status": "ok"}


def get_me(token: str) -> Dict[str, Any]:
    """GET /auth/me → UserPublic."""
    with httpx.Client(timeout=_AUTH_TIMEOUT) as c:
        r = c.get(f"{BASE_URL}/auth/me", headers=_auth_headers(token))
        r.raise_for_status()
        return r.json()


# ── Query ─────────────────────────────────────────────────────────────────────

def query(
    text: str,
    session_id: str,
    token: str,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """POST /query → full AgentResponse dict (answer, decision, confidence, sources, latency)."""
    payload: Dict[str, Any] = {"query": text, "session_id": session_id}
    if sources:
        payload["sources"] = sources
    with httpx.Client(timeout=_QUERY_TIMEOUT) as c:
        r = c.post(f"{BASE_URL}/query", json=payload, headers=_auth_headers(token))
        r.raise_for_status()
        return r.json()


# ── Knowledge Base ────────────────────────────────────────────────────────────

def list_kb(token: str) -> Dict[str, Any]:
    """GET /knowledge-base → {files: [{filename, modality, size_mb, ...}]}."""
    with httpx.Client(timeout=_KB_TIMEOUT) as c:
        r = c.get(f"{BASE_URL}/knowledge-base", headers=_auth_headers(token))
        r.raise_for_status()
        return r.json()


def ingest(file_path: str, session_id: str, token: str) -> Dict[str, Any]:
    """POST /ingest (multipart) → ingest result dict."""
    path = Path(file_path)
    with path.open("rb") as fh:
        with httpx.Client(timeout=_UPLOAD_TIMEOUT) as c:
            r = c.post(
                f"{BASE_URL}/ingest",
                files={"file": (path.name, fh, _guess_mime(path.suffix))},
                data={"session_id": session_id},
                headers=_auth_headers(token),
            )
            return r.json()


def delete_kb_file(filename: str, token: str) -> Dict[str, Any]:
    """DELETE /knowledge-base/{filename} → purge from disk + Qdrant + BM25."""
    with httpx.Client(timeout=_KB_TIMEOUT) as c:
        r = c.delete(f"{BASE_URL}/knowledge-base/{filename}", headers=_auth_headers(token))
        r.raise_for_status()
        return r.json()


def clear_memory(session_id: str, token: str) -> Dict[str, Any]:
    """POST /memory/clear — wipes Redis short-term memory for this session."""
    with httpx.Client(timeout=_AUTH_TIMEOUT) as c:
        r = c.post(
            f"{BASE_URL}/memory/clear",
            json={"session_id": session_id},
            headers=_auth_headers(token),
        )
        return r.json()


# ── Helpers ───────────────────────────────────────────────────────────────────

_MIME: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md":  "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".avi": "video/x-msvideo", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
}


def _guess_mime(ext: str) -> str:
    return _MIME.get(ext.lower(), "application/octet-stream")
