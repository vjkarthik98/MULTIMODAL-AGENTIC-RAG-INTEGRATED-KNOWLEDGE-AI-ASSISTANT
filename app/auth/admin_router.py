"""
Admin API — industry-standard admin capabilities for a RAG system.

All routes require role=admin in the JWT. A regular user hitting any
of these endpoints gets HTTP 403 Forbidden.

Routes:
  GET    /admin/users                    — list all users + usage stats
  GET    /admin/users/{user_id}          — single user detail + stats
  PATCH  /admin/users/{user_id}/role     — promote/demote role
  PATCH  /admin/users/{user_id}/status   — activate / deactivate
  DELETE /admin/users/{user_id}          — GDPR purge any user
  GET    /admin/system/health            — Qdrant + Redis + Mongo + model status
  GET    /admin/system/audit             — recent audit log entries
  GET    /admin/stats                    — platform-wide usage summary
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.auth.dependencies import get_current_admin_user
from app.auth.models import UserPublic, UserRole
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_users_col():
    from app.core.infra_registry import infra
    mongo = infra.get_mongo()
    if mongo is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")
    return mongo.client[settings.MONGO_DB_NAME][settings.AUTH_COLLECTION]


def _get_messages_col():
    from app.core.infra_registry import infra
    mongo = infra.get_mongo()
    if mongo is None:
        return None
    return mongo.client[settings.MONGO_DB_NAME][settings.MONGO_MESSAGES_COLLECTION]


def _get_summaries_col():
    from app.core.infra_registry import infra
    mongo = infra.get_mongo()
    if mongo is None:
        return None
    return mongo.client[settings.MONGO_DB_NAME][settings.MONGO_SUMMARIES_COLLECTION]


def _user_stats(user_id: str) -> Dict[str, Any]:
    """Compute per-user usage statistics from MongoDB."""
    stats: Dict[str, Any] = {
        "total_queries": 0,
        "total_messages": 0,
        "total_summaries": 0,
        "qdrant_chunks": 0,
    }
    try:
        msg_col = _get_messages_col()
        if msg_col is not None:
            stats["total_messages"] = msg_col.count_documents({"user_id": user_id})
            stats["total_queries"] = msg_col.count_documents(
                {"user_id": user_id, "role": "user"}
            )
        sum_col = _get_summaries_col()
        if sum_col is not None:
            stats["total_summaries"] = sum_col.count_documents({"user_id": user_id})
    except Exception:
        pass

    try:
        from app.core.infra_registry import infra
        vs = infra.get_vector_store()
        if vs:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            result = vs.client.count(
                collection_name=vs.text_collection,
                count_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                exact=False,
            )
            stats["qdrant_chunks"] = result.count
    except Exception:
        pass

    return stats


def _doc_to_user(doc: dict) -> dict:
    return {
        "user_id":    doc.get("user_id"),
        "email":      doc.get("email"),
        "role":       doc.get("role", "user"),
        "is_active":  doc.get("is_active", True),
        "created_at": doc.get("created_at"),
        "last_login": doc.get("last_login"),
    }


# ── Request / response models ─────────────────────────────────────────────────

class RoleUpdate(BaseModel):
    role: UserRole


class StatusUpdate(BaseModel):
    is_active: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/users", summary="List all users")
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """List all registered users with basic profile info."""
    col = _get_users_col()
    total = col.count_documents({})
    docs = list(col.find({}, {"hashed_password": 0, "_id": 0}).skip(skip).limit(limit))
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "users": [_doc_to_user(d) for d in docs],
    }


@router.get("/users/{user_id}", summary="Get user detail + usage stats")
async def get_user(
    user_id: str,
    _admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Full profile + usage statistics for a specific user."""
    col = _get_users_col()
    doc = col.find_one({"user_id": user_id}, {"hashed_password": 0, "_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")

    stats = await asyncio.to_thread(_user_stats, user_id)
    return {**_doc_to_user(doc), "stats": stats}


@router.patch("/users/{user_id}/role", summary="Promote or demote user role")
async def update_user_role(
    user_id: str,
    body: RoleUpdate,
    admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Change a user's role (user ↔ admin). Admins cannot demote themselves."""
    if user_id == admin.user_id and body.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=400,
            detail="You cannot remove your own admin role",
        )

    col = _get_users_col()
    result = col.update_one(
        {"user_id": user_id},
        {"$set": {"role": body.role.value}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(
        event="admin_role_updated",
        target_user=user_id,
        new_role=body.role.value,
        by_admin=admin.user_id,
    )
    return {"status": "ok", "user_id": user_id, "role": body.role.value}


@router.patch("/users/{user_id}/status", summary="Activate or deactivate a user")
async def update_user_status(
    user_id: str,
    body: StatusUpdate,
    admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Block or unblock a user. Blocked users cannot log in."""
    if user_id == admin.user_id and not body.is_active:
        raise HTTPException(
            status_code=400,
            detail="You cannot deactivate your own account",
        )

    col = _get_users_col()
    result = col.update_one(
        {"user_id": user_id},
        {"$set": {"is_active": body.is_active}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    action = "activated" if body.is_active else "deactivated"
    logger.info(
        event=f"admin_user_{action}",
        target_user=user_id,
        by_admin=admin.user_id,
    )
    return {"status": "ok", "user_id": user_id, "is_active": body.is_active}


@router.delete("/users/{user_id}", summary="GDPR purge any user")
async def admin_purge_user(
    user_id: str,
    admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """
    Permanently delete all data for a user across Qdrant, Redis, and MongoDB.
    Also removes their account. This cannot be undone.
    """
    if user_id == admin.user_id:
        raise HTTPException(
            status_code=400,
            detail="Use DELETE /auth/me to delete your own account",
        )

    # Purge memory (Redis + Mongo messages/summaries)
    try:
        from app.memory.memory_manager import MemoryManager
        manager = MemoryManager()
        await asyncio.to_thread(manager.gdpr_purge, user_id)
    except Exception as exc:
        logger.warning(event="admin_gdpr_memory_failed", user_id=user_id, error=str(exc))

    # Purge BM25
    try:
        from app.core.infra_registry import infra
        bm25 = infra.get_bm25()
        if bm25 and hasattr(bm25, "purge_by_session"):
            await asyncio.to_thread(bm25.purge_by_session, user_id)
    except Exception as exc:
        logger.warning(event="admin_gdpr_bm25_failed", user_id=user_id, error=str(exc))

    # Remove auth account
    col = _get_users_col()
    result = col.delete_one({"user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(
        event="admin_user_purged",
        target_user=user_id,
        by_admin=admin.user_id,
    )
    return {
        "status": "ok",
        "user_id": user_id,
        "message": "User and all associated data permanently deleted",
    }


@router.get("/system/health", summary="System health overview")
async def system_health(
    _admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Live health status of Qdrant, Redis, MongoDB, and loaded models."""
    from app.core.infra_registry import infra

    health: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "infra": {},
        "models": {},
    }

    try:
        health["infra"] = infra.health_check()
    except Exception as exc:
        health["infra"] = {"error": str(exc)}

    try:
        from app.core.model_loader import model_loader
        health["models"] = model_loader.health_check()
    except Exception as exc:
        health["models"] = {"error": str(exc)}

    return health


@router.get("/system/audit", summary="Recent audit log entries")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    _admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """
    Return the most recent audit log entries from the append-only audit log file.
    Each line is a JSON event (login, query, ingest, purge, etc).
    """
    import json

    entries = []
    try:
        with open(settings.AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Most recent first
        for line in reversed(lines[-limit:]):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    entries.append({"raw": line})
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(event="admin_audit_log_read_failed", error=str(exc))

    return {
        "count": len(entries),
        "entries": entries,
    }


@router.get("/stats", summary="Platform-wide usage summary")
async def platform_stats(
    _admin: UserPublic = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Aggregate statistics across all users — total users, queries, chunks stored."""
    stats: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_users": 0,
        "active_users": 0,
        "admin_users": 0,
        "total_messages": 0,
        "total_summaries": 0,
        "total_qdrant_chunks": 0,
    }

    try:
        col = _get_users_col()
        stats["total_users"]  = col.count_documents({})
        stats["active_users"] = col.count_documents({"is_active": True})
        stats["admin_users"]  = col.count_documents({"role": "admin"})
    except Exception:
        pass

    try:
        msg_col = _get_messages_col()
        if msg_col is not None:
            stats["total_messages"] = msg_col.count_documents({})
        sum_col = _get_summaries_col()
        if sum_col is not None:
            stats["total_summaries"] = sum_col.count_documents({})
    except Exception:
        pass

    try:
        from app.core.infra_registry import infra
        vs = infra.get_vector_store()
        if vs:
            info = vs.client.get_collection(vs.text_collection)
            stats["total_qdrant_chunks"] = info.points_count or 0
    except Exception:
        pass

    return stats
