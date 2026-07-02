"""Guest session management: lifecycle, atomic limit enforcement, data migration.

Redis key schema:
  guest:{guest_id}:queries  — INCR counter, TTL = GUEST_SESSION_TTL_HOURS * 3600
  guest:{guest_id}:uploads  — INCR counter, same TTL
  guest_create:{client_ip}  — abuse guard counter, TTL = 600 (10 min)

All Redis operations fail-open: a Redis outage must not break the app.
"""
from __future__ import annotations

import shutil
import uuid
from typing import Any, Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.paths import DATA_ROOT

logger = get_logger(__name__)

_TTL = settings.GUEST_SESSION_TTL_HOURS * 3600

# Lua: atomic check-then-increment. Returns 1 if allowed, 0 if limit reached.
_LUA_INCR_IF_UNDER = """
local key   = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl   = tonumber(ARGV[2])
local cur   = tonumber(redis.call('GET', key) or '0')
if cur >= limit then return 0 end
redis.call('INCR', key)
redis.call('EXPIRE', key, ttl)
return 1
"""


def _redis():
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem is not None and mem.client is not None:
            return mem.client
    except Exception:
        pass
    return None


def _is_upstash(r) -> bool:
    # upstash-redis is a REST client with a different eval()/pipeline() API
    # shape than redis-py (eval takes keys=[...]/args=[...] lists, not
    # positional numkeys+*args; pipeline batches flush via .exec(), not a
    # no-arg .execute()). Detect by class module rather than importing
    # upstash_redis here (it may not be installed when redis-py is in use).
    return type(r).__module__.startswith("upstash_redis")


def _eval_incr_if_under(r, key: str, limit: int, ttl: int):
    if _is_upstash(r):
        return r.eval(_LUA_INCR_IF_UNDER, keys=[key], args=[str(limit), str(ttl)])
    return r.eval(_LUA_INCR_IF_UNDER, 1, key, limit, ttl)


# ── Session lifecycle ─────────────────────────────────────────────────────────

def create_guest_session() -> str:
    """Generate a guest_user_id and initialise Redis counters. Returns guest_user_id."""
    guest_id = f"guest_{uuid.uuid4().hex[:12]}"
    r = _redis()
    if r:
        try:
            if _is_upstash(r):
                # upstash-redis's Pipeline.set() queues the command; the batch
                # is sent by .exec(), not a no-arg .execute() (that name means
                # something else on this client — see _is_upstash docstring).
                pipe = r.pipeline()
                pipe.set(f"guest:{guest_id}:queries", 0, ex=_TTL)
                pipe.set(f"guest:{guest_id}:uploads", 0, ex=_TTL)
                pipe.exec()
            else:
                pipe = r.pipeline()
                pipe.set(f"guest:{guest_id}:queries", 0, ex=_TTL)
                pipe.set(f"guest:{guest_id}:uploads", 0, ex=_TTL)
                pipe.execute()
        except Exception as exc:
            logger.warning(event="guest_session_redis_init_failed", error=str(exc))
    logger.info(event="guest_session_created", guest_id=guest_id)
    return guest_id


def get_guest_limits(guest_id: str) -> Dict[str, int]:
    """Return current usage counters and remaining allowances."""
    queries_used = uploads_used = 0
    r = _redis()
    if r:
        try:
            q = r.get(f"guest:{guest_id}:queries")
            u = r.get(f"guest:{guest_id}:uploads")
            queries_used = int(q) if q else 0
            uploads_used = int(u) if u else 0
        except Exception:
            pass
    return {
        "queries_used":  queries_used,
        "uploads_used":  uploads_used,
        "queries_left":  max(0, settings.GUEST_QUERY_LIMIT - queries_used),
        "uploads_left":  max(0, settings.GUEST_FILE_LIMIT  - uploads_used),
    }


# ── Atomic limit enforcement (Lua — no TOCTOU race) ──────────────────────────

def check_and_increment_queries(guest_id: str) -> bool:
    """Atomically check and increment the query counter.
    Returns True if the query is allowed, False if the limit is reached.
    Fails open on Redis outage so a Redis failure never blocks the app.
    """
    r = _redis()
    if r is None:
        return True
    try:
        result = _eval_incr_if_under(
            r, f"guest:{guest_id}:queries", settings.GUEST_QUERY_LIMIT, _TTL,
        )
        return bool(result)
    except Exception as exc:
        logger.warning(event="guest_query_limit_check_failed", error=str(exc))
        return True


def check_and_increment_uploads(guest_id: str) -> bool:
    """Atomically check and increment the upload counter.
    Returns True if the upload is allowed, False if the limit is reached.
    Fails open on Redis outage.
    """
    r = _redis()
    if r is None:
        return True
    try:
        result = _eval_incr_if_under(
            r, f"guest:{guest_id}:uploads", settings.GUEST_FILE_LIMIT, _TTL,
        )
        return bool(result)
    except Exception as exc:
        logger.warning(event="guest_upload_limit_check_failed", error=str(exc))
        return True


def check_guest_create_rate(client_ip: str) -> bool:
    """IP-based abuse guard for POST /auth/guest.
    Allows up to 5 guest session creations per IP per 10 minutes.
    Fails open on Redis outage.
    """
    r = _redis()
    if r is None:
        return True
    key = f"guest_create:{client_ip}"
    try:
        result = _eval_incr_if_under(r, key, 5, 600)
        return bool(result)
    except Exception:
        return True


# ── Data migration: guest → real user ────────────────────────────────────────

def migrate_guest_to_user(guest_id: str, real_user_id: str) -> Dict[str, Any]:
    """Migrate all guest data to a real user_id after successful registration/OAuth.

    Runs in this exact order:
      1. Qdrant: overwrite user_id payload on all guest vectors (no re-embedding)
      2. Filesystem: rename data/users/{guest_id}/ → data/users/{real_user_id}/
         (merge knowledge_base/ if target already exists)
      3. Redis: delete guest counters
    Returns migration stats dict for audit logging.
    """
    stats: Dict[str, Any] = {"qdrant_chunks": 0, "files_moved": 0, "errors": []}

    # Step 1: Qdrant payload migration
    try:
        from app.core.infra_registry import infra
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        vs = infra.get_vector_store()
        if vs and vs.client:
            for collection in (vs.text_collection, vs.vision_collection):
                offset = None
                while True:
                    results, next_offset = vs.client.scroll(
                        collection_name=collection,
                        scroll_filter=Filter(must=[
                            FieldCondition(key="user_id", match=MatchValue(value=guest_id))
                        ]),
                        limit=100,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    if not results:
                        break
                    point_ids = [r.id for r in results]
                    vs.client.overwrite_payload(
                        collection_name=collection,
                        payload={"user_id": real_user_id},
                        points=point_ids,
                    )
                    stats["qdrant_chunks"] += len(point_ids)
                    if next_offset is None:
                        break
                    offset = next_offset
    except Exception as exc:
        logger.warning(event="guest_migrate_qdrant_failed", error=str(exc))
        stats["errors"].append(f"qdrant: {exc}")

    # Step 2: Filesystem rename / merge
    try:
        guest_dir = DATA_ROOT / guest_id
        real_dir  = DATA_ROOT / real_user_id
        if guest_dir.exists():
            if real_dir.exists():
                # Real user dir already exists — merge knowledge_base files only
                guest_kb = guest_dir / "knowledge_base"
                real_kb  = real_dir  / "knowledge_base"
                real_kb.mkdir(exist_ok=True)
                if guest_kb.exists():
                    for f in guest_kb.iterdir():
                        if f.is_file():
                            dest = real_kb / f.name
                            if not dest.exists():
                                shutil.copy2(f, dest)
                                stats["files_moved"] += 1
                shutil.rmtree(str(guest_dir), ignore_errors=True)
            else:
                guest_dir.rename(real_dir)
                stats["files_moved"] = sum(1 for _ in real_dir.rglob("*") if _.is_file())
    except Exception as exc:
        logger.warning(event="guest_migrate_filesystem_failed", error=str(exc))
        stats["errors"].append(f"filesystem: {exc}")

    # Step 3: Clean up Redis guest counters
    try:
        r = _redis()
        if r:
            r.delete(
                f"guest:{guest_id}:queries",
                f"guest:{guest_id}:uploads",
            )
    except Exception:
        pass

    logger.info(
        event="guest_migrated",
        guest_id=guest_id,
        real_user_id=real_user_id,
        qdrant_chunks=stats["qdrant_chunks"],
        files_moved=stats["files_moved"],
        errors=stats["errors"],
    )
    return stats


def cleanup_expired_guest_dirs() -> int:
    """Delete data/users/guest_* directories older than GUEST_SESSION_TTL_HOURS + 1h.
    Called from the startup cleanup sweep in main.py.
    Returns count of directories removed.
    """
    import time
    cutoff_age = (settings.GUEST_SESSION_TTL_HOURS + 1) * 3600
    removed = 0
    try:
        if not DATA_ROOT.exists():
            return 0
        for guest_dir in DATA_ROOT.glob("guest_*"):
            if not guest_dir.is_dir():
                continue
            age = time.time() - guest_dir.stat().st_mtime
            if age > cutoff_age:
                shutil.rmtree(str(guest_dir), ignore_errors=True)
                removed += 1
                logger.info(event="guest_dir_expired_cleaned", path=str(guest_dir))
    except Exception as exc:
        logger.warning(event="guest_dir_cleanup_failed", error=str(exc))
    return removed
