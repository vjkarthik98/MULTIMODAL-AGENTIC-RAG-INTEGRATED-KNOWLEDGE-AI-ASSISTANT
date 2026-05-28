"""
Tenant isolation helpers — single source of truth for per-user scoping.

Every retrieval and memory call site imports from here.
No string-building is scattered across files — all filters are typed
Pydantic/Qdrant objects to prevent injection and typos.
"""
from __future__ import annotations

from typing import Any, Dict


# ── Qdrant ────────────────────────────────────────────────────────────────────

def qdrant_user_filter(user_id: str):
    """
    Return a typed Qdrant Filter that restricts results to chunks
    belonging to the given user.  Never string-built — always uses
    the typed FieldCondition + MatchValue objects.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    return Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            )
        ]
    )


def qdrant_user_and_source_filter(user_id: str, source_substrings: list[str]):
    """Filter by user_id AND one of the given source substrings (for /query?sources=)."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue, MatchText
    must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if source_substrings:
        should = [
            FieldCondition(key="source", match=MatchText(text=s))
            for s in source_substrings
        ]
        return Filter(must=must, should=should)
    return Filter(must=must)


# ── Redis ─────────────────────────────────────────────────────────────────────

def redis_key_prefix(user_id: str) -> str:
    """All Redis keys for a user are namespaced under u:{user_id}:"""
    return f"u:{user_id}:"


def redis_scoped_key(user_id: str, key: str) -> str:
    """Build a fully-scoped Redis key: u:{user_id}:{key}"""
    return f"u:{user_id}:{key}"


def redis_session_key(user_id: str, session_id: str) -> str:
    """Key for a user's conversation session turns."""
    return f"u:{user_id}:sess:{session_id}"


def redis_cache_key(user_id: str, cache_type: str, cache_hash: str) -> str:
    """Key for a user-scoped cache entry (embeddings, query responses)."""
    return f"u:{user_id}:cache:{cache_type}:{cache_hash}"


# ── MongoDB ───────────────────────────────────────────────────────────────────

def mongo_user_query(user_id: str, base_query: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Merge user_id into a MongoDB query dict.
    Always call this before any find/find_one/update/delete.
    """
    q = dict(base_query) if base_query else {}
    q["user_id"] = user_id
    return q


def mongo_session_query(user_id: str, session_id: str) -> Dict[str, Any]:
    """Scoped query for a specific user + session."""
    return {"user_id": user_id, "session_id": session_id}
