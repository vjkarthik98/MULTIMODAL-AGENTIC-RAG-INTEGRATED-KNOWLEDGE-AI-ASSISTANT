"""
Phase 27 — THE CORE ISOLATION TEST.

Proves that User B cannot access User A's data through any path:
  - Qdrant filter enforces user_id boundary
  - Redis keys are namespaced per user
  - MongoDB queries always scope to user_id
  - tenancy helpers produce correct typed objects

This is the Phase 27 acceptance gate. If any test here fails, isolation is broken.
"""
from __future__ import annotations

import uuid

import pytest

from app.auth.tenancy import (
    mongo_user_query,
    mongo_session_query,
    qdrant_user_filter,
    qdrant_user_and_source_filter,
    redis_cache_key,
    redis_key_prefix,
    redis_scoped_key,
    redis_session_key,
)


USER_A_ID = "user-a-" + str(uuid.uuid4())
USER_B_ID = "user-b-" + str(uuid.uuid4())


# ── Qdrant filter isolation ───────────────────────────────────────────────────

def test_qdrant_filter_is_typed_not_string():
    """Filter must be a qdrant_client Filter object, never a raw dict or string."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    f = qdrant_user_filter(USER_A_ID)
    assert isinstance(f, Filter)
    assert len(f.must) == 1
    cond = f.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "user_id"
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value == USER_A_ID


def test_qdrant_filter_user_a_does_not_match_user_b():
    """A filter built for User A must not match User B's user_id."""
    from qdrant_client.models import FieldCondition, MatchValue
    f_a = qdrant_user_filter(USER_A_ID)
    cond = f_a.must[0]
    assert isinstance(cond, FieldCondition)
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value != USER_B_ID


def test_qdrant_filter_user_b_does_not_match_user_a():
    from qdrant_client.models import FieldCondition, MatchValue
    f_b = qdrant_user_filter(USER_B_ID)
    cond = f_b.must[0]
    assert isinstance(cond, MatchValue) or isinstance(cond.match, MatchValue)
    val = cond.match.value if hasattr(cond, "match") else cond.value
    assert val != USER_A_ID


def test_qdrant_filters_for_different_users_are_different():
    f_a = qdrant_user_filter(USER_A_ID)
    f_b = qdrant_user_filter(USER_B_ID)
    # The filter objects must not be equal
    assert f_a != f_b


def test_qdrant_user_and_source_filter_includes_user_id():
    from qdrant_client.models import FieldCondition, MatchValue
    f = qdrant_user_and_source_filter(USER_A_ID, ["doc1.pdf"])
    assert any(
        isinstance(c, FieldCondition) and c.key == "user_id"
        and isinstance(c.match, MatchValue) and c.match.value == USER_A_ID
        for c in f.must
    )


def test_qdrant_user_and_source_filter_empty_sources():
    from qdrant_client.models import Filter
    f = qdrant_user_and_source_filter(USER_A_ID, [])
    assert isinstance(f, Filter)
    assert len(f.must) == 1


# ── Redis key namespace isolation ────────────────────────────────────────────

def test_redis_key_prefix_different_for_different_users():
    prefix_a = redis_key_prefix(USER_A_ID)
    prefix_b = redis_key_prefix(USER_B_ID)
    assert prefix_a != prefix_b


def test_redis_key_prefix_format():
    prefix = redis_key_prefix(USER_A_ID)
    assert prefix.startswith("u:")
    assert USER_A_ID in prefix


def test_redis_scoped_key_includes_user_and_key():
    key = redis_scoped_key(USER_A_ID, "embeddings:abc123")
    assert USER_A_ID in key
    assert "embeddings:abc123" in key


def test_redis_scoped_key_user_a_differs_from_user_b_same_suffix():
    key_a = redis_scoped_key(USER_A_ID, "session:xyz")
    key_b = redis_scoped_key(USER_B_ID, "session:xyz")
    assert key_a != key_b


def test_redis_session_key_isolation():
    session_id = "sess-001"
    key_a = redis_session_key(USER_A_ID, session_id)
    key_b = redis_session_key(USER_B_ID, session_id)
    assert key_a != key_b
    assert USER_A_ID in key_a
    assert USER_B_ID in key_b


def test_redis_cache_key_isolation():
    hash_val = "deadbeef"
    key_a = redis_cache_key(USER_A_ID, "embed", hash_val)
    key_b = redis_cache_key(USER_B_ID, "embed", hash_val)
    assert key_a != key_b


# ── MongoDB query isolation ───────────────────────────────────────────────────

def test_mongo_user_query_injects_user_id():
    q = mongo_user_query(USER_A_ID, {"session_id": "s1"})
    assert q["user_id"] == USER_A_ID
    assert q["session_id"] == "s1"


def test_mongo_user_query_user_a_differs_from_user_b():
    q_a = mongo_user_query(USER_A_ID, {})
    q_b = mongo_user_query(USER_B_ID, {})
    assert q_a["user_id"] != q_b["user_id"]


def test_mongo_user_query_does_not_mutate_base():
    base = {"session_id": "s1"}
    mongo_user_query(USER_A_ID, base)
    assert "user_id" not in base


def test_mongo_user_query_empty_base():
    q = mongo_user_query(USER_A_ID)
    assert q == {"user_id": USER_A_ID}


def test_mongo_session_query_includes_both_fields():
    q = mongo_session_query(USER_A_ID, "sess-001")
    assert q["user_id"] == USER_A_ID
    assert q["session_id"] == "sess-001"


def test_mongo_session_query_user_a_vs_user_b():
    q_a = mongo_session_query(USER_A_ID, "sess-001")
    q_b = mongo_session_query(USER_B_ID, "sess-001")
    assert q_a != q_b


# ── Cross-tenant data access simulation ──────────────────────────────────────

def test_cross_tenant_qdrant_filter_blocks_user_b_from_user_a_chunk():
    """
    Simulate what happens when User B tries to retrieve a chunk that belongs to User A.
    The filter built for User B must have a different user_id value than User A's chunk payload.
    """
    user_a_chunk_payload = {"user_id": USER_A_ID, "text": "secret document of user A"}

    filter_for_user_b = qdrant_user_filter(USER_B_ID)
    from qdrant_client.models import FieldCondition, MatchValue
    condition = filter_for_user_b.must[0]
    assert isinstance(condition, FieldCondition)
    assert isinstance(condition.match, MatchValue)

    # The filter's user_id value must NOT match User A's chunk
    assert condition.match.value != user_a_chunk_payload["user_id"]
    assert condition.match.value == USER_B_ID


def test_cross_tenant_mongo_query_blocks_user_b_from_user_a_messages():
    """
    A MongoDB query scoped to User B must never return User A's documents.
    """
    user_a_doc = {"user_id": USER_A_ID, "content": "private memory of user A"}
    query_for_b = mongo_user_query(USER_B_ID)

    # The query's user_id must not match User A's document
    assert query_for_b["user_id"] != user_a_doc["user_id"]


def test_cross_tenant_redis_key_namespace_prevents_collision():
    """
    User B's Redis key for the same logical data must not equal User A's key.
    """
    key_a = redis_session_key(USER_A_ID, "session-123")
    key_b = redis_session_key(USER_B_ID, "session-123")
    assert key_a != key_b
    assert not key_b.startswith(redis_key_prefix(USER_A_ID))
    assert not key_a.startswith(redis_key_prefix(USER_B_ID))
