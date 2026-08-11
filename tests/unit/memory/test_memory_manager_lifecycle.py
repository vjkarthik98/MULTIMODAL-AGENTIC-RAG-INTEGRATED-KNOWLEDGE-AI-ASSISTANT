"""Unit tests for MemoryManager.clear() and MemoryManager.gdpr_purge().

Regression coverage for two real production bugs found on 2026-08-08:

1. POST /memory/clear silently no-op'd on both Redis and Mongo — clear()
   never accepted a user_id param at all, so RedisMemory.delete()/
   MongoMemory.delete()'s tenant-isolation guards ("refusing an unscoped
   delete") always fired. The endpoint still returned 200 "cleared".

2. gdpr_purge() called self.redis_memory.delete(user_id) — but
   RedisMemory.delete(session_id, user_id=None) takes session_id first, so
   this passed user_id as the *session_id* positional arg with no real
   user_id, always hitting the same unscoped-delete guard. It was masked by
   a redundant manual .keys()/.delete() block that happened to work — this
   should now call purge_user() directly instead.

MemoryManager's constructor accepts redis_memory/mongo_memory directly
(dependency injection), so — contrary to test_memory_manager.py's docstring
disclaimer for the *class itself* — these two methods ARE unit-testable
without live Redis/Mongo.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.memory.memory_manager import MemoryManager

# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestMemoryManagerClear:

    def test_clear_passes_user_id_to_redis_delete(self):
        mock_redis = MagicMock()
        mock_mongo = MagicMock()
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=mock_mongo)

        manager.clear("session1", "user1")

        mock_redis.delete.assert_called_once_with("session1", "user1")

    def test_clear_passes_user_id_to_mongo_delete(self):
        mock_redis = MagicMock()
        mock_mongo = MagicMock()
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=mock_mongo)

        manager.clear("session1", "user1")

        mock_mongo.delete.assert_called_once_with("session1", "user1")

    def test_clear_without_user_id_still_forwards_none_not_omits(self):
        """Callers that genuinely have no user_id (unlikely, but must not
        crash) still forward user_id=None explicitly rather than dropping
        the argument — the underlying stores' own guards handle refusal."""
        mock_redis = MagicMock()
        mock_mongo = MagicMock()
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=mock_mongo)

        manager.clear("session1")

        mock_redis.delete.assert_called_once_with("session1", None)
        mock_mongo.delete.assert_called_once_with("session1", None)

    def test_clear_does_not_raise_when_store_delete_raises(self):
        mock_redis = MagicMock()
        mock_redis.delete.side_effect = RuntimeError("boom")
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=None)

        manager.clear("session1", "user1")  # must not raise


# ---------------------------------------------------------------------------
# gdpr_purge()
# ---------------------------------------------------------------------------


class TestMemoryManagerGdprPurge:

    def test_gdpr_purge_calls_redis_purge_user_not_delete(self):
        mock_redis = MagicMock()
        mock_redis.purge_user.return_value = 5
        mock_mongo = MagicMock()
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=mock_mongo)

        result = manager.gdpr_purge("user1")

        mock_redis.purge_user.assert_called_once_with("user1")
        mock_redis.delete.assert_not_called()
        assert result["redis"] is True
        assert result["redis_keys_deleted"] == 5

    def test_gdpr_purge_calls_mongo_purge_user_with_user_id(self):
        mock_redis = MagicMock()
        mock_mongo = MagicMock()
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=mock_mongo)

        manager.gdpr_purge("user1")

        mock_mongo.purge_user.assert_called_once_with("user1")

    def test_gdpr_purge_reports_redis_error_without_raising(self):
        mock_redis = MagicMock()
        mock_redis.purge_user.side_effect = RuntimeError("upstash down")
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=None)

        result = manager.gdpr_purge("user1")

        assert result["redis"] is False
        assert any("upstash down" in e for e in result["errors"])

    def test_gdpr_purge_result_has_user_id(self):
        mock_redis = MagicMock()
        mock_redis.purge_user.return_value = 0
        manager = MemoryManager(redis_memory=mock_redis, mongo_memory=None)

        result = manager.gdpr_purge("user1")

        assert result["user_id"] == "user1"
