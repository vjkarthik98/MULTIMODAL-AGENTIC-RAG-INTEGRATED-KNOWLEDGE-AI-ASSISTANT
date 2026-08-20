"""Unit tests for RedisMemory.purge_user() and RedisMemory.delete()'s
tenant-isolation guard.

Regression coverage for a real production bug (2026-08-08): purge_user()
called self.client.scan_iter(...) unconditionally, but upstash_redis.Redis
(the REST client used whenever REDIS_URL+REDIS_TOKEN are set) has no
scan_iter() — only real redis.Redis does. Every other method in
redis_memory.py already branches on self._use_upstash; this was the one
place that didn't, so GDPR/session purge silently failed
(redis_purge_scan_failed) on every Upstash-backed deployment.

_FakeUpstashClient below deliberately does NOT define scan_iter, mirroring
the real upstash_redis.Redis API surface — a plain MagicMock() would not
have reproduced the bug, since MagicMock auto-creates any attribute access
instead of raising AttributeError like the real client does.
"""

from __future__ import annotations

from unittest.mock import patch

from app.memory.redis_memory import RedisMemory


class _FakeUpstashClient:
    """Mimics upstash_redis.Redis's real API surface: has .keys()/.delete(),
    deliberately has no .scan_iter()."""

    def __init__(self, existing_keys: list[str]):
        self._keys = list(existing_keys)
        self.keys_calls: list[str] = []
        self.deleted: list[str] = []

    def keys(self, pattern: str) -> list[str]:
        self.keys_calls.append(pattern)
        prefix = pattern.rstrip("*")
        return [k for k in self._keys if k.startswith(prefix)]

    def delete(self, key: str) -> int:
        self.deleted.append(key)
        if key in self._keys:
            self._keys.remove(key)
            return 1
        return 0


class _FakeRealRedisClient:
    """Mimics real redis.Redis's API surface: has .scan_iter()/.delete()."""

    def __init__(self, existing_keys: list[str]):
        self._keys = list(existing_keys)
        self.scan_iter_calls: list[str] = []
        self.deleted: list[str] = []

    def scan_iter(self, pattern: str):
        self.scan_iter_calls.append(pattern)
        prefix = pattern.rstrip("*")
        return iter([k for k in self._keys if k.startswith(prefix)])

    def delete(self, key: str) -> int:
        self.deleted.append(key)
        if key in self._keys:
            self._keys.remove(key)
            return 1
        return 0


def _make_redis_memory() -> RedisMemory:
    """Construct RedisMemory without a real network connection attempt."""
    with patch.object(RedisMemory, "_connect", lambda self: None):
        rm = RedisMemory()
    rm._enabled = True
    rm._redis_ok = True
    return rm


class TestPurgeUserUpstash:

    def test_purge_user_uses_keys_not_scan_iter(self):
        rm = _make_redis_memory()
        rm._use_upstash = True
        fake_client = _FakeUpstashClient(
            existing_keys=[
                f"{rm.prefix}:user1:sess1:history",
                f"{rm.prefix}:user1:sess2:history",
                f"{rm.prefix}:user2:sess1:history",  # different user — must survive
            ]
        )
        rm.client = fake_client

        count = rm.purge_user("user1")

        assert count == 2
        assert len(fake_client.keys_calls) == 1
        assert sorted(fake_client.deleted) == [
            f"{rm.prefix}:user1:sess1:history",
            f"{rm.prefix}:user1:sess2:history",
        ]
        # the other user's key must still be present
        assert f"{rm.prefix}:user2:sess1:history" in fake_client._keys

    def test_purge_user_on_upstash_does_not_touch_scan_iter(self):
        """A fake client with no scan_iter attribute must not raise —
        proves purge_user() takes the .keys() branch on Upstash, not the
        scan_iter() branch that crashed in production."""
        rm = _make_redis_memory()
        rm._use_upstash = True
        rm.client = _FakeUpstashClient(existing_keys=[])

        count = rm.purge_user("user1")  # must not raise AttributeError

        assert count == 0


class TestPurgeUserRealRedis:

    def test_purge_user_uses_scan_iter_on_real_redis(self):
        rm = _make_redis_memory()
        rm._use_upstash = False
        fake_client = _FakeRealRedisClient(
            existing_keys=[
                f"{rm.prefix}:user1:sess1:history",
                f"{rm.prefix}:user1:sess2:history",
            ]
        )
        rm.client = fake_client

        count = rm.purge_user("user1")

        assert count == 2
        assert len(fake_client.scan_iter_calls) == 1


class TestPurgeUserEdgeCases:

    def test_purge_user_empty_user_id_returns_zero(self):
        rm = _make_redis_memory()
        assert rm.purge_user("") == 0

    def test_purge_user_disabled_returns_zero(self):
        rm = _make_redis_memory()
        rm._enabled = False
        assert rm.purge_user("user1") == 0


# ---------------------------------------------------------------------------
# delete() tenant-isolation guard — regression for the /memory/clear bug
# ---------------------------------------------------------------------------


class TestDeleteTenantIsolationGuard:

    def test_delete_with_user_id_clears_the_key(self):
        rm = _make_redis_memory()
        rm._use_upstash = True
        key = f"{rm.prefix}:user1:sess1:history"
        fake_client = _FakeUpstashClient(existing_keys=[key])
        rm.client = fake_client

        rm.delete("sess1", "user1")

        assert key in fake_client.deleted

    def test_delete_without_user_id_does_not_touch_client(self):
        """The exact bug shape: calling delete(session_id) with no user_id
        must not attempt any client call — it should hit the tenant-
        isolation guard and no-op, not raise and not silently scope to the
        wrong key."""
        rm = _make_redis_memory()
        rm._use_upstash = True
        fake_client = _FakeUpstashClient(existing_keys=[])
        rm.client = fake_client

        rm.delete("sess1")  # user_id omitted — must not raise

        assert fake_client.deleted == []
