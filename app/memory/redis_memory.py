from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pybreaker

    _redis_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

    class _DummyBreaker:
        def __call__(self, fn):
            return fn

    _redis_breaker = _DummyBreaker()  # type: ignore[assignment]


# VALID ROLES
_VALID_ROLES = {"user", "assistant", "system"}

# VALID MODALITIES
_VALID_MODALITIES = {"text", "image", "audio", "video", "table", "document"}


# IN-MEMORY LRU FALLBACK STORE


class _InMemoryStore:

    def __init__(self, maxsize: int = 1000) -> None:
        self._store: OrderedDict = OrderedDict()
        self._maxsize: int = maxsize

    def _evict(self) -> None:
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def rpush(self, key: str, value: str) -> None:
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(value)
        self._store.move_to_end(key)
        self._evict()

    def ltrim(self, key: str, start: int, end: int) -> None:
        if key not in self._store:
            return
        lst = self._store[key]
        if end == -1:
            self._store[key] = lst[start:]
        else:
            self._store[key] = lst[start : end + 1]

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        lst = self._store.get(key, [])
        if not isinstance(lst, list):
            return []
        if end == -1:
            return lst[start:]
        return lst[start : end + 1]

    def llen(self, key: str) -> int:
        lst = self._store.get(key, [])
        return len(lst) if isinstance(lst, list) else 0

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        self._evict()

    def get(self, key: str) -> str | None:
        val = self._store.get(key)
        if isinstance(val, str):
            self._store.move_to_end(key)
            return val
        return None

    def keys_with_prefix(self, prefix: str) -> list[str]:
        return [k for k in self._store.keys() if str(k).startswith(prefix)]

    def flush(self) -> None:
        self._store.clear()


class RedisMemory:

    def __init__(self) -> None:
        self._enabled: bool = settings.USE_REDIS
        self.client = None
        self._use_upstash: bool = False
        self._redis_ok: bool = False
        self._fallback = _InMemoryStore(maxsize=settings.LRU_CACHE_MAXSIZE)

        self.max_messages: int = settings.MAX_HISTORY_MESSAGES
        self.ttl: int = settings.REDIS_TTL_SECONDS
        self.query_cache_ttl: int = settings.REDIS_QUERY_CACHE_TTL
        self.embed_cache_ttl: int = settings.REDIS_EMBEDDING_CACHE_TTL
        self.prefix: str = settings.REDIS_KEY_PREFIX

        if not self._enabled:
            logger.info(event="redis_disabled_memory_using_no_op_fallback")
            return

        self._connect()

    # CONNECTION

    def _connect(self) -> None:
        # UPSTASH REST CLIENT — CLOUD PRIORITY
        if settings.REDIS_URL and settings.REDIS_TOKEN:
            try:
                from upstash_redis import Redis as UpstashRedis

                self.client = UpstashRedis(
                    url=settings.REDIS_URL,
                    token=settings.REDIS_TOKEN,
                )
                self._use_upstash = True
                self._redis_ok = True
                logger.info(
                    event="redis_upstash_connected",
                    url=settings.REDIS_URL[:40],
                )
                return
            except ImportError:
                logger.warning(
                    event="upstash_redis_not_installed",
                    hint="pip install upstash-redis",
                )
            except Exception as exc:
                logger.warning(event="upstash_connect_failed", error=str(exc))

        # STANDARD REDIS — LOCAL OR REMOTE
        try:
            import redis as redis_lib

            pool = redis_lib.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=settings.REDIS_TIMEOUT,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                retry_on_timeout=True,
            )
            self.client = redis_lib.Redis(connection_pool=pool)
            self.client.ping()
            self._redis_ok = True

            logger.info(
                event="redis_connected",
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
            )

        except Exception as exc:
            self.client = None
            self._redis_ok = False
            logger.warning(
                event="redis_unavailable_using_fallback",
                error=str(exc),
                hint="Set REDIS_URL + REDIS_TOKEN for Upstash, or ensure local Redis is running",
            )

    # AVAILABILITY

    def _is_available(self) -> bool:
        return self._redis_ok and self.client is not None

    # TENACITY RETRY WRAPPER

    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            min=settings.RETRY_WAIT_MIN_SEC,
            max=settings.RETRY_WAIT_MAX_SEC,
        ),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _retry(self, fn):
        def _do():
            return fn()

        if _PYBREAKER_AVAILABLE:
            return _redis_breaker(_do)()
        return _do()

    # HELPERS

    def _clean(self, text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFC", str(text or ""))
        return " ".join(text.strip().split())

    def _role(self, role: str) -> str:
        role = str(role or "user").lower().strip()
        return role if role in _VALID_ROLES else "user"

    def _modality(self, modality: str) -> str:
        m = str(modality or "text").lower().strip()
        return m if m in _VALID_MODALITIES else "text"

    def _key(self, session_id: str, user_id: str | None = None) -> str:
        if not user_id:
            logger.error(
                event="redis_key_missing_user_id",
                msg="user_id is required — refusing an unscoped key",
                session_id=session_id,
            )
            raise ValueError(
                "TENANT_ISOLATION_VIOLATION: user_id is required to build a Redis memory key"
            )
        return f"{self.prefix}:{user_id}:{session_id}:history"

    def _hash_msg(self, msg: dict) -> str:
        base = f"{msg.get('role')}|{str(msg.get('content', ''))[:200]}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _valid_embedding(self, emb: Any) -> bool:
        return isinstance(emb, list) and len(emb) in (
            settings.TEXT_EMBEDDING_DIM,
            settings.VISION_EMBEDDING_DIM,
        )

    # PIPELINE WRITE — RPUSH + LTRIM + EXPIRE
    # Upstash HTTP client has no pipeline(); issue commands individually.

    def _pipeline_write(self, key: str, payload: str) -> None:
        def _do():
            if self._use_upstash:
                self.client.rpush(key, payload)
                self.client.ltrim(key, -self.max_messages, -1)
                if self.ttl:
                    self.client.expire(key, self.ttl)
            else:
                pipe = self.client.pipeline()
                pipe.rpush(key, payload)
                pipe.ltrim(key, -self.max_messages, -1)
                if self.ttl:
                    pipe.expire(key, self.ttl)
                pipe.execute()

        self._retry(_do)

    # ADD MESSAGE

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: list[float] | None = None,
        modality: str = "text",
        importance: float = 1.0,
        extra: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> None:
        if not self._enabled:
            return
        if not session_id or not content:
            return

        content = self._clean(content)
        if len(content) < 2:
            return

        content = content[: settings.MAX_PROMPT_CHARS]
        try:
            key = self._key(session_id, user_id)
        except ValueError:
            return

        message: dict[str, Any] = {
            "role": self._role(role),
            "content": content,
            "timestamp": time.time(),
            "modality": self._modality(modality),
            "importance": max(0.0, min(float(importance), 1.0)),
        }

        if self._valid_embedding(embedding):
            message["embedding"] = embedding

        if isinstance(extra, dict) and extra:
            message["extra"] = extra

        payload = json.dumps(message, ensure_ascii=False)

        if self._is_available():
            try:
                self._pipeline_write(key, payload)
                return
            except Exception as exc:
                logger.warning(
                    event="redis_add_failed_using_fallback",
                    error=str(exc),
                    session_id=session_id,
                )
                self._redis_ok = False

        # FALLBACK
        self._fallback.rpush(key, payload)
        self._fallback.ltrim(key, -self.max_messages, -1)

    # APPEND ALIAS — USED BY MEMORY MANAGER

    def append(self, session_id: str, message: dict[str, Any], user_id: str | None = None) -> None:
        if not self._enabled:
            return
        self.add_message(
            session_id=session_id,
            role=message.get("role", "user"),
            content=message.get("content", ""),
            modality=message.get("modality", "text"),
            importance=message.get("importance", 1.0),
            user_id=user_id or message.get("user_id"),
        )

    # GET HISTORY

    def get_history(self, session_id: str, user_id: str | None = None) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        try:
            key = self._key(session_id, user_id)
        except ValueError:
            return []

        if self._is_available():
            try:

                def _do():
                    return self.client.lrange(key, 0, -1)

                data = self._retry(_do)
                return self._parse_messages(data)
            except Exception as exc:
                logger.warning(
                    event="redis_fetch_failed_using_fallback",
                    error=str(exc),
                    session_id=session_id,
                )
                self._redis_ok = False

        data = self._fallback.lrange(key, 0, -1)
        return self._parse_messages(data)

    # GET ALIAS — USED BY MEMORY MANAGER

    def get(self, session_id: str, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_history(session_id, user_id)

    # GET LAST K MESSAGES

    def get_last_k(
        self,
        session_id: str,
        k: int | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        k = k or settings.MEMORY_TOP_K
        try:
            key = self._key(session_id, user_id)
        except ValueError:
            return []

        if self._is_available():
            try:

                def _do():
                    return self.client.lrange(key, -k, -1)

                data = self._retry(_do)
                return self._parse_messages(data)
            except Exception as exc:
                logger.warning(
                    event="redis_get_last_k_fallback",
                    error=str(exc),
                    session_id=session_id,
                )

        data = self._fallback.lrange(key, -k, -1)
        return self._parse_messages(data)

    # PARSE MESSAGES — DEDUP + VALIDATE

    def _parse_messages(self, data: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set = set()

        for item in data:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8", errors="replace")

                parsed = json.loads(item)

                if not isinstance(parsed, dict):
                    continue

                parsed["content"] = self._clean(parsed.get("content", ""))
                if not parsed["content"]:
                    continue

                h = self._hash_msg(parsed)
                if h in seen:
                    continue

                seen.add(h)
                out.append(parsed)

            except (json.JSONDecodeError, Exception):
                continue

        return out

    # MEMORY SIZE

    def get_memory_size(self, session_id: str, user_id: str | None = None) -> int:
        if not self._enabled:
            return 0
        try:
            key = self._key(session_id, user_id)
        except ValueError:
            return 0

        if self._is_available():
            try:

                def _do():
                    return self.client.llen(key)

                return int(self._retry(_do))
            except Exception:
                pass

        return self._fallback.llen(key)

    # CLEAR MEMORY

    def clear_memory(self, session_id: str, user_id: str | None = None) -> None:
        if not self._enabled:
            return
        try:
            key = self._key(session_id, user_id)
        except ValueError:
            return

        if self._is_available():
            try:

                def _do():
                    return self.client.delete(key)

                self._retry(_do)
            except Exception as exc:
                logger.warning(
                    event="redis_clear_failed",
                    error=str(exc),
                    session_id=session_id,
                )

        self._fallback.delete(key)

        logger.debug(
            event="redis_memory_cleared",
            session_id=session_id,
        )

    # DELETE ALIAS — USED BY MEMORY MANAGER

    def delete(self, session_id: str, user_id: str | None = None) -> None:
        self.clear_memory(session_id, user_id)

    # GDPR PURGE — ALL DATA FOR USER_ID PREFIX

    def purge_user(self, user_id: str) -> None:
        if not self._enabled or not user_id:
            return

        prefix = f"{self.prefix}:{user_id}:"
        purged = 0

        if self._is_available():
            try:

                def _scan():
                    return list(self.client.scan_iter(f"{prefix}*"))

                keys = self._retry(_scan)
                for key in keys:
                    try:

                        def _del(k=key):
                            return self.client.delete(k)

                        self._retry(_del)
                        purged += 1
                    except Exception as exc:
                        logger.warning(
                            event="redis_purge_key_failed",
                            key=str(key),
                            error=str(exc),
                        )
            except Exception as exc:
                logger.warning(
                    event="redis_purge_scan_failed",
                    user_id=user_id,
                    error=str(exc),
                )

        # FALLBACK PURGE
        fallback_keys = self._fallback.keys_with_prefix(prefix)
        for key in fallback_keys:
            self._fallback.delete(key)
            purged += 1

        logger.info(
            event="redis_user_purged",
            user_id=user_id,
            keys_deleted=purged,
        )

    # QUERY RESULT CACHE

    def cache_set(
        self,
        cache_key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        ttl = ttl or self.query_cache_ttl

        try:
            payload = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning(event="cache_serialize_failed", error=str(exc))
            return

        if self._is_available():
            try:

                def _do():
                    return self.client.setex(cache_key, ttl, payload)

                self._retry(_do)
                return
            except Exception as exc:
                logger.warning(event="redis_cache_set_failed", error=str(exc))

        self._fallback.setex(cache_key, ttl, payload)

    def cache_get(self, cache_key: str) -> Any | None:
        if not self._enabled:
            return None
        if self._is_available():
            try:

                def _do():
                    return self.client.get(cache_key)

                raw = self._retry(_do)
                if raw:
                    return json.loads(raw)
                return None
            except Exception as exc:
                logger.warning(event="redis_cache_get_failed", error=str(exc))

        raw = self._fallback.get(cache_key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None

    def cache_delete(self, cache_key: str) -> None:
        if not self._enabled:
            return
        if self._is_available():
            try:

                def _do():
                    return self.client.delete(cache_key)

                self._retry(_do)
            except Exception as exc:
                logger.warning(event="redis_cache_delete_failed", error=str(exc))
        self._fallback.delete(cache_key)

    def cache_flush_query_cache(self) -> int:
        """Delete all query-response cache entries (keys prefixed 'qresp:').
        Uses SCAN to avoid blocking. Returns number of keys deleted."""
        deleted = 0
        if self._is_available():
            try:
                cursor = 0
                while True:
                    result = self.client.scan(cursor, match="qresp:*", count=100)
                    # Upstash HTTP client returns [cursor, [keys]]
                    if isinstance(result, (list, tuple)) and len(result) == 2:
                        cursor, keys = result
                    else:
                        break
                    if keys:
                        for k in keys:
                            self.client.delete(k)
                        deleted += len(keys)
                    if int(cursor) == 0:
                        break
                logger.info(event="query_cache_flushed", deleted=deleted)
                return deleted
            except Exception as exc:
                logger.warning(event="query_cache_flush_failed", error=str(exc))

        # Fallback in-memory: delete keys starting with qresp:
        fallback_keys = [k for k in list(self._fallback._store.keys()) if k.startswith("qresp:")]
        for k in fallback_keys:
            self._fallback.delete(k)
        deleted += len(fallback_keys)
        return deleted

    # EMBEDDING CACHE

    def embedding_cache_set(
        self,
        text_hash: str,
        embedding: list[float],
    ) -> None:
        if not self._enabled:
            return
        if not self._valid_embedding(embedding):
            return
        key = f"{self.prefix}:emb:{text_hash}"
        self.cache_set(key, embedding, ttl=self.embed_cache_ttl)

    def embedding_cache_get(self, text_hash: str) -> list[float] | None:
        if not self._enabled:
            return None
        key = f"{self.prefix}:emb:{text_hash}"
        result = self.cache_get(key)
        if isinstance(result, list) and self._valid_embedding(result):
            return result
        return None

    # ASYNC WRAPPERS

    async def async_add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        modality: str = "text",
        importance: float = 1.0,
        user_id: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.add_message,
            session_id,
            role,
            content,
            None,
            modality,
            importance,
            None,
            user_id,
        )

    async def async_get_history(
        self, session_id: str, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.get_history, session_id, user_id)

    async def async_clear_memory(self, session_id: str, user_id: str | None = None) -> None:
        await asyncio.to_thread(self.clear_memory, session_id, user_id)

    async def async_purge_user(self, user_id: str) -> None:
        await asyncio.to_thread(self.purge_user, user_id)

    # HEALTH CHECK

    def health_check(self) -> dict[str, Any]:
        return {
            "redis_ok": self._redis_ok,
            "use_upstash": self._use_upstash,
            "client_type": (
                "upstash" if self._use_upstash else ("redis" if self._redis_ok else "fallback")
            ),
            "fallback_active": not self._redis_ok,
            "fallback_size": len(self._fallback._store),
            "circuit_breaker": _PYBREAKER_AVAILABLE,
            "max_messages": self.max_messages,
            "ttl_seconds": self.ttl,
        }
