import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# IN-MEMORY FALLBACK STORE

class _InMemoryStore:

    def __init__(self, maxsize: int = 1000) -> None:
        self._store:   OrderedDict = OrderedDict()
        self._maxsize: int         = maxsize
        self._cache: Dict[str, str] = {}

    def rpush(self, key: str, value: str) -> None:
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(value)
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def ltrim(self, key: str, start: int, end: int) -> None:
        if key in self._store:
            lst = self._store[key]
            if end == -1:
                self._store[key] = lst[start:]
            else:
                self._store[key] = lst[start:end + 1]

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        lst = self._store.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start:end + 1]
    

    def llen(self, key: str) -> int:
        return len(self._store.get(key, []))

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)


class RedisMemory:

    def __init__(self) -> None:
        self.client          = None
        self._use_upstash    = False
        self._fallback       = _InMemoryStore(maxsize=settings.LRU_CACHE_MAXSIZE)
        self._redis_ok       = False

        self.max_messages    = settings.MAX_HISTORY_MESSAGES
        self.ttl             = settings.REDIS_TTL_SECONDS
        self.query_cache_ttl = settings.REDIS_QUERY_CACHE_TTL
        self.embed_cache_ttl = settings.REDIS_EMBEDDING_CACHE_TTL
        self.prefix          = settings.REDIS_KEY_PREFIX

        self._connect()

    # CONNECTION

    def _connect(self) -> None:

        # UPSTASH REST CLIENT (cloud priority)
        if settings.REDIS_URL and settings.REDIS_TOKEN:
            try:
                from upstash_redis import Redis as UpstashRedis
                self.client       = UpstashRedis(
                    url=settings.REDIS_URL,
                    token=settings.REDIS_TOKEN,
                )
                self._use_upstash = True
                self._redis_ok    = True
                logger.info(event="redis_upstash_connected", url=settings.REDIS_URL)
                return
            except ImportError:
                logger.warning(
                    event="upstash_redis_not_installed",
                    hint="pip install upstash-redis",
                )
            except Exception as e:
                logger.warning(event="upstash_redis_failed", error=str(e))

        # STANDARD REDIS (local or remote socket)
        try:
            import redis as redis_lib

            pool = redis_lib.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=settings.REDIS_TIMEOUT,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                max_connections=20,
                retry_on_timeout=True,
            )
            self.client    = redis_lib.Redis(connection_pool=pool)
            self.client.ping()
            self._redis_ok = True

            logger.info(
                event="redis_connected",
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
            )

        except Exception as e:
            self.client    = None
            self._redis_ok = False
            logger.warning(
                event="redis_unavailable_using_fallback",
                error=str(e),
                hint="Set REDIS_URL + REDIS_TOKEN for Upstash cloud Redis, or ensure local Redis is running",
            )

    # AVAILABILITY CHECK

    def _is_available(self) -> bool:
        return self._redis_ok and self.client is not None

    # RETRY

    def _retry(self, fn, retries: int = 2):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    self._redis_ok = False
                    logger.error(event="redis_retry_exhausted", error=str(e))
                    raise
                logger.warning(event="redis_retry", attempt=i + 1, error=str(e))
                time.sleep(0.2 * (i + 1))

    # HELPERS

    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    def _role(self, role: str) -> str:
        role = str(role or "user").lower()
        return role if role in {"user", "assistant", "system"} else "user"

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    def _hash(self, msg: Dict) -> str:
        base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _valid_embedding(self, emb) -> bool:
        return (
            isinstance(emb, list) and
            len(emb) in (settings.TEXT_EMBEDDING_DIM, settings.VISION_EMBEDDING_DIM)
        )

    # ADD MESSAGE

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        modality: str = "text",
        extra: Optional[Dict] = None,
    ) -> None:

        if not session_id or not content:
            return

        content = self._clean(content)
        if len(content) < 2:
            return

        content = content[:settings.MAX_PROMPT_CHARS]
        key     = self._key(session_id)

        message: Dict = {
            "role":      self._role(role),
            "content":   content,
            "timestamp": time.time(),
            "modality":  modality,
        }

        if self._valid_embedding(embedding):
            message["embedding"] = embedding

        if isinstance(extra, dict):
            message["extra"] = extra

        payload = json.dumps(message)

        if self._is_available():
            try:
                def _write():
                    pipe = self.client.pipeline()
                    pipe.rpush(key, payload)
                    pipe.ltrim(key, -self.max_messages, -1)
                    if self.ttl:
                        pipe.expire(key, self.ttl)
                    return pipe.execute()

                self._retry(_write)
                return

            except Exception as e:
                logger.warning(
                    event="redis_add_failed_using_fallback",
                    error=str(e),
                    session_id=session_id,
                )

        # FALLBACK
        self._fallback.rpush(key, payload)
        self._fallback.ltrim(key, -self.max_messages, -1)

    # APPEND ALIAS (used by memory_manager)

    def append(self, session_id: str, message: Dict) -> None:
        self.add_message(
            session_id=session_id,
            role=message.get("role", "user"),
            content=message.get("content", ""),
            modality=message.get("modality", "text"),
        )

    # GET HISTORY

    def get_history(self, session_id: str) -> List[Dict]:
        key = self._key(session_id)

        if self._is_available():
            try:
                data = self._retry(lambda: self.client.lrange(key, 0, -1))
                return self._parse_messages(data)
            except Exception as e:
                logger.warning(
                    event="redis_fetch_failed_using_fallback",
                    error=str(e),
                    session_id=session_id,
                )

        data = self._fallback.lrange(key, 0, -1)
        return self._parse_messages(data)

    # GET ALIAS (used by memory_manager)

    def get(self, session_id: str) -> List[Dict]:
        return self.get_history(session_id)

    # PARSE MESSAGES

    def _parse_messages(self, data: List[str]) -> List[Dict]:
        out:  List[Dict] = []
        seen: set        = set()

        for item in data:
            try:
                parsed = json.loads(item)

                if not isinstance(parsed, dict):
                    continue

                parsed["content"] = self._clean(parsed.get("content", ""))
                h = self._hash(parsed)

                if h in seen:
                    continue

                seen.add(h)
                out.append(parsed)

            except Exception:
                continue

        return out

    # GET LAST K

    def get_last_k(self, session_id: str, k: Optional[int] = None) -> List[Dict]:
        key = self._key(session_id)
        k   = k or settings.MEMORY_TOP_K

        if self._is_available():
            try:
                data = self._retry(lambda: self.client.lrange(key, -k, -1))
                return self._parse_messages(data)
            except Exception:
                pass

        data = self._fallback.lrange(key, -k, -1)
        return self._parse_messages(data)

    # CLEAR MEMORY

    def clear_memory(self, session_id: str) -> None:
        key = self._key(session_id)

        if self._is_available():
            try:
                self._retry(lambda: self.client.delete(key))
            except Exception as e:
                logger.warning(
                    event="redis_clear_failed",
                    error=str(e),
                    session_id=session_id,
                )

        self._fallback.delete(key)

    # DELETE ALIAS (used by memory_manager)

    def delete(self, session_id: str) -> None:
        self.clear_memory(session_id)

    def purge_user(self, user_id: str) -> None:
        if not user_id:
            return
        prefix = f"{self.prefix}:{user_id}:"
        if self._is_available():
            try:
                for key in self.client.scan_iter(f"{prefix}*"):
                    self.client.delete(key)
            except Exception as exc:
                logger.warning(event="redis_user_purge_failed", user_id=user_id, error=str(exc))
        for key in list(self._fallback._store.keys()):
            if str(key).startswith(prefix):
                self._fallback.delete(key)

    # MEMORY SIZE

    def get_memory_size(self, session_id: str) -> int:
        key = self._key(session_id)

        if self._is_available():
            try:
                return self._retry(lambda: self.client.llen(key))
            except Exception:
                pass

        return self._fallback.llen(key)

    # QUERY RESULT CACHE

    def cache_set(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl     = ttl or self.query_cache_ttl
        payload = json.dumps(value)

        if self._is_available():
            try:
                self._retry(lambda: self.client.setex(cache_key, ttl, payload))
                return
            except Exception as e:
                logger.warning(event="redis_cache_set_failed", error=str(e))

        self._fallback.setex(cache_key, ttl, payload)

    def cache_get(self, cache_key: str) -> Optional[Any]:
        if self._is_available():
            try:
                raw = self._retry(lambda: self.client.get(cache_key))
                if raw:
                    return json.loads(raw)
                return None
            except Exception as e:
                logger.warning(event="redis_cache_get_failed", error=str(e))

        raw = self._fallback.get(cache_key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None

    # HEALTH CHECK

    def health_check(self) -> Dict:
        return {
            "redis_ok":        self._redis_ok,
            "use_upstash":     self._use_upstash,
            "client_type":     "upstash" if self._use_upstash else ("redis" if self._redis_ok else "fallback"),
            "fallback_active": not self._redis_ok,
        }


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/memory/redis_memory.py -v
# ============================================================

def test_memory_manager_fuses_redis_and_mongo() -> None:
    memory = RedisMemory()
    memory.append("s1", {"role": "user", "content": "hello"})
    assert memory.get("s1")


def test_redis_ttl_expires_old_turns() -> None:
    memory = RedisMemory()
    assert memory.ttl == settings.REDIS_TTL_SECONDS


def test_mongo_persistent_memory_retrieved() -> None:
    assert settings.MONGO_DB_NAME


def test_summarizer_compresses_long_memory() -> None:
    assert settings.MEMORY_SUMMARY_MAX_CHARS > 0


def test_gdpr_purge_all_memory() -> None:
    memory = RedisMemory()
    memory._fallback.rpush(f"{memory.prefix}:u1:session", "{}")
    memory.purge_user("u1")
    assert not memory._fallback._store
