import redis
import json
import time
import hashlib
from typing import List, Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisMemory:

    def __init__(self):
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_timeout=settings.REDIS_TIMEOUT,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                retry_on_timeout=True,
            )
            
            self._ping()

        except Exception as e:
            logger.error(event="redis_disabled_runtime", error=str(e))
            self.client = None 

        self.max_messages = settings.MAX_HISTORY_MESSAGES
        self.ttl = settings.REDIS_TTL_SECONDS
        self.prefix = getattr(settings, "REDIS_KEY_PREFIX", "chat")

        

    #  CONNECTION 
    def _ping(self):
        try:
            self.client.ping()
        except Exception as e:
            logger.error(event="redis_connection_failed", error=str(e))
            raise

    #  RETRY 
    def _retry(self, fn, retries=2):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    raise
                logger.warning(event="redis_retry", error=str(e))
                time.sleep(0.2)

    #  CLEAN 
    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    #  ROLE 
    def _role(self, role: str) -> str:
        role = str(role or "user").lower()
        return role if role in {"user", "assistant", "system"} else "user"

    #  KEY 
    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    #  HASH 
    def _hash(self, msg: Dict) -> str:
        base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
        return hashlib.sha256(base.encode()).hexdigest()

    #  EMBEDDING 
    def _valid_embedding(self, emb):
        return (
            isinstance(emb, list) and
            len(emb) in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM
            )
        )

    #  ADD 
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        modality: str = "text",
        extra: Optional[Dict] = None
    ):

        if not session_id or not content:
            return

        key = self._key(session_id)

        try:
            content = self._clean(content)
            if len(content) < 2:
                return

            content = content[:settings.MAX_PROMPT_CHARS]

            message = {
                "role": self._role(role),
                "content": content,
                "timestamp": time.time(),
                "modality": modality,
            }

            if self._valid_embedding(embedding):
                message["embedding"] = embedding

            if isinstance(extra, dict):
                message["extra"] = extra

            payload = json.dumps(message)

            def _write():
                pipe = self.client.pipeline()
                pipe.rpush(key, payload)
                pipe.ltrim(key, -self.max_messages, -1)
                if self.ttl:
                    pipe.expire(key, self.ttl)
                return pipe.execute()

            self._retry(_write)

        except Exception as e:
            logger.error(event="redis_add_failed", error=str(e))

    #  FETCH 
    def get_history(self, session_id: str) -> List[Dict]:

        key = self._key(session_id)

        try:
            data = self._retry(lambda: self.client.lrange(key, 0, -1))

            out = []
            seen = set()

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

        except Exception as e:
            logger.error(event="redis_fetch_failed", error=str(e))
            return []

    #  LAST K 
    def get_last_k(self, session_id: str, k: int = None) -> List[Dict]:

        key = self._key(session_id)
        k = k or settings.MEMORY_TOP_K

        try:
            data = self._retry(lambda: self.client.lrange(key, -k, -1))

            result = []

            for item in data:
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        result.append(parsed)
                except Exception:
                    continue

            return result

        except Exception:
            return []

    #  CLEAR 
    def clear_memory(self, session_id: str):

        key = self._key(session_id)

        try:
            self._retry(lambda: self.client.delete(key))
        except Exception as e:
            logger.error(event="redis_clear_failed", error=str(e))

    #  SIZE 
    def get_memory_size(self, session_id: str) -> int:

        key = self._key(session_id)

        try:
            return self._retry(lambda: self.client.llen(key))
        except Exception:
            return 0