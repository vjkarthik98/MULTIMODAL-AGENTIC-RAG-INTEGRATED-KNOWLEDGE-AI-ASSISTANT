import redis
import json
import time
from typing import List, Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class RedisMemory:
    def __init__(self):
        logger.info("[RedisMemory] connecting")

        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=settings.REDIS_TIMEOUT,
            socket_connect_timeout=settings.REDIS_TIMEOUT,
            retry_on_timeout=True,
        )

        self.max_messages = settings.MAX_HISTORY_MESSAGES
        self.ttl = settings.REDIS_TTL_SECONDS

        self.key_prefix = getattr(settings, "REDIS_KEY_PREFIX", "chat")

        logger.info("[RedisMemory] initialized")

    # KEY 
    def _get_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"

    # ADD MESSAGE 
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

        key = self._get_key(session_id)

        try:
            # Truncate content
            if len(content) > settings.MAX_PROMPT_CHARS:
                content = content[:settings.MAX_PROMPT_CHARS]

            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "modality": modality,
            }

            # Validate embedding
            if embedding and isinstance(embedding, list):
                if len(embedding) in (
                    settings.TEXT_EMBEDDING_DIM,
                    settings.VISION_EMBEDDING_DIM,
                ):
                    message["embedding"] = embedding

            if extra and isinstance(extra, dict):
                message["extra"] = extra

            payload = json.dumps(message)

            pipe = self.client.pipeline()

            pipe.rpush(key, payload)
            pipe.ltrim(key, -self.max_messages, -1)

            if self.ttl:
                pipe.expire(key, self.ttl)

            pipe.execute()

            logger.debug(
                "[RedisMemory] added | session_id=%s | role=%s",
                session_id,
                role
            )

        except Exception as e:
            logger.error(
                "[RedisMemory] add failed | session_id=%s | %s",
                session_id,
                str(e)
            )

    # GET HISTORY 
    def get_history(self, session_id: str) -> List[Dict]:
        key = self._get_key(session_id)

        try:
            data = self.client.lrange(key, 0, -1)

            history = []

            for item in data:
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        history.append(parsed)
                except Exception:
                    continue

            logger.debug(
                "[RedisMemory] fetched | session_id=%s | count=%s",
                session_id,
                len(history)
            )

            return history

        except Exception as e:
            logger.error(
                "[RedisMemory] fetch failed | session_id=%s | %s",
                session_id,
                str(e)
            )
            return []

    # GET LAST K 
    def get_last_k(self, session_id: str, k: int = None) -> List[Dict]:
        key = self._get_key(session_id)

        k = k or settings.MEMORY_TOP_K

        try:
            data = self.client.lrange(key, -k, -1)

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

    # CLEAR MEMORY 
    def clear_memory(self, session_id: str):
        key = self._get_key(session_id)

        try:
            self.client.delete(key)

            logger.info(
                "[RedisMemory] cleared | session_id=%s",
                session_id
            )

        except Exception as e:
            logger.error(
                "[RedisMemory] clear failed | session_id=%s | %s",
                session_id,
                str(e)
            )

    # MEMORY SIZE 
    def get_memory_size(self, session_id: str) -> int:
        key = self._get_key(session_id)

        try:
            return self.client.llen(key)
        except Exception:
            return 0