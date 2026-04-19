import redis
import json
import time
from typing import List, Dict, Optional
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


class RedisMemory:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_messages: int = 20,
        ttl_seconds: Optional[int] = 86400 # 24h default
    ):
        logger.info(f"[RedisMemory] Connecting to Redis | host={host} port={port} db={db}")

        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.max_messages = max_messages
        self.ttl = ttl_seconds

        logger.info("[RedisMemory] Initialized successfully")

    
    # KEY
    def _get_key(self, session_id: str) -> str:
        return f"chat:{session_id}"
    

    # ADD MESSAGE (ATOMIC)
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        modality: str = "text",
        extra: Optional[Dict] = None
    ):
        key = self._get_key(session_id)

        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "modality": modality
            }

            if embedding:
                message["embedding"] = embedding

            if extra:
                message["extra"] = extra

            # PUSH TO REDIS LIST
            self.client.rpush(key, json.dumps(message))

            # Trim to max_messages
            self.client.ltrim(key, -self.max_messages, -1)

            # TTL (Optional)
            if self.ttl:
                self.client.expire(key, self.ttl)

            logger.debug(
                f"[RedisMemory] session_id={session_id} | Added | role={role}"
            )

        except Exception as e:
            logger.error(
                f"[RedisMemory] session_id={session_id} | Add failed | {str(e)}"
            )
            raise

    # GET HISTORY   


    def get_history(self, session_id: str) -> List[Dict]:

        key = self._get_key(session_id)

        try:
            data = self.client.lrange(key, 0, -1)

            history = []
            for item in data:
                try:
                    history.append(json.loads(item))
                except Exception:
                    continue
            logger.debug(
                f"[RedisMemory] session_id={session_id} | Retrieved | count={len(history)}"
            )

            return history
        
        except Exception as e:
            logger.error(
                f"[RedisMemory] session_id={session_id} | Fetch failed | {str(e)}"
            )
            return []

    
    # CLEAR MEMORY

    def clear_memory(self, session_id: str):
        key = self._get_key(session_id)

        try:
            self.client.delete(key)

            logger.info(f"[RedisMemory] session_id={session_id} | Memory cleared")

        except Exception as e:
            logger.error(
                f"[RedisMemory] session_id={session_id} | Clear failed | {str(e)}"
            )
            raise

    # MEMORY SIZE (DEBUG/MONITORING)
    def get_memory_size(self, session_id: str) -> int:
        key = self._get_key(session_id)

        try:
            size = self.client.llen(key)
            return size
        except Exception:
            return 0
    
    # LAST N MESSAGES 
    def get_last_k(self, session_id: str, k: int = 5) -> List[Dict]:
        key = self._get_key(session_id)

        try:
            data = self.client.lrange(key, -k, -1)

            return [json.loads(x) for x in data if x]
        
        except Exception:
            return []