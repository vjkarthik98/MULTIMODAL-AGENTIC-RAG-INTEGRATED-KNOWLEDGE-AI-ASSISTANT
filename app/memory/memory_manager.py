from typing import List, Dict
import time
import hashlib

from app.utils.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class MemoryManager:

    def __init__(self, redis_memory=None, mongo_memory=None):
        self.redis_memory = redis_memory
        self.mongo_memory = mongo_memory

    #  HASH 
    def _hash(self, msg: Dict) -> str:
        base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
        return hashlib.sha256(base.encode()).hexdigest()

    #  VALID 
    def _valid(self, content: str) -> bool:
        return isinstance(content, str) and len(content.strip()) > 2

    #  STORE 
    def add_message(self, session_id: str, role: str, content: str):

        if not self._valid(content):
            return

        message = {
            "role": role,
            "content": content.strip(),
            "timestamp": time.time()
        }

        try:
            #  REDIS (SHORT TERM) 
            if self.redis_memory:
                self.redis_memory.append(session_id, message)

            #  MONGO (LONG TERM) 
            if self.mongo_memory:
                self.mongo_memory.insert(session_id, message)

        except Exception as e:
            logger.error(event="memory_store_failed", error=str(e))

    #  INTERACTION 
    def add_interaction(self, session_id: str, query: str, response: str):

        try:
            self.add_message(session_id, "user", query)
            self.add_message(session_id, "assistant", response)
        except Exception as e:
            logger.error(event="memory_interaction_failed", error=str(e))

    #  DEDUP 
    def _dedup(self, history: List[Dict]) -> List[Dict]:

        seen = set()
        out = []

        for msg in history:
            try:
                h = self._hash(msg)
                if h in seen:
                    continue
                seen.add(h)
                out.append(msg)
            except Exception:
                continue

        return out

    #  GET 
    def get_history(self, session_id: str, limit: int = None) -> List[Dict]:

        limit = limit or settings.MAX_HISTORY_MESSAGES

        try:
            history = []

            #  REDIS PRIMARY 
            if self.redis_memory:
                data = self.redis_memory.get(session_id)
                if isinstance(data, list):
                    history = data

            #  FALLBACK MONGO 
            if not history and self.mongo_memory:
                data = self.mongo_memory.get(session_id)
                if isinstance(data, list):
                    history = data

            if not history:
                return []

            history = self._dedup(history)

            return history[-limit:]

        except Exception as e:
            logger.error(event="memory_fetch_failed", error=str(e))
            return []

    #  CLEAR 
    def clear(self, session_id: str):

        try:
            if self.redis_memory:
                self.redis_memory.delete(session_id)

            if self.mongo_memory:
                self.mongo_memory.delete(session_id)

        except Exception as e:
            logger.error(event="memory_clear_failed", error=str(e))