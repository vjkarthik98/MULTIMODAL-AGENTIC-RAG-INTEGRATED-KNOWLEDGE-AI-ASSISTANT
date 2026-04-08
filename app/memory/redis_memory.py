import redis
import json
from typing import List, Dict
import logging

# Logger
logger = logging.getLogger(__name__)


class RedisMemory:
    """
    Redis-based memory for storing recent chat history.
    """

    def __init__(self, host="localhost", port=6379, db=0, max_messages=10):
        logger.info(f"[RedisMemory] Connecting to Redis | host={host} port={port}")

        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.max_messages = max_messages

        logger.info("[RedisMemory] Connection established")

    def _get_key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    def get_history(self, session_id: str):
        """
        Retrieve chat history for a session.
        """
        key = self._get_key(session_id)

        try:
            logger.debug(f"[RedisMemory] session_id={session_id} | Fetching history")

            data = self.client.get(key)

            if not data:
                return []

            history = json.loads(data)

            logger.debug(
                f"[RedisMemory] session_id={session_id} | History fetched | count={len(history)}"
            )

            return history

        except Exception as e:
            logger.error(
                f"[RedisMemory] session_id={session_id} | Fetch error | error={str(e)}"
            )
            return []

    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to memory.
        """
        key = self._get_key(session_id)

        try:
            data = self.client.get(key)

            if data:
                try:
                    history = json.loads(data)
                except Exception:
                    history = []
            else:
                history = []

            # Append new message
            history.append({
                "role": role,
                "content": content
            })

            # Keep only last N messages
            history = history[-self.max_messages:]

            self.client.set(key, json.dumps(history))

            logger.debug(
                f"[RedisMemory] session_id={session_id} | Stored message | length={len(history)}"
            )

        except Exception as e:
            logger.error(
                f"[RedisMemory] session_id={session_id} | Store error | error={str(e)}"
            )
            raise

    def clear_memory(self, session_id: str):
        """
        Clear session memory.
        """
        key = self._get_key(session_id)

        try:
            self.client.delete(key)

            logger.info(f"[RedisMemory] session_id={session_id} | Memory cleared")

        except Exception as e:
            logger.error(
                f"[RedisMemory] session_id={session_id} | Clear error | error={str(e)}"
            )
            raise