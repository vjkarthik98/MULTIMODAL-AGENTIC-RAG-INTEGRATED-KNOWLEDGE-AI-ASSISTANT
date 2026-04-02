import redis
import json
from typing import List, Dict

class RedisMemory:
    """
    Redis-based memory for storing recent chat history.
    Each session_id maps to a list of messages:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    def __init__(self, host="localhost", port=6379, db=0, max_messages=10):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.max_messages = max_messages

    def _get_key(self, session_id: str) -> str:
        return f"chat: {session_id}"
    
    def get_history(self, session_id: str) -> List[Dict]:
        """
        Retrieve chat history for a session.
        """
        key = self._get_key(session_id)
        data=self.client.get(key)

        if not data:
            return []
        
        return json.loads(data)
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to memory.
        """
        history = self.get_history(session_id)

        history.append({
            "role": role,
            "content": content
        })

        # Keep only last N messages
        history = history[-self.max_messages:]

        key = self._get_key(session_id)
        self.client.set(key, json.dumps(history))

    def clear_memory(self, session_id: str):
        """
        Clear session memory.
        """
        key = self._get_key(session_id)
        self.client.delete(key)
    