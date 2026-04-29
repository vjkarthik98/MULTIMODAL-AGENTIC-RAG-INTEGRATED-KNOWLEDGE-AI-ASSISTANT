from app.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:

    # INIT
    def __init__(self, redis_memory=None, mongo_memory=None):
        self.redis_memory = redis_memory
        self.mongo_memory = mongo_memory

    # STORE MESSAGE
    def add_message(self, session_id: str, role: str, content: str):

        message = {
            "role": role,
            "content": content
        }

        try:
            # REDIS (SHORT TERM)
            if self.redis_memory:
                self.redis_memory.append(session_id, message)

            # MONGO (LONG TERM)
            if self.mongo_memory:
                self.mongo_memory.insert(session_id, message)

        except Exception as e:
            logger.error("[MemoryManager] add_message failed | %s", str(e))


    # ADD INTERACTION
    def add_interaction(self, session_id: str, query: str, response: str):

        try:
            # store user query
            self.add_message(session_id, "user", query)

            # store assistant response
            self.add_message(session_id, "assistant", response)

        except Exception as e:
            logger.error("[MemoryManager] add_interaction failed | %s", str(e))
    
    # GET HISTORY
    def get_history(self, session_id: str, limit: int = 5):

        try:
            history = []

            # REDIS (PRIMARY)
            if self.redis_memory:
                data = self.redis_memory.get(session_id)

                if isinstance(data, list):
                    history = data

            # FALLBACK TO MONGO
            if not history and self.mongo_memory:
                data = self.mongo_memory.get(session_id)

                if isinstance(data, list):
                    history = data

            # ALWAYS RETURN LIST (SAFE)
            return history[-limit:] if history else []

        except Exception as e:
            logger.error("[MemoryManager] get_history failed | %s", str(e))
            return []

    # CLEAR MEMORY
    def clear(self, session_id: str):

        try:
            if self.redis_memory:
                self.redis_memory.delete(session_id)

            if self.mongo_memory:
                self.mongo_memory.delete(session_id)

        except Exception as e:
            logger.error("[MemoryManager] clear failed | %s", str(e))