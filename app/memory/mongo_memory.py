from pymongo import MongoClient
from datetime import datetime
import logging

# Logger
logger = logging.getLogger(__name__)


class MongoMemory:
    def __init__(self, uri="mongodb://localhost:27017"):
        logger.info(f"[MongoMemory] Connecting to MongoDB | uri={uri}")

        self.client = MongoClient(uri)
        self.db = self.client["rag_memory"]
        self.collection = self.db["chat_history"]

        logger.info("[MongoMemory] Connection established")

    def store_message(self, session_id: str, role: str, content: str):
        try:
            logger.debug(f"[MongoMemory] session_id={session_id} | Storing message | role={role}")

            self.collection.insert_one({
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow()
            })

        except Exception as e:
            logger.error(
                f"[MongoMemory] session_id={session_id} | Store failed | error={str(e)}"
            )
            raise

    def get_history(self, session_id: str, limit: int = 50):
        try:
            logger.debug(
                f"[MongoMemory] session_id={session_id} | Fetching history | limit={limit}"
            )

            cursor = self.collection.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).limit(limit)

            history = []
            for doc in reversed(list(cursor)):
                history.append({
                    "role": doc["role"],
                    "content": doc["content"]
                })

            logger.debug(
                f"[MongoMemory] session_id={session_id} | History fetched | count={len(history)}"
            )

            return history

        except Exception as e:
            logger.error(
                f"[MongoMemory] session_id={session_id} | Fetch failed | error={str(e)}"
            )
            raise