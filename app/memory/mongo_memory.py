from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import List, Dict, Optional
import time

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MongoMemory:

    def __init__(self):

        self.client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=settings.DB_TIMEOUT_MS,
            maxPoolSize=settings.DB_MAX_POOL_SIZE,
            connect=True,
        )

        self._ping()

        self.db = self.client[settings.MONGO_DB_NAME]
        self.messages = self.db["messages"]
        self.summaries = self.db["summaries"]

        self._indexes()

    #  CONNECTION 
    def _ping(self):
        try:
            self.client.admin.command("ping")
        except Exception as e:
            logger.error(event="mongo_connection_failed", error=str(e))
            raise

    #  RETRY 
    def _retry(self, fn, retries=2):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    raise
                logger.warning(event="mongo_retry", error=str(e))
                time.sleep(0.3)

    #  CLEAN 
    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    #  ROLE 
    def _role(self, role: str) -> str:
        role = str(role or "user").lower()
        return role if role in {"user", "assistant", "system"} else "user"

    #  IMPORTANCE 
    def _importance(self, val: float) -> float:
        try:
            return max(0.0, min(float(val), 1.0))
        except Exception:
            return 0.5

    #  EMBEDDING 
    def _valid_embedding(self, emb):
        return (
            isinstance(emb, list) and
            len(emb) in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM
            )
        )

    #  INDEX 
    def _indexes(self):

        self.messages.create_index(
            [("session_id", ASCENDING), ("timestamp", DESCENDING)]
        )

        self.messages.create_index([("importance", DESCENDING)])

        self.summaries.create_index(
            [("session_id", ASCENDING), ("timestamp", DESCENDING)]
        )

    #  STORE MESSAGE 
    def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        modality: str = "text",
        importance: float = 1.0,
        extra: Optional[Dict] = None
    ):

        if not session_id or not content:
            return

        try:
            content = self._clean(content)
            if len(content) < 2:
                return

            content = content[:settings.MAX_PROMPT_CHARS]

            doc = {
                "session_id": session_id,
                "role": self._role(role),
                "content": content,
                "timestamp": datetime.utcnow(),
                "modality": modality,
                "importance": self._importance(importance),
            }

            if self._valid_embedding(embedding):
                doc["embedding"] = embedding

            if isinstance(extra, dict):
                doc["extra"] = extra

            self._retry(lambda: self.messages.insert_one(doc))

        except Exception as e:
            logger.error(event="mongo_store_failed", error=str(e))

    #  STORE SUMMARY 
    def store_summary(self, session_id: str, summary: str, embedding=None):

        if not session_id or not summary:
            return

        try:
            summary = self._clean(summary)

            if len(summary) < 5:
                return

            summary = summary[:settings.MEMORY_SUMMARY_MAX_CHARS]

            doc = {
                "session_id": session_id,
                "summary": summary,
                "timestamp": datetime.utcnow(),
            }

            if self._valid_embedding(embedding):
                doc["embedding"] = embedding

            self._retry(lambda: self.summaries.insert_one(doc))

        except Exception as e:
            logger.error(event="mongo_summary_failed", error=str(e))

    #  FETCH 
    def get_recent_history(self, session_id: str, limit: int = None) -> List[Dict]:

        limit = min(limit or settings.MAX_HISTORY_MESSAGES, settings.MAX_HISTORY_MESSAGES)

        try:
            cursor = self._retry(lambda: self.messages.find(
                {"session_id": session_id},
                {"_id": 0}
            ).sort("timestamp", DESCENDING).limit(limit))

            result = []

            for doc in reversed(list(cursor)):
                result.append({
                    "role": doc.get("role"),
                    "content": self._clean(doc.get("content")),
                    "embedding": doc.get("embedding"),
                    "modality": doc.get("modality"),
                    "timestamp": doc.get("timestamp").timestamp()
                    if doc.get("timestamp") else None
                })

            return result

        except Exception as e:
            logger.error(event="mongo_fetch_failed", error=str(e))
            return []

    #  CLEAR 
    def clear_memory(self, session_id: str):

        try:
            self._retry(lambda: self.messages.delete_many({"session_id": session_id}))
            self._retry(lambda: self.summaries.delete_many({"session_id": session_id}))
        except Exception as e:
            logger.error(event="mongo_clear_failed", error=str(e))