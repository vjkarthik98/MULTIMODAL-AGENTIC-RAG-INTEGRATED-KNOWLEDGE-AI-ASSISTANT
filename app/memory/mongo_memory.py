from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import List, Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class MongoMemory:
    def __init__(self):
        uri = settings.MONGO_URI

        logger.info("[MongoMemory] Connecting")

        self.client = MongoClient(
            uri,
            serverSelectionTimeoutMS=settings.DB_TIMEOUT_MS,
            maxPoolSize=settings.DB_MAX_POOL_SIZE,
        )

        self.db = self.client[settings.MONGO_DB_NAME]

        self.messages = self.db["messages"]
        self.summaries = self.db["summaries"]

        self._ensure_indexes()

        logger.info("[MongoMemory] initialized")

    def _ensure_indexes(self):
        # Messages
        self.messages.create_index(
            [("session_id", ASCENDING), ("timestamp", DESCENDING)]
        )
        self.messages.create_index([("importance", DESCENDING)])

        # Summaries
        self.summaries.create_index(
            [("session_id", ASCENDING), ("timestamp", DESCENDING)]
        )

    # STORE MESSAGE 
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
            # Truncate content
            if len(content) > settings.MAX_PROMPT_CHARS:
                content = content[:settings.MAX_PROMPT_CHARS]

            doc = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow(),
                "modality": modality,
                "importance": float(importance),
            }

            # Validate embedding
            if embedding and isinstance(embedding, list):
                if len(embedding) in (
                    settings.TEXT_EMBEDDING_DIM,
                    settings.VISION_EMBEDDING_DIM,
                ):
                    doc["embedding"] = embedding

            if extra and isinstance(extra, dict):
                doc["extra"] = extra

            self.messages.insert_one(doc)

            logger.debug(
                "[MongoMemory] stored | session_id=%s | role=%s",
                session_id,
                role
            )

        except Exception as e:
            logger.error(
                "[MongoMemory] store failed | session_id=%s | %s",
                session_id,
                str(e)
            )

    #  STORE SUMMARY 
    def store_summary(
        self,
        session_id: str,
        summary: str,
        embedding: Optional[List[float]] = None
    ):
        if not session_id or not summary:
            return

        try:
            if len(summary) > settings.MEMORY_SUMMARY_MAX_CHARS:
                summary = summary[:settings.MEMORY_SUMMARY_MAX_CHARS]

            doc = {
                "session_id": session_id,
                "summary": summary,
                "timestamp": datetime.utcnow(),
            }

            if embedding and isinstance(embedding, list):
                if len(embedding) in (
                    settings.TEXT_EMBEDDING_DIM,
                    settings.VISION_EMBEDDING_DIM,
                ):
                    doc["embedding"] = embedding

            self.summaries.insert_one(doc)

            logger.debug(
                "[MongoMemory] summary stored | session_id=%s",
                session_id
            )

        except Exception as e:
            logger.error(
                "[MongoMemory] summary failed | session_id=%s | %s",
                session_id,
                str(e)
            )

    #  GET RECENT HISTORY 
    def get_recent_history(
        self,
        session_id: str,
        limit: int = None
    ) -> List[Dict]:

        limit = limit or settings.MAX_HISTORY_MESSAGES

        try:
            cursor = self.messages.find(
                {"session_id": session_id}
            ).sort("timestamp", DESCENDING).limit(limit)

            history = []

            for doc in reversed(list(cursor)):
                history.append({
                    "role": doc.get("role"),
                    "content": doc.get("content"),
                    "embedding": doc.get("embedding"),
                    "modality": doc.get("modality"),
                    "timestamp": doc.get("timestamp").timestamp()
                    if doc.get("timestamp") else None
                })

            return history

        except Exception as e:
            logger.error("[MongoMemory] fetch failed | %s", str(e))
            return []

    #  GET LATEST SUMMARY 
    def get_latest_summary(self, session_id: str) -> str:
        try:
            doc = self.summaries.find_one(
                {"session_id": session_id},
                sort=[("timestamp", DESCENDING)]
            )

            return doc.get("summary", "") if doc else ""

        except Exception as e:
            logger.error("[MongoMemory] summary fetch failed | %s", str(e))
            return ""

    #  CLEAR MEMORY 
    def clear_memory(self, session_id: str):
        try:
            self.messages.delete_many({"session_id": session_id})
            self.summaries.delete_many({"session_id": session_id})

            logger.info(
                "[MongoMemory] cleared | session_id=%s",
                session_id
            )

        except Exception as e:
            logger.error("[MongoMemory] clear failed | %s", str(e))