from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import List, Dict, Optional
import time

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class MongoMemory:

    def __init__(self):

        logger.info("[MongoMemory] connecting")

        self.client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=settings.DB_TIMEOUT_MS,
            maxPoolSize=settings.DB_MAX_POOL_SIZE,
            connect=True,  
        )

        self._validate_connection()

        self.db = self.client[settings.MONGO_DB_NAME]

        self.messages = self.db["messages"]
        self.summaries = self.db["summaries"]

        self._ensure_indexes()

        logger.info("[MongoMemory] initialized")

    
    # CONNECTION CHECK
    def _validate_connection(self):
        try:
            self.client.admin.command("ping")
        except Exception as e:
            logger.error("[MongoMemory] connection failed | %s", str(e))
            raise

    
    # RETRY
    def _retry(self, fn, retries=2):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    raise
                logger.warning("[MongoMemory][RETRY] %s", str(e))
                time.sleep(0.3)

    
    # CLEAN TEXT
    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    
    # ROLE NORMALIZATION
    def _normalize_role(self, role: str) -> str:
        role = str(role or "user").lower()
        return role if role in {"user", "assistant", "system"} else "user"

    
    # IMPORTANCE NORMALIZATION
    def _normalize_importance(self, value: float) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except Exception:
            return 0.5

    
    # INDEXES
    def _ensure_indexes(self):
        self.messages.create_index(
            [("session_id", ASCENDING), ("timestamp", DESCENDING)]
        )
        self.messages.create_index([("importance", DESCENDING)])

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
            role = self._normalize_role(role)
            content = self._clean(content)

            if len(content) < 2:
                return

            if len(content) > settings.MAX_PROMPT_CHARS:
                content = content[:settings.MAX_PROMPT_CHARS]

            doc = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow(),
                "modality": modality,
                "importance": self._normalize_importance(importance),
            }

            if embedding and isinstance(embedding, list):
                if len(embedding) in (
                    settings.TEXT_EMBEDDING_DIM,
                    settings.VISION_EMBEDDING_DIM,
                ):
                    doc["embedding"] = embedding

            if extra and isinstance(extra, dict):
                doc["extra"] = extra

            self._retry(lambda: self.messages.insert_one(doc))

        except Exception as e:
            logger.error("[MongoMemory] store failed | %s", str(e))

    
    # STORE SUMMARY
    def store_summary(self, session_id: str, summary: str, embedding=None):

        if not session_id or not summary:
            return

        try:
            summary = self._clean(summary)

            if len(summary) < 5:
                return

            if len(summary) > settings.MEMORY_SUMMARY_MAX_CHARS:
                summary = summary[:settings.MEMORY_SUMMARY_MAX_CHARS]

            doc = {
                "session_id": session_id,
                "summary": summary,
                "timestamp": datetime.utcnow(),
            }

            if embedding and isinstance(embedding, list):
                doc["embedding"] = embedding

            self._retry(lambda: self.summaries.insert_one(doc))

        except Exception as e:
            logger.error("[MongoMemory] summary failed | %s", str(e))

    
    # GET HISTORY
    def get_recent_history(self, session_id: str, limit: int = None) -> List[Dict]:

        limit = min(
            limit or settings.MAX_HISTORY_MESSAGES,
            settings.MAX_HISTORY_MESSAGES
        )

        try:
            cursor = self._retry(lambda: self.messages.find(
                {"session_id": session_id}
            ).sort("timestamp", DESCENDING).limit(limit))

            history = []

            for doc in reversed(list(cursor)):
                history.append({
                    "role": doc.get("role"),
                    "content": self._clean(doc.get("content")),
                    "embedding": doc.get("embedding"),
                    "modality": doc.get("modality"),
                    "timestamp": doc.get("timestamp").timestamp()
                    if doc.get("timestamp") else None
                })

            return history

        except Exception as e:
            logger.error("[MongoMemory] fetch failed | %s", str(e))
            return []

    
    # CLEAR
    def clear_memory(self, session_id: str):

        try:
            self._retry(lambda: self.messages.delete_many({"session_id": session_id}))
            self._retry(lambda: self.summaries.delete_many({"session_id": session_id}))
        except Exception as e:
            logger.error("[MongoMemory] clear failed | %s", str(e))