import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MongoMemory:

    def __init__(self) -> None:
        self._mongo_ok = False
        self.client    = None
        self.db        = None
        self.messages  = None
        self.summaries = None

        self._connect()

    # CONNECTION

    def _connect(self) -> None:
        try:
            self.client = MongoClient(
                settings.MONGO_URI,
                serverSelectionTimeoutMS=settings.DB_TIMEOUT_MS,
                maxPoolSize=settings.DB_MAX_POOL_SIZE,
                connect=True,
            )

            self._ping()

            self.db        = self.client[settings.MONGO_DB_NAME]
            self.messages  = self.db["messages"]
            self.summaries = self.db["summaries"]

            self._indexes()
            self._mongo_ok = True

            logger.info(event="mongo_connected", db=settings.MONGO_DB_NAME)

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self._mongo_ok = False
            logger.error(event="mongo_connection_failed", error=str(e))
            raise

        except Exception as e:
            self._mongo_ok = False
            logger.error(event="mongo_init_failed", error=str(e))
            raise

    # AVAILABILITY

    def _is_available(self) -> bool:
        return self._mongo_ok and self.messages is not None

    # PING

    def _ping(self) -> None:
        self.client.admin.command("ping")

    # RETRY

    def _retry(self, fn, retries: int = 2):
        for i in range(retries):
            try:
                return fn()
            except Exception as e:
                if i == retries - 1:
                    raise
                logger.warning(event="mongo_retry", attempt=i + 1, error=str(e))
                time.sleep(0.3 * (i + 1))

    # HELPERS

    def _clean(self, text: str) -> str:
        return " ".join(str(text or "").strip().split())

    def _role(self, role: str) -> str:
        role = str(role or "user").lower()
        return role if role in {"user", "assistant", "system"} else "user"

    def _importance(self, val: Any) -> float:
        try:
            return max(0.0, min(float(val), 1.0))
        except Exception:
            return 0.5

    def _valid_embedding(self, emb) -> bool:
        return (
            isinstance(emb, list) and
            len(emb) in (settings.TEXT_EMBEDDING_DIM, settings.VISION_EMBEDDING_DIM)
        )

    # INDEXES

    def _indexes(self) -> None:
        try:
            # PRIMARY QUERY INDEX
            self.messages.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="session_timestamp",
            )

            # IMPORTANCE INDEX
            self.messages.create_index(
                [("importance", DESCENDING)],
                name="importance",
            )

            # ROLE FILTER INDEX
            self.messages.create_index(
                [("session_id", ASCENDING), ("role", ASCENDING)],
                name="session_role",
            )

            # TTL INDEX: auto-expire messages after REDIS_TTL_SECONDS
            self.messages.create_index(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=settings.REDIS_TTL_SECONDS,
                name="ttl_expire",
            )

            # SUMMARIES INDEX
            self.summaries.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="summary_session_timestamp",
            )

            logger.info(event="mongo_indexes_created")

        except Exception as e:
            if "already exists" in str(e) or "IndexOptionsConflict" in str(e):
                logger.info(event="mongo_index_verified_existing")
            else:
                logger.warning(event="mongo_index_creation_failed", error=str(e))

    # STORE MESSAGE

    def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        modality: str = "text",
        importance: float = 1.0,
        extra: Optional[Dict] = None,
    ) -> None:

        if not session_id or not content:
            return

        if not self._is_available():
            logger.warning(event="mongo_store_skipped_unavailable", session_id=session_id)
            return

        try:
            content = self._clean(content)
            if len(content) < 2:
                return

            content = content[:settings.MAX_PROMPT_CHARS]

            doc: Dict = {
                "session_id": session_id,
                "role":       self._role(role),
                "content":    content,
                "timestamp":  datetime.utcnow(),
                "modality":   modality,
                "importance": self._importance(importance),
            }

            if self._valid_embedding(embedding):
                doc["embedding"] = embedding

            if isinstance(extra, dict):
                doc["extra"] = extra

            self._retry(lambda: self.messages.insert_one(doc))

        except Exception as e:
            logger.error(
                event="mongo_store_failed",
                session_id=session_id,
                error=str(e),
            )

    # INSERT ALIAS (used by memory_manager)

    def insert(self, session_id: str, message: Dict) -> None:
        self.store_message(
            session_id=session_id,
            role=message.get("role", "user"),
            content=message.get("content", ""),
            modality=message.get("modality", "text"),
            importance=message.get("importance", 1.0),
        )

    # STORE SUMMARY

    def store_summary(
        self,
        session_id: str,
        summary: str,
        embedding: Optional[List[float]] = None,
    ) -> None:

        if not session_id or not summary:
            return

        if not self._is_available():
            return

        try:
            summary = self._clean(summary)

            if len(summary) < 5:
                return

            summary = summary[:settings.MEMORY_SUMMARY_MAX_CHARS]

            doc: Dict = {
                "session_id": session_id,
                "summary":    summary,
                "timestamp":  datetime.utcnow(),
            }

            if self._valid_embedding(embedding):
                doc["embedding"] = embedding

            self._retry(lambda: self.summaries.insert_one(doc))

        except Exception as e:
            logger.error(
                event="mongo_summary_store_failed",
                session_id=session_id,
                error=str(e),
            )

    # GET RECENT HISTORY

    def get_recent_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict]:

        if not self._is_available():
            return []

        limit = min(limit or settings.MAX_HISTORY_MESSAGES, settings.MAX_HISTORY_MESSAGES)

        try:
            cursor = self._retry(
                lambda: self.messages.find(
                    {"session_id": session_id},
                    {"_id": 0},
                ).sort("timestamp", DESCENDING).limit(limit)
            )

            result: List[Dict] = []

            for doc in reversed(list(cursor)):
                ts = doc.get("timestamp")
                result.append({
                    "role":       doc.get("role"),
                    "content":    self._clean(doc.get("content", "")),
                    "embedding":  doc.get("embedding"),
                    "modality":   doc.get("modality", "text"),
                    "importance": doc.get("importance", 1.0),
                    "timestamp":  ts.timestamp() if isinstance(ts, datetime) else None,
                })

            logger.debug(
                event="mongo_history_fetched",
                doc_count=len(result),
                session_id=session_id,
            )

            return result

        except Exception as e:
            logger.error(
                event="mongo_fetch_failed",
                session_id=session_id,
                error=str(e),
            )
            return []

    # GET ALIAS (used by memory_manager)

    def get(self, session_id: str) -> List[Dict]:
        return self.get_recent_history(session_id)

    # GET LATEST SUMMARY (used by memory_fusion)

    def get_latest_summary(self, session_id: str) -> str:
        if not self._is_available():
            return ""

        try:
            doc = self._retry(
                lambda: self.summaries.find_one(
                    {"session_id": session_id},
                    {"_id": 0, "summary": 1},
                    sort=[("timestamp", DESCENDING)],
                )
            )
            return self._clean(doc.get("summary", "")) if doc else ""

        except Exception as e:
            logger.error(
                event="mongo_summary_fetch_failed",
                session_id=session_id,
                error=str(e),
            )
            return ""

    # CLEAR MEMORY

    def clear_memory(self, session_id: str) -> None:
        if not self._is_available():
            return

        try:
            self._retry(lambda: self.messages.delete_many({"session_id": session_id}))
            self._retry(lambda: self.summaries.delete_many({"session_id": session_id}))

            logger.info(event="mongo_memory_cleared", session_id=session_id)

        except Exception as e:
            logger.error(
                event="mongo_clear_failed",
                session_id=session_id,
                error=str(e),
            )

    # DELETE ALIAS (used by memory_manager)

    def delete(self, session_id: str) -> None:
        self.clear_memory(session_id)

    # HEALTH CHECK

    def health_check(self) -> Dict:
        status = {"mongo_ok": self._mongo_ok}

        if self._is_available():
            try:
                status["messages_count"]  = self.messages.estimated_document_count()
                status["summaries_count"] = self.summaries.estimated_document_count()
            except Exception:
                status["count_error"] = True

        return status