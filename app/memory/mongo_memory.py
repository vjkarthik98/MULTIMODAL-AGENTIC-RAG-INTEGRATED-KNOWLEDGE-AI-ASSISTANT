from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.errors import (
        BulkWriteError,
        ConnectionFailure,
        DuplicateKeyError,
        OperationFailure,
        ServerSelectionTimeoutError,
    )

    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False
    ASCENDING = 1
    DESCENDING = -1
    MongoClient = None  # type: ignore[assignment]

    class ConnectionFailure(Exception):  # type: ignore[no-redef]  # noqa: N818 -- name deliberately matches pymongo.errors.ConnectionFailure so except clauses work identically whether pymongo is installed or not
        pass

    class ServerSelectionTimeoutError(Exception):  # type: ignore[no-redef]
        pass

    class OperationFailure(Exception):  # type: ignore[no-redef]  # noqa: N818 -- name deliberately matches pymongo.errors.OperationFailure, same fallback-shim reasoning as ConnectionFailure above
        pass

    class DuplicateKeyError(Exception):  # type: ignore[no-redef]
        pass

    class BulkWriteError(Exception):  # type: ignore[no-redef]
        pass


try:
    import pybreaker

    _mongo_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

    class _DummyBreaker:
        def __call__(self, fn):
            return fn

    _mongo_breaker = _DummyBreaker()  # type: ignore[assignment]


# VALID ROLES
_VALID_ROLES = {"user", "assistant", "system"}

# VALID MODALITIES
_VALID_MODALITIES = {"text", "image", "audio", "video", "table", "document"}

# CAP ON EMBEDDED MESSAGES PER CHAT SESSION DOCUMENT (≈200 turns)
_MAX_SESSION_MESSAGES = 400


class MongoMemory:

    def __init__(self) -> None:
        self._enabled: bool = bool(settings.MONGO_URI) and settings.USE_MONGO
        self._mongo_ok: bool = False
        self.client = None
        self.db = None
        self.messages = None
        self.summaries = None
        self.sessions = None
        self.feedback = None

        if not self._enabled:
            logger.info(event="mongo_disabled_no_uri_or_flag")
            return

        if not _PYMONGO_AVAILABLE:
            logger.warning(event="pymongo_not_installed")
            return

        self._connect()

    # CONNECTION

    def _connect(self) -> None:
        try:
            self.client = MongoClient(
                settings.MONGO_URI,
                serverSelectionTimeoutMS=settings.DB_TIMEOUT_MS,
                maxPoolSize=settings.DB_MAX_POOL_SIZE,
                connect=True,
                tz_aware=True,
            )
            self._ping()

            self.db = self.client[settings.MONGO_DB_NAME]
            self.messages = self.db[settings.MONGO_MEMORY_COLLECTION]
            self.summaries = self.db[settings.MONGO_SUMMARY_COLLECTION]
            self.sessions = self.db[settings.MONGO_SESSIONS_COLLECTION]
            self.feedback = self.db[settings.MONGO_FEEDBACK_COLLECTION]

            self._ensure_indexes()
            self._mongo_ok = True

            logger.info(
                event="mongo_connected",
                db=settings.MONGO_DB_NAME,
                messages_col=settings.MONGO_MEMORY_COLLECTION,
                summaries_col=settings.MONGO_SUMMARY_COLLECTION,
            )

        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            self._mongo_ok = False
            logger.warning(event="mongo_connection_failed", error=str(exc))

        except Exception as exc:
            self._mongo_ok = False
            logger.warning(event="mongo_init_failed", error=str(exc))

    # AVAILABILITY CHECK

    def _is_available(self) -> bool:
        return self._mongo_ok and self.messages is not None

    # PING WITH CIRCUIT BREAKER

    def _ping(self) -> None:
        def _do():
            self.client.admin.command("ping")

        if _PYBREAKER_AVAILABLE:
            _mongo_breaker(_do)()
        else:
            _do()

    # TENACITY RETRY WRAPPER

    @retry(
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            min=settings.RETRY_WAIT_MIN_SEC,
            max=settings.RETRY_WAIT_MAX_SEC,
        ),
        retry=retry_if_exception_type((ConnectionFailure, ServerSelectionTimeoutError)),
        reraise=True,
    )
    def _retry(self, fn):
        def _do():
            return fn()

        if _PYBREAKER_AVAILABLE:
            return _mongo_breaker(_do)()
        return _do()

    # HELPERS

    def _clean(self, text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFC", str(text or ""))
        return " ".join(text.strip().split())

    def _role(self, role: str) -> str:
        role = str(role or "user").lower().strip()
        return role if role in _VALID_ROLES else "user"

    def _modality(self, modality: str) -> str:
        m = str(modality or "text").lower().strip()
        return m if m in _VALID_MODALITIES else "text"

    def _importance(self, val: Any) -> float:
        try:
            return max(0.0, min(float(val), 1.0))
        except Exception:
            return 0.5

    def _valid_embedding(self, emb: Any) -> bool:
        return isinstance(emb, list) and len(emb) in (
            settings.TEXT_EMBEDDING_DIM,
            settings.VISION_EMBEDDING_DIM,
        )

    def _utcnow(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    # ENSURE INDEXES

    def _ensure_indexes(self) -> None:
        # Only ever called from _connect(), immediately after the 4
        # collections below are assigned from a live self.db — guaranteed
        # non-None here, but that invariant crosses a method boundary mypy
        # can't see on its own, hence the assert (also a real, cheap safety
        # net if that call-order invariant is ever broken by a future edit).
        assert (
            self.messages is not None
            and self.summaries is not None
            and self.sessions is not None
            and self.feedback is not None
        )
        try:
            # PRIMARY QUERY INDEX
            self.messages.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_session_timestamp",
                background=True,
            )

            # USER + SESSION INDEX
            self.messages.create_index(
                [("user_id", ASCENDING), ("session_id", ASCENDING)],
                name="idx_user_session",
                background=True,
            )

            # IMPORTANCE INDEX
            self.messages.create_index(
                [("importance", DESCENDING)],
                name="idx_importance",
                background=True,
            )

            # ROLE FILTER INDEX
            self.messages.create_index(
                [("session_id", ASCENDING), ("role", ASCENDING)],
                name="idx_session_role",
                background=True,
            )

            # MODALITY FILTER INDEX
            self.messages.create_index(
                [("modality", ASCENDING)],
                name="idx_modality",
                background=True,
            )

            # TTL AUTO-EXPIRE INDEX
            self.messages.create_index(
                [("expire_at", ASCENDING)],
                name="idx_ttl_expire",
                expireAfterSeconds=0,
                background=True,
            )

            # SUMMARIES INDEXES
            self.summaries.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_summary_session_timestamp",
                background=True,
            )
            self.summaries.create_index(
                [("user_id", ASCENDING)],
                name="idx_summary_user_id",
                background=True,
            )

            # CHAT SESSION INDEXES — RECENTS (permanent transcripts, no TTL)
            self.sessions.create_index(
                [("user_id", ASCENDING), ("session_id", ASCENDING)],
                name="idx_session_user_session",
                unique=True,
                background=True,
            )
            self.sessions.create_index(
                [("user_id", ASCENDING), ("updated_at", DESCENDING)],
                name="idx_session_user_updated",
                background=True,
            )

            # FEEDBACK INDEXES
            self.feedback.create_index(
                [("user_id", ASCENDING), ("created_at", DESCENDING)],
                name="idx_feedback_user_created",
                background=True,
            )
            self.feedback.create_index(
                [("session_id", ASCENDING)],
                name="idx_feedback_session",
                background=True,
            )

            logger.info(event="mongo_indexes_ensured")

        except Exception as exc:
            if "already exists" in str(exc) or "IndexOptionsConflict" in str(exc):
                logger.debug(event="mongo_indexes_already_exist")
            else:
                logger.warning(event="mongo_index_creation_warning", error=str(exc))

    # STORE MESSAGE

    def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: list[float] | None = None,
        modality: str = "text",
        importance: float = 1.0,
        user_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not session_id or not content:
            return

        if not self._enabled:
            return

        if not user_id:
            logger.error(
                event="mongo_store_missing_user_id",
                session_id=session_id,
                error="store_message called without user_id — refusing to store an unscoped message",
            )
            return

        if not self._is_available():
            logger.warning(
                event="mongo_store_skipped_unavailable",
                session_id=session_id,
            )
            return

        try:
            content = self._clean(content)
            if len(content) < 2:
                return

            content = content[: settings.MAX_PROMPT_CHARS]
            now = self._utcnow()

            # COMPUTE TTL EXPIRY
            ttl_seconds = getattr(settings, "MEMORY_REDIS_TTL", settings.REDIS_TTL_SECONDS)
            expire_at = datetime.fromtimestamp(
                time.time() + ttl_seconds,
                tz=timezone.utc,
            )

            doc: dict[str, Any] = {
                "session_id": session_id,
                "user_id": user_id,
                "role": self._role(role),
                "content": content,
                "timestamp": now,
                "expire_at": expire_at,
                "modality": self._modality(modality),
                "importance": self._importance(importance),
            }

            if self._valid_embedding(embedding):
                doc["embedding"] = embedding

            if isinstance(extra, dict) and extra:
                doc["extra"] = extra

            def _do():
                self.messages.insert_one(doc)

            self._retry(_do)

        except Exception as exc:
            logger.warning(
                event="mongo_store_failed",
                session_id=session_id,
                error=str(exc),
            )

    # INSERT ALIAS — USED BY MEMORY MANAGER

    def insert(self, session_id: str, message: dict[str, Any]) -> None:
        self.store_message(
            session_id=session_id,
            role=message.get("role", "user"),
            content=message.get("content", ""),
            modality=message.get("modality", "text"),
            importance=message.get("importance", 1.0),
            user_id=message.get("user_id"),
        )

    # STORE SUMMARY

    def store_summary(
        self,
        session_id: str,
        summary: str,
        embedding: list[float] | None = None,
        user_id: str | None = None,
    ) -> None:
        if not session_id or not summary:
            return

        if not user_id:
            logger.error(
                event="mongo_summary_store_missing_user_id",
                session_id=session_id,
                error="store_summary called without user_id — refusing to store an unscoped summary",
            )
            return

        if not self._enabled or not self._is_available():
            return

        try:
            summary = self._clean(summary)
            if len(summary) < settings.MIN_SUMMARY_LENGTH:
                return

            summary = summary[: settings.MEMORY_SUMMARY_MAX_CHARS]
            now = self._utcnow()

            doc: dict[str, Any] = {
                "session_id": session_id,
                "user_id": user_id,
                "summary": summary,
                "timestamp": now,
            }

            if self._valid_embedding(embedding):
                doc["embedding"] = embedding

            def _do():
                self.summaries.insert_one(doc)

            self._retry(_do)

            logger.debug(
                event="mongo_summary_stored",
                session_id=session_id,
                length=len(summary),
            )

        except Exception as exc:
            logger.warning(
                event="mongo_summary_store_failed",
                session_id=session_id,
                error=str(exc),
            )

    # GET RECENT HISTORY

    def get_recent_history(
        self,
        session_id: str,
        limit: int | None = None,
        role_filter: str | None = None,
        modality_filter: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._is_available():
            return []

        if not user_id:
            logger.error(
                event="mongo_history_missing_user_id",
                session_id=session_id,
                error="get_recent_history called without user_id — refusing an unscoped query",
            )
            return []

        limit = min(limit or settings.MAX_HISTORY_MESSAGES, settings.MAX_HISTORY_MESSAGES)

        try:
            query: dict[str, Any] = {"session_id": session_id, "user_id": user_id}

            if role_filter and role_filter in _VALID_ROLES:
                query["role"] = role_filter

            if modality_filter and modality_filter in _VALID_MODALITIES:
                query["modality"] = modality_filter

            def _do():
                return list(
                    self.messages.find(
                        query,
                        {"_id": 0},
                    )
                    .sort("timestamp", DESCENDING)
                    .limit(limit)
                )

            raw = self._retry(_do)
            result: list[dict[str, Any]] = []

            for doc in reversed(raw):
                ts = doc.get("timestamp")
                unix_ts: float | None = None
                if isinstance(ts, datetime):
                    unix_ts = ts.timestamp()

                result.append(
                    {
                        "role": doc.get("role"),
                        "content": self._clean(doc.get("content", "")),
                        "embedding": doc.get("embedding"),
                        "modality": doc.get("modality", "text"),
                        "importance": doc.get("importance", 1.0),
                        "timestamp": unix_ts,
                        "user_id": doc.get("user_id"),
                    }
                )

            logger.debug(
                event="mongo_history_fetched",
                doc_count=len(result),
                session_id=session_id,
            )

            return result

        except Exception as exc:
            logger.warning(
                event="mongo_fetch_failed",
                session_id=session_id,
                error=str(exc),
            )
            return []

    # GET ALIAS — USED BY MEMORY MANAGER

    def get(self, session_id: str, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.get_recent_history(session_id, user_id=user_id)

    # GET LATEST SUMMARY — USED BY MEMORY FUSION

    def get_latest_summary(self, session_id: str, user_id: str | None = None) -> str:
        if not self._is_available():
            return ""

        if not user_id:
            logger.error(
                event="mongo_summary_missing_user_id",
                session_id=session_id,
                error="get_latest_summary called without user_id — refusing an unscoped query",
            )
            return ""

        try:
            _q: dict[str, Any] = {"session_id": session_id, "user_id": user_id}

            def _do():
                return self.summaries.find_one(
                    _q,
                    {"_id": 0, "summary": 1},
                    sort=[("timestamp", DESCENDING)],
                )

            doc = self._retry(_do)
            return self._clean(doc.get("summary", "")) if doc else ""

        except Exception as exc:
            logger.warning(
                event="mongo_summary_fetch_failed",
                session_id=session_id,
                error=str(exc),
            )
            return ""

    # GET ALL SUMMARIES FOR SESSION

    def get_summaries(
        self,
        session_id: str,
        limit: int = 5,
        user_id: str | None = None,
    ) -> list[str]:
        if not self._is_available():
            return []

        if not user_id:
            logger.error(
                event="mongo_summaries_missing_user_id",
                session_id=session_id,
                error="get_summaries called without user_id — refusing an unscoped query",
            )
            return []

        try:
            _q: dict[str, Any] = {"session_id": session_id, "user_id": user_id}

            def _do():
                return list(
                    self.summaries.find(
                        _q,
                        {"_id": 0, "summary": 1},
                    )
                    .sort("timestamp", DESCENDING)
                    .limit(limit)
                )

            docs = self._retry(_do)
            return [self._clean(d.get("summary", "")) for d in docs if d.get("summary")]

        except Exception as exc:
            logger.error(
                event="mongo_summaries_fetch_failed",
                session_id=session_id,
                error=str(exc),
            )
            return []

    # CLEAR SESSION MEMORY

    def clear_memory(self, session_id: str, user_id: str | None = None) -> None:
        if not self._is_available():
            return

        if not user_id:
            logger.error(
                event="mongo_clear_missing_user_id",
                session_id=session_id,
                error="clear_memory called without user_id — refusing an unscoped delete",
            )
            return

        try:
            _q: dict[str, Any] = {"session_id": session_id, "user_id": user_id}

            def _del_msgs():
                return self.messages.delete_many(_q)

            def _del_sums():
                return self.summaries.delete_many(_q)

            r1 = self._retry(_del_msgs)
            r2 = self._retry(_del_sums)

            logger.info(
                event="mongo_memory_cleared",
                session_id=session_id,
                messages_deleted=r1.deleted_count,
                summaries_deleted=r2.deleted_count,
            )

        except Exception as exc:
            logger.warning(
                event="mongo_clear_failed",
                session_id=session_id,
                error=str(exc),
            )

    # DELETE ALIAS — USED BY MEMORY MANAGER

    def delete(self, session_id: str, user_id: str | None = None) -> None:
        self.clear_memory(session_id, user_id)

    # GDPR PURGE — ALL DATA FOR USER

    def purge_user(self, user_id: str) -> None:
        if not user_id or not self._is_available():
            return

        try:

            def _del_msgs():
                return self.messages.delete_many({"user_id": user_id})

            def _del_sums():
                return self.summaries.delete_many({"user_id": user_id})

            def _del_sessions():
                return self.sessions.delete_many({"user_id": user_id})

            r1 = self._retry(_del_msgs)
            r2 = self._retry(_del_sums)
            r3 = self._retry(_del_sessions)

            logger.info(
                event="mongo_user_purged",
                user_id=user_id,
                messages_deleted=r1.deleted_count,
                summaries_deleted=r2.deleted_count,
                sessions_deleted=r3.deleted_count,
            )

        except Exception as exc:
            logger.error(
                event="mongo_user_purge_failed",
                user_id=user_id,
                error=str(exc),
            )

    # CHAT SESSIONS — RECENTS (permanent transcripts, decoupled from the
    # short-term `messages` TTL so old chats stay browsable indefinitely)

    def _auto_title(self, text: str, max_words: int = 8, max_chars: int = 48) -> str:
        text = self._clean(text)
        if not text:
            return "New chat"
        words = text.split(" ")[:max_words]
        title = " ".join(words)
        if len(title) > max_chars:
            title = title[:max_chars].rstrip()
        return f"{title}…" if len(title) < len(text) else title

    def save_chat_turn(
        self,
        session_id: str,
        user_id: str | None,
        query: str,
        answer: str,
        sources: list | None = None,
    ) -> None:
        if not session_id or not query.strip() or not answer.strip():
            return
        if not user_id:
            logger.error(
                event="mongo_save_chat_turn_missing_user_id",
                session_id=session_id,
                error="save_chat_turn called without user_id — refusing to store an unscoped chat turn",
            )
            return
        if not self._enabled or not self._is_available():
            return

        try:
            now = self._utcnow()
            q = self._clean(query)[: settings.MAX_PROMPT_CHARS]
            a = self._clean(answer)[: settings.MAX_PROMPT_CHARS]
            src = sources if isinstance(sources, list) else []

            def _do():
                self.sessions.update_one(
                    {"user_id": user_id, "session_id": session_id},
                    {
                        "$setOnInsert": {
                            "title": self._auto_title(q),
                            "created_at": now,
                            "pinned": False,
                            "archived": False,
                        },
                        "$set": {"updated_at": now},
                        "$inc": {"message_count": 2},
                        "$push": {
                            "messages": {
                                "$each": [
                                    {
                                        "role": "user",
                                        "content": q,
                                        "timestamp": now,
                                        "msg_id": str(uuid.uuid4()),
                                    },
                                    {
                                        "role": "assistant",
                                        "content": a,
                                        "timestamp": now,
                                        "sources": src,
                                        "msg_id": str(uuid.uuid4()),
                                        "vote": None,
                                    },
                                ],
                                "$slice": -_MAX_SESSION_MESSAGES,
                            }
                        },
                    },
                    upsert=True,
                )

            self._retry(_do)

        except Exception as exc:
            logger.warning(
                event="chat_session_save_failed",
                session_id=session_id,
                error=str(exc),
            )

    def list_chat_sessions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if not user_id or not self._is_available():
            return []
        try:

            def _do():
                cursor = (
                    self.sessions.find({"user_id": user_id}, {"messages": 0, "_id": 0})
                    .sort([("pinned", DESCENDING), ("updated_at", DESCENDING)])
                    .limit(limit)
                )
                return list(cursor)

            docs = self._retry(_do)
            return [
                {
                    "session_id": d.get("session_id"),
                    "title": d.get("title") or "New chat",
                    "created_at": d.get("created_at"),
                    "updated_at": d.get("updated_at"),
                    "message_count": d.get("message_count", 0),
                    "pinned": bool(d.get("pinned", False)),
                    "archived": bool(d.get("archived", False)),
                }
                for d in docs
            ]

        except Exception as exc:
            logger.warning(event="chat_sessions_list_failed", user_id=user_id, error=str(exc))
            return []

    def get_chat_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        if not user_id or not session_id or not self._is_available():
            return None
        try:

            def _do():
                return self.sessions.find_one(
                    {"user_id": user_id, "session_id": session_id}, {"_id": 0}
                )

            doc = self._retry(_do)
            if not doc:
                return None

            return {
                "session_id": doc.get("session_id"),
                "title": doc.get("title") or "New chat",
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "messages": [
                    {
                        "role": m.get("role"),
                        "content": m.get("content", ""),
                        "timestamp": m.get("timestamp"),
                        "sources": m.get("sources") or [],
                        "msg_id": m.get("msg_id"),
                        "vote": m.get("vote"),
                    }
                    for m in doc.get("messages", [])
                ],
            }

        except Exception as exc:
            logger.warning(event="chat_session_fetch_failed", session_id=session_id, error=str(exc))
            return None

    def patch_last_assistant_message(
        self,
        session_id: str,
        user_id: str | None,
        content: str,
        sources: list | None = None,
        msg_id: str | None = None,
    ) -> bool:
        """Overwrite the last assistant message in the session transcript.
        Called after streaming completes so reload shows the exact streamed answer.
        If msg_id is provided it is stamped onto the message so the frontend ID
        and DB msg_id stay in sync for vote persistence."""
        if not session_id or not content or not self._is_available():
            return False
        if not user_id:
            logger.error(
                event="mongo_patch_missing_user_id",
                session_id=session_id,
                error="patch_last_assistant_message called without user_id — refusing an unscoped update",
            )
            return False
        try:

            def _do():
                doc = self.sessions.find_one(
                    {"session_id": session_id, "user_id": user_id},
                    {"messages": 1, "_id": 1},
                )
                if not doc:
                    return False
                messages = list(doc.get("messages") or [])
                updated = False
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "assistant":
                        patch = {
                            **messages[i],
                            "content": str(content)[: settings.MAX_PROMPT_CHARS],
                            "sources": sources if isinstance(sources, list) else [],
                        }
                        if msg_id:
                            patch["msg_id"] = msg_id
                        messages[i] = patch
                        updated = True
                        break
                if updated:
                    self.sessions.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"messages": messages}},
                    )
                return updated

            return bool(self._retry(_do))
        except Exception as exc:
            logger.warning(
                event="patch_last_msg_failed",
                session_id=session_id,
                error=str(exc),
            )
            return False

    def delete_chat_session(self, user_id: str, session_id: str) -> bool:
        """Delete a chat session and ALL associated data from every collection.

        Removes from:
          - sessions   (metadata / Recents entry)
          - messages   (full Q&A history)
          - summaries  (compressed context summaries)
        Returns True only if the sessions record was found and deleted.
        """
        if not user_id or not session_id or not self._is_available():
            return False
        try:
            q = {"user_id": user_id, "session_id": session_id}

            def _do():
                session_result = self.sessions.delete_one(q)
                messages_result = self.messages.delete_many(q)
                summaries_result = (
                    self.summaries.delete_many(q) if self.summaries is not None else None
                )
                return session_result, messages_result, summaries_result

            session_r, msg_r, sum_r = self._retry(_do)
            found = session_r is not None and session_r.deleted_count > 0
            logger.info(
                event="chat_session_deleted",
                session_id=session_id,
                user_id=user_id,
                messages_deleted=msg_r.deleted_count if msg_r else 0,
                summaries_deleted=sum_r.deleted_count if sum_r else 0,
            )
            return found

        except Exception as exc:
            logger.warning(
                event="chat_session_delete_failed", session_id=session_id, error=str(exc)
            )
            return False

    def delete_chat_session_all(self, user_id: str, session_id: str) -> None:
        """Unconditional wipe: removes the session from every collection.
        Never raises — errors are logged and swallowed so the API can always
        return 200 OK (idempotent delete)."""
        if not user_id or not session_id:
            return
        if not self._is_available():
            return
        try:
            q = {"user_id": user_id, "session_id": session_id}

            def _do():
                s_r = self.sessions.delete_one(q)
                m_r = self.messages.delete_many(q)
                sum_r = self.summaries.delete_many(q) if self.summaries is not None else None
                return s_r, m_r, sum_r

            s_r, m_r, sum_r = self._retry(_do)
            logger.info(
                event="chat_session_deleted",
                session_id=session_id,
                user_id=user_id,
                sessions_deleted=s_r.deleted_count if s_r else 0,
                messages_deleted=m_r.deleted_count if m_r else 0,
                summaries_deleted=sum_r.deleted_count if sum_r else 0,
            )
        except Exception as exc:
            logger.warning(
                event="chat_session_delete_failed", session_id=session_id, error=str(exc)
            )

    def delete_all_chat_sessions(self, user_id: str) -> int:
        """Delete every chat session (and associated messages/summaries) for a user.
        Returns the number of sessions deleted."""
        if not user_id or not self._is_available():
            return 0
        assert (
            self.sessions is not None and self.messages is not None and self.summaries is not None
        )
        try:
            s_r = self.sessions.delete_many({"user_id": user_id})
            self.messages.delete_many({"user_id": user_id})
            self.summaries.delete_many({"user_id": user_id})
            count = s_r.deleted_count if s_r else 0
            logger.info(event="all_chat_sessions_deleted", user_id=user_id, count=count)
            return count
        except Exception as exc:
            logger.warning(event="delete_all_sessions_failed", user_id=user_id, error=str(exc))
            return 0

    _SESSION_UPDATABLE_FIELDS = {"title", "pinned", "archived"}

    def update_chat_session(self, user_id: str, session_id: str, fields: dict[str, Any]) -> bool:
        """Rename / pin / archive a chat. Does not touch updated_at so the
        Recents ordering by recency is preserved (pinned sorts separately)."""
        if not user_id or not session_id or not self._is_available():
            return False

        updates = {k: v for k, v in fields.items() if k in self._SESSION_UPDATABLE_FIELDS}
        if not updates:
            return False

        try:

            def _do():
                return self.sessions.update_one(
                    {"user_id": user_id, "session_id": session_id},
                    {"$set": updates},
                )

            result = self._retry(_do)
            return bool(result and result.matched_count)

        except Exception as exc:
            logger.warning(
                event="chat_session_update_failed", session_id=session_id, error=str(exc)
            )
            return False

    def patch_message_vote(
        self,
        user_id: str,
        session_id: str,
        msg_id: str,
        vote: str | None,
    ) -> bool:
        """Write the vote ('up'/'down'/None) onto the specific message in the session transcript."""
        if not user_id or not session_id or not msg_id or not self._is_available():
            return False
        try:

            def _do():
                return self.sessions.update_one(
                    {"user_id": user_id, "session_id": session_id, "messages.msg_id": msg_id},
                    {"$set": {"messages.$.vote": vote}},
                )

            result = self._retry(_do)
            return bool(result and result.matched_count)
        except Exception as exc:
            logger.warning(event="patch_message_vote_failed", msg_id=msg_id, error=str(exc))
            return False

    def save_feedback(
        self,
        user_id: str,
        session_id: str,
        vote: str,
        message_id: str | None,
        query: str | None,
        response_snippet: str | None,
    ) -> bool:
        """Persist a thumbs-up / thumbs-down rating for a single assistant message."""
        if not self._is_available() or self.feedback is None:
            return False
        if vote not in ("up", "down"):
            return False
        try:
            doc = {
                "user_id": user_id,
                "session_id": session_id,
                "vote": vote,
                "message_id": message_id,
                "query": (query or "")[:500],
                "response_snippet": (response_snippet or "")[:500],
                "created_at": datetime.now(timezone.utc),
            }
            self.feedback.insert_one(doc)
            logger.info(event="feedback_saved", user_id=user_id, session_id=session_id, vote=vote)
            return True
        except Exception as exc:
            logger.warning(event="feedback_save_failed", error=str(exc))
            return False

    # ASYNC WRAPPERS

    async def async_store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: list[float] | None = None,
        modality: str = "text",
        importance: float = 1.0,
        user_id: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.store_message,
            session_id,
            role,
            content,
            embedding,
            modality,
            importance,
            user_id,
        )

    async def async_get_recent_history(
        self,
        session_id: str,
        limit: int | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.get_recent_history, session_id, limit, None, None, user_id
        )

    async def async_purge_user(self, user_id: str) -> None:
        await asyncio.to_thread(self.purge_user, user_id)

    # MESSAGE COUNT

    def message_count(self, session_id: str) -> int:
        if not self._is_available():
            return 0
        try:

            def _do():
                return self.messages.count_documents({"session_id": session_id})

            return int(self._retry(_do))
        except Exception:
            return 0

    # HEALTH CHECK

    def health_check(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "mongo_ok": self._mongo_ok,
            "pymongo_available": _PYMONGO_AVAILABLE,
            "circuit_breaker": _PYBREAKER_AVAILABLE,
        }

        if self._is_available():
            assert (
                self.messages is not None
                and self.summaries is not None
                and self.sessions is not None
            )
            try:
                status["messages_count"] = self.messages.estimated_document_count()
                status["summaries_count"] = self.summaries.estimated_document_count()
                status["sessions_count"] = self.sessions.estimated_document_count()
                status["db_name"] = settings.MONGO_DB_NAME
            except Exception:
                status["count_error"] = True

        return status
