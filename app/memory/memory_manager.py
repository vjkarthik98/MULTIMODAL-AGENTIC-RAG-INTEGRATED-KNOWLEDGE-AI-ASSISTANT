from __future__ import annotations

import asyncio
import hashlib
import time
import unicodedata
from collections import deque
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS


def _get_metrics():
    try:
        from prometheus_client import Counter, Gauge, Histogram

        memory_ops = Counter(
            "memory_operations_total",
            "Memory operations by type and store",
            ["operation", "store"],
        )
        memory_latency = Histogram(
            "memory_operation_latency_seconds",
            "Memory operation latency",
            ["operation"],
        )
        memory_size = Gauge(
            "memory_session_size",
            "Number of messages in memory per session",
        )
        return {
            "memory_ops": memory_ops,
            "memory_latency": memory_latency,
            "memory_size": memory_size,
        }
    except Exception:
        return {}


_METRICS: dict[str, Any] = {}

if settings.PROMETHEUS_ENABLED:
    try:
        _METRICS = _get_metrics()
    except Exception:
        pass


def _record_op(operation: str, store: str) -> None:
    try:
        if "memory_ops" in _METRICS:
            _METRICS["memory_ops"].labels(operation=operation, store=store).inc()
    except Exception:
        pass


def _record_latency(operation: str, latency: float) -> None:
    try:
        if "memory_latency" in _METRICS:
            _METRICS["memory_latency"].labels(operation=operation).observe(latency)
    except Exception:
        pass


def _set_memory_size(size: int) -> None:
    try:
        if "memory_size" in _METRICS:
            _METRICS["memory_size"].set(size)
    except Exception:
        pass


# NORMALIZE TEXT — SECTION 2.3


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("\x00", "")
    text = text.lstrip("\ufeff\ufffe")
    return " ".join(text.strip().split())


# MESSAGE HASH FOR DEDUP — SECTION 2.2


def _hash_msg(msg: dict[str, Any]) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content', ''))[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# CONTENT VALIDATOR


def _valid_content(content: str) -> bool:
    return isinstance(content, str) and len(content.strip()) > 2


# DEDUP MESSAGES


def _dedup(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set = set()
    out: list[dict[str, Any]] = []
    for msg in history:
        try:
            h = _hash_msg(msg)
            if h in seen:
                continue
            seen.add(h)
            out.append(msg)
        except Exception:
            continue
    return out


# SLIDING WINDOW — SECTION 4.7
# deque.appendleft() keeps chronological order in a single O(n) pass,
# eliminating the second list(reversed(...)) allocation of the prior version.


def _apply_sliding_window(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_tokens = settings.SLIDING_WINDOW_MAX_TOKENS
    total = 0
    window: deque = deque()

    for msg in reversed(history):
        content = str(msg.get("content", ""))
        token_est = max(1, len(content) // 4)
        if total + token_est > max_tokens:
            break
        window.appendleft(msg)
        total += token_est

    return list(window)


# MEMORY MANAGER CLASS


class MemoryManager:

    def __init__(
        self,
        redis_memory: Any | None = None,
        mongo_memory: Any | None = None,
    ) -> None:

        # LAZY RESOLUTION FROM INFRA REGISTRY
        if redis_memory is None:
            try:
                from app.core.infra_registry import infra

                redis_memory = infra.get_memory()
            except Exception as e:
                logger.warning(event="redis_memory_init_failed", error=str(e))
                redis_memory = None

        if mongo_memory is None:
            try:
                from app.core.infra_registry import infra

                mongo_memory = infra.get_mongo()
            except Exception as e:
                logger.warning(event="mongo_memory_init_failed", error=str(e))
                mongo_memory = None

        self.redis_memory = redis_memory
        self.mongo_memory = mongo_memory
        self._semaphore = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)

    # ADD MESSAGE — SECTION 4.7

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        modality: str = "text",
        importance: float = 1.0,
        embedding: list[float] | None = None,
        user_id: str | None = None,
    ) -> None:

        if not session_id:
            logger.warning(event="memory_add_skipped_no_session")
            return

        content = _normalize(content)
        if not _valid_content(content):
            return

        content = content[: settings.MAX_PROMPT_CHARS]
        importance = max(0.0, min(float(importance), 1.0))
        role = role.lower().strip()
        if role not in {"user", "assistant", "system"}:
            role = "user"

        message: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "modality": modality,
            "importance": importance,
            "user_id": user_id or None,
        }

        if embedding is not None and isinstance(embedding, list):
            message["embedding"] = embedding

        t_start = time.time()

        # REDIS WRITE — SECTION 4.7
        if self.redis_memory:
            try:
                self.redis_memory.append(session_id, message)
                _record_op("add_message", "redis")
            except Exception as e:
                logger.error(
                    event="redis_add_message_failed",
                    session_id=session_id,
                    error=str(e),
                )

        # MONGO WRITE — SECTION 4.7
        if self.mongo_memory:
            try:
                self.mongo_memory.insert(session_id, message)
                _record_op("add_message", "mongo")
            except Exception as e:
                logger.error(
                    event="mongo_add_message_failed",
                    session_id=session_id,
                    error=str(e),
                )

        _record_latency("add_message", round(time.time() - t_start, 3))

    # ADD INTERACTION — SECTION 4.7

    def add_interaction(
        self,
        session_id: str,
        query: str,
        response: str,
        modality: str = "text",
        user_id: str | None = None,
    ) -> None:
        try:
            self.add_message(
                session_id,
                "user",
                query,
                modality=modality,
                user_id=user_id,
            )
            self.add_message(
                session_id,
                "assistant",
                response,
                modality=modality,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(
                event="memory_interaction_failed",
                session_id=session_id,
                error=str(e),
            )

    # ASYNC ADD INTERACTION — SECTION 4.7

    async def add_interaction_async(
        self,
        session_id: str,
        query: str,
        response: str,
        modality: str = "text",
        user_id: str | None = None,
    ) -> None:
        async with self._semaphore:
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.add_interaction(session_id, query, response, modality, user_id),
            )

    # GET HISTORY — SECTION 4.7

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:

        limit = limit or settings.MAX_HISTORY_MESSAGES
        t_start = time.time()

        try:
            history: list[dict[str, Any]] = []
            source = "empty"

            # REDIS PRIMARY — SECTION 4.7
            if self.redis_memory:
                try:
                    data = self.redis_memory.get(session_id, user_id)
                    if isinstance(data, list) and data:
                        history = data
                        source = "redis"
                        _record_op("get_history", "redis")
                except Exception as e:
                    logger.warning(
                        event="redis_history_fetch_failed",
                        session_id=session_id,
                        error=str(e),
                    )

            # MONGO FALLBACK OR MERGE — SECTION 4.7
            if self.mongo_memory:
                try:
                    mongo_data = self.mongo_memory.get(session_id, user_id)
                    if isinstance(mongo_data, list) and mongo_data:
                        if not history:
                            history = mongo_data
                            source = "mongo"
                            _record_op("get_history", "mongo")
                        else:
                            # MERGE + DEDUP — SECTION 4.7
                            combined = history + mongo_data
                            history = _dedup(combined)
                            source = "merged"
                            _record_op("get_history", "merged")
                except Exception as e:
                    logger.warning(
                        event="mongo_history_fetch_failed",
                        session_id=session_id,
                        error=str(e),
                    )

            if not history:
                if source == "empty":
                    logger.info(
                        event="memory_context_empty_all_services_unavailable",
                        session_id=session_id,
                    )
                return []

            history = _dedup(history)
            history = _apply_sliding_window(history)
            history = history[-limit:]

            _set_memory_size(len(history))
            _record_latency("get_history", round(time.time() - t_start, 3))

            logger.debug(
                event="memory_history_fetched",
                source=source,
                count=len(history),
                session_id=session_id,
            )

            return history

        except Exception as e:
            logger.warning(
                event="memory_fetch_failed",
                session_id=session_id,
                error=str(e),
            )
            return []

    # ASYNC GET HISTORY — SECTION 4.7

    async def get_history_async(
        self,
        session_id: str,
        limit: int | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        async with self._semaphore:
            return await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.get_history(session_id, limit, user_id),
            )

    # CLEAR SESSION — SECTION 4.7

    def clear(self, session_id: str, user_id: str | None = None) -> None:
        # Both RedisMemory.delete() and MongoMemory.delete() refuse an
        # unscoped delete when user_id is missing (tenant-isolation guard) —
        # without it this silently no-ops on both stores while still logging
        # "memory_cleared" as if it worked. Caller MUST pass the
        # authenticated user's user_id.
        cleared: list[str] = []
        t_start = time.time()

        try:
            if self.redis_memory:
                self.redis_memory.delete(session_id, user_id)
                cleared.append("redis")
                _record_op("clear", "redis")

            if self.mongo_memory:
                self.mongo_memory.delete(session_id, user_id)
                cleared.append("mongo")
                _record_op("clear", "mongo")

            _record_latency("clear", round(time.time() - t_start, 3))

            logger.info(
                event="memory_cleared",
                stores=cleared,
                session_id=session_id,
                user_id=user_id,
            )

        except Exception as e:
            logger.error(
                event="memory_clear_failed",
                session_id=session_id,
                error=str(e),
            )

    # GDPR PURGE — SECTION 4.7 / SECTION 5

    def gdpr_purge(self, user_id: str) -> dict[str, Any]:
        purged: dict[str, Any] = {
            "user_id": user_id,
            "redis": False,
            "mongo": False,
            "errors": [],
            "purged_at": time.time(),
        }

        logger.info(event="gdpr_purge_start", user_id=user_id)

        # REDIS PURGE — delete all keys matching user prefix.
        # NOTE: this used to call self.redis_memory.delete(user_id) — that's
        # RedisMemory.delete(session_id, user_id=None), so it was passing
        # user_id as the *session_id* positional arg with no actual user_id,
        # which always hit the "refusing an unscoped key" guard and silently
        # no-op'd. The manual .keys()/.delete() block right below was a
        # working duplicate of what purge_user() is supposed to do — now
        # that purge_user() is fixed (branches on Upstash vs real redis) and
        # reports an accurate count, call it directly instead of
        # reimplementing it here.
        if self.redis_memory:
            try:
                redis_purged_count = self.redis_memory.purge_user(user_id)
                purged["redis"] = True
                purged["redis_keys_deleted"] = redis_purged_count
                _record_op("gdpr_purge", "redis")
            except Exception as e:
                purged["errors"].append(f"redis: {e}")
                logger.error(event="gdpr_purge_redis_failed", user_id=user_id, error=str(e))

        # MONGO PURGE — SECTION 4.7 / SECTION 5
        # NOTE: must filter by user_id (not session_id) — clear_memory(session_id, ...)
        # takes a session id as its first arg, so calling it with user_id would query
        # {"session_id": user_id}, which never matches real session ids and silently
        # purges nothing. purge_user(user_id) is the correctly user-scoped method —
        # it covers messages, summaries AND chat_sessions (Recents transcripts).
        if self.mongo_memory:
            try:
                if hasattr(self.mongo_memory, "purge_user"):
                    self.mongo_memory.purge_user(user_id)
                else:
                    self.mongo_memory.delete(user_id)
                purged["mongo"] = True
                _record_op("gdpr_purge", "mongo")
            except Exception as e:
                purged["errors"].append(f"mongo: {e}")
                logger.error(event="gdpr_purge_mongo_failed", user_id=user_id, error=str(e))

        logger.info(
            event="gdpr_purge_complete",
            user_id=user_id,
            redis_purged=purged["redis"],
            mongo_purged=purged["mongo"],
            errors=purged["errors"],
        )

        return purged

    # ASYNC GDPR PURGE — SECTION 4.7

    async def gdpr_purge_async(self, user_id: str) -> dict[str, Any]:
        async with self._semaphore:
            return await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.gdpr_purge(user_id),
            )

    # SUMMARIZE AND COMPRESS — SECTION 4.7

    def summarize_and_compress(
        self,
        session_id: str,
        llm: Any,
        user_id: str | None = None,
    ) -> str | None:
        try:
            from app.memory.summarizer import summarize_conversation

            history = self.get_history(session_id, user_id=user_id)
            if not history:
                return None

            summary = summarize_conversation(
                llm=llm,
                history=history,
                session_id=session_id,
                mongo_memory=self.mongo_memory,
                user_id=user_id,
            )

            logger.info(
                event="memory_summarized",
                session_id=session_id,
                summary_len=len(summary) if summary else 0,
            )

            return summary

        except Exception as e:
            logger.error(
                event="memory_summarize_failed",
                session_id=session_id,
                error=str(e),
            )
            return None

    # GET LAST K MESSAGES — SECTION 4.7

    def get_last_k(
        self,
        session_id: str,
        k: int,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        history = self.get_history(session_id, limit=k, user_id=user_id)
        return history[-k:] if history else []

    # GET CONTEXT AS STRING — SECTION 4.7
    # Always returns a string (may be empty). Never raises.

    def get_context(
        self, session_id: str, limit: int | None = None, user_id: str | None = None
    ) -> str:
        try:
            history = self.get_history(session_id, limit=limit, user_id=user_id)
            if not history:
                return ""
            parts: list[str] = []
            for msg in history:
                role = msg.get("role", "user").capitalize()
                content = str(msg.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}")
            return "\n".join(parts)
        except Exception as e:
            logger.warning(event="memory_get_context_failed", session_id=session_id, error=str(e))
            return ""

    # STORE TURN — SECTION 4.7
    # Always succeeds as no-op if services unavailable. Never raises.

    def store_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        modality: str = "text",
    ) -> None:
        try:
            self.add_interaction(session_id, query, response, modality=modality)
        except Exception as e:
            logger.warning(event="memory_store_turn_failed", session_id=session_id, error=str(e))

    # MEMORY SIZE — SECTION 4.7

    def get_memory_size(self, session_id: str, user_id: str | None = None) -> int:
        if self.redis_memory and hasattr(self.redis_memory, "get_memory_size"):
            try:
                return self.redis_memory.get_memory_size(session_id, user_id)
            except Exception:
                pass
        # Use raw document count — get_history() applies sliding-window trimming
        # which causes the count to never land on an exact multiple, breaking the
        # modulo-based summarization trigger.
        if self.mongo_memory and hasattr(self.mongo_memory, "message_count"):
            try:
                raw = self.mongo_memory.message_count(session_id, user_id)
                if raw > 0:
                    return raw
            except Exception:
                pass
        return len(self.get_history(session_id, user_id=user_id))

    # HEALTH CHECK — SECTION 4.7

    def health_check(self) -> dict[str, Any]:
        redis_health: dict[str, Any] = {}
        mongo_health: dict[str, Any] = {}

        if self.redis_memory and hasattr(self.redis_memory, "health_check"):
            try:
                redis_health = self.redis_memory.health_check()
            except Exception as e:
                redis_health = {"error": str(e)}

        if self.mongo_memory and hasattr(self.mongo_memory, "health_check"):
            try:
                mongo_health = self.mongo_memory.health_check()
            except Exception as e:
                mongo_health = {"error": str(e)}

        return {
            "redis_available": self.redis_memory is not None,
            "mongo_available": self.mongo_memory is not None,
            "redis_health": redis_health,
            "mongo_health": mongo_health,
        }
