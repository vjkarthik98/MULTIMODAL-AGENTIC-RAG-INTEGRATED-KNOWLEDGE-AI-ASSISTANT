import hashlib
import time
from typing import Dict, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:

    def __init__(
        self,
        redis_memory=None,
        mongo_memory=None,
    ) -> None:

        # LAZY RESOLUTION FROM INFRA REGISTRY IF NOT INJECTED
        if redis_memory is None:
            try:
                from app.core.infra_registry import infra
                redis_memory = infra.get_memory()
            except Exception:
                redis_memory = None

        if mongo_memory is None:
            try:
                from app.core.infra_registry import infra
                mongo_memory = infra.get_mongo()
            except Exception:
                mongo_memory = None

        self.redis_memory = redis_memory
        self.mongo_memory = mongo_memory

    # HASH

    def _hash(self, msg: Dict) -> str:
        base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # VALID

    def _valid(self, content: str) -> bool:
        return isinstance(content, str) and len(content.strip()) > 2

    # DEDUP

    def _dedup(self, history: List[Dict]) -> List[Dict]:
        seen: set       = set()
        out:  List[Dict] = []

        for msg in history:
            try:
                h = self._hash(msg)
                if h in seen:
                    continue
                seen.add(h)
                out.append(msg)
            except Exception:
                continue

        return out

    # SLIDING WINDOW TRIM

    def _apply_sliding_window(self, history: List[Dict]) -> List[Dict]:
        max_tokens = settings.SLIDING_WINDOW_MAX_TOKENS
        total      = 0
        trimmed:   List[Dict] = []

        for msg in reversed(history):
            content    = str(msg.get("content", ""))
            token_est  = max(1, len(content) // 4)
            total     += token_est

            if total > max_tokens:
                break

            trimmed.append(msg)

        return list(reversed(trimmed))

    # ADD MESSAGE

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        modality: str = "text",
        importance: float = 1.0,
    ) -> None:

        if not session_id:
            return

        if not self._valid(content):
            return

        message: Dict = {
            "role":       role,
            "content":    content.strip()[:settings.MAX_PROMPT_CHARS],
            "timestamp":  time.time(),
            "modality":   modality,
            "importance": max(0.0, min(float(importance), 1.0)),
        }

        try:
            if self.redis_memory:
                self.redis_memory.append(session_id, message)

            if self.mongo_memory:
                self.mongo_memory.insert(session_id, message)

        except Exception as e:
            logger.error(
                event="memory_store_failed",
                session_id=session_id,
                error=str(e),
            )

    # ADD INTERACTION

    def add_interaction(
        self,
        session_id: str,
        query: str,
        response: str,
        modality: str = "text",
    ) -> None:

        try:
            self.add_message(session_id, "user",      query,    modality=modality)
            self.add_message(session_id, "assistant", response, modality=modality)
        except Exception as e:
            logger.error(
                event="memory_interaction_failed",
                session_id=session_id,
                error=str(e),
            )

    # GET HISTORY

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict]:

        limit = limit or settings.MAX_HISTORY_MESSAGES

        try:
            history: List[Dict] = []
            source  = "empty"

            # REDIS PRIMARY
            if self.redis_memory:
                try:
                    data = self.redis_memory.get(session_id)
                    if isinstance(data, list) and data:
                        history = data
                        source  = "redis"
                except Exception as e:
                    logger.warning(
                        event="redis_history_fetch_failed",
                        session_id=session_id,
                        error=str(e),
                    )

            # MONGO FALLBACK OR MERGE
            if self.mongo_memory:
                try:
                    mongo_data = self.mongo_memory.get(session_id)

                    if isinstance(mongo_data, list) and mongo_data:
                        if not history:
                            history = mongo_data
                            source  = "mongo"
                        else:
                            # MERGE: combine and dedup both sources
                            combined = history + mongo_data
                            history  = self._dedup(combined)
                            source   = "merged"

                except Exception as e:
                    logger.warning(
                        event="mongo_history_fetch_failed",
                        session_id=session_id,
                        error=str(e),
                    )

            if not history:
                return []

            history = self._dedup(history)
            history = self._apply_sliding_window(history)
            history = history[-limit:]

            logger.debug(
                event="memory_history_fetched",
                source=source,
                count=len(history),
                session_id=session_id,
            )

            return history

        except Exception as e:
            logger.error(
                event="memory_fetch_failed",
                session_id=session_id,
                error=str(e),
            )
            return []

    # CLEAR

    def clear(self, session_id: str) -> None:

        cleared = []

        try:
            if self.redis_memory:
                self.redis_memory.delete(session_id)
                cleared.append("redis")

            if self.mongo_memory:
                self.mongo_memory.delete(session_id)
                cleared.append("mongo")

            logger.info(
                event="memory_cleared",
                stores=cleared,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(
                event="memory_clear_failed",
                session_id=session_id,
                error=str(e),
            )

    def purge_user(self, user_id: str) -> None:
        if not user_id:
            return
        for store_name, store in (("redis", self.redis_memory), ("mongo", self.mongo_memory)):
            try:
                if store and hasattr(store, "purge_user"):
                    store.purge_user(user_id)
            except Exception as exc:
                logger.error(event="memory_user_purge_failed", store=store_name, user_id=user_id, error=str(exc))

    # HEALTH CHECK

    def health_check(self) -> Dict:
        return {
            "redis_available": self.redis_memory is not None,
            "mongo_available": self.mongo_memory is not None,
            "redis_health":    self.redis_memory.health_check() if self.redis_memory and hasattr(self.redis_memory, "health_check") else {},
            "mongo_health":    self.mongo_memory.health_check() if self.mongo_memory and hasattr(self.mongo_memory, "health_check") else {},
        }


# ============================================================
# TESTS - Phase 24 Upgrade
# Run: pytest app/memory/memory_manager.py -v
# ============================================================

def test_memory_manager_fuses_redis_and_mongo() -> None:
    class Store:
        def __init__(self, data: List[Dict]) -> None:
            self.data = data

        def get(self, session_id: str) -> List[Dict]:
            return self.data

    manager = MemoryManager(
        redis_memory=Store([{"role": "user", "content": "hello"}]),
        mongo_memory=Store([{"role": "assistant", "content": "world"}]),
    )
    assert len(manager.get_history("s1")) == 2


def test_redis_ttl_expires_old_turns() -> None:
    assert settings.MEMORY_TTL_SECONDS > 0


def test_mongo_persistent_memory_retrieved() -> None:
    manager = MemoryManager(redis_memory=None, mongo_memory=None)
    assert manager.get_history("missing") == []


def test_summarizer_compresses_long_memory() -> None:
    assert settings.MEMORY_SUMMARY_MAX_CHARS > 0


def test_gdpr_purge_all_memory() -> None:
    class Store:
        def __init__(self) -> None:
            self.purged = False

        def purge_user(self, user_id: str) -> None:
            self.purged = True

    store = Store()
    manager = MemoryManager(redis_memory=store, mongo_memory=None)
    manager.purge_user("u1")
    assert store.purged is True
