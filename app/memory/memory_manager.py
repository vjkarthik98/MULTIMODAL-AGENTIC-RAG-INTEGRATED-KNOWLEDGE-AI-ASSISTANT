from typing import Dict, List
import time

from app.core.config import settings
from app.memory.redis_memory import RedisMemory
from app.memory.mongo_memory import MongoMemory
from app.memory.summarizer import summarize_conversation
from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.utils.logger import get_logger


logger = get_logger(__name__)


class MemoryManager:
    def __init__(self, llm, embedder):
        self.redis = RedisMemory()
        self.mongo = MongoMemory()

        self.llm = llm
        self.embedder = embedder

        self.summary_trigger = settings.MEMORY_SUMMARY_THRESHOLD

    # Add interaction
    def add_interaction(
        self,
        session_id: str,
        query: str,
        response: str
    ) -> Dict:
        if not query or not response:
            logger.warning("[MemoryManager] Empty interaction skipped")
            return {"summarized": False}

        try:
            logger.info(f"[MemoryManager][ADD] session_id={session_id}")

            # Safe truncation before embedding
            query = query[:settings.MEMORY_MAX_CONTEXT_CHARS]
            response = response[:settings.MEMORY_MAX_CONTEXT_CHARS]

            # Embeddings (fail-safe)
            try:
                query_emb = self.embedder.embed_query(query)
                response_emb = self.embedder.embed_query(response)
            except Exception as e:
                logger.warning(f"[MemoryManager] Embedding failed | {str(e)}")
                query_emb = None
                response_emb = None

            # Redis (short-term)
            self.redis.add_message(
                session_id,
                role="user",
                content=query,
                embedding=query_emb
            )

            self.redis.add_message(
                session_id,
                role="assistant",
                content=response,
                embedding=response_emb
            )

            # Mongo (long-term)
            self.mongo.store_message(
                session_id,
                role="user",
                content=query,
                embedding=query_emb
            )

            self.mongo.store_message(
                session_id,
                role="assistant",
                content=response,
                embedding=response_emb
            )

            # Check summarization
            history = self.redis.get_history(session_id)

            if len(history) >= self.summary_trigger:
                return self._summarize_and_reset(session_id, history)

            return {"summarized": False}

        except Exception as e:
            logger.error(f"[MemoryManager][ADD_FAIL] {str(e)}")
            raise

    # Summarization flow
    def _summarize_and_reset(self, session_id: str, history: List[Dict]) -> Dict:
        try:
            logger.info("[MemoryManager] Triggering summarization")

            start = time.time()

            summary = summarize_conversation(self.llm, history)

            latency = round(time.time() - start, 2)
            logger.info(f"[MemoryManager] Summary generated | {latency}s")

            # Store summary
            self.mongo.store_summary(session_id, summary)

            # Reset Redis and inject summary
            self.redis.clear_memory(session_id)

            self.redis.add_message(
                session_id,
                role="system",
                content=summary
            )

            return {
                "summarized": True,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"[MemoryManager][SUMMARY_FAIL] {str(e)}")
            return {"summarized": False}

    # Get memory context
    def get_memory_context(self, session_id: str, query: str) -> str:
        try:
            logger.info(f"[MemoryManager][FETCH] session_id={session_id}")

            recent_history = self.redis.get_history(session_id)
            summary = self.mongo.get_latest_summary(session_id)

            filtered = filter_relevant_history(
                query=query,
                history=recent_history,
                embedder=self.embedder
            )

            context = build_memory_context(
                summary=summary,
                filtered_history=filtered
            )

            return context[:settings.MEMORY_MAX_CONTEXT_CHARS]

        except Exception as e:
            logger.error(f"[MemoryManager][FETCH_FAIL] {str(e)}")
            return ""

    # Debug snapshot
    def get_debug_snapshot(self, session_id: str) -> Dict:
        return {
            "redis_size": self.redis.get_memory_size(session_id),
            "recent_messages": self.redis.get_last_k(session_id, 5),
            "summary": self.mongo.get_latest_summary(session_id)
        }