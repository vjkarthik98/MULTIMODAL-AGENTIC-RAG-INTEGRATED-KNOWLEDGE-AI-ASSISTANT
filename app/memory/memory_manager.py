from typing import Dict, List
from app.memory.redis_memory import RedisMemory
from app.memory.mongo_memory import MongoMemory
from app.memory.summarizer import summarize_conversation
from app.memory.memory_filter import filter_relevant_history
from app.memory.memory_fusion import build_memory_context
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


class MemoryManager:
    def __init__(self, llm, embedder, max_messages=20, summary_trigger=10):
        self.redis = RedisMemory(max_messages=max_messages)
        self.mongo = MongoMemory()
        self.llm = llm
        self.embedder = embedder
        self.summary_trigger = summary_trigger


    # ADD INTERACTION
    def add_interaction(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str
    ) -> Dict:
        
        try:
            logger.info(f"[MemoryManager][ADD] session_id={session_id}")

            # STEP 1: Embed Messages 
            user_emb = self.embedder.embed_query(user_msg)
            assistant_emb = self.embedder.embed_query(assistant_msg)

            # STEP 2: Store in Redis (short-term)
            self.redis.add_message(
                session_id,
                role="user",
                content=user_msg,
                embedding=user_emb
            )

            self.redis.add_message(
                session_id,
                role="assistant",
                content=assistant_msg,
                embedding=assistant_emb
            )

            # STEP 3: Store in Mongo (long-term)
            self.mongo.store_message(
                session_id,
                role="user",
                content=user_msg,
                embedding=user_emb
            )

            self.mongo.store_message(
                session_id,
                role="assistant",
                content=assistant_msg,
                embedding=assistant_emb
            )

            # STEP 4: Check summarization trigger
            history = self.redis.get_history(session_id)

            if len(history) >= self.summary_trigger:
                logger.info("[MemoryManager] Triggering summarization")

                summary  = summarize_conversation(self.llm, history)

                # Store summary in Mongo
                self.mongo.store_summary(session_id, summary)

                # Reset Redis but keep summary
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
            
            return {"summarized": False}
        
        except Exception as e:
            logger.error(f"[MemoryManager][ADD_FAIL] {str(e)}")
            raise

    # GET MEMORY CONTEXT 
    def get_memory_context(self, session_id: str, query: str) -> str:
        try:
            logger.info(f"[MemoryManager][FETCH] session_id={session_id}")

            # STEP 1: Get Short-term memory
            recent_history = self.redis.get_history(session_id)

            # STEP 2: Get long-term summary
            summary = self.mongo.get_latest_summary(session_id)

            # STEP 3: Filter relevant messages
            filtered = filter_relevant_history(
                query=query,
                history=recent_history,
                embedder=self.embedder
            )

            # STEP 4: Build Final Context
            memory_context = build_memory_context(
                summary=summary,
                filtered_history=filtered
            )

            return memory_context
        
        except Exception as e:
            logger.error(f"[MemoryManager][FETCH_FAIL] {str(e)}")
            return ""
    
    # DEBUG / OBSERVABILITY
    def get_debug_snapshot(self, session_id: str) -> Dict:
        return {
            "redis_size": self.redis.get_memory_size(session_id),
            "recent_messages": self.redis.get_last_k(session_id, 5),
            "summary": self.mongo.get_latest_summary(session_id)
        }

    