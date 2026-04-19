from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from typing import List, Dict, Optional
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


class MongoMemory:
    def __init__(self, uri="mongodb://localhost:27017"):
        logger.info(f"[MongoMemory] Connecting to MongoDB | uri={uri}")

        self.client = MongoClient(uri)
        self.db = self.client["rag_memory"]

        # separate Collections
        self.messages = self.db["messages"]
        self.summaries = self.db["summaries"]

        self._ensure_indexes()

        logger.info("[MongoMemory] Initialized Successfully")

    # INDEXES
    def _ensure_indexes(self):
        self.messages.create_index([("session_id", ASCENDING)])
        self.messages.create_index([("timestamp", DESCENDING)])
        self.messages.create_index([("importance", DESCENDING)])

        self.summaries.create_index([("session_id", ASCENDING)])


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
        try:
            doc = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow(),
                "modality": modality,
                "importance": importance
            }

            if embedding:
                doc["embedding"] = embedding

            if extra:
                doc["extra"] = extra

            self.messages.insert_one(doc)

            logger.debug(
                f"[MongoMemory] Stored | session_id={session_id} | role={role}"
            )

        except Exception as e:
            logger.error(
                f"[MongoMemory] Store failed | session_id={session_id} | {str(e)}"
            )
            raise

    # STORE SUMMARY
    def store_summary(
        self,
        session_id: str,
        summary: str,
        embedding: Optional[List[float]] = None
    ):
        
        try:
            doc = {
                "session_id": session_id,
                "summary": summary,
                "timestamp": datetime.utcnow()
            }

            if embedding:
                doc["embedding"] = embedding

            self.summaries.insert_one(doc)

            logger.info(
                f"[MongoMemory] Summary stored | session_id={session_id}"
            )

        except Exception as e:
            logger.error(
                f"[MongoMemory] Summary failed | session_id={session_id} | {str(e)}"
            )
            raise

    
    # GET RECENT HISTORY
    def get_recent_history(
            self,
            session_id: str,
            limit: int = 20
    ) -> List[Dict]:
        try:
            cursor = self.messages.find(
                {"session_id": session_id}
            ).sort("timestamp", DESCENDING).limit(limit)

            history = []
            for doc in reversed(list(cursor)):
                history.append({
                    "role": doc["role"],
                    "content": doc["content"],
                    "embedding": doc.get("embedding"),
                    "modality": doc.get("modality")
                })

            return history
        
        except Exception as e:
            logger.error(
                f"[MongoMemory] Fetch recent failed | {str(e)}"
            )
            return []

    # GET LATEST SUMMARY
    def get_latest_summary(self, session_id: str) -> str:
        try:
            doc = self.summaries.find_one(
                {"session_id": session_id},
                sort=[("timestamp", DESCENDING)]
            )

            if doc:
                return doc.get("summary", "")
            
            return ""
        
        except Exception as e:
            logger.error(
                f"[MongoMemory] Summary fetch failed | {str(e)}"
            )
            return ""
        
    # CLEAR MEMORY
    def clear_memory(self, session_id: str):
        try:
            self.messages.delete_many({"session_id": session_id})
            self.summaries.delete_many({"session_id": session_id})

            logger.info(
                f"[MongoMemory] Cleared | session_id={session_id}"
            )

        except Exception as e:
            logger.error(
                f"[MongoMemory] Clear failed | {str(e)}"

            )
            raise