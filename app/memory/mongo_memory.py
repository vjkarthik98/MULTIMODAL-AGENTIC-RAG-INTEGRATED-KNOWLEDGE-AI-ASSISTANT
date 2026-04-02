from pymongo import MongoClient
from datetime import datetime

class MongoMemory:
    def __init__(self, uri="mongodb://localhost:27017"):
        self.client = MongoClient(uri)
        self.db = self.client["rag_memory"]
        self.collection = self.db["chat_history"]

    def store_message(self, session_id: str, role: str, content: str):
        self.collection.insert_one({
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        })
    
    def get_history(self, session_id: str, limit: int = 50):
        cursor = self.collection.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit)

        history = []
        for doc in reversed(list(cursor)):
            history.append({
                "role": doc["role"],
                "content": doc["content"]
            })
        return history
