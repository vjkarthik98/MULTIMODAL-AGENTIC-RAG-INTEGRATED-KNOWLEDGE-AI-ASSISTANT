from app.memory.redis_memory import RedisMemory
from app.memory.summarizer import summarize_conversation
import json


class MemoryManager:
    def __init__(self, llm, max_messages=10, summary_trigger=8):
        self.memory = RedisMemory(max_messages=max_messages)
        self.llm = llm
        self.summary_trigger = summary_trigger

    def get_history(self, session_id):
        return self.memory.get_history(session_id)
    
    def add_interaction(self, session_id, user_msg, assistant_msg):
        """
        Add new interaction and trigger summarization if needed.
        """
        # Step 1: Add messages
        self.memory.add_message(session_id, "user", user_msg)
        self.memory.add_message(session_id, "assistant", assistant_msg)

        # Step 2: Get updated history
        history = self.memory.get_history(session_id)

        print(f"[MemoryManager] History length: {len(history)}")

        # Step 3: trigger summarization
        if len(history) >= self.summary_trigger:
            print("[MemoryManager]  Trigerring summarization....")
            summary = summarize_conversation(self.llm, history)

            # Clear old memory
            self.memory.clear_memory(session_id)

            # Store summary as system message
            self.memory.add_message(session_id, "system", summary)

            return {
                "summarized": True,
                "summary": summary
            }
        return {
            "summarized": False
        }
        
