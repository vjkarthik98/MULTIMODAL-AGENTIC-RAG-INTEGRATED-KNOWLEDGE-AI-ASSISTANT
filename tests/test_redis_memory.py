from app.memory.redis_memory import RedisMemory

memory = RedisMemory()

session_id = "test_user"

memory.add_message(session_id, "user", "Hello")
memory.add_message(session_id, "assistant", "Hi there!")


history = memory.get_history(session_id)

print(history)
