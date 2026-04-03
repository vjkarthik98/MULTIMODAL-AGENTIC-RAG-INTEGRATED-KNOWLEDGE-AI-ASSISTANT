from app.memory.redis_memory import RedisMemory

memory = RedisMemory()

session_id = "gradio_user"

history = memory.get_history(session_id)

print("\n=== MEMORY BEFORE PROMPT ===")
for msg in history:
    print(msg)

print("\n=== CURRENT MEMORY===\n")
for msg in history:
    print(msg)




