from app.memory.memory_fusion import build_memory_context

summary = "User discussed AI and ML."

history = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is artificial intelligence."}
]

context = build_memory_context(summary, history)

print(context)