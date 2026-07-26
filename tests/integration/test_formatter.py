from app.memory.formatter import format_history

history = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assitant", "content": "AI is artificial intelligence."}
]

print(format_history(history))