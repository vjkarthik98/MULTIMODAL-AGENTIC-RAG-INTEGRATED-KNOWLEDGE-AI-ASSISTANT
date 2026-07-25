from app.memory.summarizer import summarize_conversation
from app.llm.gguf_model import GGUFModel

# Step 1: Initialize LLM
llm = GGUFModel()

# Step 2: Dummy History
history = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assitant", "content": "AI is artificial intelligence."},
    {"role": "user", "content": "Explain Machine Learning"},
    {"role": "assitant", "content": "ML is a subset of AI."}
]


summary = summarize_conversation(llm, history)
print("\nSUMMARY:\n", summary)