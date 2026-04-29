from app.chunking.chunker import chunk_text

text = """
Artificial intelligence is transforming industries at a rapid pace. 
It enables machines to learn from data and improve over time. 
Machine learning, a subset of AI, focuses on building models that can generalize from data. 
Deep learning, a further subset, uses neural networks with many layers.

In healthcare, AI is used for diagnosis, drug discovery, and personalized medicine. 
In finance, it helps in fraud detection, algorithmic trading, and risk management. 
Automation powered by AI is also transforming manufacturing and logistics.

Despite its benefits, AI also raises ethical concerns such as bias, transparency, and job displacement. 
Researchers and policymakers are working to ensure responsible AI development."""

chunks = chunk_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk}\n")