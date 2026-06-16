"""Integration test — text chunking via TxtChunker."""
from app.chunking.txt_chunker import TxtChunker
from app.ingestion.schema import IngestedDocument

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


def test_txt_chunker_splits_text():
    doc = IngestedDocument(
        text=text,
        metadata={
            "source_file": "test.txt",
            "modality": "txt",
            "user_id": "test_user",
            "session_id": "test_session",
        },
    )
    chunker = TxtChunker()
    chunks = chunker.chunk([doc])
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.text.strip()
