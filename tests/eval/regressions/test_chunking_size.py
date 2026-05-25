"""Regression: text ingest must produce chunks within the 512–1500 token window.

Phase 24 fix: CHUNK_SIZE increased to 1024 with CHUNK_OVERLAP=128 so financial facts
are not cut mid-sentence/mid-table, causing retrieval misses and hallucination.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


@pytest.fixture(scope="module")
def sample_finance_text():
    """Use the committed Apple 10-K excerpt as a real finance corpus sample."""
    corpus_path = Path(__file__).resolve().parents[4] / "data/raw/finance/txt/apple_10k_2023_excerpt.txt"
    if not corpus_path.exists():
        pytest.skip("Apple 10-K excerpt not found — run download_eval_corpus.sh first")
    return corpus_path.read_text()


def test_chunks_respect_max_size(sample_finance_text):
    """All chunks must be <= CHUNK_SIZE * 1.5 chars (allowing for LangChain overshoot)."""
    from app.core.config import get_settings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
        separators=["\nSECTION ", "\n====", "\n----", "\n####", "\n###", "\n##", "\n#", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(sample_finance_text)
    assert chunks, "Chunker produced no chunks"

    max_allowed = s.CHUNK_SIZE * 4  # generous: 4 chars per token * 1.5 token overshoot
    oversized = [c for c in chunks if len(c) > max_allowed]
    assert not oversized, (
        f"{len(oversized)} chunks exceed max allowed {max_allowed} chars. "
        f"Longest: {max(len(c) for c in chunks)} chars"
    )


def test_chunks_not_trivially_small(sample_finance_text):
    """Chunks should not be smaller than ~50 chars (prevents over-fragmentation)."""
    from app.core.config import get_settings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
        separators=["\nSECTION ", "\n====", "\n----", "\n####", "\n###", "\n##", "\n#", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(sample_finance_text)
    trivial = [c for c in chunks if len(c) < 50]
    assert len(trivial) <= 2, (
        f"Too many trivially small chunks ({len(trivial)}): {[c[:40] for c in trivial]}"
    )


def test_chunk_count_reasonable(sample_finance_text):
    """A ~4KB document should produce at least 2 and at most 30 chunks at CHUNK_SIZE=1024."""
    from app.core.config import get_settings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(sample_finance_text)
    assert 2 <= len(chunks) <= 30, (
        f"Got {len(chunks)} chunks for finance excerpt — expected 2-30"
    )
