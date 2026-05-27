"""Shared pytest fixtures for all unit tests — Phase 24.9."""
from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


# LLM MOCK

@pytest.fixture
def mock_llm() -> MagicMock:
    m = MagicMock()
    m.generate = MagicMock(return_value="Test answer from LLM.")
    return m


# EMBEDDER MOCK — returns 384-dim vectors (TEXT_EMBEDDING_DIM default)

@pytest.fixture
def mock_embedder() -> MagicMock:
    m = MagicMock()
    m.embed_query = MagicMock(return_value=[0.1] * 384)
    m.embed_documents = MagicMock(return_value=[[0.1] * 384] * 10)
    return m


# QDRANT MOCK

@pytest.fixture
def mock_qdrant() -> AsyncMock:
    m = AsyncMock()
    m.search = AsyncMock(
        return_value=[
            {"text": "result 1", "score": 0.9, "metadata": {"modality": "text", "page_number": 1}},
            {"text": "result 2", "score": 0.8, "metadata": {"modality": "text", "page_number": 2}},
        ]
    )
    return m


# SAMPLE CONVERSATION HISTORY

@pytest.fixture
def sample_history() -> List[Dict[str, Any]]:
    return [
        {"role": "user",      "content": "Hello there",          "timestamp": 1_000_000.0},
        {"role": "assistant", "content": "Hi there, how can I help?", "timestamp": 1_000_001.0},
        {"role": "user",      "content": "What is RAG?",          "timestamp": 1_000_002.0},
        {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation.", "timestamp": 1_000_003.0},
    ]


# SAMPLE HISTORY WITH EMBEDDINGS (pre-computed 384-dim)

@pytest.fixture
def sample_history_with_embeddings() -> List[Dict[str, Any]]:
    base = [0.1] * 384
    high = [0.9] * 384
    return [
        {"role": "user",      "content": "What is RAG?",   "timestamp": time.time(), "embedding": high},
        {"role": "assistant", "content": "RAG is a technique.", "timestamp": time.time(), "embedding": base},
        {"role": "user",      "content": "Tell me more.",   "timestamp": time.time(), "embedding": base},
    ]


# SIMPLE DOC OBJECT FOR CHUNKER TESTS

class _Doc:
    """Minimal document-like object for chunker tests."""
    def __init__(self, text: str, modality: str = "text", chunk_id: int = None):
        self.text     = text
        self.modality = modality
        self.chunk_id = chunk_id


@pytest.fixture
def doc_factory():
    return _Doc
