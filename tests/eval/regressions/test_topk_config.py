"""Regression: top-k config values must match Phase 24 tuned values.

Phase 24 fix: DEFAULT_TOP_K=8, VECTOR_TOP_K=15, RAG_TOP_K=8, RERANK_TOP_K=8
(was 5/10/5/5 — too small to surface correct finance chunks alongside noise)
"""
import pytest
from app.core.config import get_settings


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def test_chunk_size_is_1024(settings):
    assert settings.CHUNK_SIZE >= 1024, (
        f"CHUNK_SIZE={settings.CHUNK_SIZE} — Phase 24 fix set it to 1024 to prevent facts being cut mid-sentence"
    )


def test_chunk_overlap_is_128(settings):
    assert settings.CHUNK_OVERLAP >= 128, (
        f"CHUNK_OVERLAP={settings.CHUNK_OVERLAP} — Phase 24 fix set it to 128 (10-15% of 1024)"
    )


def test_chunk_overlap_less_than_chunk_size(settings):
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE, (
        "CHUNK_OVERLAP must be < CHUNK_SIZE"
    )


def test_default_top_k_at_least_8(settings):
    assert settings.DEFAULT_TOP_K >= 8, (
        f"DEFAULT_TOP_K={settings.DEFAULT_TOP_K} — Phase 24 fix raised to 8 to surface relevant chunks alongside noise"
    )


def test_vector_top_k_at_least_15(settings):
    assert settings.VECTOR_TOP_K >= 15, (
        f"VECTOR_TOP_K={settings.VECTOR_TOP_K} — Phase 24 fix raised to 15 for broad initial recall"
    )


def test_rag_top_k_at_least_8(settings):
    assert settings.RAG_TOP_K >= 8, (
        f"RAG_TOP_K={settings.RAG_TOP_K} — Phase 24 fix raised to 8"
    )


def test_rerank_top_k_at_least_8(settings):
    assert settings.RERANK_TOP_K >= 8, (
        f"RERANK_TOP_K={settings.RERANK_TOP_K} — Phase 24 fix raised to 8"
    )
