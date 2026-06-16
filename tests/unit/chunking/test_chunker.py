"""Unit tests for app/chunking/chunker.py — Phase 24.9."""
from __future__ import annotations

import pytest

from app.chunking.base_chunker import chunk_documents


# ── Minimal doc-like object ───────────────────────────────────────────────────

class _Doc:
    def __init__(self, text: str, modality: str = "text", chunk_id=None):
        self.text     = text
        self.modality = modality
        self.chunk_id = chunk_id


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChunkDocuments:

    def test_returns_list(self):
        docs = [_Doc("Hello world this is some text.")]
        result = chunk_documents(docs)
        assert isinstance(result, list)

    def test_valid_docs_pass_through(self):
        docs = [
            _Doc("First document text content here."),
            _Doc("Second document content here."),
        ]
        result = chunk_documents(docs)
        assert len(result) == 2

    def test_empty_document_skipped(self):
        docs = [_Doc(""), _Doc("Valid content here.")]
        result = chunk_documents(docs)
        assert len(result) == 1

    def test_whitespace_only_document_skipped(self):
        docs = [_Doc("   \n\t  "), _Doc("Valid content.")]
        result = chunk_documents(docs)
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        result = chunk_documents([])
        assert result == []

    def test_chunk_id_assigned_when_missing(self):
        doc = _Doc("Some text content here.", chunk_id=None)
        result = chunk_documents([doc])
        assert len(result) == 1
        assert result[0].chunk_id is not None

    def test_chunk_id_preserved_when_set(self):
        doc = _Doc("Some text.", chunk_id=99)
        result = chunk_documents([doc])
        assert result[0].chunk_id == 99

    def test_chunk_id_assigned_sequentially(self):
        docs = [_Doc(f"Document number {i}") for i in range(3)]
        result = chunk_documents(docs)
        ids = [r.chunk_id for r in result]
        assert ids == list(range(3))

    def test_modality_breakdown_tracked(self, caplog):
        docs = [
            _Doc("Text doc one.", modality="text"),
            _Doc("Image doc.", modality="image"),
            _Doc("Another text.", modality="text"),
        ]
        result = chunk_documents(docs)
        assert len(result) == 3

    def test_token_estimate_is_positive(self):
        docs = [_Doc("The quick brown fox jumps over the lazy dog.")]
        chunk_documents(docs)

    def test_all_empty_returns_empty(self):
        docs = [_Doc(""), _Doc("   "), _Doc("\n")]
        result = chunk_documents(docs)
        assert result == []

    def test_preserves_modality_attribute(self):
        docs = [
            _Doc("audio transcript text.", modality="audio"),
            _Doc("pdf page text.", modality="pdf"),
        ]
        result = chunk_documents(docs)
        modalities = {r.modality for r in result}
        assert "audio" in modalities
        assert "pdf" in modalities

    def test_mixed_valid_and_empty(self):
        docs = [
            _Doc("Valid first document content."),
            _Doc(""),
            _Doc("Valid second document content."),
            _Doc("   "),
        ]
        result = chunk_documents(docs)
        assert len(result) == 2

    def test_doc_without_text_attribute_skipped(self):
        class _NoText:
            modality = "text"
            chunk_id = None
        result = chunk_documents([_NoText()])
        assert result == []
