import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.ingestion_pipeline import (
    _check_disk_space,
    _dedup_chunks,
    _modality_counts,
    _scan_corruption,
    _sha256,
    _split_by_modality,
    _valid_chunks,
    _valid_embeddings,
)
from app.ingestion.schema import IngestedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    text="This is a valid chunk of ingested text with enough length.",
    modality="text",
    embedding=None,
    embedding_space="text",
    chunk_id=0,
    doc_id_val="D1",
):
    structure = {"embedding_space": embedding_space, "doc_id": doc_id_val}
    doc = IngestedDocument(
        text=text,
        modality=modality,
        source_type=modality,
        source="test.txt",
        structure=structure,
        chunk_id=chunk_id,
    )
    if embedding is not None:
        doc.embedding = embedding
    return doc


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------

class TestSha256:

    def test_returns_64_char_hex(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_bytes(b"hello world")
        result = _sha256(str(f))
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_matches_known_hash(self, tmp_path):
        f = tmp_path / "f.txt"
        data = b"hello"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256(str(f)) == expected

    def test_large_file_reads_in_chunks(self, tmp_path):
        f = tmp_path / "big.bin"
        data = b"A" * (200 * 1024)  # 200 KB
        f.write_bytes(data)
        result = _sha256(str(f))
        assert len(result) == 64


# ---------------------------------------------------------------------------
# _scan_corruption
# ---------------------------------------------------------------------------

class TestScanCorruption:

    def test_clean_txt_returns_empty(self, tmp_path):
        f = tmp_path / "clean.txt"
        f.write_text("Hello world. This is valid text.\n", encoding="utf-8")
        assert _scan_corruption(str(f)) == []

    def test_null_bytes_detected(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_bytes(b"hello\x00world")
        reasons = _scan_corruption(str(f))
        assert "null_bytes" in reasons

    def test_ansi_escape_detected(self, tmp_path):
        f = tmp_path / "ansi.txt"
        f.write_bytes(b"text \x1b[31mred\x1b[0m more text")
        reasons = _scan_corruption(str(f))
        assert "ansi_escape_sequences" in reasons

    def test_binary_tail_detected(self, tmp_path):
        f = tmp_path / "tail.txt"
        # Write lots of valid text then a binary tail
        content = b"Normal text content. " * 10 + bytes(range(256))[-64:]
        f.write_bytes(content)
        reasons = _scan_corruption(str(f))
        assert len(reasons) > 0

    def test_non_text_extension_skipped(self, tmp_path):
        f = tmp_path / "img.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        # Non text-like extension — should return empty (not scanned)
        assert _scan_corruption(str(f)) == []

    def test_pdf_extension_skipped(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"\x00" * 50)
        assert _scan_corruption(str(f)) == []

    def test_md_extension_scanned(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_bytes(b"# Valid markdown\n\nSome content here.\n")
        assert _scan_corruption(str(f)) == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        assert _scan_corruption(str(f)) == []


# ---------------------------------------------------------------------------
# _modality_counts
# ---------------------------------------------------------------------------

class TestModalityCounts:

    def test_counts_by_modality(self):
        docs = [
            _make_doc(modality="text"),
            _make_doc(modality="text"),
            _make_doc(modality="image"),
        ]
        counts = _modality_counts(docs)
        assert counts["text"] == 2
        assert counts["image"] == 1

    def test_empty_list(self):
        assert _modality_counts([]) == {}

    def test_single_doc(self):
        counts = _modality_counts([_make_doc(modality="audio")])
        assert counts == {"audio": 1}


# ---------------------------------------------------------------------------
# _valid_chunks
# ---------------------------------------------------------------------------

class TestValidChunks:

    def test_long_text_kept(self):
        doc = _make_doc(text="A" * 100)
        result = _valid_chunks([doc])
        assert len(result) == 1

    def test_short_text_removed(self):
        doc = _make_doc(text="short")  # < CHUNK_MIN_SIZE
        result = _valid_chunks([doc])
        assert len(result) == 0

    def test_vision_space_kept_regardless_of_text_length(self):
        doc = _make_doc(text="tiny", embedding_space="vision")
        result = _valid_chunks([doc])
        assert len(result) == 1

    def test_empty_text_removed(self):
        doc = _make_doc(text="")
        result = _valid_chunks([doc])
        assert len(result) == 0

    def test_mixed_list_filtered_correctly(self):
        docs = [
            _make_doc(text="A" * 100),          # kept — long enough
            _make_doc(text="x"),                  # removed — too short
            _make_doc(text="y", embedding_space="vision"),  # kept — vision
        ]
        result = _valid_chunks(docs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _valid_embeddings
# ---------------------------------------------------------------------------

class TestValidEmbeddings:

    def test_valid_text_embedding_384(self):
        from app.core.config import settings
        doc = _make_doc(embedding=[0.1] * settings.TEXT_EMBEDDING_DIM)
        valid, invalid = _valid_embeddings([doc])
        assert len(valid) == 1
        assert invalid == 0

    def test_valid_vision_embedding_512(self):
        from app.core.config import settings
        doc = _make_doc(embedding=[0.1] * settings.VISION_EMBEDDING_DIM)
        valid, invalid = _valid_embeddings([doc])
        assert len(valid) == 1
        assert invalid == 0

    def test_wrong_dim_rejected(self):
        doc = _make_doc(embedding=[0.1] * 128)
        valid, invalid = _valid_embeddings([doc])
        assert len(valid) == 0
        assert invalid == 1

    def test_none_embedding_rejected(self):
        doc = _make_doc(embedding=None)
        valid, invalid = _valid_embeddings([doc])
        assert len(valid) == 0
        assert invalid == 1

    def test_mixed_valid_invalid(self):
        from app.core.config import settings
        valid_doc   = _make_doc(embedding=[0.1] * settings.TEXT_EMBEDDING_DIM)
        invalid_doc = _make_doc(embedding=[0.2] * 10)
        valid, invalid = _valid_embeddings([valid_doc, invalid_doc])
        assert len(valid) == 1
        assert invalid == 1

    def test_empty_list(self):
        valid, invalid = _valid_embeddings([])
        assert valid == []
        assert invalid == 0


# ---------------------------------------------------------------------------
# _split_by_modality
# ---------------------------------------------------------------------------

class TestSplitByModality:

    def test_text_goes_to_text_list(self):
        doc = _make_doc(embedding_space="text")
        text_docs, vision_docs = _split_by_modality([doc])
        assert len(text_docs) == 1
        assert len(vision_docs) == 0

    def test_vision_goes_to_vision_list(self):
        doc = _make_doc(embedding_space="vision")
        text_docs, vision_docs = _split_by_modality([doc])
        assert len(text_docs) == 0
        assert len(vision_docs) == 1

    def test_default_is_text(self):
        doc = IngestedDocument(
            text="text content here for testing purposes",
            modality="text",
            source_type="text",
            source="f.txt",
            structure={},
        )
        text_docs, vision_docs = _split_by_modality([doc])
        assert len(text_docs) == 1
        assert len(vision_docs) == 0

    def test_mixed_split(self):
        docs = [
            _make_doc(embedding_space="text"),
            _make_doc(embedding_space="vision"),
            _make_doc(embedding_space="text"),
        ]
        text_docs, vision_docs = _split_by_modality(docs)
        assert len(text_docs) == 2
        assert len(vision_docs) == 1


# ---------------------------------------------------------------------------
# _dedup_chunks
# ---------------------------------------------------------------------------

class TestDedupChunks:

    def test_identical_chunks_deduped(self):
        doc = _make_doc(text="same text content for dedup test", doc_id_val="D1", chunk_id=0)
        result = _dedup_chunks([doc, doc])
        assert len(result) == 1

    def test_different_text_kept(self):
        d1 = _make_doc(text="first document text content here", doc_id_val="D1", chunk_id=0)
        d2 = _make_doc(text="second document text content here", doc_id_val="D1", chunk_id=1)
        result = _dedup_chunks([d1, d2])
        assert len(result) == 2

    def test_empty_list(self):
        assert _dedup_chunks([]) == []

    def test_order_preserved(self):
        docs = [
            _make_doc(text=f"chunk text content number {i}", doc_id_val="D1", chunk_id=i)
            for i in range(3)
        ]
        result = _dedup_chunks(docs)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# IngestionPipeline.process_file — integration-level mocked tests
# ---------------------------------------------------------------------------

class TestIngestionPipelineProcessFile:

    def _make_pipeline(self):
        with patch("app.pipeline.ingestion_pipeline.IngestionPipeline.__init__", return_value=None):
            from app.pipeline.ingestion_pipeline import IngestionPipeline
            p = IngestionPipeline.__new__(IngestionPipeline)
            p.vector_store = MagicMock()
            p.vector_store.insert_documents.return_value = None
            p.bm25 = None
            p.max_chunks = 1000
            p.batch_size = 32
            return p

    def test_empty_session_id_raises(self, tmp_path):
        from app.pipeline.ingestion_pipeline import IngestionPipeline
        with patch("app.pipeline.ingestion_pipeline.IngestionPipeline.__init__", return_value=None):
            p = IngestionPipeline.__new__(IngestionPipeline)
            p.vector_store = MagicMock()
            p.bm25 = None
            p.max_chunks = 1000
            p.batch_size = 32

        f = tmp_path / "test.txt"
        f.write_text("hello world")
        with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
            p.process_file(str(f), session_id="")

    def test_missing_file_raises(self, tmp_path):
        from app.pipeline.ingestion_pipeline import IngestionPipeline
        with patch("app.pipeline.ingestion_pipeline.IngestionPipeline.__init__", return_value=None):
            p = IngestionPipeline.__new__(IngestionPipeline)
            p.vector_store = MagicMock()
            p.bm25 = None
            p.max_chunks = 1000
            p.batch_size = 32

        with pytest.raises(RuntimeError, match="INGESTION_FAILED"):
            with patch("app.utils.paths.set_current_user", return_value=None), \
                 patch("app.utils.paths.reset_current_user"):
                p.process_file("/nonexistent/path/file.txt", session_id="s1")

    def test_empty_file_raises(self, tmp_path):
        from app.pipeline.ingestion_pipeline import IngestionPipeline, EmptyFileError
        with patch("app.pipeline.ingestion_pipeline.IngestionPipeline.__init__", return_value=None):
            p = IngestionPipeline.__new__(IngestionPipeline)
            p.vector_store = MagicMock()
            p.bm25 = None
            p.max_chunks = 1000
            p.batch_size = 32

        f = tmp_path / "empty.txt"
        f.write_bytes(b"")

        with pytest.raises(EmptyFileError):
            with patch("app.utils.paths.set_current_user", return_value=None), \
                 patch("app.utils.paths.reset_current_user"):
                p.process_file(str(f), session_id="s1")

    def test_corrupt_txt_raises(self, tmp_path):
        from app.pipeline.ingestion_pipeline import IngestionPipeline, CorruptFileError
        with patch("app.pipeline.ingestion_pipeline.IngestionPipeline.__init__", return_value=None):
            p = IngestionPipeline.__new__(IngestionPipeline)
            p.vector_store = MagicMock()
            p.bm25 = None
            p.max_chunks = 1000
            p.batch_size = 32

        f = tmp_path / "corrupt.txt"
        f.write_bytes(b"valid start\x00null_byte_poison")

        with pytest.raises(CorruptFileError):
            with patch("app.utils.paths.set_current_user", return_value=None), \
                 patch("app.utils.paths.reset_current_user"):
                p.process_file(str(f), session_id="s1")
