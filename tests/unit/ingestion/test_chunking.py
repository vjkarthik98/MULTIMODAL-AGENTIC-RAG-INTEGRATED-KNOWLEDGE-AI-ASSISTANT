import pytest

from app.ingestion.chunking import (
    chunk_text,
    chunk_documents
)
from app.ingestion.schema import IngestedDocument



# TEXT CHUNKING 


def test_chunk_text_success():
    text = "This is a sample text. " * 100

    chunks = chunk_text(text)

    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_text_empty():
    with pytest.raises(ValueError):
        chunk_text("")


def test_chunk_text_limit():
    text = "word " * 10000

    chunks = chunk_text(text, max_chunks=10)

    assert len(chunks) == 10



# DOCUMENT CHUNKING


def test_chunk_documents_text():
    doc = IngestedDocument(
        text="This is a long document. " * 100,
        modality="text",
        subtype="page",
        source_type="pdf",
        structure={"session_id": "test"}
    )

    result = chunk_documents([doc])

    assert len(result) > 1

    for d in result:
        assert d.modality == "text"
        assert d.structure is not None
        assert "chunk_index" in d.structure
        assert "total_chunks" in d.structure


def test_chunk_documents_table():
    doc = IngestedDocument(
        text="col1 col2\n1 2\n3 4",
        modality="table",
        subtype="structured",
        source_type="pdf",
        structure={"session_id": "test"}
    )

    result = chunk_documents([doc])

    assert len(result) == 1
    assert result[0].chunk_id == 0
    assert result[0].structure["parent_modality"] == "table"


def test_chunk_documents_image():
    doc = IngestedDocument(
        text="A dog running in a park",
        modality="image",
        subtype="caption",
        source_type="image",
        structure={"session_id": "test"}
    )

    result = chunk_documents([doc])

    assert len(result) == 1
    assert result[0].chunk_id == 0
    assert result[0].structure["parent_modality"] == "image"


def test_chunk_documents_mixed_modalities():
    docs = [
        IngestedDocument(
            text="Text data " * 50,
            modality="text",
            subtype="page",
            source_type="pdf",
            structure={"session_id": "test"}
        ),
        IngestedDocument(
            text="table data",
            modality="table",
            subtype="structured",
            source_type="pdf",
            structure={"session_id": "test"}
        ),
        IngestedDocument(
            text="image caption",
            modality="image",
            subtype="caption",
            source_type="image",
            structure={"session_id": "test"}
        )
    ]

    result = chunk_documents(docs)

    assert len(result) >= 3  

    modalities = [d.modality for d in result]

    assert "text" in modalities
    assert "table" in modalities
    assert "image" in modalities


def test_chunk_documents_empty_input():
    with pytest.raises(ValueError):
        chunk_documents([])


def test_chunk_documents_metadata_preserved():
    doc = IngestedDocument(
        text="Sample text " * 50,
        modality="text",
        subtype="page",
        source_type="pdf",
        structure={"session_id": "test", "doc_id": "123"}
    )

    result = chunk_documents([doc])

    for d in result:
        assert d.structure["session_id"] == "test"
        assert d.structure["doc_id"] == "123"