import pytest
from app.ingestion.chunking import chunk_documents
from app.ingestion.schema import IngestedDocument


def test_audio_chunking():
    docs = [
        IngestedDocument(
            text="Audio speech from 0s to 2s: hello world",
            modality="audio",
            subtype="speech",
            source_type="audio",
            source="test.wav",
            page=None,
            chunk_id=None,
            structure={
                "segment_index": 0,
                "session_id": "test"
            }
        )
    ]

    chunks = chunk_documents(docs)

    assert isinstance(chunks, list)
    assert len(chunks) == 1

    doc = chunks[0]

    assert doc.modality == "audio"
    assert doc.chunk_id == 0
    assert doc.structure["parent_modality"] == "audio"
    assert doc.structure["content_type"] == "speech_segment"