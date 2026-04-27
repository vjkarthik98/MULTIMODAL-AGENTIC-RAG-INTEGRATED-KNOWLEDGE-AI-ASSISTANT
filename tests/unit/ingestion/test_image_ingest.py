import pytest
from app.ingestion.image_ingest import ingest
from PIL import Image


def test_image_ingest_basic(tmp_path):
    file_path = tmp_path / "test.png"

    img = Image.new("RGB", (100, 100), color="white")
    img.save(file_path)

    docs = ingest(str(file_path), session_id="test")

    assert isinstance(docs, list)
    assert len(docs) > 0

    for d in docs:
        assert d.modality == "image"
        assert "doc_id" in d.structure
        assert "session_id" in d.structure


def test_invalid_image():
    with pytest.raises(ValueError):
        ingest("invalid.jpg", session_id="test")