import pytest
from app.embeddings.image_embedder import ImageEmbedder
from PIL import Image


def test_image_embedding(tmp_path):
    file_path = tmp_path / "test.png"

    img = Image.new("RGB", (100, 100), color="white")
    img.save(file_path)

    embedder = ImageEmbedder()
    emb = embedder.embed(str(file_path))

    assert emb is not None
    assert isinstance(emb, list)
    assert len(emb) > 0