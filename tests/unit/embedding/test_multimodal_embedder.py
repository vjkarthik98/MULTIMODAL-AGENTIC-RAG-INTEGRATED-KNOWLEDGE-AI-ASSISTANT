import pytest
from app.embeddings.multimodal_embedder import MultimodalEmbedder
from app.ingestion.schema import IngestedDocument
from PIL import Image


def test_multimodal_embedding(tmp_path):
    # create image
    file_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(file_path)

    docs = [
        IngestedDocument(
            text="A white image",
            modality="image",
            subtype="caption",
            source=str(file_path),
            source_type="image",
            structure={"session_id": "test"}
        )
    ]

    embedder = MultimodalEmbedder()

    text_docs, vision_docs = embedder.embed_documents(
        docs,
        session_id="test"
    )

    assert isinstance(text_docs, list)
    assert isinstance(vision_docs, list)