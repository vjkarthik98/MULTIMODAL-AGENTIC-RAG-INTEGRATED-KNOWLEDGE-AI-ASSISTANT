"""Phase E — real token-aware chunking: long prose splits with overlap while
preserving locator metadata; structured/temporal/visual units stay atomic."""
from app.ingestion.schema import IngestedDocument
from app.chunking.chunker import chunk_documents, _split_text
from app.core.config import settings


def _doc(text, modality="text", subtype=None, structure=None):
    return IngestedDocument(
        text=text, modality=modality, subtype=subtype, source="f.pdf",
        structure=structure or {"doc_id": "d1", "session_id": "s", "content_type": "x",
                                "embedding_space": "text"},
    ).finalize()


def test_split_text_respects_size_and_overlap():
    size, overlap = 200, 40
    text = ". ".join(f"sentence number {i} with some filler words here" for i in range(60))
    parts = _split_text(text, size, overlap)
    assert len(parts) > 1
    assert all(len(p) <= size + 60 for p in parts)   # ~size, allow sentence spill


def test_long_prose_is_split_and_keeps_locators():
    long_text = " ".join(f"This is paragraph sentence {i} about revenue and growth." for i in range(200))
    d = _doc(long_text, modality="text", structure={
        "doc_id": "d1", "session_id": "s", "content_type": "pdf",
        "embedding_space": "text", "page_number": 4, "section_title": "Overview",
    })
    out = chunk_documents([d])
    assert len(out) > 1                              # actually split
    ids = [o.chunk_id for o in out]
    assert len(ids) == len(set(ids))                 # unique chunk_ids
    for o in out:                                    # locators preserved on every part
        assert o.structure.get("page_number") == 4
        assert o.structure.get("section_title") == "Overview"
        assert o.modality == "text"


def test_table_is_atomic():
    d = _doc("[Sheet: Sales, Rows 1-5]\n" + "a b c\n" * 50, modality="table",
             subtype="structured", structure={
                 "doc_id": "d2", "session_id": "s", "content_type": "excel_sheet",
                 "embedding_space": "text", "sheet": "Sales", "section_title": "Sales"})
    out = chunk_documents([d])
    assert len(out) == 1                             # never split
    assert out[0].text.startswith("[Sheet: Sales")
    assert out[0].structure.get("section_title") == "Sales"


def test_short_doc_unchanged():
    d = _doc("Short answer about net sales.", modality="text")
    out = chunk_documents([d])
    assert len(out) == 1
    assert out[0].text == "Short answer about net sales."


def test_audio_segment_atomic():
    d = _doc("a transcript segment " * 100, modality="audio", subtype="speech",
             structure={"doc_id": "d3", "session_id": "s", "content_type": "audio",
                        "embedding_space": "text", "timestamp_start": 5.0})
    out = chunk_documents([d])
    assert len(out) == 1
    assert out[0].structure.get("timestamp_start") == 5.0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
