"""Phase D — per-modality source-chip locators must be accurate:
pdf->page, docx->heading, xlsx->sheet, jpg->caption, audio/video->timestamp."""
from app.pipeline.rag_pipeline import _build_p248_sources
from app.pipeline.query_pipeline import _build_sources_array


def _doc(meta, text=""):
    return {"text": text, "score": 0.9, "metadata": meta}


def _both(meta, text=""):
    """Run a doc through both source builders (stream + meta) and return chips."""
    return (
        _build_p248_sources([_doc(meta, text)])[0],
        _build_sources_array([_doc(meta, text)])[0],
    )


def test_pdf_page_locator():
    for chip in _both({"source": "report.pdf", "modality": "text", "page": 7}):
        assert chip["source"] == "report.pdf"
        assert chip["page_number"] == 7


def test_docx_heading_locator():
    for chip in _both({"source": "memo.docx", "modality": "text",
                       "section_title": "Risk Factors"}):
        assert chip["section_title"] == "Risk Factors"


def test_xlsx_sheet_locator_from_metadata():
    # New ingestion path: sheet name is in section_title (payload-backed).
    for chip in _both({"source": "sales.xlsx", "modality": "table",
                       "section_title": "Q3 Sales"}):
        assert chip["section_title"] == "Q3 Sales"


def test_xlsx_sheet_locator_from_text_prefix_legacy():
    # Already-indexed data: sheet name only in the chunk-text prefix.
    for chip in _both({"source": "sales.xlsx", "modality": "table"},
                      text="[Sheet: Legacy Sheet, Rows 1-5]\nfoo bar"):
        assert chip["section_title"] == "Legacy Sheet"


def test_image_title_locator():
    # An image is cited by `image_title` (the chart's own printed title), NOT
    # by section_title: the caption is a multi-paragraph VLM analysis dump and
    # deliberately never lands on the chip. This test still asserted the old
    # caption->section_title contract and had been failing since that change.
    for chip in _both({"source": "gdp.jpg", "modality": "image",
                       "image_title": "Bar chart of GDP growth",
                       "caption": "A bar chart showing GDP growth. " * 20}):
        assert chip["image_title"] == "Bar chart of GDP growth"
        assert chip["section_title"] is None, "the caption dump must not become the locator"


def test_audio_timestamp_locator():
    for chip in _both({"source": "talk.mp3", "modality": "audio",
                       "timestamp_start": 73.5, "timestamp_end": 79.0}):
        assert chip["start_time"] == 73.5 and chip["end_time"] == 79.0


def test_video_timestamp_locator():
    for chip in _both({"source": "clip.mp4", "modality": "video",
                       "start_time": 12.0, "end_time": 15.5}):
        assert chip["start_time"] == 12.0 and chip["end_time"] == 15.5


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
