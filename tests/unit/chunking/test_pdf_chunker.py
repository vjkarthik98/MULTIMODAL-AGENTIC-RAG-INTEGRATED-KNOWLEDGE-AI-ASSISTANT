"""Unit tests for app/chunking/pdf_chunker.py — error_markers wiring for low
OCR confidence (hallucination-reduction initiative, Phase 5, 2026-08-13).

pdf_ingest.py already computes ocr_confidence per page (RawExtract.extra) but
pdf_chunker.py previously never read it. Now surfaces low-confidence pages
into the prompt via the existing error_markers -> "⚠ ERROR_MARKERS="
mechanism (rag_pipeline.py::_build_context) — flag-only, no retrieval-ranking
change.
"""

from __future__ import annotations

from app.chunking.pdf_chunker import PdfChunker
from app.ingestion.schema import RawExtract, UniversalMetadata


def _meta() -> UniversalMetadata:
    return UniversalMetadata(source_path="/tmp/report.pdf", modality="pdf")


def _prose_extract(text: str, page: int, ocr_confidence: float | None) -> RawExtract:
    return RawExtract(
        text=text,
        extract_type="prose",
        page=page,
        raw_source_ref=f"pdf:report.pdf|page:{page}",
        extra={"ocr_confidence": ocr_confidence, "is_ocr": ocr_confidence is not None},
    )


class TestPdfChunkerOcrErrorMarkers:
    def test_low_confidence_prose_page_flagged(self):
        chunker = PdfChunker()
        extracts = [_prose_extract("Apple reported net sales of $383,285 million.", 1, 0.2)]
        docs = chunker.chunk(extracts, _meta())
        assert len(docs) >= 1
        assert docs[0].structure["error_markers"] == ["low_ocr_confidence"]
        assert docs[0].structure["ocr_confidence"] == 0.2

    def test_high_confidence_prose_page_not_flagged(self):
        chunker = PdfChunker()
        extracts = [_prose_extract("Apple reported net sales of $383,285 million.", 1, 0.95)]
        docs = chunker.chunk(extracts, _meta())
        assert len(docs) >= 1
        assert docs[0].structure["error_markers"] == []

    def test_no_ocr_confidence_not_flagged(self):
        # Native-text (non-OCR) PDF pages carry no ocr_confidence at all —
        # must not be treated as "low confidence" by omission.
        chunker = PdfChunker()
        extracts = [_prose_extract("Apple reported net sales of $383,285 million.", 1, None)]
        docs = chunker.chunk(extracts, _meta())
        assert len(docs) >= 1
        assert docs[0].structure["error_markers"] == []
        assert docs[0].structure["ocr_confidence"] is None

    def test_worst_confidence_wins_across_extracts_on_same_page(self):
        # flush_prose() buffers text across multiple extracts feeding one
        # page — the flagging must reflect the WORST confidence among them,
        # not just the last one seen.
        chunker = PdfChunker()
        extracts = [
            _prose_extract("First paragraph on the page.", 1, 0.9),
            _prose_extract("Second paragraph, harder to OCR.", 1, 0.2),
        ]
        docs = chunker.chunk(extracts, _meta())
        assert len(docs) >= 1
        assert docs[0].structure["error_markers"] == ["low_ocr_confidence"]
        assert docs[0].structure["ocr_confidence"] == 0.2

    def test_scanned_page_low_confidence_flagged(self):
        chunker = PdfChunker()
        extracts = [
            RawExtract(
                text="Scanned page fallback text.",
                extract_type="scanned_page",
                page=1,
                raw_source_ref="pdf:report.pdf|page:1",
                extra={"ocr_confidence": 0.15, "is_ocr": True},
            )
        ]
        docs = chunker.chunk(extracts, _meta())
        assert len(docs) >= 1
        assert docs[0].structure["error_markers"] == ["low_ocr_confidence"]
