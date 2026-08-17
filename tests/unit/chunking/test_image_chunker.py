"""Unit tests for app/chunking/image_chunker.py — error_markers wiring for
low OCR confidence (hallucination-reduction initiative, Phase 5, 2026-08-13).

EasyOCR's per-detection confidence (r[2], 0-1 scale) was already being read
in _get_ocr_boxes()'s results only to apply the >0.3 inclusion filter, then
thrown away — mirrors the identical fix already made in pdf_ingest.py's
_ocr_page_image for pytesseract. Surfaced into the prompt via the existing
error_markers -> "⚠ ERROR_MARKERS=" mechanism (rag_pipeline.py::_build_context).
"""

from __future__ import annotations

import io
from unittest.mock import patch

from PIL import Image

from app.chunking.image_chunker import ImageChunker
from app.ingestion.schema import RawExtract, UniversalMetadata


def _png_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _image_raw_extract() -> RawExtract:
    return RawExtract(
        text="",
        extract_type="image_raw",
        raw_source_ref="image:chart.png",
        raw_bytes=_png_bytes(),
        extra={},
    )


def _meta() -> UniversalMetadata:
    return UniversalMetadata(source_path="/tmp/chart.png", modality="image")


class TestImageChunkerOcrErrorMarkers:
    def _run(self, ocr_boxes):
        with patch("app.chunking.image_chunker._get_ocr_boxes", return_value=ocr_boxes), \
             patch(
                 "app.chunking.image_chunker._ocr_text_from_boxes",
                 return_value="Revenue grew 10 percent" if ocr_boxes else "",
             ), \
             patch("app.chunking.image_chunker.ocr", return_value=""), \
             patch(
                 "app.chunking.image_chunker._qwen2vl_caption_for_image",
                 return_value="A chart showing revenue growth.",
             ), \
             patch("app.chunking.image_chunker.classify_image_type", return_value="bar_chart"):
            return ImageChunker().chunk([_image_raw_extract()], _meta())

    def test_low_confidence_ocr_flagged(self):
        # bbox format: (points, text, confidence) — matches EasyOCR's readtext()
        boxes = [([[0, 0], [10, 0], [10, 10], [0, 10]], "Revenue grew 10 percent", 0.35)]
        docs = self._run(boxes)
        assert len(docs) >= 1
        caption_doc = next(d for d in docs if d.structure.get("ocr_text"))
        assert caption_doc.structure["error_markers"] == ["low_ocr_confidence"]
        assert caption_doc.structure["ocr_confidence"] == 0.35

    def test_high_confidence_ocr_not_flagged(self):
        boxes = [([[0, 0], [10, 0], [10, 10], [0, 10]], "Revenue grew 10 percent", 0.95)]
        docs = self._run(boxes)
        assert len(docs) >= 1
        caption_doc = next(d for d in docs if d.structure.get("ocr_text"))
        assert caption_doc.structure["error_markers"] == []
        assert caption_doc.structure["ocr_confidence"] == 0.95

    def test_no_ocr_boxes_not_flagged(self):
        # TrOCR-fallback path (no boxes at all) — ocr_confidence must be None,
        # not treated as "low confidence" by omission.
        docs = self._run([])
        assert len(docs) >= 1
        assert docs[0].structure["ocr_confidence"] is None
        assert docs[0].structure["error_markers"] == []
