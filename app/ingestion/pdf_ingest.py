"""
PDF ingestor — Phase 1 per-modality refactor.

PdfIngestor.extract() → List[RawExtract]   (extraction only; no chunking)
ingest()              → List[IngestedDocument]  (backward-compat; full pipeline)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.core.config import settings
from app.ingestion.base_ingest import BaseIngestor
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

_ingest_duration = Histogram(
    "pdf_ingest_duration_seconds",
    "PDF ingestion duration",
    ["status"],
)
_ingest_errors = Counter(
    "pdf_ingest_errors_total",
    "PDF ingestion errors by type",
    ["error_type"],
)
_ocr_invocations = Counter(
    "pdf_ocr_invocations_total",
    "OCR invocations during PDF ingestion",
    [],
)
_EXTRACTS_TOTAL = Counter("magik_pdf_extracts_total", "Total extracts produced by pdf ingestor")
_EXTRACT_ERRORS = Counter("magik_pdf_extract_errors_total", "Errors in pdf ingestor")

_semaphore = asyncio.Semaphore(5)

_PDF_WATERMARK_STRIP = {
    "DRAFT", "CONFIDENTIAL", "FOR DISCUSSION ONLY", "PRELIMINARY",
    "RESTRICTED", "CLASSIFIED", "DO NOT COPY", "SAMPLE",
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _quality(text: str) -> float:
    l = len(text)
    if l < 50:
        return 0.2
    if l < 200:
        return 0.5
    return 1.0


def _is_pdf_encrypted(file_path: str) -> bool:
    try:
        import fitz
        doc = fitz.open(file_path)
        enc = doc.is_encrypted
        doc.close()
        return enc
    except Exception:
        return False


def _check_pdf_javascript(file_path: str) -> bool:
    try:
        import fitz
        doc = fitz.open(file_path)
        has_js = False
        for page in doc:
            annots = page.annots()
            if annots:
                for a in annots:
                    if a.info.get("content", "").lower().find("javascript") != -1:
                        has_js = True
                        break
        doc.close()
        return has_js
    except Exception:
        return False


def _is_pdfa(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(8192).decode("latin-1", errors="ignore")
        return "PDF/A" in header or "pdfa" in header.lower()
    except Exception:
        return False


def _has_xfa(file_path: str) -> bool:
    try:
        import fitz
        doc = fitz.open(file_path)
        result = False
        for page in doc:
            if "XFA" in (page.get_text() or ""):
                result = True
                break
        doc.close()
        return result
    except Exception:
        return False


def _repair_pdf(file_path: str) -> str:
    try:
        repaired = file_path + ".repaired.pdf"
        subprocess.run(
            ["qpdf", "--replace-input", file_path, "--", repaired],
            capture_output=True, text=True, timeout=30,
        )
        if os.path.exists(repaired):
            return repaired
    except Exception as exc:
        logger.warning("pdf_repair_failed", error=str(exc))
    return file_path


def _get_page_rotation(page: Any) -> int:
    try:
        return page.rotation
    except Exception:
        return 0


def _text_density(text: str, page_area: float) -> float:
    if page_area <= 0:
        return 0.0
    return len(text.strip()) / page_area


def _ocr_page_image(pix: Any, page_num: int) -> Tuple[str, float]:
    try:
        import pytesseract
        from PIL import Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text = (pytesseract.image_to_string(img) or "").strip()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        confidence = round(sum(confs) / max(len(confs), 1) / 100.0, 3) if confs else 0.5
        return ocr_text, confidence
    except Exception as exc:
        logger.warning("ocr_page_failed", page=page_num, error=str(exc))
        return "", 0.0


def _extract_pdf_text_multicolumn(page: Any) -> str:
    try:
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        if not blocks or page_width <= 0:
            return (page.get_text() or "").strip()
        left: List[Tuple[float, str]] = []
        right: List[Tuple[float, str]] = []
        for b in blocks:
            if b[6] != 0:
                continue
            x_center = (b[0] + b[2]) / 2
            if x_center < page_width / 2:
                left.append((b[1], b[4]))
            else:
                right.append((b[1], b[4]))
        left.sort(key=lambda x: x[0])
        right.sort(key=lambda x: x[0])
        parts = [t.strip() for _, t in left + right if t.strip()]
        return "\n\n".join(parts) if parts else (page.get_text() or "").strip()
    except Exception:
        return (page.get_text() or "").strip()


def _correct_reading_order(text: str) -> str:
    lines = text.split("\n")
    output = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                output.append(buffer.strip())
                buffer = ""
            output.append("")
        elif len(stripped) < 40 and not stripped.endswith("."):
            buffer += " " + stripped
        else:
            if buffer:
                output.append(buffer.strip())
                buffer = ""
            output.append(stripped)
    if buffer:
        output.append(buffer.strip())
    return "\n".join(output)


def _table_to_text(rows: Iterable[Iterable[object]]) -> str:
    cleaned = [
        [str(cell or "").strip() for cell in row]
        for row in (rows or [])
        if any(cell for cell in row)
    ]
    if not cleaned:
        return ""
    return "\n".join(" | ".join(row) for row in cleaned)


def _table_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    separator = ["---"] * len(header)
    body = rows[1:] if len(rows) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _detect_font_size(page: Any, text_block: str) -> Optional[float]:
    try:
        spans = page.get_text("dict").get("blocks", [])
        for block in spans:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if text_block[:20].strip() in (span.get("text") or ""):
                        return round(span.get("size", 0.0), 1)
    except Exception:
        pass
    return None


def _detect_is_bold(page: Any, text_block: str) -> bool:
    try:
        spans = page.get_text("dict").get("blocks", [])
        for block in spans:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if text_block[:20].strip() in (span.get("text") or ""):
                        flags = span.get("flags", 0)
                        return bool(flags & 2**4)  # bold bit
    except Exception:
        pass
    return False


def _detect_heading(text: str, font_size: Optional[float], is_bold: bool,
                    body_font_size: Optional[float]) -> bool:
    t = text.strip()
    if not t or len(t) > 120:
        return False
    words = t.split()
    if not (1 <= len(words) <= 15):
        return False
    if t[-1] in ".,;":
        return False
    if font_size and body_font_size and font_size > body_font_size + 1.5:
        return True
    if is_bold and len(words) <= 10:
        return True
    if t == t.upper() and len(words) <= 8:
        return True
    return False


# ─── Phase 1: PdfIngestor ─────────────────────────────────────────────────────

class PdfIngestor(BaseIngestor):
    """Extracts raw content from PDF files → List[RawExtract].

    Phase 1 responsibility: file I/O, page parsing, security gates.
    Does NOT chunk. The chunker (Phase 2) handles splitting.
    """

    def health_check(self) -> dict:
        return {
            "modality": "pdf",
            "status": "ok",
            "class": self.__class__.__name__,
        }

    async def extract(
        self,
        path: Path,
        metadata: UniversalMetadata,
    ) -> List[RawExtract]:
        source = path.name
        file_path = str(path)
        logger.info(event="extraction_start", modality="pdf", file=str(path), size=path.stat().st_size)
        try:
            if _is_pdf_encrypted(file_path):
                raise ValueError(f"PASSWORD_PROTECTED_PDF: {path.name}")

            has_js = _check_pdf_javascript(file_path)
            if has_js:
                logger.warning("pdf_javascript_detected", file=path.name)

            is_pdfa = _is_pdfa(file_path)
            is_xfa = _has_xfa(file_path)

            import fitz
            try:
                pdf = fitz.open(file_path)
            except Exception:
                logger.warning("pdf_corrupt_attempting_repair", file=path.name)
                repaired_path = await asyncio.get_event_loop().run_in_executor(
                    None, _repair_pdf, file_path
                )
                try:
                    pdf = fitz.open(repaired_path)
                except Exception as exc:
                    raise ValueError(f"PDF_CORRUPT_UNRECOVERABLE: {exc}")

            total_pages = len(pdf)
            extracts: List[RawExtract] = []
            header_footer_counts: Dict[str, int] = {}

            # Estimate body font size from first few text-heavy pages
            body_font_size: Optional[float] = None
            try:
                for pg in list(pdf)[:5]:
                    for block in pg.get_text("dict").get("blocks", []):
                        if block.get("type") != 0:
                            continue
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                sz = span.get("size", 0)
                                if sz > 8 and (body_font_size is None or sz < body_font_size):
                                    body_font_size = round(sz, 1)
            except Exception:
                pass

            for i, page in enumerate(pdf, start=1):
                # Auto-rotation
                rotation = _get_page_rotation(page)
                if rotation not in (0, 360):
                    page.set_rotation(0)

                raw_text = (page.get_text() or "").strip()

                # Injection guard on extracted text
                if raw_text:
                    raw_text = self._sanitize(raw_text, surface="pdf_ingest")

                rect = page.rect
                page_area = max(rect.width * rect.height, 1)
                density = _text_density(raw_text, page_area)
                ocr_conf = 1.0
                is_ocr = False

                if not raw_text:
                    # Scanned page — full OCR
                    _ocr_invocations.inc()
                    is_ocr = True
                    try:
                        pix = page.get_pixmap(dpi=200)
                        ocr_result, ocr_conf = _ocr_page_image(pix, i)
                        if ocr_result:
                            raw_text = ocr_result
                    except Exception as exc:
                        logger.warning("pdf_ocr_fallback_failed", page=i, error=str(exc))
                elif density < 0.01:
                    _ocr_invocations.inc()
                    try:
                        pix = page.get_pixmap(dpi=200)
                        ocr_result, ocr_conf = _ocr_page_image(pix, i)
                        if ocr_result and len(ocr_result) > len(raw_text):
                            raw_text = ocr_result
                            is_ocr = True
                    except Exception as exc:
                        logger.warning("pdf_ocr_supplemental_failed", page=i, error=str(exc))

                if not raw_text:
                    continue

                # PII scrub
                raw_text = self._scrub_pii(raw_text, surface="pdf_ingest")

                # Track header/footer candidates
                lines = raw_text.split("\n")
                if len(lines) > 2:
                    header_footer_counts[lines[0]] = header_footer_counts.get(lines[0], 0) + 1
                    header_footer_counts[lines[-1]] = header_footer_counts.get(lines[-1], 0) + 1

                # Reading-order correction
                page_text = _correct_reading_order(raw_text)

                # Multi-column reading order via block bboxes
                try:
                    mc_text = _extract_pdf_text_multicolumn(page)
                    if mc_text and len(mc_text) >= len(page_text):
                        page_text = mc_text
                except Exception:
                    pass

                # Determine extract_type: scanned vs prose vs heading
                font_sz = _detect_font_size(page, page_text[:50])
                is_bold = _detect_is_bold(page, page_text[:50])
                is_heading = _detect_heading(page_text, font_sz, is_bold, body_font_size)

                if is_ocr and not raw_text.strip():
                    ext_type = "scanned_page"
                elif is_heading:
                    ext_type = "heading"
                else:
                    ext_type = "prose"

                extracts.append(RawExtract(
                    text=page_text,
                    extract_type=ext_type,
                    page=i,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    font_size=font_sz,
                    is_bold=is_bold,
                    raw_source_ref=f"pdf:{path.name}|page:{i}",
                    extra={
                        "total_pages": total_pages,
                        "ocr_confidence": ocr_conf,
                        "is_ocr": is_ocr,
                        "is_pdfa": is_pdfa,
                        "has_xfa": is_xfa,
                        "has_javascript": has_js,
                    },
                ))

                # Embedded images → image_raw extracts
                for img_idx, img_ref in enumerate(page.get_images(full=True)):
                    try:
                        xref = img_ref[0]
                        base = pdf.extract_image(xref)
                        img_bytes = base.get("image")
                        if not img_bytes or len(img_bytes) < 256:
                            continue
                        extracts.append(RawExtract(
                            text="",
                            extract_type="image_region",
                            page=i,
                            raw_source_ref=f"pdf:{path.name}|page:{i}|img:{img_idx}",
                            raw_bytes=img_bytes,
                            extra={
                                "img_ext": base.get("ext", "png"),
                                "context_text": page_text[:500],
                            },
                        ))
                    except Exception as exc:
                        logger.warning("pdf_image_extract_failed", page=i,
                                       img_idx=img_idx, error=str(exc))

            # Hyperlinks as footnote extracts
            try:
                for i, page in enumerate(pdf, start=1):
                    for link in page.get_links():
                        uri = link.get("uri", "")
                        if uri:
                            uri = self._sanitize(uri, surface="pdf_hyperlink_ingest")
                            extracts.append(RawExtract(
                                text=f"[HYPERLINK page={i}] {uri}",
                                extract_type="footnote",
                                page=i,
                                raw_source_ref=f"pdf:{path.name}|page:{i}|link",
                            ))
            except Exception as exc:
                logger.warning("pdf_hyperlink_extraction_failed", error=str(exc))
            finally:
                pdf.close()

            # Strip repeated header/footer text from prose extracts
            repeated = {k for k, v in header_footer_counts.items() if v > 3 and k.strip()}
            if repeated:
                for ex in extracts:
                    if ex.extract_type in ("prose", "heading"):
                        for r in repeated:
                            ex.text = ex.text.replace(r, "").strip()

            # Tables via pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf_p:
                    for i, page in enumerate(pdf_p.pages, start=1):
                        for t_idx, table in enumerate(page.extract_tables() or []):
                            rows = [[str(cell or "").strip() for cell in row] for row in table]
                            txt = _table_to_text(rows)
                            md = _table_to_markdown(rows)
                            if not txt:
                                continue
                            combined = self._sanitize(f"{txt}\n\n{md}", surface="pdf_table_ingest")
                            combined = self._scrub_pii(combined, surface="pdf_table_ingest")
                            if combined:
                                extracts.append(RawExtract(
                                    text=combined,
                                    extract_type="table_row",
                                    page=i,
                                    raw_source_ref=f"pdf:{path.name}|page:{i}|table:{t_idx}",
                                    extra={"markdown": md, "table_index": t_idx},
                                ))

                    # Outline/bookmarks
                    try:
                        outline = getattr(pdf_p, "outline", None)
                        if outline:
                            outline_text = self._sanitize(str(outline)[:2000], surface="pdf_outline_ingest")
                            outline_text = self._scrub_pii(outline_text, surface="pdf_outline_ingest")
                            if outline_text:
                                extracts.append(RawExtract(
                                    text=f"[OUTLINE]\n{outline_text}",
                                    extract_type="heading",
                                    raw_source_ref=f"pdf:{path.name}|outline",
                                    extra={"is_outline": True},
                                ))
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("pdf_table_extraction_failed", error=str(exc))

            if not extracts:
                raise ValueError("NO_EXTRACTS_PRODUCED")

            _EXTRACTS_TOTAL.inc(len(extracts))
            logger.info(event="extraction_complete", modality="pdf", file=str(path), extracts=len(extracts))
            return extracts
        except Exception as _exc:
            _EXTRACT_ERRORS.inc()
            logger.error(event="extraction_failed", modality="pdf", source=source, error=str(_exc))
            raise


# ─── Backward-compat ingest() — full pipeline ─────────────────────────────────

def _base_structure(doc_id: str, session_id: str, source_path: str, **extra: Any) -> Dict[str, Any]:
    return {"doc_id": doc_id, "session_id": session_id, "source_path": source_path, **extra}


def _redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    if not settings.PII_DETECTION_ENABLED:
        return text, {}
    entity_counts: Dict[str, int] = {}
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        entities = getattr(settings, "PII_ENTITIES", [
            "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IP_ADDRESS",
        ])
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text, entities=entities, language="en")
        for r in results:
            entity_counts[r.entity_type] = entity_counts.get(r.entity_type, 0) + 1
        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pii_redaction_failed", error=str(exc))
    return text, entity_counts


def _pii_scrub(text: str, surface: str) -> str:
    try:
        from app.guardrails.pii import scrub_pii as _gp_scrub
        cleaned, _ = _gp_scrub(text)
        return cleaned
    except Exception:
        cleaned, _ = _redact_pii(text)
        return cleaned


def _sanitize_text(text: str, surface: str) -> str:
    try:
        from app.guardrails.input_guard import sanitize as _g
        return _g(text, surface=surface)
    except Exception:
        return text


async def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:
    """Backward-compatible entry point. Router imports this until Phase 8."""
    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("EMPTY_FILE")
    if file_size > settings.MAX_FILE_SIZE_PDF:
        raise ValueError(f"FILE_TOO_LARGE: {file_size}")

    with tracer.start_as_current_span("pdf_ingest") as span:
        span.set_attribute("file.name", path.name)
        span.set_attribute("file.size", file_size)
        span.set_attribute("session.id", session_id)
        start = time.time()

        async with _semaphore:
            try:
                logger.info("pdf_ingest_start", file=path.name, size=file_size, session_id=session_id)

                import fitz
                import pdfplumber

                doc_id = str(uuid.uuid4())
                source_name = path.name
                source_path_str = str(path.resolve())
                fhash = _file_hash(file_path)

                # Security
                if _is_pdf_encrypted(file_path):
                    raise ValueError(f"PASSWORD_PROTECTED_PDF: {path.name}")
                has_js = _check_pdf_javascript(file_path)
                if has_js:
                    logger.warning("pdf_javascript_detected", file=path.name)
                is_pdfa = _is_pdfa(file_path)
                is_xfa = _has_xfa(file_path)

                from app.utils.paths import resolved_images_dir
                image_dir = resolved_images_dir() / doc_id
                image_dir.mkdir(parents=True, exist_ok=True)

                active_path = file_path
                try:
                    pdf = fitz.open(active_path)
                except Exception:
                    logger.warning("pdf_corrupt_attempting_repair", file=path.name)
                    active_path = _repair_pdf(file_path)
                    try:
                        pdf = fitz.open(active_path)
                    except Exception as exc:
                        raise ValueError(f"PDF_CORRUPT_UNRECOVERABLE: {exc}")

                total_pages = len(pdf)
                documents: List[IngestedDocument] = []
                header_footer_counts: Dict[str, int] = {}

                for i, page in enumerate(pdf, start=1):
                    rotation = _get_page_rotation(page)
                    if rotation not in (0, 360):
                        page.set_rotation(0)

                    raw_text = (page.get_text() or "").strip()
                    if raw_text:
                        raw_text = _sanitize_text(raw_text, surface="pdf_ingest")

                    rect = page.rect
                    page_area = max(rect.width * rect.height, 1)
                    density = _text_density(raw_text, page_area)
                    ocr_conf = 1.0

                    if not raw_text:
                        _ocr_invocations.inc()
                        try:
                            pix = page.get_pixmap(dpi=200)
                            ocr_result, ocr_conf = _ocr_page_image(pix, i)
                            if ocr_result:
                                raw_text = ocr_result
                        except Exception as exc:
                            logger.warning("pdf_ocr_fallback_failed", page=i, error=str(exc))
                    elif density < 0.01:
                        _ocr_invocations.inc()
                        try:
                            pix = page.get_pixmap(dpi=200)
                            ocr_result, ocr_conf = _ocr_page_image(pix, i)
                            if ocr_result and len(ocr_result) > len(raw_text):
                                raw_text = ocr_result
                        except Exception as exc:
                            logger.warning("pdf_ocr_supplemental_failed", page=i, error=str(exc))

                    page_text = raw_text.strip()
                    if page_text:
                        page_text = _pii_scrub(page_text, surface="pdf_ingest")

                    lines = page_text.split("\n") if page_text else []
                    if len(lines) > 2:
                        header_footer_counts[lines[0]] = header_footer_counts.get(lines[0], 0) + 1
                        header_footer_counts[lines[-1]] = header_footer_counts.get(lines[-1], 0) + 1

                    page_text = _correct_reading_order(page_text) if page_text else page_text

                    if page_text:
                        page_sub_chunks: List[str] = []
                        if len(page_text) > settings.CHUNK_SIZE:
                            try:
                                from langchain_text_splitters import RecursiveCharacterTextSplitter
                                _splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=settings.CHUNK_SIZE,
                                    chunk_overlap=settings.CHUNK_OVERLAP,
                                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                                )
                                page_sub_chunks = [c.strip() for c in _splitter.split_text(page_text) if c.strip()]
                            except Exception:
                                pass
                        if not page_sub_chunks:
                            page_sub_chunks = [page_text]

                        for sub_idx, sub_chunk in enumerate(page_sub_chunks):
                            documents.append(
                                IngestedDocument(
                                    text=sub_chunk,
                                    modality="text",
                                    subtype="page",
                                    source_type="pdf",
                                    source=source_name,
                                    page=i,
                                    structure=_base_structure(
                                        doc_id, session_id, source_path_str,
                                        page=i,
                                        page_number=i,
                                        total_pages=total_pages,
                                        sub_chunk_index=sub_idx,
                                        total_sub_chunks=len(page_sub_chunks),
                                        ocr_confidence=ocr_conf,
                                        is_pdfa=is_pdfa,
                                        has_xfa=is_xfa,
                                        has_javascript=has_js,
                                        section_title=None,
                                        ingestion_timestamp=time.time(),
                                        language="en",
                                        file_size_bytes=file_size,
                                        content_type="pdf_page",
                                        ingestion_time=time.time(),
                                    ),
                                    extra_metadata={
                                        "data_quality_score": _quality(sub_chunk),
                                        "importance_score": _quality(sub_chunk),
                                        "modality_weight": 1.0,
                                    },
                                ).finalize()
                            )

                    # Embedded images
                    for img_idx, img_ref in enumerate(page.get_images(full=True)):
                        try:
                            xref = img_ref[0]
                            base = pdf.extract_image(xref)
                            img_bytes = base.get("image")
                            if not img_bytes:
                                continue
                            img_path = image_dir / f"p{i}_img{img_idx}.png"
                            img_path.write_bytes(img_bytes)
                            from app.ingestion.image_ingest import ingest as image_ingest
                            img_docs = image_ingest(str(img_path), session_id)
                            for d in img_docs:
                                d.structure.update({
                                    "doc_id": doc_id,
                                    "page": i,
                                    "context_text": page_text[:500] if page_text else "",
                                    "source_path": source_path_str,
                                })
                                documents.append(d)
                        except Exception as exc:
                            logger.warning("pdf_image_extract_failed", page=i,
                                           img_idx=img_idx, error=str(exc))

                # Strip repeated headers/footers
                repeated = {k for k, v in header_footer_counts.items() if v > 3 and k.strip()}
                if repeated:
                    for d in documents:
                        if d.modality == "text":
                            for r in repeated:
                                d.text = d.text.replace(r, "").strip()

                # Tables via pdfplumber
                try:
                    with pdfplumber.open(active_path) as pdf_p:
                        for i, page in enumerate(pdf_p.pages, start=1):
                            for t_idx, table in enumerate(page.extract_tables() or []):
                                rows = [[str(cell or "").strip() for cell in row] for row in table]
                                txt = _table_to_text(rows)
                                md = _table_to_markdown(rows)
                                if not txt:
                                    continue
                                combined = _sanitize_text(f"{txt}\n\n{md}", surface="pdf_table_ingest")
                                combined = _pii_scrub(combined, surface="pdf_table_ingest")
                                documents.append(
                                    IngestedDocument(
                                        text=f"[TABLE page {i}]\n{combined}",
                                        modality="table",
                                        subtype="structured",
                                        source_type="pdf",
                                        source=source_name,
                                        page=i,
                                        structure=_base_structure(
                                            doc_id, session_id, source_path_str,
                                            page=i,
                                            page_number=i,
                                            total_pages=total_pages,
                                            table_index=t_idx,
                                            section_title=None,
                                            ingestion_timestamp=time.time(),
                                            language="en",
                                            file_size_bytes=file_size,
                                            content_type="pdf_table",
                                            ingestion_time=time.time(),
                                        ),
                                        extra_metadata={
                                            "data_quality_score": _quality(txt),
                                            "importance_score": _quality(txt),
                                            "modality_weight": 1.0,
                                        },
                                    ).finalize()
                                )

                        try:
                            outline = getattr(pdf_p, "outline", None)
                            if outline:
                                outline_text = _sanitize_text(str(outline)[:2000], surface="pdf_outline_ingest")
                                outline_text = _pii_scrub(outline_text, surface="pdf_outline_ingest")
                                documents.append(
                                    IngestedDocument(
                                        text=f"[OUTLINE]\n{outline_text}",
                                        modality="text",
                                        subtype="heading",
                                        source_type="pdf",
                                        source=source_name,
                                        structure=_base_structure(
                                            doc_id, session_id, source_path_str,
                                            content_type="pdf_outline",
                                            ingestion_time=time.time(),
                                        ),
                                        extra_metadata={
                                            "data_quality_score": 0.8,
                                            "importance_score": 0.8,
                                            "modality_weight": 1.0,
                                        },
                                    ).finalize()
                                )
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("pdf_table_extraction_failed", error=str(exc))

                # Hyperlinks
                try:
                    pdf_reopen = fitz.open(active_path)
                    for i, page in enumerate(pdf_reopen, start=1):
                        for link in page.get_links():
                            uri = link.get("uri", "")
                            if uri:
                                uri = _sanitize_text(uri, surface="pdf_hyperlink_ingest")
                                uri = _pii_scrub(uri, surface="pdf_hyperlink_ingest")
                                documents.append(
                                    IngestedDocument(
                                        text=f"[HYPERLINK page={i}] {uri}",
                                        modality="text",
                                        subtype="chunk",
                                        source_type="pdf",
                                        source=source_name,
                                        page=i,
                                        structure=_base_structure(
                                            doc_id, session_id, source_path_str,
                                            page=i,
                                            content_type="pdf_hyperlink",
                                            ingestion_time=time.time(),
                                        ),
                                        extra_metadata={
                                            "data_quality_score": 0.6,
                                            "importance_score": 0.6,
                                            "modality_weight": 0.8,
                                        },
                                    ).finalize()
                                )
                    pdf_reopen.close()
                except Exception as exc:
                    logger.warning("pdf_hyperlink_extraction_failed", error=str(exc))
                finally:
                    try:
                        pdf.close()
                    except Exception:
                        pass

                if not documents:
                    raise ValueError("NO_CONTENT_EXTRACTED")

                latency = round(time.time() - start, 2)
                _ingest_duration.labels(status="success").observe(latency)
                span.set_attribute("docs.count", len(documents))
                span.set_status(Status(StatusCode.OK))
                logger.info("pdf_ingest_success", file=path.name, docs=len(documents),
                            pages=total_pages, latency=latency, session_id=session_id)
                return documents

            except Exception as exc:
                latency = round(time.time() - start, 2)
                error_type = type(exc).__name__
                _ingest_duration.labels(status="error").observe(latency)
                _ingest_errors.labels(error_type=error_type).inc()
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                logger.error("pdf_ingest_failed", file=path.name, session_id=session_id,
                             error=str(exc), error_type=error_type, latency=latency)
                raise


def ingest_sync(file_path: str, session_id: str) -> List[IngestedDocument]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, ingest(file_path, session_id))
                return future.result()
        return loop.run_until_complete(ingest(file_path, session_id))
    except RuntimeError:
        return asyncio.run(ingest(file_path, session_id))
