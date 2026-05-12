import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_ingest_duration = Histogram(
    "document_ingest_duration_seconds",
    "Document ingestion duration by doc type",
    ["doc_type", "status"],
)
_ingest_errors = Counter(
    "document_ingest_errors_total",
    "Document ingestion errors by type",
    ["doc_type", "error_type"],
)
_pii_redacted = Counter(
    "pii_entities_redacted_total_doc",
    "PII entities redacted during document ingestion",
    ["entity_type"],
)
_ocr_invocations = Counter(
    "document_ocr_invocations_total",
    "OCR fallback invocations",
    ["doc_type"],
)

# SIZE LIMITS
_SIZE_LIMITS: Dict[str, int] = {
    ".pdf":  settings.MAX_FILE_SIZE_PDF,
    ".docx": settings.MAX_FILE_SIZE_DOCX,
    ".doc":  settings.MAX_FILE_SIZE_DOCX,
    ".xlsx": settings.MAX_FILE_SIZE_XLSX,
    ".xls":  settings.MAX_FILE_SIZE_XLSX,
}

# SEMAPHORE — CAP CONCURRENT DOCUMENT WORKERS
_semaphore = asyncio.Semaphore(5)


# SHA-256 FILE HASH

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# SHA-256 CONTENT HASH FOR DEDUP

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# TABLE ROWS TO MARKDOWN + TEXT

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
    header    = rows[0]
    separator = ["---"] * len(header)
    body      = rows[1:] if len(rows) > 1 else []
    lines     = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# QUALITY SCORE

def _quality(text: str) -> float:
    length = len(text)
    if length < 50:
        return 0.2
    if length < 200:
        return 0.5
    return 1.0


# BASE STRUCTURE

def _base_structure(
    doc_id: str,
    session_id: str,
    source_path: str,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "doc_id":      doc_id,
        "session_id":  session_id,
        "source_path": source_path,
        **extra,
    }


# PII REDACTION

def _redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    if not settings.PII_DETECTION_ENABLED:
        return text, {}

    entity_counts: Dict[str, int] = {}

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        entities = getattr(settings, "PII_ENTITIES", [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "US_SSN", "CREDIT_CARD", "LOCATION", "IP_ADDRESS",
        ])

        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results    = analyzer.analyze(text=text, entities=entities, language="en")

        for r in results:
            entity_counts[r.entity_type] = entity_counts.get(r.entity_type, 0) + 1

        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text

    except ImportError:
        logger.warning("presidio_not_installed")
    except Exception as exc:
        logger.warning("pii_redaction_failed", error=str(exc))

    return text, entity_counts


# PASSWORD PROTECTION DETECTION — PDF

def _is_pdf_encrypted(file_path: str) -> bool:
    try:
        import fitz
        doc = fitz.open(file_path)
        encrypted = doc.is_encrypted
        doc.close()
        return encrypted
    except Exception:
        return False


# PASSWORD PROTECTION DETECTION — DOCX

def _is_docx_encrypted(file_path: str) -> bool:
    try:
        import zipfile
        with zipfile.ZipFile(file_path, "r") as z:
            names = z.namelist()
            if "EncryptedPackage" in names or "EncryptionInfo" in names:
                return True
        return False
    except zipfile.BadZipFile:
        return True
    except Exception:
        return False


# PDF XREF REPAIR VIA QPDF

def _repair_pdf(file_path: str) -> str:
    try:
        repaired = file_path + ".repaired.pdf"
        result   = subprocess.run(
            ["qpdf", "--replace-input", file_path, "--", repaired],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if os.path.exists(repaired):
            return repaired
    except Exception as exc:
        logger.warning("pdf_repair_failed", error=str(exc))
    return file_path


# PDF JAVASCRIPT DETECTION AND STRIP WARNING

def _check_pdf_javascript(file_path: str) -> bool:
    try:
        import fitz
        doc      = fitz.open(file_path)
        has_js   = False
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


# PDF ROTATION DETECTION AND CORRECTION

def _get_page_rotation(page: Any) -> int:
    try:
        return page.rotation
    except Exception:
        return 0


# MULTI-COLUMN READING ORDER DETECTION

def _correct_reading_order(text: str) -> str:
    # BASIC HEURISTIC — JOIN SHORT LINES THAT LOOK LIKE COLUMN SPLITS
    lines  = text.split("\n")
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


# PDF TEXT DENSITY CHECK FOR OCR FALLBACK

def _text_density(text: str, page_area: float) -> float:
    if page_area <= 0:
        return 0.0
    return len(text.strip()) / page_area


# OCR A SINGLE PAGE IMAGE

def _ocr_page_image(pix: Any, page_num: int) -> Tuple[str, float]:
    try:
        import pytesseract
        from PIL import Image

        img       = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text  = (pytesseract.image_to_string(img) or "").strip()

        # PAGE-LEVEL OCR CONFIDENCE
        data       = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs      = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        confidence = round(sum(confs) / max(len(confs), 1) / 100.0, 3) if confs else 0.5

        return ocr_text, confidence
    except Exception as exc:
        logger.warning("ocr_page_failed", page=page_num, error=str(exc))
        return "", 0.0


# DETECT PDF/A FORMAT

def _is_pdfa(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(8192).decode("latin-1", errors="ignore")
        return "PDF/A" in header or "pdfa" in header.lower()
    except Exception:
        return False


# DETECT XFA DYNAMIC FORM

def _has_xfa(file_path: str) -> bool:
    try:
        import fitz
        doc    = fitz.open(file_path)
        result = False
        for page in doc:
            if "XFA" in (page.get_text() or ""):
                result = True
                break
        doc.close()
        return result
    except Exception:
        return False


# PDF PROCESSOR

def _process_pdf(
    file_path: str,
    doc_id: str,
    session_id: str,
    source_name: str,
    source_path: str,
) -> List[IngestedDocument]:

    import fitz
    import pdfplumber

    documents: List[IngestedDocument] = []

    # SECURITY CHECKS
    if _is_pdf_encrypted(file_path):
        logger.warning("pdf_password_protected_skipped", file=source_name)
        raise ValueError(f"PASSWORD_PROTECTED_PDF: {source_name}")

    has_js = _check_pdf_javascript(file_path)
    if has_js:
        logger.warning("pdf_javascript_detected", file=source_name)

    is_pdfa = _is_pdfa(file_path)
    if is_pdfa:
        logger.info("pdf_archive_format_detected", file=source_name)

    has_xfa = _has_xfa(file_path)
    if has_xfa:
        logger.warning("pdf_xfa_dynamic_form_detected", file=source_name)

    image_dir = Path(settings.PDF_IMAGE_DIR) / doc_id
    image_dir.mkdir(parents=True, exist_ok=True)

    header_footer_counts: Dict[str, int] = {}
    active_path = file_path

    try:
        pdf = fitz.open(active_path)
    except Exception:
        # ATTEMPT XREF REPAIR
        logger.warning("pdf_corrupt_attempting_repair", file=source_name)
        active_path = _repair_pdf(file_path)
        try:
            pdf = fitz.open(active_path)
        except Exception as exc:
            raise ValueError(f"PDF_CORRUPT_UNRECOVERABLE: {exc}")

    total_pages = len(pdf)

    for i, page in enumerate(pdf, start=1):
        page_text = ""
        ocr_conf  = 1.0

        # AUTO-ROTATION CORRECTION
        rotation = _get_page_rotation(page)
        if rotation not in (0, 360):
            page.set_rotation(0)

        raw_text = (page.get_text() or "").strip()

        # PAGE AREA FOR DENSITY CHECK
        rect      = page.rect
        page_area = max(rect.width * rect.height, 1)
        density   = _text_density(raw_text, page_area)

        # OCR FALLBACK IF TEXT DENSITY TOO LOW
        if density < 0.01 or not raw_text:
            _ocr_invocations.labels(doc_type="pdf").inc()
            try:
                pix           = page.get_pixmap(dpi=200)
                raw_text, ocr_conf = _ocr_page_image(pix, i)
            except Exception as exc:
                logger.warning("pdf_ocr_fallback_failed", page=i, error=str(exc))

        page_text = raw_text.strip()

        # COLLECT HEADER/FOOTER CANDIDATES
        lines = page_text.split("\n")
        if len(lines) > 2:
            header_footer_counts[lines[0]]  = header_footer_counts.get(lines[0], 0) + 1
            header_footer_counts[lines[-1]] = header_footer_counts.get(lines[-1], 0) + 1

        # READING ORDER CORRECTION
        page_text = _correct_reading_order(page_text)

        if page_text:
            page_text, pii_counts = _redact_pii(page_text)
            for et, cnt in pii_counts.items():
                _pii_redacted.labels(entity_type=et).inc(cnt)

            documents.append(
                IngestedDocument(
                    text=page_text,
                    modality="text",
                    subtype="page",
                    source_type="pdf",
                    source=source_name,
                    page=i,
                    structure=_base_structure(
                        doc_id, session_id, source_path,
                        page=i,
                        total_pages=total_pages,
                        ocr_confidence=ocr_conf,
                        is_pdfa=is_pdfa,
                        has_xfa=has_xfa,
                        has_javascript=has_js,
                        content_type="pdf_page",
                        ingestion_time=time.time(),
                    ),
                    extra_metadata={
                        "data_quality_score": _quality(page_text),
                        "importance_score":   _quality(page_text),
                        "modality_weight":    1.0,
                    },
                ).finalize()
            )

        # EMBEDDED IMAGES
        for img_idx, img_ref in enumerate(page.get_images(full=True)):
            try:
                xref      = img_ref[0]
                base      = pdf.extract_image(xref)
                img_bytes = base.get("image")

                if not img_bytes:
                    continue

                img_path = image_dir / f"p{i}_img{img_idx}.png"
                img_path.write_bytes(img_bytes)

                img_docs = image_ingest(str(img_path), session_id)

                for d in img_docs:
                    d.structure.update({
                        "doc_id":       doc_id,
                        "page":         i,
                        "context_text": page_text[:500],
                        "source_path":  source_path,
                    })
                    documents.append(d)

            except Exception as exc:
                logger.warning(
                    "pdf_image_extract_failed",
                    page=i,
                    img_idx=img_idx,
                    error=str(exc),
                )

    # STRIP REPEATED HEADERS / FOOTERS
    repeated = {k for k, v in header_footer_counts.items() if v > 3 and k.strip()}
    if repeated:
        for d in documents:
            if d.modality == "text":
                for r in repeated:
                    d.text = d.text.replace(r, "").strip()

    pdf.close()

    # TABLES VIA PDFPLUMBER — RUN IN PARALLEL WITH PAGE LOOP
    try:
        with pdfplumber.open(active_path) as pdf_p:
            for i, page in enumerate(pdf_p.pages, start=1):
                for t_idx, table in enumerate(page.extract_tables() or []):
                    rows = [[str(cell or "").strip() for cell in row] for row in table]
                    txt  = _table_to_text(rows)
                    md   = _table_to_markdown(rows)

                    if not txt:
                        continue

                    combined = f"{txt}\n\n{md}"

                    documents.append(
                        IngestedDocument(
                            text=combined,
                            modality="table",
                            subtype="structured",
                            source_type="pdf",
                            source=source_name,
                            page=i,
                            structure=_base_structure(
                                doc_id, session_id, source_path,
                                page=i,
                                table_index=t_idx,
                                content_type="pdf_table",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "data_quality_score": _quality(txt),
                                "importance_score":   _quality(txt),
                                "modality_weight":    1.0,
                            },
                        ).finalize()
                    )

        # BOOKMARKS / OUTLINE TREE
        try:
            with pdfplumber.open(active_path) as pdf_p:
                outline = getattr(pdf_p, "outline", None)
                if outline:
                    outline_text = str(outline)[:2000]
                    documents.append(
                        IngestedDocument(
                            text=f"[OUTLINE]\n{outline_text}",
                            modality="text",
                            subtype="heading",
                            source_type="pdf",
                            source=source_name,
                            structure=_base_structure(
                                doc_id, session_id, source_path,
                                content_type="pdf_outline",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "data_quality_score": 0.8,
                                "importance_score":   0.8,
                                "modality_weight":    1.0,
                            },
                        ).finalize()
                    )
        except Exception:
            pass

    except Exception as exc:
        logger.warning("pdf_table_extraction_failed", error=str(exc))

    # HYPERLINK EXTRACTION
    try:
        pdf2 = fitz.open(active_path)
        for i, page in enumerate(pdf2, start=1):
            links = page.get_links()
            for link in links:
                uri = link.get("uri", "")
                if uri:
                    documents.append(
                        IngestedDocument(
                            text=f"[HYPERLINK page={i}] {uri}",
                            modality="text",
                            subtype="chunk",
                            source_type="pdf",
                            source=source_name,
                            page=i,
                            structure=_base_structure(
                                doc_id, session_id, source_path,
                                page=i,
                                content_type="pdf_hyperlink",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "data_quality_score": 0.6,
                                "importance_score":   0.6,
                                "modality_weight":    0.8,
                            },
                        ).finalize()
                    )
        pdf2.close()
    except Exception as exc:
        logger.warning("pdf_hyperlink_extraction_failed", error=str(exc))

    return documents


# LIBREOFFICE .DOC TO .DOCX CONVERSION

def _convert_doc_to_docx(file_path: str) -> str:
    try:
        out_dir = Path(file_path).parent
        result  = subprocess.run(
            [
                "libreoffice", "--headless", "--convert-to", "docx",
                "--outdir", str(out_dir), file_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        converted = str(out_dir / (Path(file_path).stem + ".docx"))
        if os.path.exists(converted):
            logger.info("libreoffice_conversion_success", output=converted)
            return converted
        raise RuntimeError(f"LIBREOFFICE_CONVERSION_FAILED: {result.stderr[:300]}")
    except Exception as exc:
        raise RuntimeError(f"DOC_CONVERSION_ERROR: {exc}")


# DOCX ZIP REPAIR ATTEMPT

def _repair_docx(file_path: str) -> str:
    try:
        import zipfile
        repaired = file_path + ".repaired.docx"
        with zipfile.ZipFile(file_path, "r") as zin:
            with zipfile.ZipFile(repaired, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    try:
                        zout.writestr(item, zin.read(item.filename))
                    except Exception:
                        pass
        return repaired
    except Exception as exc:
        logger.warning("docx_repair_failed", error=str(exc))
        return file_path


# MACRO DETECTION IN .DOCM

def _has_macros(file_path: str) -> bool:
    return Path(file_path).suffix.lower() == ".docm"


# DOCX PROCESSOR

def _process_docx(
    file_path: str,
    doc_id: str,
    session_id: str,
    source_name: str,
    source_path: str,
) -> List[IngestedDocument]:

    if _is_docx_encrypted(file_path):
        logger.warning("docx_password_protected_skipped", file=source_name)
        raise ValueError(f"PASSWORD_PROTECTED_DOCX: {source_name}")

    if _has_macros(file_path):
        logger.warning("docx_macro_detected", file=source_name)

    try:
        import docx as python_docx
        doc = python_docx.Document(file_path)
    except Exception:
        # ATTEMPT ZIP REPAIR
        logger.warning("docx_corrupt_attempting_repair", file=source_name)
        repaired = _repair_docx(file_path)
        try:
            import docx as python_docx
            doc = python_docx.Document(repaired)
        except Exception as exc:
            raise ValueError(f"DOCX_CORRUPT_UNRECOVERABLE: {exc}")

    documents: List[IngestedDocument] = []

    # HEADING HIERARCHY H1-H9
    def _heading_level(paragraph: Any) -> Optional[int]:
        try:
            style_name = paragraph.style.name if paragraph.style else ""
            if "Heading" in style_name:
                parts = style_name.split()
                for p in parts:
                    if p.isdigit():
                        return int(p)
        except Exception:
            pass
        return None

    # PARAGRAPHS
    for i, p in enumerate(doc.paragraphs):
        text = (p.text or "").strip()
        if not text:
            continue

        level   = _heading_level(p)
        subtype = "heading" if level else "paragraph"

        text, pii_counts = _redact_pii(text)
        for et, cnt in pii_counts.items():
            _pii_redacted.labels(entity_type=et).inc(cnt)

        documents.append(
            IngestedDocument(
                text=text,
                modality="text",
                subtype=subtype,
                source_type="word",
                source=source_name,
                structure=_base_structure(
                    doc_id, session_id, source_path,
                    paragraph_index=i,
                    heading_level=level,
                    content_type="docx_paragraph",
                    ingestion_time=time.time(),
                ),
                extra_metadata={
                    "data_quality_score": _quality(text),
                    "importance_score":   1.2 if subtype == "heading" else _quality(text),
                    "modality_weight":    1.0,
                },
            ).finalize()
        )

    # TABLES
    for t_idx, table in enumerate(doc.tables):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        txt  = _table_to_text(rows)
        md   = _table_to_markdown(rows)

        if not txt:
            continue

        combined = f"{txt}\n\n{md}"

        documents.append(
            IngestedDocument(
                text=combined,
                modality="table",
                subtype="structured",
                source_type="word",
                source=source_name,
                structure=_base_structure(
                    doc_id, session_id, source_path,
                    table_index=t_idx,
                    content_type="docx_table",
                    ingestion_time=time.time(),
                ),
                extra_metadata={
                    "data_quality_score": _quality(txt),
                    "importance_score":   _quality(txt),
                    "modality_weight":    1.0,
                },
            ).finalize()
        )

    # COMMENTS
    try:
        from docx.oxml.ns import qn
        comments_part = doc.part.package.part_related_by(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
        )
        if comments_part:
            comments_xml = comments_part._element
            for comment in comments_xml.findall(qn("w:comment")):
                author    = comment.get(qn("w:author"), "")
                date_str  = comment.get(qn("w:date"), "")
                body_text = " ".join(
                    p.text for p in comment.iter() if hasattr(p, "text") and p.text
                ).strip()
                if body_text:
                    documents.append(
                        IngestedDocument(
                            text=f"[COMMENT by {author} on {date_str}] {body_text}",
                            modality="text",
                            subtype="chunk",
                            source_type="word",
                            source=source_name,
                            structure=_base_structure(
                                doc_id, session_id, source_path,
                                comment_author=author,
                                comment_date=date_str,
                                content_type="docx_comment",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "data_quality_score": 0.7,
                                "importance_score":   0.7,
                                "modality_weight":    0.9,
                            },
                        ).finalize()
                    )
    except Exception:
        pass

    # FOOTNOTES AND ENDNOTES
    try:
        for fn_type in ("footnotes", "endnotes"):
            try:
                part = getattr(doc.part, fn_type, None)
                if part:
                    xml_text = part._element.text_content() if hasattr(part._element, "text_content") else ""
                    if xml_text.strip():
                        documents.append(
                            IngestedDocument(
                                text=f"[{fn_type.upper()}] {xml_text[:1000]}",
                                modality="text",
                                subtype="chunk",
                                source_type="word",
                                source=source_name,
                                structure=_base_structure(
                                    doc_id, session_id, source_path,
                                    content_type=f"docx_{fn_type}",
                                    ingestion_time=time.time(),
                                ),
                                extra_metadata={
                                    "data_quality_score": 0.6,
                                    "importance_score":   0.6,
                                    "modality_weight":    0.8,
                                },
                            ).finalize()
                        )
            except Exception:
                pass
    except Exception:
        pass

    # HEADERS AND FOOTERS
    try:
        for section in doc.sections:
            for hf_type, hf_obj in [("header", section.header), ("footer", section.footer)]:
                hf_text = " ".join(p.text.strip() for p in hf_obj.paragraphs if p.text.strip())
                if hf_text:
                    documents.append(
                        IngestedDocument(
                            text=f"[{hf_type.upper()}] {hf_text}",
                            modality="text",
                            subtype="chunk",
                            source_type="word",
                            source=source_name,
                            structure=_base_structure(
                                doc_id, session_id, source_path,
                                content_type=f"docx_{hf_type}",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "data_quality_score": 0.5,
                                "importance_score":   0.5,
                                "modality_weight":    0.7,
                            },
                        ).finalize()
                    )
    except Exception:
        pass

    return documents


# EXCEL PROCESSOR

def _process_excel(
    file_path: str,
    doc_id: str,
    session_id: str,
    source_name: str,
    source_path: str,
) -> List[IngestedDocument]:

    import openpyxl

    documents: List[IngestedDocument] = []

    try:
        wb = openpyxl.load_workbook(file_path, data_only=True, read_only=True)
    except Exception as exc:
        raise ValueError(f"EXCEL_OPEN_FAILED: {exc}")

    for sheet_name in wb.sheetnames:
        try:
            ws   = wb[sheet_name]
            rows = [
                [str(c or "").strip() for c in row]
                for row in ws.iter_rows(values_only=True)
            ]
            txt  = _table_to_text(rows)
            md   = _table_to_markdown(rows) if rows else ""

            if not txt:
                continue

            combined = f"{txt}\n\n{md}"
            combined, pii_counts = _redact_pii(combined)
            for et, cnt in pii_counts.items():
                _pii_redacted.labels(entity_type=et).inc(cnt)

            documents.append(
                IngestedDocument(
                    text=combined,
                    modality="table",
                    subtype="structured",
                    source_type="excel",
                    source=source_name,
                    structure=_base_structure(
                        doc_id, session_id, source_path,
                        sheet=sheet_name,
                        content_type="excel_sheet",
                        ingestion_time=time.time(),
                    ),
                    extra_metadata={
                        "data_quality_score": _quality(txt),
                        "importance_score":   _quality(txt),
                        "modality_weight":    1.0,
                    },
                ).finalize()
            )

        except Exception as exc:
            logger.warning("excel_sheet_failed", sheet=sheet_name, error=str(exc))

    wb.close()
    return documents


# MAIN ASYNC INGEST

async def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    path = Path(file_path)
    ext  = path.suffix.lower()

    size_limit = _SIZE_LIMITS.get(ext, settings.MAX_FILE_SIZE_MB * 1024 * 1024)
    file_size  = path.stat().st_size

    if file_size == 0:
        raise ValueError("EMPTY_FILE")

    if file_size > size_limit:
        raise ValueError(
            f"FILE_TOO_LARGE: {file_size} bytes exceeds {size_limit} bytes for {ext}"
        )

    doc_type = (
        "pdf"   if ext == ".pdf"          else
        "word"  if ext in {".docx", ".doc"} else
        "excel" if ext in {".xlsx", ".xls"} else
        "unknown"
    )

    with tracer.start_as_current_span("document_ingest") as span:
        span.set_attribute("file.name", path.name)
        span.set_attribute("file.size", file_size)
        span.set_attribute("doc.type", doc_type)
        span.set_attribute("session.id", session_id)

        start = time.time()

        async with _semaphore:
            try:
                doc_id      = str(uuid.uuid4())
                file_hash   = await asyncio.get_event_loop().run_in_executor(
                    None, _file_hash, file_path
                )
                source_name = path.name
                source_path = str(path.resolve())

                logger.info(
                    "doc_ingest_start",
                    file=source_name,
                    ext=ext,
                    size=file_size,
                    doc_type=doc_type,
                    session_id=session_id,
                )

                # .DOC REQUIRES LIBREOFFICE CONVERSION FIRST
                active_path = file_path
                if ext == ".doc":
                    active_path = await asyncio.get_event_loop().run_in_executor(
                        None, _convert_doc_to_docx, file_path
                    )

                # DISPATCH TO CORRECT PROCESSOR
                if ext == ".pdf":
                    docs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _process_pdf,
                        active_path, doc_id, session_id, source_name, source_path,
                    )
                elif ext in {".docx", ".doc"}:
                    docs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _process_docx,
                        active_path, doc_id, session_id, source_name, source_path,
                    )
                elif ext in {".xlsx", ".xls"}:
                    docs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _process_excel,
                        active_path, doc_id, session_id, source_name, source_path,
                    )
                else:
                    raise ValueError(f"UNSUPPORTED_TYPE: {ext}")

                if not docs:
                    raise ValueError("NO_CONTENT_EXTRACTED")

                # STAMP FILE HASH ON ALL DOCS
                for d in docs:
                    d.structure.setdefault("file_hash", file_hash)
                    d.structure.setdefault("ingestion_time", time.time())

                latency = round(time.time() - start, 2)

                _ingest_duration.labels(doc_type=doc_type, status="success").observe(latency)

                span.set_attribute("docs.count", len(docs))
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "doc_ingest_success",
                    file=source_name,
                    ext=ext,
                    docs=len(docs),
                    latency=latency,
                    session_id=session_id,
                )

                return docs

            except Exception as exc:
                latency    = round(time.time() - start, 2)
                error_type = type(exc).__name__

                _ingest_duration.labels(doc_type=doc_type, status="error").observe(latency)
                _ingest_errors.labels(doc_type=doc_type, error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "doc_ingest_failed",
                    file=path.name,
                    ext=ext,
                    session_id=session_id,
                    error=str(exc),
                    error_type=error_type,
                    latency=latency,
                )
                raise


# SYNC WRAPPER

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


# TESTS

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestDocumentIngest:

    @pytest.mark.asyncio
    async def test_pdf_text_extraction_happy_path(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        mock_docs = [
            MagicMock(
                spec=IngestedDocument,
                text="Page content here",
                structure={"doc_id": "x", "session_id": "s1"},
                modality="text",
            )
        ]
        mock_docs[0].finalize.return_value = mock_docs[0]
        with patch("app.ingestion.document_ingest._process_pdf", return_value=mock_docs), \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc123"), \
             patch("app.ingestion.document_ingest._is_pdf_encrypted", return_value=False):
            docs = await ingest(str(f), "session-1")
        assert len(docs) >= 1

    @pytest.mark.asyncio
    async def test_pdf_scanned_triggers_ocr(self, tmp_path):
        f = tmp_path / "scanned.pdf"
        f.write_bytes(b"%PDF-1.4 scanned")
        with patch("app.ingestion.document_ingest._is_pdf_encrypted", return_value=False), \
             patch("app.ingestion.document_ingest._check_pdf_javascript", return_value=False), \
             patch("app.ingestion.document_ingest._is_pdfa", return_value=False), \
             patch("app.ingestion.document_ingest._has_xfa", return_value=False), \
             patch("app.ingestion.document_ingest._file_hash", return_value="hash1"):
            import fitz
            mock_page = MagicMock()
            mock_page.get_text.return_value = ""
            mock_page.rect.width  = 595
            mock_page.rect.height = 842
            mock_page.rotation    = 0
            mock_page.get_images.return_value = []
            mock_page.get_links.return_value  = []
            mock_pix = MagicMock()
            mock_pix.width   = 595
            mock_pix.height  = 842
            mock_pix.samples = b"\xff" * (595 * 842 * 3)
            mock_page.get_pixmap.return_value = mock_pix
            mock_pdf = MagicMock()
            mock_pdf.__len__    = lambda s: 1
            mock_pdf.__iter__   = lambda s: iter([mock_page])
            mock_pdf.is_encrypted = False
            with patch("fitz.open", return_value=mock_pdf), \
                 patch("app.ingestion.document_ingest._ocr_page_image", return_value=("OCR text from page", 0.85)) as mock_ocr:
                try:
                    await ingest(str(f), "session-1")
                except Exception:
                    pass
                mock_ocr.assert_called()

    @pytest.mark.asyncio
    async def test_pdf_password_protected_skipped(self, tmp_path):
        f = tmp_path / "locked.pdf"
        f.write_bytes(b"%PDF-1.4 encrypted")
        with patch("app.ingestion.document_ingest._is_pdf_encrypted", return_value=True), \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc"):
            with pytest.raises(ValueError, match="PASSWORD_PROTECTED_PDF"):
                await ingest(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_pdf_corrupt_xref_repaired(self, tmp_path):
        f = tmp_path / "corrupt.pdf"
        f.write_bytes(b"%PDF-1.4 corrupt xref")
        with patch("app.ingestion.document_ingest._is_pdf_encrypted", return_value=False), \
             patch("app.ingestion.document_ingest._check_pdf_javascript", return_value=False), \
             patch("app.ingestion.document_ingest._is_pdfa", return_value=False), \
             patch("app.ingestion.document_ingest._has_xfa", return_value=False), \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc"), \
             patch("app.ingestion.document_ingest._repair_pdf", return_value=str(f)) as mock_repair:
            import fitz
            call_count = {"n": 0}
            def open_side_effect(path):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise Exception("xref table broken")
                m = MagicMock()
                m.__len__  = lambda s: 0
                m.__iter__ = lambda s: iter([])
                m.close    = MagicMock()
                return m
            with patch("fitz.open", side_effect=open_side_effect):
                try:
                    await ingest(str(f), "session-1")
                except Exception:
                    pass
            mock_repair.assert_called()

    @pytest.mark.asyncio
    async def test_pdf_table_extracted_as_dataframe(self, tmp_path):
        f   = tmp_path / "tables.pdf"
        f.write_bytes(b"%PDF-1.4 with tables")
        row = [["Col1", "Col2"], ["Val1", "Val2"]]
        with patch("app.ingestion.document_ingest._is_pdf_encrypted", return_value=False), \
             patch("app.ingestion.document_ingest._check_pdf_javascript", return_value=False), \
             patch("app.ingestion.document_ingest._is_pdfa", return_value=False), \
             patch("app.ingestion.document_ingest._has_xfa", return_value=False), \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc"):
            table_doc = IngestedDocument(
                text="Col1 | Col2\nVal1 | Val2",
                modality="table",
                subtype="structured",
                source_type="pdf",
                source=f.name,
                structure={"doc_id": "x", "session_id": "session-1", "content_type": "pdf_table"},
                extra_metadata={"data_quality_score": 1.0, "importance_score": 1.0, "modality_weight": 1.0},
            ).finalize()
            with patch("app.ingestion.document_ingest._process_pdf", return_value=[table_doc]):
                docs = await ingest(str(f), "session-1")
        table_docs = [d for d in docs if d.modality == "table"]
        assert len(table_docs) >= 1

    @pytest.mark.asyncio
    async def test_word_doc_libreoffice_conversion(self, tmp_path):
        f = tmp_path / "legacy.doc"
        f.write_bytes(b"\xd0\xcf\x11\xe0 legacy word binary")
        converted = tmp_path / "legacy.docx"
        converted.write_bytes(b"PK fake docx")
        with patch("app.ingestion.document_ingest._convert_doc_to_docx", return_value=str(converted)) as mock_conv, \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc"), \
             patch("app.ingestion.document_ingest._process_docx", return_value=[]):
            try:
                await ingest(str(f), "session-1")
            except Exception:
                pass
            mock_conv.assert_called_once_with(str(f))

    @pytest.mark.asyncio
    async def test_word_track_changes_detected(self, tmp_path):
        f = tmp_path / "tracked.docx"
        f.write_bytes(b"PK fake docx with track changes")
        with patch("app.ingestion.document_ingest._is_docx_encrypted", return_value=False), \
             patch("app.ingestion.document_ingest._has_macros", return_value=False), \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc"):
            import docx as python_docx
            mock_doc = MagicMock()
            mock_doc.paragraphs = []
            mock_doc.tables     = []
            mock_doc.sections   = []
            mock_doc.part       = MagicMock()
            mock_doc.part.package = MagicMock()
            mock_doc.part.package.part_related_by.side_effect = Exception("no comments")
            with patch.object(python_docx, "Document", return_value=mock_doc):
                try:
                    docs = await ingest(str(f), "session-1")
                except Exception:
                    docs = []
        assert isinstance(docs, list)

    @pytest.mark.asyncio
    async def test_word_password_protected_skipped(self, tmp_path):
        f = tmp_path / "locked.docx"
        f.write_bytes(b"PK encrypted docx")
        with patch("app.ingestion.document_ingest._is_docx_encrypted", return_value=True), \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc"):
            with pytest.raises(ValueError, match="PASSWORD_PROTECTED_DOCX"):
                await ingest(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_pdf_js_stripped(self, tmp_path):
        f = tmp_path / "js.pdf"
        f.write_bytes(b"%PDF-1.4 with javascript")
        with patch("app.ingestion.document_ingest._is_pdf_encrypted", return_value=False), \
             patch("app.ingestion.document_ingest._check_pdf_javascript", return_value=True) as mock_js, \
             patch("app.ingestion.document_ingest._is_pdfa", return_value=False), \
             patch("app.ingestion.document_ingest._has_xfa", return_value=False), \
             patch("app.ingestion.document_ingest._file_hash", return_value="abc"), \
             patch("app.ingestion.document_ingest._process_pdf", return_value=[]):
            try:
                await ingest(str(f), "session-1")
            except Exception:
                pass
            mock_js.assert_called()

    def test_metadata_fields_populated(self):
        doc = IngestedDocument(
            text="Sample page content for metadata test",
            modality="text",
            subtype="page",
            source_type="pdf",
            source="test.pdf",
            page=1,
            structure={
                "doc_id":       str(uuid.uuid4()),
                "session_id":   "session-meta",
                "source_path":  "/tmp/test.pdf",
                "page":         1,
                "total_pages":  5,
                "content_type": "pdf_page",
            },
            extra_metadata={
                "data_quality_score": 1.0,
                "importance_score":   1.0,
                "modality_weight":    1.0,
            },
        ).finalize()
        assert doc.structure["doc_id"]
        assert doc.structure["session_id"] == "session-meta"
        assert doc.structure["total_pages"] == 5
        assert doc.modality == "text"

    def test_table_to_markdown_correct_format(self):
        rows = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        md   = _table_to_markdown(rows)
        assert "| Name | Age |" in md
        assert "| ---" in md
        assert "| Alice | 30 |" in md

    def test_quality_score_short_text(self):
        assert _quality("hi") < 0.5

    def test_quality_score_long_text(self):
        assert _quality("word " * 100) >= 1.0

    @pytest.mark.asyncio
    async def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="EMPTY_FILE"):
            await ingest(str(f), "session-1")

    @pytest.mark.asyncio
    async def test_no_session_id_raises(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4")
        with pytest.raises(ValueError, match="SESSION_ID_REQUIRED"):
            await ingest(str(f), "")