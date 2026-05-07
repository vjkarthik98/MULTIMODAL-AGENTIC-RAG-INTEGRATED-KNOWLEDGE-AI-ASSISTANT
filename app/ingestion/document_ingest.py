import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.core.config import settings
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


# SIZE LIMITS

_SIZE_LIMITS: Dict[str, int] = {
    ".pdf":  settings.MAX_FILE_SIZE_PDF,
    ".docx": settings.MAX_FILE_SIZE_DOCX,
    ".xlsx": settings.MAX_FILE_SIZE_XLSX,
    ".xls":  settings.MAX_FILE_SIZE_XLSX,
}


# HASH

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# TABLE

def _table_to_text(rows: Iterable[Iterable[object]]) -> str:
    cleaned = [
        [str(cell or "").strip() for cell in row]
        for row in (rows or [])
        if any(cell for cell in row)
    ]
    if not cleaned:
        return ""
    return "\n".join(" | ".join(row) for row in cleaned)


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
    **extra,
) -> Dict:
    return {
        "doc_id":      doc_id,
        "session_id":  session_id,
        "source_path": source_path,
        **extra,
    }


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
    import pytesseract
    from PIL import Image

    documents: List[IngestedDocument] = []

    image_dir = settings.PDF_IMAGE_DIR / doc_id
    image_dir.mkdir(parents=True, exist_ok=True)

    header_footer_counts: Dict[str, int] = {}

    with fitz.open(file_path) as pdf:

        total_pages = len(pdf)

        for i, page in enumerate(pdf, start=1):

            text = (page.get_text() or "").strip()

            # OCR FALLBACK FOR SCANNED PAGES
            if not text:
                try:
                    pix = page.get_pixmap(dpi=150)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = (pytesseract.image_to_string(img) or "").strip()
                except Exception as e:
                    logger.warning(event="pdf_ocr_fallback_failed", page=i, error=str(e))

            text = text.strip()

            # HEADER / FOOTER DETECTION
            lines = text.split("\n")
            if len(lines) > 2:
                header_footer_counts[lines[0]]  = header_footer_counts.get(lines[0], 0) + 1
                header_footer_counts[lines[-1]] = header_footer_counts.get(lines[-1], 0) + 1

            if text:
                documents.append(
                    IngestedDocument(
                        text=text,
                        modality="text",
                        subtype="page",
                        source_type="pdf",
                        source=source_name,
                        page=i,
                        structure=_base_structure(
                            doc_id, session_id, source_path,
                            page=i,
                            total_pages=total_pages,
                            content_type="pdf_page",
                        ),
                        extra_metadata={
                            "data_quality_score": _quality(text),
                            "importance_score":   _quality(text),
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
                            "context_text": text[:500],
                            "source_path":  source_path,
                        })
                        documents.append(d)

                except Exception as e:
                    logger.warning(event="pdf_image_extract_failed", page=i, img_idx=img_idx, error=str(e))

    # STRIP REPEATED HEADERS / FOOTERS
    repeated = {k for k, v in header_footer_counts.items() if v > 3 and k.strip()}
    if repeated:
        for d in documents:
            if d.modality == "text":
                for r in repeated:
                    d.text = d.text.replace(r, "").strip()

    # TABLES VIA PDFPLUMBER
    try:
        with pdfplumber.open(file_path) as pdf_p:
            for i, page in enumerate(pdf_p.pages, start=1):
                for t_idx, table in enumerate(page.extract_tables() or []):
                    txt = _table_to_text(table)
                    if not txt:
                        continue
                    documents.append(
                        IngestedDocument(
                            text=txt,
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
                            ),
                            extra_metadata={
                                "data_quality_score": _quality(txt),
                                "importance_score":   _quality(txt),
                                "modality_weight":    1.0,
                            },
                        ).finalize()
                    )
    except Exception as e:
        logger.warning(event="pdf_table_extraction_failed", error=str(e))

    return documents


# DOCX PROCESSOR

def _process_docx(
    file_path: str,
    doc_id: str,
    session_id: str,
    source_name: str,
    source_path: str,
) -> List[IngestedDocument]:

    import docx as python_docx

    documents: List[IngestedDocument] = []
    doc = python_docx.Document(file_path)

    # PARAGRAPHS
    for i, p in enumerate(doc.paragraphs):
        text = (p.text or "").strip()
        if not text:
            continue

        subtype = "heading" if p.style and "Heading" in p.style.name else "paragraph"

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
                    content_type="docx_paragraph",
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
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        txt  = _table_to_text(rows)
        if not txt:
            continue

        documents.append(
            IngestedDocument(
                text=txt,
                modality="table",
                subtype="structured",
                source_type="word",
                source=source_name,
                structure=_base_structure(
                    doc_id, session_id, source_path,
                    table_index=t_idx,
                    content_type="docx_table",
                ),
                extra_metadata={
                    "data_quality_score": _quality(txt),
                    "importance_score":   _quality(txt),
                    "modality_weight":    1.0,
                },
            ).finalize()
        )

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
    except Exception as e:
        raise ValueError(f"EXCEL_OPEN_FAILED: {e}")

    for sheet_name in wb.sheetnames:
        try:
            ws   = wb[sheet_name]
            rows = [[str(c or "").strip() for c in row] for row in ws.iter_rows(values_only=True)]
            txt  = _table_to_text(rows)

            if not txt:
                continue

            documents.append(
                IngestedDocument(
                    text=txt,
                    modality="table",
                    subtype="structured",
                    source_type="excel",
                    source=source_name,
                    structure=_base_structure(
                        doc_id, session_id, source_path,
                        sheet=sheet_name,
                        content_type="excel_sheet",
                    ),
                    extra_metadata={
                        "data_quality_score": _quality(txt),
                        "importance_score":   _quality(txt),
                        "modality_weight":    1.0,
                    },
                ).finalize()
            )

        except Exception as e:
            logger.warning(event="excel_sheet_failed", sheet=sheet_name, error=str(e))

    wb.close()
    return documents


# MAIN

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

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

    start       = time.time()
    doc_id      = str(uuid.uuid4())
    file_hash   = _file_hash(file_path)
    source_name = path.name
    source_path = str(path.resolve())

    logger.info(
        event="doc_ingest_start",
        file=source_name,
        ext=ext,
        size=file_size,
        session_id=session_id,
    )

    try:
        if ext == ".pdf":
            docs = _process_pdf(file_path, doc_id, session_id, source_name, source_path)

        elif ext == ".docx":
            docs = _process_docx(file_path, doc_id, session_id, source_name, source_path)

        elif ext in {".xlsx", ".xls"}:
            docs = _process_excel(file_path, doc_id, session_id, source_name, source_path)

        else:
            raise ValueError(f"UNSUPPORTED_TYPE: {ext}")

        if not docs:
            raise ValueError("NO_CONTENT_EXTRACTED")

        # STAMP FILE HASH ON ALL DOCS
        for d in docs:
            d.structure.setdefault("file_hash", file_hash)
            d.structure.setdefault("ingestion_time", time.time())

        latency = round(time.time() - start, 2)

        logger.info(
            event="doc_ingest_success",
            file=source_name,
            ext=ext,
            docs=len(docs),
            latency=latency,
            session_id=session_id,
        )

        return docs

    except Exception as e:
        logger.error(
            event="doc_ingest_failed",
            file=source_name,
            ext=ext,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 2),
        )
        raise