import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, List

import chardet

from app.core.config import settings
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  HASH 
def _generate_file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


#  TABLE 
def _table_to_text(rows: Iterable[Iterable[object]]) -> str:
    rows = [
        [str(cell or "").strip() for cell in row]
        for row in rows or []
        if any(cell for cell in row)
    ]
    if not rows:
        return ""

    return "\n".join(" | ".join(row) for row in rows)


#  QUALITY 
def _quality(text: str) -> float:
    l = len(text)
    if l < 50:
        return 0.2
    if l < 200:
        return 0.5
    return 1.0


#  PDF 
def _process_pdf(file_path, doc_id, session_id, source_name, source_path):
    import fitz
    import pdfplumber
    import pytesseract
    from PIL import Image

    documents = []
    image_dir = settings.DATA_DIR / "images" / doc_id
    image_dir.mkdir(parents=True, exist_ok=True)

    header_footer_candidates = {}

    with fitz.open(file_path) as pdf:
        for i, page in enumerate(pdf, start=1):

            text = (page.get_text() or "").strip()

            # OCR fallback
            if not text:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)

            text = text.strip()

            # collect header/footer patterns
            lines = text.split("\n")
            if len(lines) > 2:
                header_footer_candidates[lines[0]] = header_footer_candidates.get(lines[0], 0) + 1
                header_footer_candidates[lines[-1]] = header_footer_candidates.get(lines[-1], 0) + 1

            if text:
                documents.append(
                    IngestedDocument(
                        text=text,
                        modality="text",
                        subtype="page",
                        source_type="pdf",
                        source=source_name,
                        page=i,
                        structure={
                            "doc_id": doc_id,
                            "session_id": session_id,
                            "source_path": source_path,
                            "page": i,
                        },
                        extra_metadata={"data_quality_score": _quality(text)},
                    ).finalize()
                )

            # images
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base = pdf.extract_image(xref)
                    img_bytes = base["image"]

                    img_path = image_dir / f"{i}_{img_idx}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    img_docs = image_ingest(str(img_path), session_id)

                    for d in img_docs:
                        d.structure.update({
                            "doc_id": doc_id,
                            "page": i,
                            "context_text": text[:500]
                        })
                        documents.append(d)

                except Exception as e:
                    logger.warning(event="pdf_image_fail", error=str(e))

    # remove headers/footers
    repeated = {k for k, v in header_footer_candidates.items() if v > 3}
    for d in documents:
        for r in repeated:
            d.text = d.text.replace(r, "").strip()

    # tables
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for t_idx, table in enumerate(page.extract_tables() or []):
                txt = _table_to_text(table)
                if txt:
                    documents.append(
                        IngestedDocument(
                            text=txt,
                            modality="table",
                            subtype="structured",
                            source_type="pdf",
                            source=source_name,
                            page=i,
                            structure={
                                "doc_id": doc_id,
                                "session_id": session_id,
                                "page": i,
                                "table_index": t_idx,
                            },
                            extra_metadata={"data_quality_score": _quality(txt)},
                        ).finalize()
                    )

    return documents


#  WORD 
def _process_docx(file_path, doc_id, session_id, source_name, source_path):
    import docx

    documents = []
    doc = docx.Document(file_path)

    # paragraphs
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
                structure={
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "paragraph_index": i,
                },
                extra_metadata={"data_quality_score": _quality(text)},
            ).finalize()
        )

    # tables
    for t_idx, table in enumerate(doc.tables):
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        txt = _table_to_text(rows)
        if txt:
            documents.append(
                IngestedDocument(
                    text=txt,
                    modality="table",
                    subtype="structured",
                    source_type="word",
                    source=source_name,
                    structure={
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "table_index": t_idx,
                    },
                    extra_metadata={"data_quality_score": _quality(txt)},
                ).finalize()
            )

    return documents


#  EXCEL 
def _process_excel(file_path, doc_id, session_id, source_name):
    import openpyxl

    documents = []
    wb = openpyxl.load_workbook(file_path, data_only=True)

    for sheet in wb.sheetnames:
        ws = wb[sheet]

        rows = []
        for row in ws.iter_rows(values_only=True):
            rows.append([str(c or "") for c in row])

        txt = _table_to_text(rows)

        if txt:
            documents.append(
                IngestedDocument(
                    text=txt,
                    modality="table",
                    subtype="structured",
                    source_type="excel",
                    source=source_name,
                    structure={
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "sheet": sheet,
                    },
                    extra_metadata={"data_quality_score": _quality(txt)},
                ).finalize()
            )

    return documents


#  MAIN 
def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError("FILE_TOO_LARGE")

    start = time.time()

    path = Path(file_path)
    ext = path.suffix.lower()

    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    source_name = path.name
    source_path = str(path.resolve())

    logger.info(event="doc_ingest_start", file=file_path)

    try:
        if ext == ".pdf":
            docs = _process_pdf(file_path, doc_id, session_id, source_name, source_path)

        elif ext == ".docx":
            docs = _process_docx(file_path, doc_id, session_id, source_name, source_path)

        elif ext in {".xlsx", ".xls"}:
            docs = _process_excel(file_path, doc_id, session_id, source_name)

        else:
            raise ValueError("UNSUPPORTED_TYPE")

        if not docs:
            raise ValueError("NO_CONTENT")

        latency = round(time.time() - start, 2)

        logger.info(event="doc_ingest_success", docs=len(docs), latency=latency)

        return docs

    except Exception as e:
        logger.error(event="doc_ingest_failed", error=str(e))
        raise