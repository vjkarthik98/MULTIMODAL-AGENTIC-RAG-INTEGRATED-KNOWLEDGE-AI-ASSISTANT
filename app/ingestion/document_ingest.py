import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, List

from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 10


def _get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def _generate_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as file_handle:
        return hashlib.md5(file_handle.read()).hexdigest()


def _table_to_text(rows: Iterable[Iterable[object]]) -> str:
    normalized_rows = []
    for row in rows or []:
        normalized_row = [str(cell or "").strip() for cell in row]
        if any(normalized_row):
            normalized_rows.append(normalized_row)

    if not normalized_rows:
        return ""

    header = normalized_rows[0]
    body = normalized_rows[1:]

    try:
        import pandas as pd

        if body and header and len(set(header)) == len(header):
            dataframe = pd.DataFrame(body, columns=header)
        else:
            dataframe = pd.DataFrame(normalized_rows)
        return dataframe.fillna("").to_string(index=False)
    except Exception:
        return "\n".join("\t".join(row) for row in normalized_rows)


def _make_document(
    *,
    text: str,
    modality: str,
    subtype: str,
    source_type: str,
    source: str,
    doc_id: str,
    session_id: str,
    file_hash: str,
    source_path: str,
    structure: dict,
) -> IngestedDocument:
    return IngestedDocument(
        text=text,
        modality=modality,
        subtype=subtype,
        source_type=source_type,
        source=source,
        page=structure.get("page"),
        chunk_id=structure.get("chunk_index"),
        structure={
            "doc_id": doc_id,
            "session_id": session_id,
            "file_hash": file_hash,
            "source_path": source_path,
            **structure,
        },
    )


def ingest(file_path: str, session_id: str = "default") -> List[IngestedDocument]:
    if not session_id:
        raise ValueError("session_id is required")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    file_size = _get_file_size_mb(file_path)
    if file_size > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size:.2f} MB")

    start_time = time.time()
    path = Path(file_path)
    ext = path.suffix.lower()
    source_name = path.name
    source_path = str(path.resolve())
    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)
    documents: List[IngestedDocument] = []

    try:
        logger.info("[DocumentIngest][START] session_id=%s | file=%s", session_id, file_path)

        if ext == ".pdf":
            import fitz
            import pdfplumber

            image_dir = Path("data/images") / doc_id
            image_dir.mkdir(parents=True, exist_ok=True)

            with fitz.open(file_path) as pdf_document:
                for page_index, page in enumerate(pdf_document, start=1):
                    text = (page.get_text() or "").strip()
                    if text:
                        documents.append(
                            _make_document(
                                text=text,
                                modality="text",
                                subtype="page",
                                source_type="pdf",
                                source=source_name,
                                doc_id=doc_id,
                                session_id=session_id,
                                file_hash=file_hash,
                                source_path=source_path,
                                structure={"page": page_index, "content_type": "page_text"},
                            )
                        )

                    for image_index, image_meta in enumerate(page.get_images(full=True)):
                        try:
                            xref = image_meta[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_path = image_dir / f"pdf_{page_index}_{image_index}.png"

                            with open(image_path, "wb") as image_handle:
                                image_handle.write(image_bytes)

                            image_docs = image_ingest(str(image_path), session_id=session_id)
                            for image_doc in image_docs:
                                structure = dict(image_doc.structure or {})
                                structure.update(
                                    {
                                        "doc_id": doc_id,
                                        "session_id": session_id,
                                        "file_hash": file_hash,
                                        "page": page_index,
                                        "source_path": source_path,
                                        "asset_path": str(image_path),
                                        "parent_source_type": "pdf",
                                    }
                                )
                                image_doc.structure = structure
                                image_doc.source = source_name
                                image_doc.source_type = "pdf"
                                image_doc.page = page_index
                                documents.append(image_doc)
                        except Exception as exc:
                            logger.error("[DocumentIngest][PDF_IMAGE_FAIL] session_id=%s | error=%s", session_id, exc)

            with pdfplumber.open(file_path) as pdf_plumber:
                for page_index, page in enumerate(pdf_plumber.pages, start=1):
                    tables = page.extract_tables() or []
                    for table_index, table in enumerate(tables):
                        table_text = _table_to_text(table)
                        if table_text:
                            documents.append(
                                _make_document(
                                    text=table_text,
                                    modality="table",
                                    subtype="structured",
                                    source_type="pdf",
                                    source=source_name,
                                    doc_id=doc_id,
                                    session_id=session_id,
                                    file_hash=file_hash,
                                    source_path=source_path,
                                    structure={
                                        "page": page_index,
                                        "table_index": table_index,
                                        "content_type": "table",
                                    },
                                )
                            )

        elif ext == ".docx":
            import docx

            document = docx.Document(file_path)
            for paragraph_index, paragraph in enumerate(document.paragraphs):
                text = (paragraph.text or "").strip()
                if not text:
                    continue

                subtype = "heading" if paragraph.style and paragraph.style.name.startswith("Heading") else "paragraph"
                documents.append(
                    _make_document(
                        text=text,
                        modality="text",
                        subtype=subtype,
                        source_type="word",
                        source=source_name,
                        doc_id=doc_id,
                        session_id=session_id,
                        file_hash=file_hash,
                        source_path=source_path,
                        structure={
                            "paragraph_index": paragraph_index,
                            "content_type": "paragraph",
                        },
                    )
                )

            for table_index, table in enumerate(document.tables):
                table_text = _table_to_text(
                    [[cell.text for cell in row.cells] for row in table.rows]
                )
                if table_text:
                    documents.append(
                        _make_document(
                            text=table_text,
                            modality="table",
                            subtype="structured",
                            source_type="word",
                            source=source_name,
                            doc_id=doc_id,
                            session_id=session_id,
                            file_hash=file_hash,
                            source_path=source_path,
                            structure={"table_index": table_index, "content_type": "table"},
                        )
                    )

        elif ext in {".xlsx", ".xls"}:
            import pandas as pd

            workbook = pd.ExcelFile(file_path)
            for sheet_name in workbook.sheet_names:
                sheet = workbook.parse(sheet_name).fillna("")
                table_text = sheet.to_string(index=False)
                if table_text.strip():
                    documents.append(
                        _make_document(
                            text=table_text,
                            modality="table",
                            subtype="structured",
                            source_type="excel",
                            source=source_name,
                            doc_id=doc_id,
                            session_id=session_id,
                            file_hash=file_hash,
                            source_path=source_path,
                            structure={"sheet": sheet_name, "content_type": "sheet"},
                        )
                    )

        else:
            raise ValueError(f"Unsupported document type: {ext}")

        if not documents:
            raise ValueError("No extractable content found in document")

        latency = time.time() - start_time
        logger.info(
            "[DocumentIngest][SUCCESS] session_id=%s | docs=%s | latency=%.2fs",
            session_id,
            len(documents),
            latency,
        )
        return documents

    except Exception as exc:
        logger.error("[DocumentIngest][FAILED] session_id=%s | error=%s", session_id, exc)
        raise
