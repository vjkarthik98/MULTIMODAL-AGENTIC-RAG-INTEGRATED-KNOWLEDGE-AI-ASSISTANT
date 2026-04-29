import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, List

from app.core.config import settings
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger


logger = get_logger(__name__)


# GET FILE SIZE
def _get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


# GENERATE FILE HASH
def _generate_file_hash(file_path: str) -> str:
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# CONVERT TABLE TO TEXT
def _table_to_text(rows: Iterable[Iterable[object]]) -> str:
    normalized = [
        [str(cell or "").strip() for cell in row]
        for row in rows or []
        if any(cell for cell in row)
    ]

    if not normalized:
        return ""

    try:
        import pandas as pd
        return pd.DataFrame(normalized).fillna("").to_string(index=False)
    except Exception:
        return "\n".join("\t".join(row) for row in normalized)


# CREATE DOCUMENT OBJECT
def _make_document(**kwargs) -> IngestedDocument:
    return IngestedDocument(**kwargs)


# MAIN INGEST FUNCTION
def ingest(file_path: str, session_id: str = "default") -> List[IngestedDocument]:

    # VALIDATE SESSION
    if not session_id:
        raise ValueError("SESSION_ID REQUIRED")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} NOT FOUND")

    file_size = _get_file_size_mb(file_path)

    # FILE SIZE LIMIT
    if file_size > settings.MAX_FILE_SIZE_MB:
        raise ValueError(f"FILE TOO LARGE: {file_size:.2f} MB")

    start = time.time()

    path = Path(file_path)
    ext = path.suffix.lower()

    source_name = path.name
    source_path = str(path.resolve())

    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    documents: List[IngestedDocument] = []

    # LIMIT CONFIG
    max_pages = getattr(settings, "MAX_PAGES", 200)
    max_tables = getattr(settings, "MAX_TABLES", 200)
    max_images = getattr(settings, "MAX_IMAGES", 200)

    global_image_count = 0

    try:
        logger.info("[DocumentIngest][START] session_id=%s", session_id)

        # PDF PROCESSING
        if ext == ".pdf":
            import fitz
            import pdfplumber

            image_dir = settings.DATA_DIR / "images" / doc_id
            image_dir.mkdir(parents=True, exist_ok=True)

            with fitz.open(file_path) as pdf:

                for page_idx, page in enumerate(pdf, start=1):

                    if page_idx > max_pages:
                        logger.warning("[DocumentIngest] PAGE LIMIT REACHED")
                        break

                    page_text = (page.get_text() or "").strip()

                    # ADD PAGE TEXT
                    if page_text and len(page_text) > 20:
                        documents.append(
                            _make_document(
                                text=page_text,
                                modality="text",
                                subtype="page",
                                source_type="pdf",
                                source=source_name,
                                page=page_idx,
                                structure={
                                    "doc_id": doc_id,
                                    "session_id": session_id,
                                    "file_hash": file_hash,
                                    "source_path": source_path,
                                    "page": page_idx,
                                    "content_type": "page_text",
                                },
                            )
                        )

                    # IMAGE EXTRACTION
                    for img_idx, img_meta in enumerate(page.get_images(full=True)):

                        if global_image_count >= max_images:
                            break

                        try:
                            xref = img_meta[0]
                            base_img = pdf.extract_image(xref)
                            img_bytes = base_img["image"]

                            img_path = image_dir / f"{page_idx}_{img_idx}.png"

                            with open(img_path, "wb") as f:
                                f.write(img_bytes)

                            image_docs = image_ingest(str(img_path), session_id=session_id)

                            for d in image_docs:

                                # CONTEXTUAL LINKING (IMPORTANT)
                                structure = dict(d.structure or {})
                                structure.update({
                                    "doc_id": doc_id,
                                    "page": page_idx,
                                    "asset_path": str(img_path),
                                    "context_text": page_text[:500],  # LINK TEXT CONTEXT
                                })

                                d.structure = structure
                                d.source = source_name

                                documents.append(d)

                            global_image_count += 1

                        except Exception as e:
                            logger.warning("[DocumentIngest][IMG_FAIL] %s", str(e))

            # TABLE EXTRACTION
            with pdfplumber.open(file_path) as pdf:

                for page_idx, page in enumerate(pdf.pages, start=1):

                    tables = page.extract_tables() or []

                    for table_idx, table in enumerate(tables):

                        if table_idx >= max_tables:
                            break

                        text = _table_to_text(table)

                        if text and len(text) > 20:
                            documents.append(
                                _make_document(
                                    text=text,
                                    modality="table",
                                    subtype="structured",
                                    source_type="pdf",
                                    source=source_name,
                                    page=page_idx,
                                    structure={
                                        "doc_id": doc_id,
                                        "session_id": session_id,
                                        "file_hash": file_hash,
                                        "source_path": source_path,
                                        "page": page_idx,
                                        "table_index": table_idx,
                                        "context_text": (page.extract_text() or "")[:500],
                                        "content_type": "table",
                                    },
                                )
                            )

        # DOCX PROCESSING
        elif ext == ".docx":
            import docx

            doc = docx.Document(file_path)

            for i, p in enumerate(doc.paragraphs):
                text = (p.text or "").strip()

                if not text:
                    continue

                subtype = "heading" if p.style and "Heading" in p.style.name else "paragraph"

                documents.append(
                    _make_document(
                        text=text,
                        modality="text",
                        subtype=subtype,
                        source_type="word",
                        source=source_name,
                        structure={
                            "doc_id": doc_id,
                            "session_id": session_id,
                            "file_hash": file_hash,
                            "source_path": source_path,
                            "paragraph_index": i,
                        },
                    )
                )

        # EXCEL PROCESSING
        elif ext in {".xlsx", ".xls"}:
            import pandas as pd

            workbook = pd.ExcelFile(file_path)

            for sheet in workbook.sheet_names:
                df = workbook.parse(sheet).fillna("")
                text = df.to_string(index=False)

                if text.strip():
                    documents.append(
                        _make_document(
                            text=text,
                            modality="table",
                            subtype="structured",
                            source_type="excel",
                            source=source_name,
                            structure={
                                "doc_id": doc_id,
                                "session_id": session_id,
                                "file_hash": file_hash,
                                "source_path": source_path,
                                "sheet": sheet,
                            },
                        )
                    )

        else:
            raise ValueError(f"UNSUPPORTED FILE TYPE: {ext}")

        # FINAL VALIDATION
        if not documents:
            raise ValueError("NO CONTENT EXTRACTED")

        logger.info(
            "[DocumentIngest][SUCCESS] docs=%s | latency=%.2fs",
            len(documents),
            time.time() - start
        )

        return documents

    except Exception as e:
        logger.error("[DocumentIngest][FAILED] %s", str(e))
        raise