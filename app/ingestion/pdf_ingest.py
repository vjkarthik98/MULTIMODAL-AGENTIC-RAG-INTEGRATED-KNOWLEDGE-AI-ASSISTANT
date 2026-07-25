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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
    "DRAFT",
    "CONFIDENTIAL",
    "FOR DISCUSSION ONLY",
    "PRELIMINARY",
    "RESTRICTED",
    "CLASSIFIED",
    "DO NOT COPY",
    "SAMPLE",
}

# Character-level translation for EDGAR/PDF ligatures and smart punctuation.
# Applied before any NLP/guardrail pass so downstream tokenisers see plain ASCII.
_LIGATURE_MAP = str.maketrans(
    {
        '\xa0': ' ',  # non-breaking space  →  regular space  (very common in EDGAR: "September\xa028")
        '\xad': '',  # soft hyphen          →  drop
        '’': "'",  # right single quote
        '‘': "'",  # left  single quote
        '“': '"',  # left  double quote
        '”': '"',  # right double quote
        '–': '-',  # en dash
        '—': '--',  # em dash
        '•': '-',  # bullet
        '…': '...',  # ellipsis
        'ﬁ': 'fi',  # fi ligature
        'ﬂ': 'fl',  # fl ligature
        'ﬃ': 'ffi',  # ffi ligature
        'ﬄ': 'ffl',  # ffl ligature
    }
)


def _normalize_pdf_text(text: str) -> str:
    """Normalise EDGAR-specific ligatures, smart quotes, and non-breaking spaces.

    Must run before the guardrail / finance-number pass so the finance regex
    can match patterns like '$391,035' that would otherwise contain '\xa0'.
    """
    text = text.translate(_LIGATURE_MAP)
    text = re.sub(r' {2,}', ' ', text)  # collapse multiple spaces (keep newlines)
    return text


# ─── SEC 10-K section taxonomy ───────────────────────────────────────────────

_SEC_PART_RE = re.compile(r'^PART\s+([IVX]+)\s*$', re.IGNORECASE)
_SEC_ITEM_RE = re.compile(r'^Item\s+(\d+[A-Z]?)\.?\s*(.*)', re.IGNORECASE)

_SEC_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
}


def _classify_10k_section(text: str) -> str | None:
    """If text matches a SEC 10-K Part or Item heading, return a canonical label.

    Returns None when the text is ordinary prose, not a section marker.
    """
    t = text.strip()
    m = _SEC_PART_RE.match(t)
    if m:
        return f"PART {m.group(1).upper()}"
    m = _SEC_ITEM_RE.match(t)
    if m:
        num = m.group(1).upper()
        desc = (m.group(2) or "").strip()
        canon = _SEC_ITEMS.get(num, desc)
        return f"Item {num}. {canon or desc}"
    return None


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
    """Return True only when the PDF requires a password to open.

    EDGAR filings (including Apple 10-K) carry Standard RC4 encryption metadata
    for digital-rights tracking but are fully accessible without a password.
    PyMuPDF's `needs_pass` is the correct check — `is_encrypted` is True for
    any encryption layer regardless of whether a password is actually required,
    and would incorrectly reject every SEC filing.
    """
    try:
        import fitz

        doc = fitz.open(file_path)
        needs_pw = doc.needs_pass
        doc.close()
        return needs_pw
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
            capture_output=True,
            text=True,
            timeout=30,
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


def _ocr_page_image(pix: Any, page_num: int) -> tuple[str, float]:
    try:
        import pytesseract
        from PIL import Image

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Explicit DPI silences Tesseract's "Estimating resolution as N" warnings
        # (pixmaps carry no DPI metadata).
        _ocr_cfg = "--dpi 200"
        ocr_text = (pytesseract.image_to_string(img, config=_ocr_cfg) or "").strip()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=_ocr_cfg)
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
        left: list[tuple[float, str]] = []
        right: list[tuple[float, str]] = []
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


def _coalesce_leading_labels(cells: list[str]) -> list[str]:
    """Merge consecutive leading non-numeric cells into one label cell.

    Fixes pdfplumber splitting long segment names across columns, e.g.:
    ['Wearables,', 'Home and', 'Accessories', '$37,005', ...] →
    ['Wearables, Home and Accessories', '$37,005', ...]
    """
    import re as _re_cl

    if len(cells) <= 1:
        return cells
    label_parts: list[str] = []
    j = 0
    while j < len(cells) and not _re_cl.search(r'[\d$]', cells[j]):
        label_parts.append(cells[j])
        j += 1
    if len(label_parts) <= 1:
        return cells
    combined = " ".join(p.rstrip(",").strip() for p in label_parts).strip()
    return [combined] + list(cells[j:])


def _merge_row_cells(row: list[str]) -> list[str]:
    """Merge adjacent currency/percent fragments so '$' + '201,183' → '$201,183'.

    pdfplumber splits PDF table cells at glyph boundaries, producing many None/empty
    columns and patterns like ['$', '201,183'] or ['(2)', '%']. We strip empty strings
    first so adjacency checks work even when None/empty columns separate real values.

    Additional text-strategy splits handled:
    - '$ 2' + '9,749' → '$29,749'  (split mid-dollar-amount)
    - '$16,74' + '1' → '$16,741'   (split mid-3-digit group)
    """
    import re as _re_merge

    # Collapse empty strings so adjacency checks below work correctly.
    cells = [c for c in row if c]
    merged: list[str] = []
    i = 0
    while i < len(cells):
        cell = cells[i]
        # Comma-terminated cell — either numeric or text-label continuation.
        if merged and merged[-1].endswith(","):
            # Numeric: "$109," + "633" → "$109,633"
            if cell.replace(".", "").isdigit():
                merged[-1] += cell
                i += 1
                continue
            # Text label: "Wearables," + "Home and" → "Wearables, Home and"
            if not any(ch.isdigit() for ch in cell) and not cell.startswith("$"):
                merged[-1] += " " + cell
                i += 1
                continue
        # "$" followed by any non-empty cell → "$12,345"
        if cell == "$" and i + 1 < len(cells):
            merged.append("$" + cells[i + 1])
            i += 2
            continue
        # "$ N" followed by "M,NNN..." — text strategy splits "$ 29,749" into "$ 2"+"9,749"
        # Pattern: cell is "$ " + digits (no comma), next cell starts with digit + comma groups
        if (
            _re_merge.match(r'^\$\s*\d+$', cell)
            and i + 1 < len(cells)
            and _re_merge.match(r'^\d(?:,\d{3})*$', cells[i + 1])
        ):
            merged.append('$' + cell[1:].strip() + cells[i + 1])
            i += 2
            continue
        # "$N,NN" followed by short digit — split mid 3-digit group: "$16,74"+"1"→"$16,741"
        if (
            _re_merge.match(r'^\$\d{1,3},\d{1,2}$', cell)
            and i + 1 < len(cells)
            and _re_merge.match(r'^\d{1,2}$', cells[i + 1])
        ):
            merged.append(cell + cells[i + 1])
            i += 2
            continue
        # "(N)" followed by "%" → "(N)%"
        if (
            cell.startswith("(")
            and cell.endswith(")")
            and i + 1 < len(cells)
            and cells[i + 1] == "%"
        ):
            merged.append(cell + "%")
            i += 2
            continue
        # Any value followed by standalone "%" → "N%"
        if i + 1 < len(cells) and cells[i + 1] == "%":
            merged.append(cell + "%")
            i += 2
            continue
        merged.append(cell)
        i += 1
    # Coalesce any remaining consecutive leading text-label cells (e.g. after
    # comma-merge only partially collapsed a multi-word segment name).
    return _coalesce_leading_labels(merged)


def _table_to_text(rows: Iterable[Iterable[object]]) -> str:
    import re as _re

    cleaned = [
        [str(cell or "").strip() for cell in row]
        for row in (rows or [])
        if any(cell for cell in row)
    ]
    if not cleaned:
        return ""
    result_rows = []
    for row in cleaned:
        merged = _merge_row_cells(row)
        if not merged:
            continue
        # Skip rows that contain only prose fragments with no financial content.
        # Text-strategy pdfplumber may include title/header text as table rows;
        # we keep only rows that have at least one digit, $, or % character.
        if not _re.search(r'[\d$%]', " ".join(merged)):
            continue
        result_rows.append(" | ".join(merged))
    return "\n".join(result_rows)


def _table_to_markdown(rows: list[list[str]]) -> str:
    import re as _re_md

    if not rows:
        return ""
    # Apply cell merging to every row before building markdown
    cleaned_rows = [_merge_row_cells([str(c or "").strip() for c in row]) for row in rows]
    cleaned_rows = [r for r in cleaned_rows if r]
    if not cleaned_rows:
        return ""

    # Detect data-first tables: the first row is actual financial data, not a header.
    # Signal: first cell is a pure text label (no digits/$ sign) while the remaining
    # cells are financial values ($ or 5+ digit numbers). This pattern appears in
    # Apple 10-K gross-margin and product net-sales tables where pdfplumber finds no
    # separate header row. Using the data row as the markdown header confuses the LLM
    # into treating segment names as column labels.
    first_row = cleaned_rows[0]
    first_cell = first_row[0] if first_row else ""
    rest_cells = first_row[1:] if len(first_row) > 1 else []
    _label_first = bool(first_cell) and not _re_md.search(r'[\d$]', first_cell)
    _finance_values = bool(rest_cells) and bool(
        _re_md.search(r'\$\d|\b\d{5,}', " ".join(rest_cells))
    )
    _is_data_first = _label_first and _finance_values

    if _is_data_first:
        # All rows are data; generate synthetic fiscal-year header sized to the
        # widest row so no values are truncated.
        ncols = max(len(r) for r in cleaned_rows)
        _base = ["Segment", "FY2024", "FY2023", "FY2022"]
        header = (_base + [f"Col{j+5}" for j in range(max(0, ncols - 4))])[:ncols]
        body = cleaned_rows
        separator = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            padded = list(row) + [""] * max(0, ncols - len(row))
            lines.append("| " + " | ".join(padded[:ncols]) + " |")
    else:
        # Scan all rows for a fiscal-year header row.
        # Some tables (e.g. Note 2 – Revenue on page 38) have many prose paragraph
        # rows before the actual data table, so limiting to the first N rows would
        # miss the year header. We scan the entire cleaned_rows list.
        # We distinguish year-header rows from title/description rows by requiring:
        #   (a) 2+ distinct fiscal years (20XX) when cells are joined, AND
        #   (b) no individual cell longer than 30 chars (description rows have long
        #       first cells; prose paragraphs are merged into one very long cell).
        _year_header_idx = None
        _years_found: list[str] = []
        for _scan_idx, _scan_row in enumerate(cleaned_rows):
            _joined = "".join(_scan_row).replace(" ", "")
            _yrs = sorted(set(_re_md.findall(r'20\d{2}', _joined)), reverse=True)
            if len(_yrs) >= 2:
                _max_cell = max((len(c) for c in _scan_row), default=0)
                if _max_cell <= 30:
                    _year_header_idx = _scan_idx
                    _years_found = _yrs
                    break

        if _year_header_idx is not None:
            # Data rows follow the year-header row.
            _data_rows = cleaned_rows[_year_header_idx + 1 :]
            # Compute ncols from financial data rows only, within the first 20 rows
            # after the year header.  The thousands-separator pattern (\d{1,3}(?:,\d{3})+)
            # matches "37,005", "$391,035", "$201,183" etc., but NOT year values like
            # "2024", "2023," or sentence fragments that incidentally contain digits.
            _fin_rows = [
                r
                for r in _data_rows[:20]
                if any(_re_md.search(r'\d{1,3}(?:,\d{3})+', c) for c in r)
            ]
            ncols = max(
                (len(r) for r in _fin_rows),
                default=max((len(r) for r in _data_rows[:20] if r), default=4),
            )
            # Include Change columns only when the table is wide enough for them.
            _add_change = ncols > len(_years_found) + 1
            _clean_hdr: list[str] = ["Segment"]
            for yr in _years_found:
                _clean_hdr.append(f"FY{yr}")
                if _add_change and len(_clean_hdr) < ncols:
                    _clean_hdr.append("Change")
            while len(_clean_hdr) < ncols:
                _clean_hdr.append("")
            header = _clean_hdr[:ncols]
            body = _data_rows
        else:
            header = cleaned_rows[0]
            body = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

        separator = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _detect_fiscal_years_from_rows(rows: list[list[str]]) -> list[str]:
    """Return sorted list of FY-prefixed fiscal years found in the first 15 rows."""
    import re as _re_fy

    text = " ".join(str(c or "") for row in rows[:15] for c in row)
    years = sorted(set(_re_fy.findall(r'20\d{2}', text)), reverse=True)
    return [f"FY{y}" for y in years[:5]]


def _infer_table_title(rows: list[list[str]], section_title: str = "") -> str:
    """Classify a financial table by its dominant keyword pattern."""
    text = " ".join(str(c or "") for row in rows[:6] for c in row).lower()
    if "gross margin" in text:
        return "Gross Margin by Segment"
    if "net sales" in text and "cost of sales" not in text:
        return "Net Sales by Product Category"
    if "provision for income" in text or "effective tax" in text:
        return "Income Tax Provision"
    if "repurchase" in text or "capital return" in text:
        return "Capital Return Program"
    if "dividend" in text:
        return "Dividends and Capital Return"
    if "cost of sales" in text or "operating income" in text:
        return "Consolidated Statements of Operations"
    if "cash and cash equivalent" in text or "operating activities" in text:
        return "Cash Flow Statement"
    if "shares" in text and ("retained" in text or "equity" in text):
        return "Shareholders Equity Statement"
    if section_title:
        return section_title
    return "Financial Table"


def _table_to_nl_summary(md: str, page: int, table_title: str, doc_name: str = "") -> str:
    """Convert a clean markdown table into NL sentences for semantic retrieval.

    Each data row becomes: "- Segment: FY2024 $37,005M, FY2023 $39,845M, FY2022 $41,241M"
    so that embedding similarity aligns tightly with queries like "Wearables revenue 2024".
    """
    import re as _re_nl

    lines = [l.strip() for l in md.strip().split("\n") if l.strip()]
    if not lines:
        return ""

    # Locate the clean header row that contains FY20XX tokens
    header_cells: list[str] | None = None
    data_start = 0
    for i, line in enumerate(lines):
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells:
            continue
        if all(_re_nl.match(r"-+$", c) for c in cells):
            continue
        if _re_nl.search(r"FY20\d{2}|Segment", " ".join(cells), _re_nl.I):
            header_cells = cells
            data_start = i + 2  # skip header + separator
            break

    if header_cells is None:
        return ""

    # Identify which column indexes carry FY20XX values
    year_cols: list[str] = []
    year_idxs: list[int] = []
    for j, c in enumerate(header_cells):
        if _re_nl.match(r"FY20\d{2}$", c):
            year_cols.append(c)
            year_idxs.append(j)

    if not year_cols:
        return ""

    loc = ", ".join(p for p in [doc_name, f"Page {page}" if page else ""] if p)
    header_str = " | ".join(header_cells)
    out = [f"{table_title} ({loc})" if loc else table_title, f"Columns: {header_str}", ""]

    for line in lines[data_start:]:
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells or all(_re_nl.match(r"-+$", c) for c in cells):
            continue
        segment = cells[0]
        # Skip empty, disclosure/footnote rows (very long first cells)
        if not segment or not any(ch.isalpha() for ch in segment) or len(segment) > 70:
            continue
        val_parts: list[str] = []
        for yr, idx in zip(year_cols, year_idxs):
            if idx < len(cells):
                v = cells[idx]
                if not v or v in ("---", "—", ""):
                    continue
                # Normalize "$ 383,285" (space after $) → "$383,285"
                v = _re_nl.sub(r'^\$\s+', '$', v)
                # Accept comma-formatted amounts: "37,005", "$201,183", "$383,285"
                _plain = _re_nl.match(r"^\d{1,3}(?:,\d{3})+$", v)
                _dollar = _re_nl.match(r"^\$\d{1,3}(?:,\d{3})+$", v)
                # Accept percentage values: "37.2%", "24.1%", "46.2%", "(7)%"
                _pct = _re_nl.match(r"^-?\d+(?:\.\d+)?%$", v) or _re_nl.match(
                    r"^\(\d+(?:\.\d+)?\)%$", v
                )
                if not (_plain or _dollar or _pct):
                    continue
                if _plain:
                    v = f"${v}M"
                elif _dollar:
                    v = f"{v}M"
                # Percentages kept as-is (e.g. "37.2%")
                val_parts.append(f"{yr} {v}")
        # Only emit the row when at least one fiscal-year value is a real financial amount
        if val_parts:
            out.append(f"- {segment}: {', '.join(val_parts)}")

    return "\n".join(out) if len(out) > 3 else ""


def _detect_font_size(page: Any, text_block: str) -> float | None:
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


def _detect_heading(
    text: str, font_size: float | None, is_bold: bool, body_font_size: float | None
) -> bool:
    t = text.strip()
    if not t or len(t) > 250:
        return False
    # SEC 10-K Part / Item headings are always headings regardless of length or terminal char.
    # "Item 7. Management's Discussion and Analysis of Financial Condition and Results of
    # Operations" is 88 chars / 15+ words — the generic rules below would miss it.
    if _classify_10k_section(t) is not None:
        return True
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
    ) -> list[RawExtract]:
        source = path.name
        file_path = str(path)
        logger.info(
            event="extraction_start", modality="pdf", file=str(path), size=path.stat().st_size
        )
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
            extracts: list[RawExtract] = []
            header_footer_counts: dict[str, int] = {}

            # Estimate body font size from first few text-heavy pages
            body_font_size: float | None = None
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

            # PASS 1: collect page metadata + pixmap bytes for OCR pages.
            # All PyMuPDF calls stay on main thread (fitz is not thread-safe).
            _page_records: list[dict] = []
            for i, page in enumerate(pdf, start=1):
                rotation = _get_page_rotation(page)
                if rotation not in (0, 360):
                    page.set_rotation(0)
                raw_text = (page.get_text() or "").strip()
                if raw_text:
                    # Normalise EDGAR ligatures/non-breaking spaces BEFORE guardrail
                    # so the finance regex sees plain '$391,035' not '$\xa0391,035'.
                    raw_text = _normalize_pdf_text(raw_text)
                    raw_text = self._sanitize(raw_text, surface="pdf_ingest")
                rect = page.rect
                page_area = max(rect.width * rect.height, 1)
                density = _text_density(raw_text, page_area)
                needs_full_ocr = not raw_text
                # Supplemental OCR re-runs Tesseract on pages that already have a
                # text layer but low density. Off by default — it's slow on filings
                # (10-Ks) and pages with text rarely need it. Gated behind config.
                needs_supl_ocr = (
                    settings.PDF_SUPPLEMENTAL_OCR_ENABLED and bool(raw_text) and density < 0.01
                )
                pix_info: tuple[bytes, int, int] | None = None
                if needs_full_ocr or needs_supl_ocr:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        pix_info = (bytes(pix.samples), pix.width, pix.height)
                    except Exception as exc:
                        logger.warning("pdf_pixmap_failed", page=i, error=str(exc))
                _page_records.append(
                    {
                        "page_num": i,
                        "page": page,
                        "raw_text": raw_text,
                        "rect": rect,
                        "density": density,
                        "needs_full_ocr": needs_full_ocr,
                        "needs_supl_ocr": needs_supl_ocr,
                        "pix_info": pix_info,
                    }
                )

            # PASS 2: parallel Tesseract OCR — Tesseract is thread-safe, semaphore
            # bounds CPU concurrency to settings.PDF_OCR_WORKERS processes.
            _ocr_sem = asyncio.Semaphore(settings.PDF_OCR_WORKERS)

            async def _ocr_bytes_async(
                samples: bytes, w: int, h: int, page_num: int
            ) -> tuple[int, tuple[str, float]]:
                async with _ocr_sem:

                    def _run() -> tuple[str, float]:
                        try:
                            import pytesseract
                            from PIL import Image as _PILImg

                            img = _PILImg.frombytes("RGB", [w, h], samples)
                            # Pixmaps are rendered at dpi=200 but carry no DPI
                            # metadata; tell Tesseract explicitly to silence the
                            # "Estimating resolution as N" warnings/errors.
                            _ocr_cfg = "--dpi 200"
                            ocr_text = (
                                pytesseract.image_to_string(img, config=_ocr_cfg) or ""
                            ).strip()
                            data = pytesseract.image_to_data(
                                img, output_type=pytesseract.Output.DICT, config=_ocr_cfg
                            )
                            confs = [
                                int(c)
                                for c in data["conf"]
                                if str(c).lstrip("-").isdigit() and int(c) >= 0
                            ]
                            confidence = (
                                round(sum(confs) / max(len(confs), 1) / 100.0, 3) if confs else 0.5
                            )
                            return ocr_text, confidence
                        except Exception as exc:
                            logger.warning("ocr_page_failed", page=page_num, error=str(exc))
                            return "", 0.0

                    return page_num, await asyncio.to_thread(_run)

            _ocr_tasks = [
                _ocr_bytes_async(*rec["pix_info"], rec["page_num"])
                for rec in _page_records
                if rec["pix_info"] is not None
            ]
            _ocr_results: dict[int, tuple[str, float]] = {}
            if _ocr_tasks:
                for _pnum, _res in await asyncio.gather(*_ocr_tasks):
                    _ocr_results[_pnum] = _res

            # PASS 3: finalize extracts using parallel OCR results (main thread).
            for rec in _page_records:
                i = rec["page_num"]
                page = rec["page"]
                raw_text = rec["raw_text"]
                rect = rec["rect"]
                ocr_conf = 1.0
                is_ocr = False

                if rec["needs_full_ocr"]:
                    _ocr_invocations.inc()
                    is_ocr = True
                    if i in _ocr_results:
                        ocr_result, ocr_conf = _ocr_results[i]
                        if ocr_result:
                            raw_text = ocr_result
                    elif rec["pix_info"] is None:
                        logger.warning(
                            "pdf_ocr_fallback_failed", page=i, error="pixmap unavailable"
                        )
                elif rec["needs_supl_ocr"]:
                    _ocr_invocations.inc()
                    if i in _ocr_results:
                        ocr_result, ocr_conf = _ocr_results[i]
                        if ocr_result and len(ocr_result) > len(raw_text):
                            raw_text = ocr_result
                            is_ocr = True

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

                # Multi-column reading order via block bboxes.
                # _extract_pdf_text_multicolumn calls page.get_text() fresh,
                # so we must re-apply normalization before length comparison.
                try:
                    mc_text = _extract_pdf_text_multicolumn(page)
                    if mc_text:
                        mc_text = _normalize_pdf_text(mc_text)
                        if len(mc_text) >= len(page_text):
                            page_text = mc_text
                except Exception:
                    pass

                # Watermark stripping — remove lines that are solely watermark tokens
                _page_lines = page_text.split("\n")
                _filtered = [
                    ln for ln in _page_lines if ln.strip().upper() not in _PDF_WATERMARK_STRIP
                ]
                if len(_filtered) < len(_page_lines):
                    page_text = "\n".join(_filtered).strip()

                # Determine extract_type: scanned vs prose vs heading
                font_sz = _detect_font_size(page, page_text[:50])
                is_bold = _detect_is_bold(page, page_text[:50])
                # For heading detection, check the first non-blank line —
                # headings are at the top of the page, not the full page text.
                _first_line = next((ln.strip() for ln in page_text.split("\n") if ln.strip()), "")
                is_heading = _detect_heading(
                    _first_line or page_text, font_sz, is_bold, body_font_size
                )

                # Classify SEC 10-K Part/Item headings for downstream hierarchy building
                _sec_label = _classify_10k_section(_first_line) if _first_line else None

                if is_ocr and not raw_text.strip():
                    ext_type = "scanned_page"
                elif is_heading:
                    ext_type = "heading"
                else:
                    ext_type = "prose"

                extracts.append(
                    RawExtract(
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
                            "sec_section": _sec_label,  # canonical SEC label e.g. "Item 7. MD&A"
                        },
                    )
                )

                # Embedded images → image_raw extracts
                for img_idx, img_ref in enumerate(page.get_images(full=True)):
                    try:
                        xref = img_ref[0]
                        base = pdf.extract_image(xref)
                        img_bytes = base.get("image")
                        if not img_bytes or len(img_bytes) < 256:
                            continue
                        extracts.append(
                            RawExtract(
                                text="",
                                extract_type="image_region",
                                page=i,
                                raw_source_ref=f"pdf:{path.name}|page:{i}|img:{img_idx}",
                                raw_bytes=img_bytes,
                                extra={
                                    "img_ext": base.get("ext", "png"),
                                    "context_text": page_text[:500],
                                },
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "pdf_image_extract_failed", page=i, img_idx=img_idx, error=str(exc)
                        )

            # Hyperlinks as footnote extracts
            try:
                for i, page in enumerate(pdf, start=1):
                    for link in page.get_links():
                        uri = link.get("uri", "")
                        if uri:
                            uri = self._sanitize(uri, surface="pdf_hyperlink_ingest")
                            extracts.append(
                                RawExtract(
                                    text=f"[HYPERLINK page={i}] {uri}",
                                    extract_type="footnote",
                                    page=i,
                                    raw_source_ref=f"pdf:{path.name}|page:{i}|link",
                                )
                            )
            except Exception as exc:
                logger.warning("pdf_hyperlink_extraction_failed", error=str(exc))
            finally:
                pdf.close()

            # Strip repeated header/footer text from prose extracts.
            # Guard: minimum 4 characters to prevent single-digit footnote markers
            # (e.g. "1", "2", "3") from being treated as headers and corrupting
            # every financial number in the document via str.replace.
            repeated = {
                k
                for k, v in header_footer_counts.items()
                if v > 3
                and k.strip()
                and len(k.strip()) >= 4
                and not k.strip().replace(",", "").replace(".", "").isdigit()
            }
            if repeated:
                for ex in extracts:
                    if ex.extract_type in ("prose", "heading"):
                        for r in repeated:
                            ex.text = ex.text.replace(r, "").strip()

            # Build page → last known section title so pdfplumber table extracts
            # (appended after all prose) can carry the correct section context to the chunker.
            _page_to_section: dict[int, str] = {}
            _last_sec = ""
            for _ex in extracts:
                if _ex.extract_type == "heading" and _ex.text:
                    _last_sec = _ex.text.strip()
                if _ex.page is not None and _last_sec:
                    _page_to_section[_ex.page] = _last_sec

            # Tables via pdfplumber — dual strategy for borderless financial tables.
            # Lines strategy gives clean bordered tables; text strategy captures
            # borderless tables (e.g., Apple 10K product category breakdown) that
            # lines strategy partially misses. We use text strategy when it finds
            # more financial data rows (rows containing digits or $) for the same page.
            try:
                import re as _plumber_re

                import pdfplumber

                _TEXT_STRAT = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                }

                def _count_data_rows(tables: list) -> int:
                    return sum(
                        1
                        for t in tables
                        for r in t
                        if any(_plumber_re.search(r'[\d$]', str(c or '')) for c in r)
                    )

                with pdfplumber.open(file_path) as pdf_p:
                    for i, page in enumerate(pdf_p.pages, start=1):
                        lines_tables = page.extract_tables() or []
                        if not lines_tables:
                            continue
                        # Prefer text strategy when it captures more financial rows
                        # (handles borderless financial statement tables)
                        best_tables = lines_tables
                        try:
                            text_tables = page.extract_tables(_TEXT_STRAT) or []
                            lines_dr = _count_data_rows(lines_tables)
                            text_dr = _count_data_rows(text_tables)

                            # Use text strategy only when it finds significantly more data rows
                            # (indicates a borderless table missed by lines strategy, e.g. the
                            # Apple 10K product-category breakdown which lines strategy truncates).
                            # Threshold lines*2+2 prevents text strategy from "winning" on pages
                            # where it merely counts prose sentences containing year numbers.
                            # Quality check: reject text strategy if it still shatters dollar
                            # amounts AFTER _merge_row_cells repair. D1 fix handles common
                            # patterns ("$ 2"+"9,749"→"$29,749"); this gate catches unfixable
                            # residual fragments that confuse the LLM.
                            def _has_unrepaired_shatter(tables: list) -> bool:
                                import re as _qre

                                for t in tables:
                                    for row in t:
                                        raw = [
                                            str(c or "").strip() for c in row if (c or "").strip()
                                        ]
                                        repaired = _merge_row_cells(raw)
                                        for c in repaired:
                                            # Still lone "$ N" (< 3 digits) after repair
                                            if _qre.match(r'^\$\s*\d{1,2}$', c):
                                                return True
                                            # Still truncated comma group after repair
                                            if _qre.match(r'^\$?\d{1,3},\d{1,2}$', c):
                                                return True
                                return False

                            if text_dr > lines_dr * 2 + 2 and not _has_unrepaired_shatter(
                                text_tables
                            ):
                                best_tables = text_tables
                        except Exception:
                            pass

                        for t_idx, table in enumerate(best_tables):
                            rows = [[str(cell or "").strip() for cell in row] for row in table]
                            md = _table_to_markdown(rows)
                            if not md.strip():
                                continue
                            _sec_ctx = _page_to_section.get(i, "")
                            _t_title = _infer_table_title(rows, _sec_ctx)
                            _fiscal_yrs = _detect_fiscal_years_from_rows(rows)

                            # Store ONLY clean markdown — no garbled raw text that confuses the LLM
                            md_clean = self._sanitize(md, surface="pdf_table_ingest")
                            md_clean = self._scrub_pii(md_clean, surface="pdf_table_ingest")
                            if md_clean:
                                extracts.append(
                                    RawExtract(
                                        text=md_clean,
                                        extract_type="table_row",
                                        page=i,
                                        raw_source_ref=f"pdf:{path.name}|page:{i}|table:{t_idx}",
                                        extra={
                                            "markdown": md_clean,
                                            "table_index": t_idx,
                                            "section_title": _sec_ctx,
                                            "fiscal_years": _fiscal_yrs,
                                            "table_title": _t_title,
                                        },
                                    )
                                )

                            # NL summary: "- Wearables: FY2024 $37,005M, FY2023 $39,845M"
                            # High semantic overlap with natural language queries.
                            _nl = _table_to_nl_summary(md, i, _t_title, path.stem)
                            if _nl:
                                _nl_clean = self._sanitize(_nl, surface="pdf_table_ingest")
                                _nl_clean = self._scrub_pii(_nl_clean, surface="pdf_table_ingest")
                                if _nl_clean:
                                    extracts.append(
                                        RawExtract(
                                            text=_nl_clean,
                                            extract_type="table_summary",
                                            page=i,
                                            raw_source_ref=f"pdf:{path.name}|page:{i}|table:{t_idx}|nl",
                                            extra={
                                                "fiscal_years": _fiscal_yrs,
                                                "table_title": _t_title,
                                                "section_title": _sec_ctx,
                                            },
                                        )
                                    )

                    # Outline/bookmarks
                    try:
                        outline = getattr(pdf_p, "outline", None)
                        if outline:
                            outline_text = self._sanitize(
                                str(outline)[:2000], surface="pdf_outline_ingest"
                            )
                            outline_text = self._scrub_pii(
                                outline_text, surface="pdf_outline_ingest"
                            )
                            if outline_text:
                                extracts.append(
                                    RawExtract(
                                        text=f"[OUTLINE]\n{outline_text}",
                                        extract_type="heading",
                                        raw_source_ref=f"pdf:{path.name}|outline",
                                        extra={"is_outline": True},
                                    )
                                )
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("pdf_table_extraction_failed", error=str(exc))

            # Camelot — lattice (bordered) table fallback for tables pdfplumber misses
            # Deduplicates against pdfplumber results by first-50-char content hash.
            try:
                import camelot

                _plumber_sigs = {ex.text[:50] for ex in extracts if ex.extract_type == "table_row"}
                _camelot_tables = camelot.read_pdf(file_path, pages="1-end", flavor="lattice")
                for ct in _camelot_tables:
                    if ct.df.empty or ct.parsing_report.get("accuracy", 0) < 50:
                        continue
                    rows = [list(ct.df.columns)] + [list(r) for r in ct.df.itertuples(index=False)]
                    rows = [[str(cell or "").strip() for cell in row] for row in rows]
                    md = _table_to_markdown(rows)
                    _md_sig = md[:50]
                    if not md.strip() or _md_sig in _plumber_sigs:
                        continue
                    _t_title = _infer_table_title(rows)
                    _fiscal_yrs = _detect_fiscal_years_from_rows(rows)
                    md_clean = self._sanitize(md, surface="pdf_table_ingest")
                    md_clean = self._scrub_pii(md_clean, surface="pdf_table_ingest")
                    if md_clean:
                        extracts.append(
                            RawExtract(
                                text=md_clean,
                                extract_type="table_row",
                                page=ct.page,
                                raw_source_ref=f"pdf:{path.name}|page:{ct.page}|camelot",
                                extra={
                                    "markdown": md_clean,
                                    "extractor": "camelot",
                                    "accuracy": ct.parsing_report.get("accuracy"),
                                    "fiscal_years": _fiscal_yrs,
                                    "table_title": _t_title,
                                },
                            )
                        )
                        _plumber_sigs.add(_md_sig)
                    _nl = _table_to_nl_summary(md, ct.page, _t_title, path.stem)
                    if _nl:
                        _nl_clean = self._sanitize(_nl, surface="pdf_table_ingest")
                        _nl_clean = self._scrub_pii(_nl_clean, surface="pdf_table_ingest")
                        if _nl_clean:
                            extracts.append(
                                RawExtract(
                                    text=_nl_clean,
                                    extract_type="table_summary",
                                    page=ct.page,
                                    raw_source_ref=f"pdf:{path.name}|page:{ct.page}|camelot|nl",
                                    extra={"fiscal_years": _fiscal_yrs, "table_title": _t_title},
                                )
                            )
            except ImportError:
                pass  # camelot optional; pdfplumber handles standard tables
            except Exception as exc:
                logger.warning("pdf_camelot_extraction_failed", error=str(exc))

            if not extracts:
                raise ValueError("NO_EXTRACTS_PRODUCED")

            # PAGE-NUMBER REMAP — the extract `page` values above are 1-based PDF
            # page INDICES. Filings (10-K/10-Q) have cover/TOC pages before printed
            # page 1, so the PDF index is offset from the document's own printed
            # page number (what a reader looks up and what a citation should show).
            # Recover the printed page from each page's footer ("... 2024 Form 10-K
            # | 40" or "40 | Apple Inc. ...") and remap; where a footer is absent,
            # apply the median offset. No-op for docs whose footers carry no page.
            try:
                _foot_re = re.compile(
                    r'Form\s*10-?K\s*\|\s*(\d{1,3})\b|\b(\d{1,3})\s*\|\s*Apple\s+Inc',
                    re.IGNORECASE,
                )
                _printed: dict[int, int] = {}
                for _rec in _page_records:
                    _tail = (_rec.get("raw_text") or "")[-400:]
                    _mm = _foot_re.search(_tail)
                    if _mm:
                        try:
                            _printed[_rec["page_num"]] = int(_mm.group(1) or _mm.group(2))
                        except (TypeError, ValueError):
                            pass
                if len(_printed) >= 3:
                    _offs = sorted(idx - pp for idx, pp in _printed.items())
                    _median_off = _offs[len(_offs) // 2]
                    if _median_off != 0:
                        for _ex in extracts:
                            if getattr(_ex, "page", None) is not None:
                                _mapped = _printed.get(_ex.page, _ex.page - _median_off)
                                if _mapped and _mapped >= 1:
                                    _ex.page = _mapped
                        logger.info(
                            event="pdf_page_remap_applied",
                            median_offset=_median_off,
                            footer_pages=len(_printed),
                            file=str(path),
                        )
            except Exception as _remap_err:
                logger.warning(event="pdf_page_remap_failed", error=str(_remap_err))

            _EXTRACTS_TOTAL.inc(len(extracts))
            logger.info(
                event="extraction_complete", modality="pdf", file=str(path), extracts=len(extracts)
            )
            return extracts
        except Exception as _exc:
            _EXTRACT_ERRORS.inc()
            logger.error(event="extraction_failed", modality="pdf", source=source, error=str(_exc))
            raise


# ─── Backward-compat ingest() — full pipeline ─────────────────────────────────


def _base_structure(doc_id: str, session_id: str, source_path: str, **extra: Any) -> dict[str, Any]:
    return {"doc_id": doc_id, "session_id": session_id, "source_path": source_path, **extra}


def _redact_pii(text: str) -> tuple[str, dict[str, int]]:
    if not settings.PII_DETECTION_ENABLED:
        return text, {}
    entity_counts: dict[str, int] = {}
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        entities = getattr(
            settings,
            "PII_ENTITIES",
            [
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "CREDIT_CARD",
                "IP_ADDRESS",
            ],
        )
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


async def ingest(file_path: str, session_id: str) -> list[IngestedDocument]:
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
                logger.info(
                    "pdf_ingest_start", file=path.name, size=file_size, session_id=session_id
                )

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
                documents: list[IngestedDocument] = []
                header_footer_counts: dict[str, int] = {}

                for i, page in enumerate(pdf, start=1):
                    rotation = _get_page_rotation(page)
                    if rotation not in (0, 360):
                        page.set_rotation(0)

                    raw_text = (page.get_text() or "").strip()
                    if raw_text:
                        raw_text = _normalize_pdf_text(raw_text)
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
                        page_sub_chunks: list[str] = []
                        if len(page_text) > settings.CHUNK_SIZE:
                            try:
                                from langchain_text_splitters import RecursiveCharacterTextSplitter

                                _splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=settings.CHUNK_SIZE,
                                    chunk_overlap=settings.CHUNK_OVERLAP,
                                    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                                )
                                page_sub_chunks = [
                                    c.strip() for c in _splitter.split_text(page_text) if c.strip()
                                ]
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
                                        doc_id,
                                        session_id,
                                        source_path_str,
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
                                d.structure.update(
                                    {
                                        "doc_id": doc_id,
                                        "page": i,
                                        "context_text": page_text[:500] if page_text else "",
                                        "source_path": source_path_str,
                                    }
                                )
                                documents.append(d)
                        except Exception as exc:
                            logger.warning(
                                "pdf_image_extract_failed", page=i, img_idx=img_idx, error=str(exc)
                            )

                # Strip repeated headers/footers.
                # Guard: minimum 4 characters — see extract() for full rationale.
                repeated = {
                    k
                    for k, v in header_footer_counts.items()
                    if v > 3
                    and k.strip()
                    and len(k.strip()) >= 4
                    and not k.strip().replace(",", "").replace(".", "").isdigit()
                }
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
                                combined = _sanitize_text(
                                    f"{txt}\n\n{md}", surface="pdf_table_ingest"
                                )
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
                                            doc_id,
                                            session_id,
                                            source_path_str,
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
                                outline_text = _sanitize_text(
                                    str(outline)[:2000], surface="pdf_outline_ingest"
                                )
                                outline_text = _pii_scrub(
                                    outline_text, surface="pdf_outline_ingest"
                                )
                                documents.append(
                                    IngestedDocument(
                                        text=f"[OUTLINE]\n{outline_text}",
                                        modality="text",
                                        subtype="heading",
                                        source_type="pdf",
                                        source=source_name,
                                        structure=_base_structure(
                                            doc_id,
                                            session_id,
                                            source_path_str,
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
                                            doc_id,
                                            session_id,
                                            source_path_str,
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
                logger.info(
                    "pdf_ingest_success",
                    file=path.name,
                    docs=len(documents),
                    pages=total_pages,
                    latency=latency,
                    session_id=session_id,
                )
                return documents

            except Exception as exc:
                latency = round(time.time() - start, 2)
                error_type = type(exc).__name__
                _ingest_duration.labels(status="error").observe(latency)
                _ingest_errors.labels(error_type=error_type).inc()
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                logger.error(
                    "pdf_ingest_failed",
                    file=path.name,
                    session_id=session_id,
                    error=str(exc),
                    error_type=error_type,
                    latency=latency,
                )
                raise


def ingest_sync(file_path: str, session_id: str) -> list[IngestedDocument]:
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


# ─── Production path: PdfIngestor → PdfChunker ────────────────────────────────


async def ingest_pdf_full(file_path: str, session_id: str) -> list[IngestedDocument]:
    """Production PDF ingestion: PdfIngestor.extract() → PdfChunker.chunk().

    Replaces the backward-compat ingest() as the INGESTION_HANDLERS entry.
    Produces modality='pdf' docs with full Phase 1.2/2.2 metadata:
    section_title, section_hierarchy, chunk_type, is_ocr, has_figure,
    finance_entities, char_start, char_end, token_count, page_range, etc.
    """
    from app.chunking import chunk_raw_extracts
    from app.ingestion.schema import UniversalMetadata

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

    meta = UniversalMetadata(
        source_path=str(path.resolve()),
        modality="pdf",
        file_size_bytes=file_size,
        custom_fields={"session_id": session_id},
    )

    ingestor = PdfIngestor()
    extracts = await ingestor.extract(path, meta)

    if not extracts:
        raise ValueError("NO_EXTRACTS_PRODUCED")

    docs = chunk_raw_extracts(extracts, meta, "pdf")

    for doc in docs:
        struct = getattr(doc, "structure", None)
        if struct is not None and struct.get("session_id") in (None, "default"):
            struct["session_id"] = session_id

    logger.info(
        event="ingest_pdf_full_complete",
        file=path.name,
        extracts=len(extracts),
        docs=len(docs),
        session_id=session_id,
    )
    return docs
