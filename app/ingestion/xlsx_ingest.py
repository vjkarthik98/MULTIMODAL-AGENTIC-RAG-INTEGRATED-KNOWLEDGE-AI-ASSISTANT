"""
XLSX/XLS ingestor — Phase 1 per-modality refactor.

XlsxIngestor.extract() → List[RawExtract]   (extraction only; no chunking)
ingest()               → List[IngestedDocument]  (backward-compat; full pipeline)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "xlsx_ingest_duration_seconds",
    "XLSX ingestion duration",
    ["status"],
)
_ingest_errors = Counter(
    "xlsx_ingest_errors_total",
    "XLSX ingestion errors by type",
    ["error_type"],
)

_semaphore = asyncio.Semaphore(5)

_UNIT_SCALE_PATTERNS: List[Tuple[Any, str]] = [
    (re.compile(r'\bin billions?\b|\(\$\s*billions?\)', re.IGNORECASE), "billions"),
    (re.compile(r'\bin millions?\b|\(\$\s*millions?\)', re.IGNORECASE), "millions"),
    (re.compile(r'\bin thousands?\b|\busd ?000s?\b|\(\$\s*thousands?\)', re.IGNORECASE), "thousands"),
]

_PNL_SHEET_KW: set = {"income", "p&l", "profit", "earnings", "loss", "pnl", "pl"}

_PNL_GROUP_MARKERS: List[Tuple[Any, str]] = [
    (re.compile(r'\bnet income\b|\bnet profit\b|\bnet loss\b', re.IGNORECASE), "Per-Share Data"),
    (re.compile(r'\bebit\b|\boperating income\b|\boperating profit\b', re.IGNORECASE), "Below-the-Line"),
    (re.compile(r'\bgross profit\b', re.IGNORECASE), "Cost Lines"),
    (re.compile(r'\brevenue\b|\bnet sales\b|\btotal revenue\b', re.IGNORECASE), "Revenue Lines"),
]


# ─── Utilities ────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _quality(text: str) -> float:
    l = len(text)
    if l < 50:
        return 0.2
    if l < 200:
        return 0.5
    return 1.0


def _detect_unit_scale(sheet_name: str, rows: List[List[str]]) -> str:
    search_text = sheet_name + " " + " ".join(c for row in rows[:5] for c in row)
    for pattern, scale in _UNIT_SCALE_PATTERNS:
        if pattern.search(search_text):
            return scale
    return "units"


def _is_pnl_sheet(sheet_name: str) -> bool:
    return any(kw in sheet_name.lower() for kw in _PNL_SHEET_KW)


def _table_to_text(rows: Any) -> str:
    cleaned = [
        [str(cell or "").strip() for cell in row]
        for row in (rows or [])
        if any(cell for cell in row)
    ]
    if not cleaned:
        return ""
    return "\n".join(" | ".join(row) for row in cleaned)


def _pii_scrub(text: str, surface: str) -> str:
    try:
        from app.guardrails.pii import scrub_pii as _gp_scrub
        cleaned, _ = _gp_scrub(text)
        return cleaned
    except Exception:
        return text


def _sanitize_text(text: str, surface: str) -> str:
    try:
        from app.guardrails.input_guard import sanitize as _g
        return _g(text, surface=surface)
    except Exception:
        return text


def _try_float(v: Any) -> Optional[float]:
    try:
        f = float(str(v).replace(",", ""))
        return f if str(f) != "nan" else None
    except (ValueError, TypeError):
        return None


def _load_worksheet_rows(ws: Any) -> Tuple[List[List[str]], str]:
    """Load visible rows + detect unit scale. Returns (rows, unit_scale)."""
    from openpyxl.utils import get_column_letter
    hidden_col_letters = {
        col_letter for col_letter, cd in ws.column_dimensions.items() if cd.hidden
    }
    all_rows: List[List[str]] = []
    for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
        rd = ws.row_dimensions.get(row_idx)
        if rd and rd.hidden:
            continue
        cells = [
            str(c if c is not None else "").strip()
            for col_idx, c in enumerate(row, start=1)
            if get_column_letter(col_idx) not in hidden_col_letters
        ]
        all_rows.append(cells)
    non_empty = [r for r in all_rows if any(c for c in r)]
    unit_scale = _detect_unit_scale(ws.title, non_empty[:5]) if non_empty else "units"
    return non_empty, unit_scale


def _build_stats_summary(
    sheet_name: str, non_empty: List[List[str]]
) -> Optional[str]:
    """Compute a year-level statistics summary for time-series sheets."""
    if len(non_empty) <= 10:
        return None
    try:
        _num_cols = [
            ci for ci, _ in enumerate(non_empty[0])
            if any(_try_float(r[ci]) is not None for r in non_empty[1:min(10, len(non_empty))])
        ] if len(non_empty) > 1 else []

        _date_col = next(
            (ci for ci, _ in enumerate(non_empty[0])
             if any(str(r[ci]).startswith("20") or str(r[ci]).startswith("19")
                    for r in non_empty[1:min(5, len(non_empty))] if r[ci])),
            None,
        )

        if not _num_cols or _date_col is None:
            return None

        _hdr = non_empty[0]
        _data_rows = [r for r in non_empty[1:] if r[_date_col] and
                      any(_try_float(r[c]) is not None for c in _num_cols)]
        if len(_data_rows) <= 10:
            return None

        _nc = _num_cols[0]
        _vals = [(r[_date_col], _try_float(r[_nc])) for r in _data_rows if _try_float(r[_nc]) is not None]
        if not _vals:
            return None

        _max_v = max(_vals, key=lambda x: x[1])
        _min_v = min(_vals, key=lambda x: x[1])
        _first = _vals[0]
        _last = _vals[-1]

        _year_stats: dict = {}
        for _d, _v in _vals:
            _yr = str(_d)[:4]
            if _yr not in _year_stats:
                _year_stats[_yr] = {"first": (_d, _v), "last": (_d, _v),
                                    "sum": 0.0, "cnt": 0, "min": (_d, _v), "max": (_d, _v)}
            s = _year_stats[_yr]
            s["last"] = (_d, _v)
            s["sum"] += float(_v)
            s["cnt"] += 1
            if float(_v) < float(s["min"][1]):
                s["min"] = (_d, _v)
            if float(_v) > float(s["max"][1]):
                s["max"] = (_d, _v)

        _num_col_name = _hdr[_nc] if _hdr[_nc] else f"Column {_nc + 1}"
        _lines = [
            f"[COMPUTED SUMMARY — Sheet: {sheet_name}]",
            f"Column: {_num_col_name}",
            f"Dataset: {_first[0]} to {_last[0]} ({len(_vals)} trading days)",
            f"Overall maximum: {_max_v[1]:.2f} on {_max_v[0]}",
            f"Overall minimum: {_min_v[1]:.2f} on {_min_v[0]}",
            f"First value: {_first[1]:.2f} on {_first[0]}",
            f"Last value: {_last[1]:.2f} on {_last[0]}",
        ]
        for _yr, s in sorted(_year_stats.items()):
            _avg = s["sum"] / s["cnt"] if s["cnt"] else 0
            _pct = ((float(s["last"][1]) - float(s["first"][1])) / float(s["first"][1]) * 100
                    if float(s["first"][1]) else 0)
            _lines.append(
                f"Year {_yr}: open={float(s['first'][1]):.2f} ({s['first'][0]}), "
                f"close={float(s['last'][1]):.2f} ({s['last'][0]}), "
                f"avg={_avg:.2f}, high={float(s['max'][1]):.2f} ({s['max'][0]}), "
                f"low={float(s['min'][1]):.2f} ({s['min'][0]}), "
                f"change={_pct:+.2f}%, trading_days={s['cnt']}"
            )
        _sorted_yrs = sorted(_year_stats.keys())
        if len(_sorted_yrs) >= 2:
            for i in range(len(_sorted_yrs) - 1):
                _ya, _yb = _sorted_yrs[i], _sorted_yrs[i + 1]
                _va_open = float(_year_stats[_ya]["first"][1])
                _vb_close = float(_year_stats[_yb]["last"][1])
                if _va_open:
                    _cross_pct = (_vb_close - _va_open) / _va_open * 100
                    _lines.append(
                        f"Cross-period: start of {_ya} ({_year_stats[_ya]['first'][0]}, {_va_open:.2f}) "
                        f"to end of {_yb} ({_year_stats[_yb]['last'][0]}, {_vb_close:.2f}) "
                        f"= {_cross_pct:+.2f}%"
                    )
        return "\n".join(_lines)
    except Exception as exc:
        logger.warning("excel_stats_summary_failed", sheet=sheet_name, error=str(exc))
        return None


def _extract_chart_text(ws: Any, sheet_name: str) -> List[str]:
    chart_texts: List[str] = []
    try:
        for chart_idx, chart in enumerate(getattr(ws, "_charts", []) or []):
            try:
                def _safe_str(v: Any) -> str:
                    try:
                        return str(v) if v is not None else ""
                    except Exception:
                        return ""

                title_text = ""
                try:
                    t = chart.title
                    if t is not None:
                        if hasattr(t, "tx") and t.tx and getattr(t.tx, "rich", None):
                            runs = []
                            for p in (t.tx.rich.p or []):
                                for r in (p.r or []):
                                    runs.append(_safe_str(getattr(r, "t", "")))
                            title_text = " ".join(r for r in runs if r).strip()
                        elif hasattr(t, "tx") and t.tx and getattr(t.tx, "strRef", None):
                            title_text = _safe_str(getattr(t.tx.strRef, "f", ""))
                        else:
                            title_text = _safe_str(t)
                except Exception:
                    pass

                chart_type = type(chart).__name__
                x_axis = ""
                y_axis = ""
                try:
                    x_axis = _safe_str(getattr(getattr(chart, "x_axis", None), "title", "") or "")
                    y_axis = _safe_str(getattr(getattr(chart, "y_axis", None), "title", "") or "")
                except Exception:
                    pass

                series_names: List[str] = []
                data_refs: List[str] = []
                try:
                    for s in getattr(chart, "series", []) or []:
                        try:
                            if getattr(s, "tx", None) and getattr(s.tx, "strRef", None):
                                series_names.append(_safe_str(s.tx.strRef.f))
                        except Exception:
                            pass
                        try:
                            ref = getattr(getattr(s, "val", None), "numRef", None)
                            if ref is not None:
                                data_refs.append(_safe_str(getattr(ref, "f", "")))
                        except Exception:
                            pass
                except Exception:
                    pass

                parts: List[str] = [f"[Chart: {chart_type} on sheet {sheet_name}]"]
                if title_text:
                    parts.append(f"Title: {title_text}")
                if x_axis:
                    parts.append(f"X axis: {x_axis}")
                if y_axis:
                    parts.append(f"Y axis: {y_axis}")
                series_names = [s for s in series_names if s]
                if series_names:
                    parts.append("Series: " + ", ".join(series_names[:12]))
                data_refs = [r for r in data_refs if r]
                if data_refs:
                    parts.append("Data: " + "; ".join(data_refs[:6]))

                chart_text = "\n".join(parts)
                if len(chart_text) >= 16:
                    chart_texts.append(chart_text)
            except Exception as exc:
                logger.warning("excel_chart_failed", sheet=sheet_name, error=str(exc))
    except Exception as exc:
        logger.warning("excel_chart_scan_failed", sheet=sheet_name, error=str(exc))
    return chart_texts


# ─── Phase 1: XlsxIngestor ────────────────────────────────────────────────────

class XlsxIngestor(BaseIngestor):
    """Extracts raw content from XLSX/XLS files → List[RawExtract].

    Phase 1 responsibility: file I/O, sheet/row/chart parsing, security gates.
    Does NOT chunk. The chunker (Phase 2) handles splitting.
    """

    async def extract(
        self,
        path: Path,
        metadata: UniversalMetadata,
    ) -> List[RawExtract]:
        file_path = str(path)

        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
        except Exception as exc:
            raise ValueError(f"CORRUPTED_FILE: {exc}")

        extracts: List[RawExtract] = []

        try:
            for sheet_name in wb.sheetnames:
                try:
                    ws = wb[sheet_name]
                    non_empty, unit_scale = _load_worksheet_rows(ws)

                    if not non_empty:
                        logger.info("excel_empty_sheet_skipped", sheet=sheet_name, file=path.name)
                        continue

                    # Detect unit header (first row with scale keyword)
                    if non_empty and unit_scale != "units":
                        unit_header_text = " ".join(non_empty[0]) if non_empty else ""
                        if unit_header_text:
                            unit_header_text = self._sanitize(unit_header_text, surface="excel_ingest")
                            if unit_header_text:
                                extracts.append(RawExtract(
                                    text=unit_header_text,
                                    extract_type="unit_header",
                                    sheet=sheet_name,
                                    raw_source_ref=f"xlsx:{path.name}|sheet:{sheet_name}|unit_header",
                                    extra={"unit_scale": unit_scale},
                                ))

                    # Row extracts
                    ROWS_PER_CHUNK = settings.EXCEL_ROWS_PER_CHUNK
                    header_row = non_empty[0] if non_empty else None

                    for batch_start in range(0, len(non_empty), ROWS_PER_CHUNK):
                        batch = non_empty[batch_start:batch_start + ROWS_PER_CHUNK]
                        row_start = batch_start + 1
                        row_end = batch_start + len(batch)

                        if batch_start > 0 and header_row and batch[0] != header_row:
                            batch = [header_row] + batch

                        txt = _table_to_text(batch)
                        if not txt.strip():
                            continue

                        chunk_text = f"[Sheet: {sheet_name}, Rows {row_start}-{row_end}]\n{txt}"
                        chunk_text = self._sanitize(chunk_text, surface="excel_ingest")
                        if not chunk_text.strip():
                            continue
                        chunk_text = self._scrub_pii(chunk_text, surface="excel_ingest")

                        extracts.append(RawExtract(
                            text=chunk_text,
                            extract_type="table_row",
                            sheet=sheet_name,
                            raw_source_ref=f"xlsx:{path.name}|sheet:{sheet_name}|rows:{row_start}-{row_end}",
                            extra={
                                "unit_scale": unit_scale,
                                "row_start": row_start,
                                "row_end": row_end,
                            },
                        ))

                    # Statistics summary
                    summary = _build_stats_summary(sheet_name, non_empty)
                    if summary:
                        extracts.append(RawExtract(
                            text=summary,
                            extract_type="named_range",
                            sheet=sheet_name,
                            raw_source_ref=f"xlsx:{path.name}|sheet:{sheet_name}|stats_summary",
                            extra={"is_stats_summary": True, "unit_scale": unit_scale},
                        ))

                    # Charts
                    for chart_text in _extract_chart_text(ws, sheet_name):
                        chart_text = self._scrub_pii(chart_text, surface="excel_chart_ingest")
                        if chart_text:
                            extracts.append(RawExtract(
                                text=chart_text,
                                extract_type="chart_image",
                                sheet=sheet_name,
                                raw_source_ref=f"xlsx:{path.name}|sheet:{sheet_name}|chart",
                            ))

                    # Embedded images
                    try:
                        for img_idx, img_obj in enumerate(getattr(ws, "_images", []) or []):
                            try:
                                blob: Optional[bytes] = None
                                ext_hint = ".png"
                                data_attr = getattr(img_obj, "_data", None)
                                if callable(data_attr):
                                    try:
                                        blob = data_attr()
                                    except Exception:
                                        blob = None
                                if not blob:
                                    ref = getattr(img_obj, "ref", None) or getattr(img_obj, "path", None)
                                    if ref and hasattr(ref, "read"):
                                        try:
                                            ref.seek(0)
                                        except Exception:
                                            pass
                                        blob = ref.read()
                                    elif isinstance(ref, (str, os.PathLike)):
                                        try:
                                            with open(ref, "rb") as fh:
                                                blob = fh.read()
                                            ext_hint = os.path.splitext(str(ref))[1] or ext_hint
                                        except Exception:
                                            blob = None
                                fmt = getattr(img_obj, "format", None)
                                if fmt:
                                    ext_hint = "." + str(fmt).lower()
                                if blob and len(blob) >= 256:
                                    extracts.append(RawExtract(
                                        text="",
                                        extract_type="image_region",
                                        sheet=sheet_name,
                                        raw_source_ref=f"xlsx:{path.name}|sheet:{sheet_name}|img:{img_idx}",
                                        raw_bytes=blob,
                                        extra={"img_ext": ext_hint},
                                    ))
                            except Exception as exc:
                                logger.warning("excel_embedded_image_failed", sheet=sheet_name, error=str(exc))
                    except Exception as exc:
                        logger.warning("excel_image_scan_failed", sheet=sheet_name, error=str(exc))

                except ValueError:
                    raise
                except Exception as exc:
                    logger.warning("excel_sheet_failed", sheet=sheet_name, error=str(exc))
        finally:
            try:
                wb.close()
            except Exception:
                pass

        if not extracts:
            raise ValueError("NO_EXTRACTS_PRODUCED")

        return extracts


# ─── Backward-compat ingest() — full pipeline ─────────────────────────────────

def _base_structure(doc_id: str, session_id: str, source_path: str, **extra: Any) -> Dict[str, Any]:
    return {"doc_id": doc_id, "session_id": session_id, "source_path": source_path, **extra}


def _ingest_embedded_image_bytes(
    blob: bytes, ext_hint: str, session_id: str,
    parent_doc_id: str, parent_modality: str, parent_source: str,
    parent_page: Optional[int] = None, parent_sheet: Optional[str] = None,
) -> List[IngestedDocument]:
    if not blob or len(blob) < 256:
        return []
    safe_ext = (ext_hint or ".png").lower().strip()
    if safe_ext == ".jpeg":
        safe_ext = ".jpg"
    if safe_ext not in {".png", ".jpg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}:
        safe_ext = ".png"
    from app.utils.paths import resolved_temp_dir
    tmp_dir = resolved_temp_dir() / "embedded_images"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        tmp_dir = Path(tempfile.gettempdir())
    digest = hashlib.sha256(blob).hexdigest()
    tmp_path = tmp_dir / f"{digest}{safe_ext}"
    try:
        if not tmp_path.exists():
            tmp_path.write_bytes(blob)
        from app.ingestion.image_ingest import ingest as image_ingest
        return image_ingest(
            str(tmp_path), session_id,
            parent_doc_id=parent_doc_id, parent_modality=parent_modality,
            parent_source=parent_source, parent_page=parent_page, parent_sheet=parent_sheet,
        ) or []
    except Exception as exc:
        logger.warning("embedded_image_ingest_failed", parent_source=parent_source,
                       sheet=parent_sheet, page=parent_page, error=str(exc))
        return []


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
    if file_size > settings.MAX_FILE_SIZE_XLSX:
        raise ValueError(f"FILE_TOO_LARGE: {file_size}")

    with tracer.start_as_current_span("xlsx_ingest") as span:
        span.set_attribute("file.name", path.name)
        span.set_attribute("file.size", file_size)
        span.set_attribute("session.id", session_id)
        start = time.time()

        async with _semaphore:
            try:
                logger.info("xlsx_ingest_start", file=path.name, size=file_size, session_id=session_id)

                doc_id = str(uuid.uuid4())
                source_name = path.name
                source_path_str = str(path.resolve())
                ROWS_PER_CHUNK = settings.EXCEL_ROWS_PER_CHUNK

                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path, data_only=True)
                except Exception as exc:
                    raise ValueError(f"CORRUPTED_FILE: {exc}")

                documents: List[IngestedDocument] = []
                doc_warnings: List[str] = []

                try:
                    for sheet_name in wb.sheetnames:
                        try:
                            ws = wb[sheet_name]
                            non_empty, unit_scale = _load_worksheet_rows(ws)

                            if not non_empty:
                                doc_warnings.append(f"Empty sheet skipped: {sheet_name}")
                                continue

                            header_row = non_empty[0] if non_empty else None

                            for batch_start in range(0, len(non_empty), ROWS_PER_CHUNK):
                                batch = non_empty[batch_start:batch_start + ROWS_PER_CHUNK]
                                row_start = batch_start + 1
                                row_end = batch_start + len(batch)

                                if batch_start > 0 and header_row and batch[0] != header_row:
                                    batch = [header_row] + batch

                                txt = _table_to_text(batch)
                                if not txt.strip():
                                    continue

                                chunk_text = f"[Sheet: {sheet_name}, Rows {row_start}-{row_end}]\n{txt}"
                                try:
                                    from app.guardrails.input_guard import sanitize as _guard_sanitize
                                    _clean = _guard_sanitize(chunk_text, surface="excel_ingest")
                                    if _clean != chunk_text:
                                        logger.warning("excel_injection_sanitized", file=source_name,
                                                       sheet=sheet_name)
                                        chunk_text = _clean
                                except Exception as _ge:
                                    logger.warning("excel_guardrail_failed", file=source_name, error=str(_ge))

                                if not chunk_text.strip():
                                    continue
                                chunk_text = _pii_scrub(chunk_text, surface="excel_ingest")

                                documents.append(
                                    IngestedDocument(
                                        text=chunk_text,
                                        modality="table",
                                        subtype="structured",
                                        source_type="excel",
                                        source=source_name,
                                        structure=_base_structure(
                                            doc_id, session_id, source_path_str,
                                            sheet=sheet_name,
                                            row_start=row_start,
                                            row_end=row_end,
                                            page_number=None,
                                            total_pages=None,
                                            section_title=sheet_name,
                                            ingestion_timestamp=time.time(),
                                            language="en",
                                            file_size_bytes=file_size,
                                            content_type="excel_sheet",
                                            ingestion_time=time.time(),
                                        ),
                                        extra_metadata={
                                            "data_quality_score": _quality(txt),
                                            "importance_score": _quality(txt),
                                            "modality_weight": 1.0,
                                        },
                                    ).finalize()
                                )

                            # Stats summary
                            summary = _build_stats_summary(sheet_name, non_empty)
                            if summary:
                                documents.append(
                                    IngestedDocument(
                                        text=summary,
                                        modality="table",
                                        subtype="summary",
                                        source_type="excel",
                                        source=source_name,
                                        structure=_base_structure(
                                            doc_id, session_id, source_path_str,
                                            sheet=sheet_name,
                                            row_start=1,
                                            row_end=len(non_empty),
                                            page_number=None,
                                            total_pages=None,
                                            section_title=sheet_name,
                                            ingestion_timestamp=time.time(),
                                            language="en",
                                            file_size_bytes=file_size,
                                            content_type="excel_summary",
                                            ingestion_time=time.time(),
                                        ),
                                        extra_metadata={
                                            "data_quality_score": 1.0,
                                            "importance_score": 1.0,
                                            "modality_weight": 1.5,
                                        },
                                    ).finalize()
                                )

                            # Embedded images
                            try:
                                for img_obj in getattr(ws, "_images", []) or []:
                                    try:
                                        blob: Optional[bytes] = None
                                        ext_hint = ".png"
                                        data_attr = getattr(img_obj, "_data", None)
                                        if callable(data_attr):
                                            try:
                                                blob = data_attr()
                                            except Exception:
                                                blob = None
                                        if not blob:
                                            ref = getattr(img_obj, "ref", None) or getattr(img_obj, "path", None)
                                            if ref and hasattr(ref, "read"):
                                                try:
                                                    ref.seek(0)
                                                except Exception:
                                                    pass
                                                blob = ref.read()
                                            elif isinstance(ref, (str, os.PathLike)):
                                                try:
                                                    with open(ref, "rb") as fh:
                                                        blob = fh.read()
                                                    ext_hint = os.path.splitext(str(ref))[1] or ext_hint
                                                except Exception:
                                                    blob = None
                                        fmt = getattr(img_obj, "format", None)
                                        if fmt:
                                            ext_hint = "." + str(fmt).lower()
                                        embedded = _ingest_embedded_image_bytes(
                                            blob=blob or b"", ext_hint=ext_hint, session_id=session_id,
                                            parent_doc_id=doc_id, parent_modality="excel",
                                            parent_source=source_name, parent_sheet=sheet_name,
                                        )
                                        documents.extend(embedded)
                                    except Exception as exc:
                                        logger.warning("excel_embedded_image_failed", sheet=sheet_name, error=str(exc))
                            except Exception as exc:
                                logger.warning("excel_image_scan_failed", sheet=sheet_name, error=str(exc))

                            # Charts
                            for chart_text in _extract_chart_text(ws, sheet_name):
                                chart_text = _pii_scrub(chart_text, surface="excel_chart_ingest")
                                if len(chart_text) < 16:
                                    continue
                                documents.append(
                                    IngestedDocument(
                                        text=chart_text,
                                        modality="table",
                                        subtype="chart",
                                        source_type="excel",
                                        source=source_name,
                                        structure=_base_structure(
                                            doc_id, session_id, source_path_str,
                                            sheet=sheet_name,
                                            page_number=None,
                                            total_pages=None,
                                            section_title=None,
                                            ingestion_timestamp=time.time(),
                                            language="en",
                                            file_size_bytes=file_size,
                                            content_type="excel_chart",
                                            ingestion_time=time.time(),
                                        ),
                                        extra_metadata={
                                            "data_quality_score": _quality(chart_text),
                                            "importance_score": 0.9,
                                            "modality_weight": 1.0,
                                        },
                                    ).finalize()
                                )

                        except ValueError:
                            raise
                        except Exception as exc:
                            logger.warning("excel_sheet_failed", sheet=sheet_name, error=str(exc))
                finally:
                    try:
                        wb.close()
                    except Exception:
                        pass

                if not documents:
                    raise ValueError("NO_CONTENT_EXTRACTED")

                latency = round(time.time() - start, 2)
                _ingest_duration.labels(status="success").observe(latency)
                span.set_attribute("docs.count", len(documents))
                span.set_status(Status(StatusCode.OK))
                logger.info("xlsx_ingest_success", file=path.name, docs=len(documents),
                            latency=latency, session_id=session_id,
                            warnings=doc_warnings or None)
                return documents

            except Exception as exc:
                latency = round(time.time() - start, 2)
                error_type = type(exc).__name__
                _ingest_duration.labels(status="error").observe(latency)
                _ingest_errors.labels(error_type=error_type).inc()
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                logger.error("xlsx_ingest_failed", file=path.name, session_id=session_id,
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
