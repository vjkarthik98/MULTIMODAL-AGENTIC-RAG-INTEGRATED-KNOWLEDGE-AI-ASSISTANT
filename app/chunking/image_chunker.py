"""image_chunker.py

Finance-grade Image chunker + image captioning utilities.

All ML inference (BLIP, Qwen2-VL, TrOCR, SigLIP) lives here because
captioning is a chunking concern — it transforms raw image bytes from a
RawExtract into semantic text for IngestedDocuments.

Ingestors (image_ingest.py, video_ingest.py) call classify_and_caption /
generate_caption from here via a lazy import.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from prometheus_client import Counter

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import deterministic_chunk_id, extract_finance_entities
from app.core.config import settings
from app.ingestion.schema import EmptyContentError, IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

try:
    import torch
except ImportError:
    torch = None

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_image_chunks_total",
    "Total chunks produced by image chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_image_chunk_errors_total",
    "Total errors in image chunker",
)

# ══════════════════════════════════════════════════════════════════════════════
# OCR-CAPTION MISMATCH
# ══════════════════════════════════════════════════════════════════════════════

_NUMBER_RE = re.compile(
    r"[$€£]\d[\d,.]*[BMKbmk]?|\d[\d,.]*\s?%|\d[\d,.]*[xX]|\d[\d,.]*\s?bps",
    re.IGNORECASE,
)
_MISMATCH_THRESHOLD = 0.20

_TIME_PERIOD_RE = re.compile(
    r'\b(?:FY|Q[1-4])\s*\d{4}\b'
    r'|\b\d{4}[-–]\d{2,4}\b'
    r'|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
    r'|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+\d{1,2},?\s*\d{4}\b'
    # Stock-performance-graph date axes: "9/27/19" ... "9/28/24"
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b' r'|\b\d+[\s-]*[Yy]ear\b',
    re.IGNORECASE,
)
_DATA_SERIES_RE = re.compile(
    r'\b(Revenue|Net Income|EBITDA|EPS|FCF|Free Cash Flow|Operating Income'
    r'|Gross Profit|Net Sales|Operating Margin|EBIT|Earnings|Dividends?'
    r'|Total Assets|Market Cap|Price|Volume)\b'
    # Stock-performance-graph series: benchmark indices and issuer names,
    # e.g. "S&P 500 Index", "Dow Jones U.S. Technology Supersector Index",
    # "Apple Inc." — the near-universal SEC Item 5 cumulative-return chart.
    r'|\b[A-Z][\w&.]*(?:\s+[A-Z0-9][\w&.]*){0,6}\s+Index\b'
    r'|\b(?!(?:Among|The|And|For|With|From|This|That|These|Those|A|An)\b)'
    r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+Inc\.?\b',
)
_WATERMARK_KW = frozenset(
    {
        "CONFIDENTIAL",
        "DRAFT",
        "WATERMARK",
        "DO NOT DISTRIBUTE",
        "PROPRIETARY",
        "INTERNAL USE ONLY",
        "NOT FOR DISTRIBUTION",
    }
)


def _extract_time_period(text: str) -> str | None:
    m = _TIME_PERIOD_RE.search(text)
    return m.group(0).strip() if m else None


def _extract_data_series(text: str) -> list[str]:
    seen: set = set()
    result: list[str] = []
    for m in _DATA_SERIES_RE.finditer(text):
        label = m.group(0).strip()
        lk = label.lower()
        if lk not in seen:
            seen.add(lk)
            result.append(label)
    # Drop entries that are pure substrings of a longer match — the same
    # index name recurs across caption + OCR text and can be re-matched
    # starting mid-phrase (e.g. "P 500 Index" inside "S&P 500 Index").
    result = [
        label
        for label in result
        if not any(label.lower() in other.lower() and label != other for other in result)
    ]
    return result[:8]


def _detect_watermark(text: str) -> bool:
    upper = text.upper()
    return any(kw in upper for kw in _WATERMARK_KW)


def _numbers_in(text: str) -> set:
    return set(re.sub(r"[,\s]", "", n.lower()) for n in _NUMBER_RE.findall(text))


def _check_mismatch(ocr_text: str, caption: str) -> bool:
    ocr_nums = _numbers_in(ocr_text)
    if not ocr_nums:
        return False
    cap_nums = _numbers_in(caption)
    missing = ocr_nums - cap_nums
    return len(missing) / len(ocr_nums) > _MISMATCH_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# FINANCE IMAGE TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

_IMAGE_TYPES = (
    "bar_chart",
    "line_chart",
    "pie_chart",
    "candlestick",
    "table_image",
    "org_chart",
    "flow_diagram",
    "infographic",
)

_FIN_NUMBER_RE = re.compile(
    r'(?:'
    r'\$[\d,]+\.?\d*\s*[BMKTbmkt]?'
    r'|[\d,]+\.?\d*\s*%'
    r'|[\d,]+\.?\d*\s*[xX]'
    r'|[\d,]+\.?\d*\s*bps'
    r')',
    re.IGNORECASE,
)


def _classify_image_type(image: Image.Image, ocr_text: str = "") -> str:
    """Heuristic image type classification for finance images."""
    import numpy as np

    if ocr_text:
        nums = _FIN_NUMBER_RE.findall(ocr_text)
        lines_with_nums = [
            line for line in ocr_text.split("\n") if len(re.findall(r'\d', line)) >= 3
        ]
        if len(nums) >= 5 and len(lines_with_nums) >= 3:
            return "table_image"

    try:
        w, h = image.size
        arr = np.array(image.convert("RGB"))
        r_ch = arr[:, :, 0].astype(float)
        g_ch = arr[:, :, 1].astype(float)
        b_ch = arr[:, :, 2].astype(float)
        dark_pixels = float((b_ch < 30).mean())
        red_green_contrast = float(abs(r_ch.mean() - g_ch.mean()))
        if dark_pixels > 0.3 and red_green_contrast > 20:
            return "candlestick"
        if 0.8 <= w / max(h, 1) <= 1.2:
            white_pixels = float(((arr > 230).all(axis=2)).mean())
            if white_pixels < 0.4:
                return "pie_chart"
        col_means = arr.mean(axis=0).mean(axis=1)
        row_means = arr.mean(axis=1).mean(axis=1)
        col_var = float(col_means.var())
        row_var = float(row_means.var())
        if col_var > 800:
            return "bar_chart"
        if row_var > 800 and h > w:
            return "bar_chart"
        white_pct = float(((arr > 220).all(axis=2)).mean())
        if white_pct > 0.6 and col_var > 200:
            return "line_chart"
        if white_pct > 0.75:
            return "org_chart" if h > w * 1.2 else "flow_diagram"
    except Exception:
        pass

    return "infographic"


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC LINE-CHART DIGITIZATION
# ══════════════════════════════════════════════════════════════════════════════
# The resident VLM (Qwen2-VL-2B) cannot reliably read exact values off a line
# chart — verified: it rounds every series to the nearest gridline and swaps
# the ranking of visually-similar dashed lines. No amount of prompting fixed
# this (tested: OCR-text grounding, per-tick value tables — both made results
# worse, see project_image_modality_accuracy_phase memory). This module
# sidesteps the VLM entirely for the one sub-task it cannot do: it uses
# OCR bounding boxes (not just text) to calibrate the chart's pixel-to-value
# axes, then traces each line's actual pixel path and classifies it against
# the legend's own dash pattern. Zero model inference — pure geometry —  so
# results are exact, not guessed. Gated to image_type == "line_chart" with
# a distinct dollar y-axis and date x-axis; returns None (no-op, caller falls
# back to the VLM caption alone) if that calibration can't be established.

_DOLLAR_LABEL_RE = re.compile(r'^[$S]\s*([\d,]+)\.?0*$', re.IGNORECASE)
_DATE_LABEL_RE = re.compile(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$')
_CHART_DARK_THRESH = 100
_CHART_MIN_RUN_PX = 2
_CHART_MAX_JUMP_PX = 14
_CHART_REACQUIRE_GAP = 4
_CHART_REACQUIRE_MAX_JUMP = 60


def _chart_run_lengths(mask_1d) -> list[tuple[int, int]]:
    """Contiguous True-run (start, length) pairs in a 1D boolean array."""
    runs: list[tuple[int, int]] = []
    in_run, start = False, 0
    for i, v in enumerate(mask_1d):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            runs.append((start, i - start))
    if in_run:
        runs.append((start, len(mask_1d) - start))
    return runs


def _digitize_line_chart(image: Image.Image, ocr_boxes: list) -> dict[str, Any] | None:
    """Extract exact per-series values at each x-axis tick from a line chart.

    Returns {"series": [...], "ticks": [...], "values_by_tick": {tick: {name: value}}}
    or None if this image doesn't calibrate as a dollar-axis/date-axis line chart.
    """
    if not ocr_boxes:
        return None
    try:
        import numpy as np

        arr = np.array(image.convert("L"))
        h, w = arr.shape
        dark = arr < _CHART_DARK_THRESH

        # Y-axis calibration from "$100".."$500"-style labels (any count/spacing).
        y_points = []
        for bbox, text, _conf in ocr_boxes:
            m = _DOLLAR_LABEL_RE.match(text.strip())
            if not m:
                continue
            try:
                value = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            ys = [p[1] for p in bbox]
            y_points.append(((min(ys) + max(ys)) / 2, value))
        if len(y_points) < 2:
            return None
        ys_arr = np.array([p[0] for p in y_points])
        vals_arr = np.array([p[1] for p in y_points])
        A = np.vstack([ys_arr, np.ones_like(ys_arr)]).T
        slope, intercept = np.linalg.lstsq(A, vals_arr, rcond=None)[0]
        if slope >= 0:  # a real y-axis must decrease in value as pixel-y increases
            return None
        pixel_to_value = lambda y: slope * y + intercept  # noqa: E731

        # X-axis calibration from date labels (M/D/YY or M/D/YYYY).
        x_points = []
        for bbox, text, _conf in ocr_boxes:
            if _DATE_LABEL_RE.match(text.strip()):
                xs = [p[0] for p in bbox]
                x_points.append(((min(xs) + max(xs)) / 2, text.strip()))
        x_points.sort()
        if len(x_points) < 2:
            return None

        # Legend: series names sit in the bottom band of the image; the dash
        # swatch for each sits immediately to its left.
        legend = [
            (bbox, text.strip())
            for bbox, text, conf in ocr_boxes
            if min(p[1] for p in bbox) > h * 0.85
            and conf > 0.5
            and len(text.strip()) > 3
            and not _DATE_LABEL_RE.match(text.strip())
            and not _DOLLAR_LABEL_RE.match(text.strip())
        ]
        legend.sort(key=lambda t: min(p[0] for p in t[0]))
        if len(legend) < 2:
            return None

        series_signatures: dict[str, float] = {}
        for i, (bbox, name) in enumerate(legend):
            text_x0 = min(p[0] for p in bbox)
            text_y = int(sum(p[1] for p in bbox) / len(bbox))
            prev_end = (
                (min(p[0] for p in legend[i - 1][0]) - 20) if i > 0 else max(0, text_x0 - 130)
            )
            sx0, sx1 = int(max(prev_end, text_x0 - 130)), int(text_x0 - 8)
            if sx1 <= sx0:
                continue
            row = dark[max(0, text_y - 6) : text_y + 7, sx0:sx1].any(axis=0)
            total = sx1 - sx0
            duty = sum(run_len for _, run_len in _chart_run_lengths(row)) / total if total else 0.0
            series_signatures[name] = duty
        if len(series_signatures) < 2:
            return None

        # Curve tracing: column-scan for dark-pixel runs, track N lines by
        # nearest-y proximity from a seed column where all N are separated.
        y_top = max(0, int(min(y for y, _ in y_points)) - 20)
        y_bot = min(h, int(max(y for y, _ in y_points)) + 60)
        x_left = max(0, int(x_points[0][0]) - 20)
        x_right = min(w, int(x_points[-1][0]) + 20)
        n_lines = len(series_signatures)

        def col_centers(x: int) -> list[float]:
            col = dark[y_top:y_bot, x]
            return sorted(
                y_top + s + run_len / 2
                for s, run_len in _chart_run_lengths(col)
                if run_len >= _CHART_MIN_RUN_PX
            )

        cols = [(x, col_centers(x)) for x in range(x_left, x_right, 2)]
        seed_idx = None
        for i, (_x, centers) in enumerate(cols):
            if len(centers) == n_lines and all(
                centers[j + 1] - centers[j] > 12 for j in range(n_lines - 1)
            ):
                seed_idx = i
                break
        if seed_idx is None:
            return None

        seed_x, seed_centers = cols[seed_idx]
        track_ids = [f"line{i}" for i in range(n_lines)]
        tracks = {k: {seed_x: seed_centers[i]} for i, k in enumerate(track_ids)}

        def _trace(cols_slice, seed_centers):
            last_y = {k: seed_centers[i] for i, k in enumerate(track_ids)}
            gap_count = {k: 0 for k in track_ids}
            for x, centers in cols_slice:
                avail = list(centers)
                candidates = sorted((abs(c - last_y[k]), k, c) for k in track_ids for c in avail)
                assigned_t, assigned_c = set(), set()
                for dist, k, c in candidates:
                    if k in assigned_t or c in assigned_c or dist > _CHART_MAX_JUMP_PX:
                        continue
                    tracks[k][x] = c
                    last_y[k] = c
                    gap_count[k] = 0
                    assigned_t.add(k)
                    assigned_c.add(c)
                leftover_t = [k for k in track_ids if k not in assigned_t]
                leftover_c = [c for c in avail if c not in assigned_c]
                for k in leftover_t:
                    gap_count[k] += 1
                if (
                    leftover_t
                    and leftover_c
                    and len(leftover_t) == len(leftover_c)
                    and all(gap_count[k] >= _CHART_REACQUIRE_GAP for k in leftover_t)
                ):
                    cand2 = sorted(
                        (abs(c - last_y[k]), k, c) for k in leftover_t for c in leftover_c
                    )
                    ut, uc = set(), set()
                    for dist, k, c in cand2:
                        if k in ut or c in uc or dist > _CHART_REACQUIRE_MAX_JUMP:
                            continue
                        tracks[k][x] = c
                        last_y[k] = c
                        gap_count[k] = 0
                        ut.add(k)
                        uc.add(c)

        _trace(cols[seed_idx + 1 :], seed_centers)
        _trace(list(reversed(cols[:seed_idx])), seed_centers)

        # Classify each traced line against a legend name via duty-cycle match
        # (fraction of columns where that line has ink — distinguishes solid
        # vs. long-dash vs. short-dash lines independent of their y-position,
        # so it still works through a region where two lines' values converge).
        traj_sig = {}
        for k in track_ids:
            xs = sorted(tracks[k].keys())
            span = (xs[-1] - xs[0]) // 2 if len(xs) > 1 else 1
            traj_sig[k] = len(xs) / span if span else 0.0

        name_cands = sorted(
            (abs(traj_sig[k] - sig), name, k)
            for name, sig in series_signatures.items()
            for k in track_ids
        )
        used_names, used_tracks, name_by_track = set(), set(), {}
        for _diff, name, k in name_cands:
            if name in used_names or k in used_tracks:
                continue
            name_by_track[k] = name
            used_names.add(name)
            used_tracks.add(k)

        edge_labels = {x_points[0][1], x_points[-1][1]}
        values_by_tick: dict[str, dict[str, float]] = {}
        for tx, label in x_points:
            row: dict[str, float] = {}
            for k in track_ids:
                xs = sorted(tracks[k].keys())
                if not xs:
                    continue
                nearest = min(xs, key=lambda x: abs(x - tx))
                # Interior ticks: require a close match or skip (don't guess).
                # Edge ticks (first/last): the lines start/end bunched or run
                # off the traceable range, so fall back to this track's
                # nearest actual traced point regardless of distance rather
                # than silently omitting the series.
                if abs(nearest - tx) > 15 and label not in edge_labels:
                    continue
                row[name_by_track.get(k, k)] = round(pixel_to_value(tracks[k][nearest]))
            values_by_tick[label] = row

        # Common-base snap for indexed / cumulative-total-return charts: every
        # series is defined to start at the same base value (almost always the
        # lowest round axis value, e.g. $100), but at the first tick all lines
        # are bunched together so per-line pixel tracking is imprecise (reads
        # e.g. $111/$103/$102 instead of $100). When the first tick's reads are
        # all tightly clustered AND near a round base, snap them all to that
        # base — this makes the derived percentages exact without affecting any
        # other tick. Guarded so it never fires on a genuinely-fanned start.
        first_label = x_points[0][1]
        first_row = values_by_tick.get(first_label, {})
        if len(first_row) >= 2:
            vals = list(first_row.values())
            vmin, vmax, vmean = min(vals), max(vals), sum(vals) / len(vals)
            axis_min = float(vals_arr.min())
            base_candidates = [b for b in (100.0, axis_min, 1000.0, 10.0) if b > 0]
            base = min(base_candidates, key=lambda b: abs(b - vmean))
            if (vmax - vmin) <= 0.15 * vmean and abs(vmean - base) <= 0.15 * base:
                for name in first_row:
                    first_row[name] = round(base)

        return {
            "series": list(series_signatures.keys()),
            "ticks": [lbl for _, lbl in x_points],
            "values_by_tick": values_by_tick,
        }
    except Exception as exc:
        logger.warning(event="chart_digitize_failed", error=str(exc))
        return None


def _series_trajectory(digitized: dict[str, Any], name: str) -> list[tuple[str, float]]:
    """Ordered (tick, value) pairs for one series, skipping ticks it's absent from."""
    out: list[tuple[str, float]] = []
    for tick in digitized["ticks"]:
        row = digitized["values_by_tick"].get(tick, {})
        if name in row:
            out.append((tick, float(row[name])))
    return out


def _describe_series_trends(digitized: dict[str, Any]) -> str:
    """Deterministic natural-language trend statements per series, computed
    only from the verified pixel values. This is what lets the generation LLM
    answer 'what happened between period A and B' (drawdown) and 'when did it
    plateau' correctly — without it, the model was left to infer trends from a
    bare value table and sometimes mis-mapped periods to values.

    For each series it states: start→end, the single largest peak-to-trough
    drawdown (with the two ticks and percent drop), and the flattest multi-tick
    run (a plateau/consolidation), each anchored to the exact tick labels.

    Percent figures are spelled out as "N percent" (never "N%") and explicitly
    labelled "(a percentage, not a dollar amount)" — observed the generation
    LLM occasionally writing a percent number with a "$" in front of it (e.g.
    "~$329" when it meant "~329 percent" and the dollar total was actually
    ~$429), because "+329%" and "$429" look similar side by side. Spelling
    the unit out and labelling it removes the ambiguity at the source instead
    of hoping every generation gets it right.
    """
    lines: list[str] = [
        "CHART TRENDS (per series, computed from the pixel-read values above — "
        "state every dollar figure below with a leading '~', e.g. '~$429', since "
        "chart-reading is an approximation, not an exact-to-the-dollar figure. "
        "Percent figures are written as 'N percent' — NEVER put a '$' in front "
        "of a percent figure, they are two different units):"
    ]
    for name in digitized["series"]:
        traj = _series_trajectory(digitized, name)
        if len(traj) < 2:
            continue
        (t0, v0), (tN, vN) = traj[0], traj[-1]
        pct = ((vN - v0) / v0 * 100) if v0 else 0.0
        parts = [
            f"started ~${v0:.0f} ({t0}) and ended ~${vN:.0f} ({tN}) — "
            f"a percentage gain of approximately {pct:.0f} percent "
            f"(a percentage, not a dollar amount)"
        ]

        # Largest peak-to-trough drawdown: scan for the biggest drop from a
        # running max to a later trough.
        peak_t, peak_v = traj[0]
        worst_dd, dd_from, dd_to = 0.0, None, None
        for tick, val in traj:
            if val > peak_v:
                peak_v, peak_t = val, tick
            dd = (val - peak_v) / peak_v if peak_v else 0.0
            if dd < worst_dd:
                worst_dd, dd_from, dd_to = dd, (peak_t, peak_v), (tick, val)
        if dd_from and worst_dd <= -0.08:  # only report a meaningful (>8%) drawdown
            parts.append(
                f"declined from ~${dd_from[1]:.0f} ({dd_from[0]}) to ~${dd_to[1]:.0f} "
                f"({dd_to[0]}) — a drawdown of approximately {abs(worst_dd)*100:.0f} percent "
                f"(a percentage, not a dollar amount)"
            )

        # Flattest consecutive run of >=2 ticks (plateau/consolidation): the
        # window whose max/min spread relative to its mean is smallest.
        best_run, _best_spread = None, None
        for i in range(len(traj)):
            for j in range(i + 1, len(traj)):
                window = [v for _, v in traj[i : j + 1]]
                mean = sum(window) / len(window)
                spread = (max(window) - min(window)) / mean if mean else 1.0
                # prefer longer runs; only consider genuinely-flat (<12% spread) windows
                if spread < 0.12 and (best_run is None or (j - i) > (best_run[1] - best_run[0])):
                    best_run, _best_spread = (i, j), spread
        if best_run and (best_run[1] - best_run[0]) >= 1:
            i, j = best_run
            lo = min(v for _, v in traj[i : j + 1])
            hi = max(v for _, v in traj[i : j + 1])
            parts.append(
                f"was roughly flat (a plateau) between ~${lo:.0f} and ~${hi:.0f} "
                f"from {traj[i][0]} to {traj[j][0]}"
            )

        lines.append(f"  {name}: " + "; ".join(parts) + ".")
    return "\n".join(lines)


def _format_digitized_chart(digitized: dict[str, Any]) -> str:
    """Render the digitized series/tick table + deterministic trend statements
    as plain text for the caption and for retrieval — these pixel-calibrated
    values take precedence over anything the VLM guessed (it fabricates chart
    values outright; this is a real measurement), but a chart reading is still
    an approximation, not an exact-to-the-dollar figure, so every value here
    is prefixed with '~' and downstream generation is told to keep that prefix
    rather than state a bare, falsely-precise number."""
    lines = [
        "CHART VALUES — pixel-calibrated reads (state every figure with a leading '~', e.g. '~$429'):"
    ]
    for tick in digitized["ticks"]:
        row = digitized["values_by_tick"].get(tick, {})
        if not row:
            continue
        parts = [f"{name}=~${val:.0f}" for name, val in row.items()]
        lines.append(f"  {tick}: " + ", ".join(parts))
    trends = _describe_series_trends(digitized)
    if trends:
        lines.append("")
        lines.append(trends)
    return "\n".join(lines)


_CAPTION_TRENDS_CUT_RE = re.compile(r'\n\s*5\)\s')
# Any $-amount, optionally in a comma-separated list ("$100, $150, $200"), plus
# a leading colon that introduced it ("Apple Inc.: $100, $150" → "Apple Inc.").
_DOLLAR_CLAIM_RE = re.compile(r'\s*:?\s*\$\s?[\d,]+(?:\.\d+)?(?:\s*,\s*\$\s?[\d,]+(?:\.\d+)?)*')


def _replace_vlm_trends_with_verified(caption_text: str, digitized: dict[str, Any]) -> str:
    """Reconcile the VLM caption with verified pixel data for a digitized chart.

    The VLM (even the 7B) cannot read exact chart values — it lists axis
    gridlines as if they were series values and gives visually-similar dashed
    lines identical fabricated sequences. Since the deterministic digitizer
    supplies the real numbers (injected separately into the chunk text), this:

      1. cuts the VLM's own '5) Key trends' section (its worst numeric guesses)
         and replaces it with the verified end-of-period ranking, and
      2. strips every remaining $-amount from the VLM's prose, so the ONLY
         dollar figures anywhere in the chunk are the verified ones.

    The VLM's qualitative wording (which series leads, overall shape) is kept —
    that is where a bigger model genuinely helps.
    """
    if not caption_text or not digitized:
        return caption_text
    last_tick = digitized["ticks"][-1] if digitized.get("ticks") else None
    row = digitized["values_by_tick"].get(last_tick, {}) if last_tick else {}
    if not row:
        return caption_text

    m = _CAPTION_TRENDS_CUT_RE.search(caption_text)
    kept = caption_text[: m.start()].rstrip() if m else caption_text.rstrip()
    # Remove the VLM's fabricated dollar figures from the kept prose.
    kept = _DOLLAR_CLAIM_RE.sub("", kept)

    ranked = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
    ranked_desc = "; ".join(f"{name} at ~${val:.0f}" for name, val in ranked)
    trend_line = f"5) Key trends: as of {last_tick}, ranked highest to lowest: {ranked_desc}."
    return f"{kept}\n\n{trend_line}"


def _finance_caption_prompt(image_type: str) -> str:
    _PROMPTS = {
        "bar_chart": "a financial bar chart showing",
        "line_chart": "a financial line chart showing",
        "pie_chart": "a pie chart showing portfolio or segment allocation",
        "candlestick": "a candlestick stock price chart showing",
        "table_image": "a financial table with exact numeric values showing",
        "org_chart": "a corporate structure or organizational chart showing",
        "flow_diagram": "a business process or financial flow diagram showing",
        "infographic": "a financial infographic or dashboard showing",
    }
    return _PROMPTS.get(image_type, "a financial document showing")


# ══════════════════════════════════════════════════════════════════════════════
# CAPTION QUALITY + VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

_WEAK_PREFIXES = [
    "a blurry image of",
    "a close up of",
    "an image of",
    "a picture of",
    "a photo of",
    "photo of",
    "image of",
    "this is a",
    "this is an",
    "a view of",
    "a shot of",
]

_CAPTION_MAX_WORDS = 50
_CAPTION_MIN_WORDS = 5
_CAPTION_MIN_CHARS = 10
_SOLID_COLOR_VARIANCE_THRESHOLD = 5.0


def _cache_key(image_path: str) -> str:
    return hashlib.sha256(image_path.encode("utf-8")).hexdigest()


def _caption_cache_get(key: str) -> str | None:
    try:
        from app.core.infra_registry import infra

        mem = infra.get_memory()
        if mem:
            val = mem.cache_get(f"caption:{key}")
            return val if isinstance(val, str) else None
    except Exception:
        pass
    return None


def _caption_cache_set(key: str, caption: str) -> None:
    try:
        from app.core.infra_registry import infra

        mem = infra.get_memory()
        if mem:
            mem.cache_set(f"caption:{key}", caption, ttl=settings.REDIS_EMBEDDING_CACHE_TTL)
    except Exception:
        pass


def _remove_repetition(text: str) -> str:
    words = text.split()
    if len(words) < 6:
        return text
    half = len(words) // 2
    first = " ".join(words[:half])
    second = " ".join(words[half:])
    if first.strip().lower() == second.strip().lower():
        return first.strip()
    for n in (1, 2, 3, 4):
        if len(words) < n * 3:
            continue
        ngram = tuple(words[:n])
        run = 1
        i = n
        while i + n <= len(words):
            if tuple(words[i : i + n]) == ngram:
                run += 1
                i += n
            else:
                break
        if run >= 3:
            return " ".join(words[:n]).strip()
    return text


def _sanitize_caption(text: str) -> str:
    try:
        from app.guardrails.input_guard import sanitize as _guard_sanitize

        return _guard_sanitize(text, surface="frame_captioner")
    except Exception:
        return text


def _clean_caption(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    lower = text.lower()
    for pattern in _WEAK_PREFIXES:
        if lower.startswith(pattern):
            text = text[len(pattern) :].strip()
            lower = text.lower()
            break
    if "\x00" in text:
        text = text.replace("\x00", "")
    text = _sanitize_caption(text)
    text = _remove_repetition(text)
    words = text.split()
    if len(words) < _CAPTION_MIN_WORDS:
        return None
    text = " ".join(words[:_CAPTION_MAX_WORDS])
    text = text[0].upper() + text[1:] if text else text
    if len(text) < _CAPTION_MIN_CHARS:
        return None
    return text


def _caption_confidence(caption: str) -> float:
    words = caption.split()
    if not words:
        return 0.0
    unique = set(w.lower() for w in words)
    diversity = len(unique) / len(words)
    length_score = min(len(caption) / 100.0, 1.0)
    return round((diversity + length_score) / 2.0, 3)


def _is_solid_color(image: Image.Image) -> bool:
    try:
        import numpy as np

        arr = np.array(image.convert("RGB"), dtype=float)
        return float(arr.var()) < _SOLID_COLOR_VARIANCE_THRESHOLD
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")
    if path.stat().st_size == 0:
        raise EmptyContentError(f"EMPTY_IMAGE_FILE: {image_path}")
    if path.stat().st_size > settings.MAX_FILE_SIZE_IMAGE:
        raise ValueError(
            f"IMAGE_TOO_LARGE: {path.stat().st_size} bytes exceeds {settings.MAX_FILE_SIZE_IMAGE}"
        )
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if getattr(img, "is_animated", False):
                img.seek(0)
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if w == 0 or h == 0:
                raise ValueError("INVALID_IMAGE_DIMENSIONS: zero dimension")
            if w < 32 or h < 32:
                raise ValueError(f"IMAGE_TOO_SMALL: {w}x{h}")
            total_mp = (w * h) / 1_000_000
            if total_mp > settings.MAX_IMAGE_SIZE_MP:
                scale = (settings.MAX_IMAGE_SIZE_MP * 1_000_000 / (w * h)) ** 0.5
                new_w = max(int(w * scale), 32)
                new_h = max(int(h * scale), 32)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            elif max(w, h) > settings.MAX_IMAGE_DIM:
                img.thumbnail((settings.MAX_IMAGE_DIM, settings.MAX_IMAGE_DIM), Image.LANCZOS)
            return img.copy()
    except (FileNotFoundError, EmptyContentError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"IMAGE_LOAD_FAILED: {exc}") from exc


def _generate_thumbnail(image: Image.Image, output_path: Path) -> str | None:
    try:
        thumb = image.copy()
        thumb.thumbnail((settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT), Image.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(str(output_path), "JPEG", quality=85)
        return str(output_path)
    except Exception as exc:
        logger.warning(event="thumbnail_generation_failed", error=str(exc))
        return None


def _extract_tiff_frames(image_path: str) -> list[Image.Image]:
    frames: list[Image.Image] = []
    try:
        with Image.open(image_path) as img:
            if not hasattr(img, "n_frames") or img.n_frames <= 1:
                return []
            for i in range(img.n_frames):
                img.seek(i)
                frames.append(img.copy().convert("RGB"))
    except Exception as exc:
        logger.warning(event="tiff_frame_extraction_failed", error=str(exc))
    return frames


def _rasterize_svg(svg_path: str) -> Image.Image | None:
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(url=svg_path)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except ImportError:
        logger.warning(event="cairosvg_not_installed", hint="pip install cairosvg")
        return None
    except Exception as exc:
        logger.warning(event="svg_rasterize_failed", error=str(exc))
        return None


def _build_quality_metadata(
    caption: str,
    confidence: float,
    is_solid: bool,
    infer_ms: float,
    source: str,
) -> dict[str, Any]:
    return {
        "caption_confidence": confidence,
        "caption_word_count": len(caption.split()),
        "caption_char_count": len(caption),
        "is_solid_color": is_solid,
        "inference_ms": infer_ms,
        "caption_source": source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

_FINANCE_CAPTION_PROMPT = (
    "You are a financial analyst describing an image for a retrieval system. "
    "Describe: 1) Image type (bar chart, line chart, table, infographic, etc.) "
    "2) Exact title if visible "
    "3) All data series with EXACT numbers read from labels "
    "4) Time period covered "
    "5) Axis labels and units "
    "6) Key trends or anomalies. "
    "Be precise about numbers — do not round or paraphrase."
)

_SIGLIP_IMAGE_TYPES = [
    "bar_chart",
    "line_chart",
    "pie_chart",
    "candlestick_chart",
    "table_image",
    "org_chart",
    "flow_diagram",
    "infographic",
    "scanned_document",
    "dashboard_screenshot",
]


def blip_caption(image: Image.Image, prompt: str | None = None) -> str:
    """Generate a finance-aware caption using BLIP. Returns '' on failure."""
    try:
        from app.core.model_loader import model_loader

        processor, model, device = model_loader.get_blip()
    except Exception as exc:
        logger.warning(event="blip_unavailable", error=str(exc))
        return ""
    try:
        import torch as _torch

        # BLIP-1 cannot follow instruction prompts — it echoes them. So unless a
        # short caller-supplied prefix is given, caption UNCONDITIONALLY (no text),
        # which is what blip-image-captioning-large is actually trained for.
        text_prompt = prompt
        if text_prompt:
            inputs = processor(image, text=text_prompt, return_tensors="pt").to(device)
        else:
            inputs = processor(image, return_tensors="pt").to(device)
        with _torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=settings.BLIP_MAX_TOKENS,
                num_beams=settings.BLIP_NUM_BEAMS,
            )
        result = processor.decode(out[0], skip_special_tokens=True).strip()
        # Strip any echoed prompt (case-insensitive — BLIP often lowercases it).
        if text_prompt and result.lower().startswith(text_prompt.lower()):
            result = result[len(text_prompt) :].strip()
        return result
    except Exception as exc:
        logger.warning(event="blip_caption_failed", error=str(exc))
        return ""


def classify_image_type(image: Image.Image, ocr_text: str = "") -> str:
    """Zero-shot image type classification via SigLIP; falls back to heuristic."""
    try:
        import torch as _torch

        from app.core.model_loader import model_loader

        processor, clip_model, device = model_loader.get_siglip()
        inputs = processor(
            text=_SIGLIP_IMAGE_TYPES,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with _torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]
        return _SIGLIP_IMAGE_TYPES[probs.argmax().item()]
    except Exception:
        pass
    return _classify_image_type(image, ocr_text)


# EasyOCR singleton — loaded once per process; avoids repeated ~2 s init cost.
_easyocr_reader: Any | None = None


def _get_easyocr() -> Any:
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        _easyocr_reader = easyocr.Reader(
            ["en"], gpu=False, verbose=False, model_storage_directory=settings.EASYOCR_MODEL_DIR
        )
    return _easyocr_reader


_DOLLAR_MISREAD_RE = re.compile(r'(?<![A-Za-z0-9])S(\d{2,4}(?:\.\d+)?)\b')


def _normalize_ocr_text(text: str) -> str:
    """Fix the common EasyOCR '$' → 'S' misread on bold axis-label glyphs.

    A bare "S" immediately followed by 2-4 digits (e.g. "S500") is never a
    real word in finance chart labels — it is virtually always a misread
    dollar sign (e.g. "$500"), so this substitution is safe.
    """
    return _DOLLAR_MISREAD_RE.sub(r"$\1", text)


def ocr(image: Image.Image) -> str:
    """Run EasyOCR → TrOCR fallback on a PIL Image. Returns '' on failure.

    EasyOCR handles charts, axis labels, tables, and mixed-layout images —
    i.e. almost every finance image type this corpus contains (see
    _IMAGE_TYPES). TrOCR is a single-line printed-text recognizer; run only
    as a fallback since it degenerates to near-empty output on multi-region
    layouts like charts (verified: returns a single stray digit on a clean
    5-series line chart that EasyOCR reads correctly end-to-end).
    """
    img_rgb = image.convert("RGB")

    # EasyOCR (preferred — handles multi-region chart/table layouts)
    try:
        import numpy as np

        reader = _get_easyocr()
        results = reader.readtext(np.array(img_rgb))
        easy_text = " ".join(r[1] for r in results if r[2] > 0.3).strip()
        if easy_text:
            return _normalize_ocr_text(easy_text)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(event="easyocr_ocr_failed", error=str(exc))

    # TrOCR fallback — for simple single-line printed text only
    try:
        from app.core.model_loader import model_loader

        processor, model, device = model_loader.get_trocr()
        import torch as _torch

        pixel_values = processor(images=img_rgb, return_tensors="pt").pixel_values.to(device)
        with _torch.no_grad():
            generated_ids = model.generate(pixel_values)
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if result:
            logger.debug(event="trocr_fallback_used")
            return _normalize_ocr_text(result)
    except Exception as exc:
        logger.warning(event="trocr_fallback_failed", error=str(exc))

    return ""


def _get_ocr_boxes(image: Image.Image) -> list:
    """EasyOCR results with bounding boxes, for chart digitization (which
    needs pixel positions, not just joined text). Returns [] on failure —
    caller treats that as "can't digitize", not an error.
    """
    try:
        import numpy as np

        reader = _get_easyocr()
        return reader.readtext(np.array(image.convert("RGB")))
    except Exception as exc:
        logger.warning(event="ocr_boxes_failed", error=str(exc))
        return []


def _ocr_text_from_boxes(ocr_boxes: list) -> str:
    """Join EasyOCR box results into a single text string (same rule as
    ocr()), so a single OCR pass can feed both text and box-based extraction."""
    if not ocr_boxes:
        return ""
    text = " ".join(r[1] for r in ocr_boxes if r[2] > 0.3).strip()
    return _normalize_ocr_text(text) if text else ""


def _extract_title_from_ocr_boxes(ocr_boxes: list, image_height: int) -> str | None:
    """Extract the chart/figure title from OCR — the text lines in the top
    band of the image (above the plot area), which for finance charts is the
    human-readable title/subtitle (e.g. "Comparison of 5-Year Cumulative Total
    Return Among Apple Inc., the S&P 500 Index and ..."). More reliable than
    trusting the VLM to quote its own title. Returns None if no clear title
    band exists (caller falls back to the VLM-quoted title).
    """
    if not ocr_boxes or image_height <= 0:
        return None
    title_lines = []
    for bbox, text, conf in ocr_boxes:
        top_y = min(p[1] for p in bbox)
        t = text.strip()
        if (
            top_y < image_height * 0.13
            and conf > 0.5
            and len(t) >= 8
            and not _DATE_LABEL_RE.match(t)
            and not _DOLLAR_LABEL_RE.match(t)
        ):
            title_lines.append((top_y, min(p[0] for p in bbox), t))
    if not title_lines:
        return None
    # reading order: top-to-bottom, then left-to-right
    title_lines.sort(key=lambda x: (round(x[0] / 15), x[1]))
    title = " ".join(t for _, _, t in title_lines).strip()
    title = _normalize_ocr_text(title)
    # An ALL-CAPS main heading reads better title-cased for display; leave
    # already-mixed-case text (subtitles, proper nouns like "S&P", "U.S.")
    # untouched.
    _SMALL_WORDS = {"of", "the", "and", "a", "an", "in", "on", "to", "for", "vs"}

    def _smart_case(word: str, is_first: bool) -> str:
        # Re-case "shouty" words — all-uppercase, only letters/digits/hyphens
        # (incl. "5-YEAR"). Skip acronym/symbol tokens like "S&P", "U.S." (they
        # contain "&"/"." and must stay as-is) and already-mixed-case words.
        if not re.fullmatch(r'[A-Z0-9]+(?:-[A-Z0-9]+)*', word):
            return word
        if not any(c.isalpha() for c in word):
            return word  # pure number like "500" — leave it
        recased = "-".join(p.capitalize() for p in word.split("-"))
        low = recased.lower()
        if not is_first and low in _SMALL_WORDS:
            return low
        return recased

    words = title.split()
    title = " ".join(_smart_case(w, i == 0) for i, w in enumerate(words))
    return title[:150] if title else None


# ══════════════════════════════════════════════════════════════════════════════
# BLIP CAPTION  (BLIP → Qwen2-VL fallback for empty/failed captions)
# ══════════════════════════════════════════════════════════════════════════════


def _blip_caption(
    image: Image.Image,
    session_id: str,
    text_prompt: str | None = None,
) -> str | None:
    if torch is None:
        logger.warning(event="torch_unavailable_blip_skipped", session_id=session_id)
        return None

    try:
        result = blip_caption(image, prompt=text_prompt)
        if result:
            logger.debug(event="blip_caption_used", session_id=session_id)
            return result
    except Exception as exc:
        logger.debug(event="blip_caption_skipped", error=str(exc))

    return None


_IMAGE_QWEN2VL_PROMPT = (
    "Analyze this financial chart or document image. Report verbatim:\n"
    "1) Chart type and title\n"
    "2) All data series with EXACT values read from axis labels\n"
    "3) Time period covered\n"
    "4) All axis labels and units\n"
    "5) Key trends or notable figures\n"
    "Be precise about numbers — do not round or paraphrase."
)


def _qwen2vl_caption_for_image(image: Image.Image) -> str:
    """Qwen2-VL captioning for images when BLIP produces empty output.

    Qwen2-VL-2B-Instruct handles dense numeric chart layouts and mixed text+visual
    layouts via instruction-following training. Returns '' on failure.

    NOTE: injecting OCR-extracted axis/legend text into this prompt as
    "grounding" was tried and measured worse, not better — the 2B model
    over-indexed on the OCR list's reading order and inverted the entire
    value ranking (Apple, the highest line, got assigned the *lowest*
    value). Without that injection the model at least correctly identifies
    Apple as the highest series, so grounding was reverted. OCR text is
    still used independently for metadata/retrieval (see chunk()).
    """
    try:
        from app.core.model_loader import model_loader

        processor, model, device = model_loader.get_qwen2_vl()
    except Exception as exc:
        logger.warning(event="qwen2vl_unavailable_for_image", error=str(exc))
        return ""
    try:
        import torch as _torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _IMAGE_QWEN2VL_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with _torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=settings.QWEN2_VL_MAX_TOKENS)
        generated_ids = [o[len(i) :] for i, o in zip(inputs.input_ids, out, strict=False)]
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        logger.debug(event="qwen2vl_fallback_caption_used")
        return decoded
    except Exception as exc:
        logger.warning(event="qwen2vl_caption_failed_for_image", error=str(exc))
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC CAPTION API  (called by image_ingest.py and video_ingest.py via lazy import)
# ══════════════════════════════════════════════════════════════════════════════


def generate_caption(
    image_path: str,
    session_id: str,
    use_cache: bool = True,
) -> str | None:
    """Generate a plain BLIP caption for an image file."""
    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    start = time.time()
    path = Path(image_path)

    try:
        logger.debug(event="caption_start", image=path.name, session_id=session_id)

        if path.suffix.lower() == ".svg":
            image = _rasterize_svg(image_path)
            if image is None:
                logger.warning(event="svg_rasterize_failed_no_caption", image=path.name)
                return None
        else:
            image = _load_image(image_path)

        is_solid = _is_solid_color(image)
        if is_solid:
            return "Solid color or blank image frame."

        if path.suffix.lower() in (".tiff", ".tif"):
            tiff_frames = _extract_tiff_frames(image_path)
            if tiff_frames:
                image = tiff_frames[0]

        try:
            from app.utils.paths import resolved_temp_dir

            thumb_path = resolved_temp_dir() / "thumbs" / f"{_cache_key(image_path)}.jpg"
            _generate_thumbnail(image, thumb_path)
        except Exception:
            pass

        image_hash = _cache_key(image_path)
        raw_caption: str | None = None
        caption_source = "blip"
        infer_ms = 0.0

        if use_cache:
            raw_caption = _caption_cache_get(image_hash)
            if raw_caption:
                caption_source = "cache"

        if raw_caption is None:
            t_infer = time.time()
            raw_caption = _blip_caption(image, session_id)
            infer_ms = round((time.time() - t_infer) * 1000, 1)
            if use_cache and raw_caption:
                _caption_cache_set(image_hash, raw_caption)

        caption = _clean_caption(raw_caption) if raw_caption else None

        if not caption:
            logger.warning(
                event="caption_rejected",
                raw=str(raw_caption)[:80] if raw_caption else "",
                session_id=session_id,
            )
            return None

        confidence = _caption_confidence(caption)
        _build_quality_metadata(
            caption=caption,
            confidence=confidence,
            is_solid=is_solid,
            infer_ms=infer_ms,
            source=caption_source,
        )

        logger.debug(
            event="caption_success",
            length=len(caption),
            words=len(caption.split()),
            confidence=confidence,
            source=caption_source,
            latency_ms=round((time.time() - start) * 1000, 1),
            session_id=session_id,
        )
        return caption

    except (FileNotFoundError, EmptyContentError, ValueError) as exc:
        logger.warning(
            event="caption_image_error", image=path.name, error=str(exc), session_id=session_id
        )
        return None
    except Exception as exc:
        logger.error(
            event="caption_failed",
            image=path.name,
            session_id=session_id,
            error=str(exc),
            latency=round(time.time() - start, 3),
        )
        return None


def classify_and_caption(
    image_path: str,
    session_id: str,
    ocr_text: str = "",
    use_cache: bool = True,
) -> tuple[str | None, str, list[str]]:
    """Generate a finance-specific caption with image type classification.

    Returns:
        (caption, image_type, extracted_numbers)
    """
    start = time.time()
    path = Path(image_path)
    image_type = "infographic"
    extracted_numbers: list[str] = []

    try:
        if path.suffix.lower() == ".svg":
            image = _rasterize_svg(image_path)
        else:
            image = _load_image(image_path)

        if image is None:
            return None, image_type, extracted_numbers

        if ocr_text:
            extracted_numbers = _FIN_NUMBER_RE.findall(ocr_text)[:20]

        image_type = _classify_image_type(image, ocr_text)
        text_prompt = _finance_caption_prompt(image_type)

        image_hash = hashlib.sha256((image_path + "|" + text_prompt).encode()).hexdigest()
        raw_caption: str | None = None

        if use_cache:
            raw_caption = _caption_cache_get(image_hash)

        if raw_caption is None:
            # Qwen2-VL-2B-Instruct is instruction-tuned and follows the multi-part
            # finance analyst prompt correctly. BLIP-1 (the configured BLIP_MODEL,
            # Salesforce/blip-image-captioning-large) CANNOT follow instructions —
            # it treats the prompt as a prefix and echoes it back as the "caption".
            # So Qwen2-VL is primary; BLIP is only an unconditional (no-prompt)
            # fallback that still yields a basic but valid caption.
            raw_caption = _qwen2vl_caption_for_image(image)
            if not raw_caption or len(raw_caption.split()) < 5:
                raw_caption = _blip_caption(image, session_id, text_prompt=None)
            if use_cache and raw_caption:
                _caption_cache_set(image_hash, raw_caption)

        caption = _clean_caption(raw_caption) if raw_caption else None

        if caption and extracted_numbers:
            caption_lower = caption.lower()
            missing = [
                n
                for n in extracted_numbers
                if re.sub(r"[\s,]", "", n.lower()) not in re.sub(r"[\s,]", "", caption_lower)
            ]
            if missing:
                caption = caption + " [OCR values: " + " ".join(missing[:8]) + "]"

        logger.debug(
            event="classify_and_caption_done",
            image=path.name,
            image_type=image_type,
            numbers=len(extracted_numbers),
            latency_ms=round((time.time() - start) * 1000, 1),
            session_id=session_id,
        )
        return caption, image_type, extracted_numbers

    except Exception as exc:
        logger.warning(
            event="classify_and_caption_failed",
            image=path.name,
            error=str(exc),
            session_id=session_id,
        )
        return None, image_type, extracted_numbers


async def classify_and_caption_async(
    image_path: str,
    session_id: str,
    ocr_text: str = "",
    use_cache: bool = True,
) -> tuple[str | None, str, list[str]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: classify_and_caption(image_path, session_id, ocr_text, use_cache),
    )


async def generate_caption_async(
    image_path: str,
    session_id: str,
    use_cache: bool = True,
) -> str | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: generate_caption(image_path, session_id, use_cache),
    )


async def generate_captions_batch(
    image_paths: list[str],
    session_id: str,
) -> list[str | None]:
    sem = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)

    async def _cap(path: str) -> str | None:
        async with sem:
            return await generate_caption_async(path, session_id)

    return await asyncio.gather(*[_cap(p) for p in image_paths], return_exceptions=False)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE CHUNKER
# ══════════════════════════════════════════════════════════════════════════════


class ImageChunker(BaseChunker):
    """Finance-grade chunker for image files (charts, infographics, scanned docs).

    Each image = exactly 1 chunk (images are atomic).
    Pipeline: TrOCR → BLIP/Qwen2-VL caption → SigLIP image type → consistency check.
    """

    def chunk(
        self,
        extracts: list[RawExtract],
        meta: UniversalMetadata,
    ) -> list[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.jpg"
        surface = "image_chunker"
        modality_var.set("image")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="image", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="image", source=source)
            return []
        try:
            docs: list[IngestedDocument] = []

            for chunk_idx, ext in enumerate(extracts):
                if ext.extract_type != "image_raw":
                    continue
                raw = ext.raw_bytes or b""
                if not raw:
                    continue

                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as exc:
                    logger.warning(event="image_chunker_open_failed", error=str(exc), source=source)
                    continue

                img_width, img_height = img.size

                # Step 1: OCR (ground-truth numbers) — one EasyOCR pass with
                # bounding boxes, reused for text, title, and digitization.
                ocr_text = ""
                ocr_boxes: list = []
                try:
                    ocr_boxes = _get_ocr_boxes(img)
                    ocr_text = _ocr_text_from_boxes(ocr_boxes)
                    if not ocr_text:  # EasyOCR found nothing → TrOCR fallback path
                        ocr_text = ocr(img)
                except Exception as exc:
                    logger.warning(event="image_ocr_failed", error=str(exc))
                    try:
                        ocr_text = ocr(img)
                    except Exception:
                        pass

                # Chart/figure title from the top-of-image OCR band (reliable,
                # human-readable — used as the citation caption for all images).
                ocr_title = _extract_title_from_ocr_boxes(ocr_boxes, img_height)

                # Step 2: caption — Qwen2-VL primary (instruction-tuned), BLIP unconditional fallback.
                # BLIP-1 (blip-image-captioning-large) cannot follow instruction prompts —
                # it echoes them. Qwen2-VL-2B-Instruct gives detailed financial analysis.
                # BLIP is kept as a no-prompt fallback for when Qwen2-VL is unavailable.
                session_id = ""
                try:
                    cf = getattr(meta, "custom_fields", {}) or {}
                    session_id = str(cf.get("session_id", "") or "")
                except Exception:
                    pass
                caption_text = _qwen2vl_caption_for_image(img)
                if not caption_text:
                    try:
                        caption_text = _blip_caption(img, session_id=session_id) or ""
                    except Exception as exc:
                        logger.warning(event="image_blip_failed", error=str(exc))

                # Step 3: SigLIP image type classification (falls back to heuristic).
                image_type = "infographic"
                try:
                    image_type = classify_image_type(img, ocr_text)
                except Exception:
                    pass

                # Step 4: OCR-caption consistency check.
                mismatch = _check_mismatch(ocr_text, caption_text)
                if mismatch:
                    logger.warning(event="ocr_caption_mismatch", source=source)

                # Step 4.5: deterministic chart digitization (line charts only).
                # The VLM caption above cannot reliably read exact values off
                # a line chart (verified — see module docstring above
                # _digitize_line_chart). For this one image_type, pixel-level
                # axis calibration + curve tracing gives exact numbers instead
                # of a guess, so it is injected ahead of the VLM's own
                # (less reliable) numeric reading.
                digitized_text = ""
                digitized = None
                if image_type == "line_chart":
                    try:
                        digitized = _digitize_line_chart(img, ocr_boxes)
                        if digitized:
                            digitized_text = _format_digitized_chart(digitized)
                            logger.info(
                                event="chart_digitized",
                                source=source,
                                series=len(digitized["series"]),
                            )
                    except Exception as exc:
                        logger.warning(event="chart_digitize_error", error=str(exc))

                # Step 4.6: when digitization succeeds, the VLM's own numeric
                # "trends" claim is not just unnecessary but actively harmful —
                # verified: with both the correct and the VLM's guessed
                # numbers present in the same context, the 7B generation model
                # sometimes quoted the wrong (VLM) ones anyway. Cut that
                # section out and replace it with one built only from the
                # verified values, so only one set of numbers ever reaches
                # generation.
                if digitized:
                    caption_text = _replace_vlm_trends_with_verified(caption_text, digitized)

                # Step 5: Combine. Verified digitized values go right after
                # OCR and BEFORE the VLM caption — text is capped at
                # QDRANT_TEXT_MAX_CHARS (2000) and truncates from the end, so
                # the exact numbers must never be the part that gets cut if a
                # caption happens to run long; the VLM's approximate
                # narrative is the least reliable part and the safest to lose.
                combined = f"{image_type}:"
                if ocr_text:
                    combined += f"\n{ocr_text}"
                if digitized_text:
                    combined += f"\n{digitized_text}"
                combined += f"\n{caption_text}"
                combined = combined.strip()
                if not combined:
                    continue

                fin_entities = extract_finance_entities(combined)
                # Combine OCR + caption (not "ocr_text or caption_text") — a
                # short but non-empty OCR read (e.g. a single stray digit)
                # would otherwise fully mask richer numbers present only in
                # the caption text.
                #
                # Digitized values go FIRST and are never dropped by the
                # [:20] cap below (deduped afterward): image_bm25.py indexes
                # every entry in this list directly, but doc.text (which also
                # carries the digitized block) is truncated to BM25_MAX_TEXT_
                # CHARS (1000) by _base_text() — verified this chart's own
                # digitized block fits inside that cap today, but a chart with
                # more series/ticks could push it past 1000 chars and silently
                # drop values from BM25 with no error. Putting them here too
                # makes retrieval of the actual answer values robust
                # regardless of doc.text length.
                digitized_numbers: list[str] = []
                if digitized:
                    for row in digitized["values_by_tick"].values():
                        for val in row.values():
                            digitized_numbers.append(f"${val:.0f}")
                extracted_numbers = list(
                    dict.fromkeys(
                        digitized_numbers + _NUMBER_RE.findall(f"{ocr_text} {caption_text}")
                    )
                )[:20]
                chunk_hash = deterministic_chunk_id(source, "image_raw_0", chunk_idx)

                # Extract image_title (the human-readable caption shown in
                # citations, for ALL images). Priority:
                #   1. OCR title band at the top of the image — most reliable,
                #      it's the chart's own printed title.
                #   2. The title Qwen2-VL quotes in its caption ("... titled X").
                #   3. First caption sentence (stripped of "1) " list markers).
                if ocr_title:
                    image_title = ocr_title[:150]
                else:
                    _quote_match = re.search(r'["“]([^"”]{5,150})["”]', caption_text or "")
                    if _quote_match:
                        image_title = _quote_match.group(1).strip()[:150]
                    else:
                        raw_title = re.sub(r'^\d+\)\s*', '', (caption_text or "").strip())
                        raw_title = raw_title.split(".")[0].split("\n")[0].strip()
                        image_title = raw_title[:150] if raw_title else source

                # Time period and data series from text (regex-based)
                full_text = f"{caption_text} {ocr_text}".strip()
                time_period = ext.extra.get("time_period") or _extract_time_period(full_text)
                data_series = ext.extra.get("data_series") or _extract_data_series(full_text)

                # Watermark detection from OCR+caption
                watermark_detected = _detect_watermark(full_text)

                # Caption confidence
                caption_confidence = _caption_confidence(caption_text) if caption_text else 0.0

                # Quality score from ingestor signals
                blur_score = ext.extra.get("blur_score")
                solid_color = ext.extra.get("solid_color", False)
                quality_score: float | None = None
                if blur_score is not None:
                    qs = float(blur_score)
                    if img_width * img_height > 500_000:
                        qs = min(qs + 0.1, 1.0)
                    if watermark_detected:
                        qs = max(qs - 0.2, 0.0)
                    if not ocr_text:
                        qs = max(qs - 0.05, 0.0)
                    if solid_color:
                        qs = max(qs - 0.3, 0.0)
                    quality_score = round(qs, 3)

                # source_path enables _resolve_asset_path() in MultimodalEmbedder so
                # this doc gets BOTH BGE text embedding AND SigLIP vision embedding.
                source_path = str(getattr(meta, "source_path", "") or "")

                structure = {
                    "chunk_hash_id": chunk_hash,
                    "source_file": source,
                    "source_path": source_path,
                    "asset_path": source_path,
                    "chunk_index": chunk_idx,
                    "image_type": image_type,
                    "image_title": image_title,
                    "caption": caption_text,
                    "caption_confidence": caption_confidence,
                    "ocr_text": ocr_text,
                    "extracted_numbers": extracted_numbers,
                    "time_period": time_period,
                    "data_series": data_series,
                    "ocr_caption_mismatch": mismatch,
                    "parent_document": ext.extra.get("parent_document"),
                    "parent_page": ext.extra.get("parent_page"),
                    "finance_entities": fin_entities,
                    "image_width": img_width,
                    "image_height": img_height,
                    # Phase 1.5 quality signals from ingestor
                    "blur_score": blur_score,
                    "quality_score": quality_score,
                    "solid_color": solid_color,
                    "watermark_detected": watermark_detected,
                    "face_count": ext.extra.get("face_count"),
                    "dominant_colors": ext.extra.get("dominant_colors", []),
                    "perceptual_hash": ext.extra.get("phash"),
                    "thumbnail_path": ext.extra.get("thumbnail_path"),
                    "digitized_chart_values": digitized_text or None,
                }

                doc = self._make_doc(
                    text=combined,
                    modality="image",
                    subtype="caption",
                    source=source,
                    page=None,
                    chunk_idx=chunk_idx,
                    structure=structure,
                    meta=meta,
                    surface=surface,
                )
                if doc:
                    docs.append(doc)

                # VISION DOC — a separate vision-space doc so the image also lands
                # in vision_collection via SigLIP. The caption doc above covers
                # text_collection. Two docs / two embedding spaces — never one
                # aliased doc (see _route_documents in base_embedder). The offset
                # chunk_idx guarantees a distinct Qdrant point id.
                if source_path and Path(source_path).exists():
                    vis_structure = dict(structure)
                    vis_structure["embedding_space"] = "vision"
                    vis_structure["chunk_index"] = chunk_idx + 1_000_000
                    vis_doc = self._make_doc(
                        text=combined or image_title or caption_text or "image",
                        modality="image",
                        subtype="image_frame",
                        source=source,
                        page=None,
                        chunk_idx=chunk_idx + 1_000_000,
                        structure=vis_structure,
                        meta=meta,
                        surface=surface,
                    )
                    if vis_doc:
                        docs.append(vis_doc)

            logger.info(event="image_chunking_done", source=source, chunks=len(docs))
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="image", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "image",
            "status": "ok",
            "class": self.__class__.__name__,
        }
