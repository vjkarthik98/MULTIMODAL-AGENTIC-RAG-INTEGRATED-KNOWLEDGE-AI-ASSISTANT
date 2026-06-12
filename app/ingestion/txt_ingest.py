"""
TXT/MD/CSV/LOG ingestor — Phase 1 per-modality refactor.

TxtIngestor.extract() → List[RawExtract]   (extraction only; no chunking)
ingest()              → List[IngestedDocument]  (backward-compat; full pipeline)
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chardet
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.ingestion.base_ingest import BaseIngestor
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
# ── TEXT REPAIR PASSES ────────────────────────────────────────────────────────

def repair_mojibake(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0
    try:
        import ftfy
        fixed = ftfy.fix_text(text)
        diff  = sum(1 for a, b in zip(text, fixed) if a != b) + abs(len(text) - len(fixed))
        return fixed, diff
    except ImportError:
        return text, 0
    except Exception as exc:
        logger.warning("mojibake_repair_failed", error=str(exc))
        return text, 0


_LOG_LINE_RE        = re.compile(r"^\s*---\s*LOG\s+ENTRY", re.IGNORECASE)
_DEBUG_LINE_RE      = re.compile(r"^\s*===\s*DEBUG[:=]", re.IGNORECASE)
_ERROR_LINE_RE      = re.compile(r"^\s*ERROR\s*:|^\s*\[ERROR\]|^\s*NULL POINTER", re.IGNORECASE)
_HTML_TAG_RE        = re.compile(r"<[^>]+>")
_BINARY_GARBAGE_RE  = re.compile(r"[\x00-\x08\x0E-\x1F]")
_HEX_ESCAPE_RE      = re.compile(r"(?:\\x[0-9a-fA-F]{2}){3,}")
_SYMBOL_LINE_RE     = re.compile(r"^[^A-Za-z0-9\s]{20,}$")


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _LOG_LINE_RE.match(s) or _DEBUG_LINE_RE.match(s) or _ERROR_LINE_RE.match(s):
        return True
    if _SYMBOL_LINE_RE.match(s) or _HEX_ESCAPE_RE.search(s):
        return True
    if len(s) >= 10 and sum(1 for c in s if c.isalnum() or c.isspace()) / len(s) < 0.4:
        return True
    return False


def strip_noise_lines(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0
    try:
        import bleach
        text = bleach.clean(text, tags=[], attributes={}, strip=True)
    except ImportError:
        text = _HTML_TAG_RE.sub(" ", text)
    except Exception:
        text = _HTML_TAG_RE.sub(" ", text)
    text = _BINARY_GARBAGE_RE.sub("", text)
    dropped = 0
    out_lines: List[str] = []
    for line in text.split("\n"):
        if _is_noise_line(line):
            dropped += 1
        else:
            out_lines.append(line)
    return "\n".join(out_lines), dropped


_MIN_RECOVERY_LEN = 200


def _whitespace_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isspace()) / len(text)


def recover_whitespace(text: str) -> Tuple[str, bool]:
    if not getattr(settings, "TEXT_REPAIR_WHITESPACE", True):
        return text, False
    if not text or len(text) < _MIN_RECOVERY_LEN or _whitespace_ratio(text) >= 0.05:
        return text, False
    try:
        from wordsegment import load as _ws_load, segment
        _ws_load()
        rebuilt: List[str] = []
        for line in text.split("\n"):
            if len(line) >= _MIN_RECOVERY_LEN and _whitespace_ratio(line) < 0.05:
                words = segment(line)
                rebuilt.append(" ".join(words) if words else line)
            else:
                rebuilt.append(line)
        return "\n".join(rebuilt), True
    except ImportError:
        return text, False
    except Exception as exc:
        logger.warning("whitespace_recovery_failed", error=str(exc))
        return text, False


_OCR_SUBS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"}


def _english_word_ratio(text: str) -> float:
    try:
        from wordfreq import top_n_list
        tokens = re.findall(r"[a-z]{3,}", text.lower())
        if not tokens:
            return 0.0
        common = set(top_n_list("en", 30000))
        return sum(1 for t in tokens if t in common) / max(1, len(tokens))
    except Exception:
        return 1.0


def normalize_ocr_noise(text: str) -> Tuple[str, bool]:
    if not getattr(settings, "TEXT_REPAIR_OCR", True):
        return text, False
    if not text or len(text) < 60 or _english_word_ratio(text) >= 0.30:
        return text, False

    def _fix_token(tok: str) -> str:
        if any(c.isalpha() for c in tok) and any(c.isdigit() or c in "@$" for c in tok):
            return "".join(_OCR_SUBS.get(c, c) for c in tok)
        return tok

    rebuilt = re.sub(r"\S+", lambda m: _fix_token(m.group(0)), text)
    return (rebuilt, True) if _english_word_ratio(rebuilt) > _english_word_ratio(text) else (text, False)


_SUPERSCRIPT_RE       = re.compile(r"^\s*[¹²³⁴⁵⁶⁷⁸⁹⁰]\s+")
_EDITOR_NOTE_RE       = re.compile(r"^\s*\[EDITOR\s+NOTE", re.IGNORECASE)
_REFERENCE_MISSING_RE = re.compile(r"^\s*\[REFERENCE\s+MISSING", re.IGNORECASE)


def strip_footnotes(text: str) -> Tuple[str, List[str]]:
    if not getattr(settings, "TEXT_REPAIR_FOOTNOTES", True) or not text:
        return text, []
    notes: List[str] = []
    cleaned: List[str] = []
    for line in text.split("\n"):
        if _SUPERSCRIPT_RE.match(line) or _EDITOR_NOTE_RE.match(line) or _REFERENCE_MISSING_RE.match(line):
            notes.append(line.strip())
        else:
            cleaned.append(line)
    return "\n".join(cleaned), notes


_PLACEHOLDER_PATTERNS = [
    re.compile(r"^\s*\.+\s*$"),
    re.compile(r"^\s*\[PAGE\s+INTENTIONALLY\s+LEFT\s+BLANK\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(?content\s+removed", re.IGNORECASE),
    re.compile(r"^\s*\[REST\s+OF\s+DOCUMENT\s+MISSING", re.IGNORECASE),
]


def is_placeholder(text: str) -> bool:
    if not getattr(settings, "TEXT_REPAIR_PLACEHOLDERS", True):
        return not text or not text.strip()
    if not text or len(text.strip()) < settings.CHUNK_MIN_SIZE:
        return True
    return any(pat.match(text.strip()) for pat in _PLACEHOLDER_PATTERNS)


def has_title_mismatch(section_title: Optional[str], keywords: List[str]) -> bool:
    if not getattr(settings, "TEXT_REPAIR_TITLE_MISMATCH", True) or not section_title or not keywords:
        return False
    title_tokens = set(re.findall(r"[a-z]{3,}", section_title.lower()))
    kw_tokens: set = set()
    for kw in keywords[:5]:
        kw_tokens.update(re.findall(r"[a-z]{3,}", str(kw).lower()))
    return bool(title_tokens and kw_tokens and title_tokens.isdisjoint(kw_tokens))


_ERROR_MARKER_PATTERNS = [
    re.compile(
        r"(?:NOTE\s*:?\s*)?[\"']?[^\"'\n]{0,40}[\"']?\s*"
        r"(?:does\s+not\s+exist|is\s+an?\s+intentional\s+error|"
        r"for\s+testing|fake|hallucinated)[^.\n]*\.?",
        re.IGNORECASE,
    ),
    re.compile(r"[→\->]+\s*WRONG\s+LABEL[^\n]*", re.IGNORECASE),
    re.compile(r"(?:this\s+claim\s+is\s+)?disputed(?:\s+in\s+some\s+literature)?", re.IGNORECASE),
    re.compile(r"\[CONFIRMED\]|\[UPDATED:[^\]]*\]|\[UNCHANGED\]", re.IGNORECASE),
]


def detect_error_markers(text: str) -> Tuple[str, List[str]]:
    if not text:
        return text, []
    found: List[str] = []
    cleaned = text
    for pat in _ERROR_MARKER_PATTERNS:
        for match in pat.finditer(cleaned):
            snippet = match.group(0).strip()
            if snippet:
                label = re.sub(r"\s+", " ", snippet).strip(" .\"'")
                if len(label) > 80:
                    label = label[:77] + "..."
                if label and label.lower() not in {f.lower() for f in found}:
                    found.append(label)
        cleaned = pat.sub(" ", cleaned)
    if found:
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    return cleaned, found


_VERSION_RE = re.compile(r"-v(\d+)|REVISED|FINAL|DRAFT", re.IGNORECASE)


def extract_version(section_id: Optional[str], section_title: Optional[str]) -> Optional[Dict[str, str]]:
    if not getattr(settings, "TEXT_REPAIR_VERSION_TAG", True):
        return None
    for c in (section_id or "", section_title or ""):
        m = _VERSION_RE.search(c)
        if m:
            if m.group(1):
                return {"version": f"v{m.group(1)}", "kind": "numeric"}
            return {"version": m.group(0).upper(), "kind": "label"}
    return None


def repair_text(raw: str) -> Tuple[str, Dict[str, int]]:
    stats: Dict[str, int] = {}
    if not raw or not getattr(settings, "TEXT_REPAIR_ENABLED", True):
        return raw, stats
    text = raw
    if getattr(settings, "TEXT_REPAIR_MOJIBAKE", True):
        text, n = repair_mojibake(text)
        if n:
            stats["mojibake_chars_changed"] = n
    if getattr(settings, "TEXT_REPAIR_NOISE_LINES", True):
        text, n = strip_noise_lines(text)
        if n:
            stats["noise_lines_dropped"] = n
    return text, stats
from app.utils.logger import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_ingest_duration = Histogram(
    "txt_ingest_duration_seconds",
    "TXT ingestion duration",
    ["status"],
)
_ingest_errors = Counter(
    "txt_ingest_errors_total",
    "TXT ingestion errors by type",
    ["error_type"],
)

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".log",
    ".json", ".yaml", ".yml",
}

BINARY_MAGIC_SIGNATURES = [
    b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"%PDF",
    b"PK\x03\x04", b"\x1f\x8b", b"BM", b"ID3",
    b"\x00\x00\x00", b"RIFF", b"\x7fELF",
]

_semaphore = asyncio.Semaphore(5)

_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠⁡⁢⁣⁤﻿­]")
_FANCY_SPACE_RE = re.compile(r"[   -   　]")

_SPEAKER_TURN_RE = re.compile(
    r"^[ \t]*(?P<speaker>[A-Z][A-Z0-9 \-\.']+(?:\s*[-–]\s*[A-Z][A-Z0-9 \-]+)?)[ \t]*:",
    re.MULTILINE,
)
_TRANSCRIPT_SEPARATORS = [
    "\nOPERATOR:", "\nCEO:", "\nCFO:", "\nCTO:", "\nCOO:",
    "\nANALYST:", "\nMODERATOR:", "\nQUESTION:", "\nANSWER:",
]
_CALL_SECTIONS = {
    "prepared_remarks": ["prepared remarks", "opening remarks", "opening statement"],
    "qa_session": ["question and answer", "q and a", "q&a session",
                   "we will now begin", "your first question"],
    "operator": ["thank you", "please stand by", "your lines have been placed"],
}

_NUMBERED_SECTION_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:SECTION\s+)?(\d+)(?:\.\d+)*\s*[:\.\-\)]",
    re.IGNORECASE,
)
_FORWARD_LOOKING_WORDS = frozenset([
    "outlook", "guidance", "forecast", "projection", "forward", "target",
    "expected", "anticipated", "objective", "strategy", "strategic",
    "plan", "planned", "pipeline", "upcoming", "going forward",
    "future", "fy2025", "fy2026", "fy2027",
])

_SECTION_HEADER_RE = re.compile(
    r"^[ \t]*\[(DOC-[A-Za-z0-9._\-]+)\][ \t]*(.*)$",
    re.MULTILINE,
)

_TABLE_SEPARATOR_RE = re.compile(r"^[\s\-=|+]{6,}$")
_TABLE_VALUE_RE = re.compile(r'\b\d{1,3}(?:,\d{3})+\b|\b\d+\.\d+\b|\b\d{4,}\b')
_PIPE_ROW_RE = re.compile(r"^\s*\|")
_PIPE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_YEAR_IN_LINE_RE = re.compile(r"\b(20\d{2})\b")
_FINANCIAL_HEADING_RE = re.compile(
    r"^(?:PART\s+[IVX]+|ITEM\s+\d+[A-Z]?\.|"
    r"CONSOLIDATED\s+STATEMENTS?\s+OF|"
    r"NOTES?\s+TO\s+(?:CONSOLIDATED\s+)?FINANCIAL|"
    r"MANAGEMENT[''S]*\s+DISCUSSION|"
    r"SELECTED\s+FINANCIAL|"
    r"RESULTS?\s+OF\s+OPERATIONS?|"
    r"LIQUIDITY\s+AND\s+CAPITAL|"
    r"CRITICAL\s+ACCOUNTING)",
    re.IGNORECASE,
)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _simhash(text: str) -> int:
    tokens = text.lower().split()
    v = [0] * 64
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fp = 0
    for i in range(64):
        if v[i] > 0:
            fp |= (1 << i)
    return fp


def _simhash_distance(h1: int, h2: int) -> int:
    x = h1 ^ h2
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        for sig in BINARY_MAGIC_SIGNATURES:
            if header.startswith(sig):
                return True
        if b"\x00" in header:
            return True
        return False
    except Exception:
        return False


def _strip_bom(text: str) -> str:
    for bom in ["﻿", "￾"]:
        if text.startswith(bom):
            return text[len(bom):]
    return text


def _strip_null_bytes(text: str) -> Tuple[str, int]:
    count = text.count("\x00")
    return text.replace("\x00", ""), count


def _detect_encoding(path: Path) -> Tuple[str, float]:
    with open(path, "rb") as f:
        raw = f.read(32768)
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    confidence = float(result.get("confidence") or 0.0)
    return encoding, confidence


def _load_text(path: Path) -> str:
    encoding, confidence = _detect_encoding(path)
    if confidence < 0.7:
        logger.warning("encoding_low_confidence", encoding=encoding,
                       confidence=round(confidence, 3), file=path.name)
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def _load_text_streaming(path: Path, max_bytes: int) -> str:
    encoding, _ = _detect_encoding(path)
    parts: List[str] = []
    total = 0
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                if len(line) > 100_000:
                    words = line.split(" ")
                    chunk = ""
                    for word in words:
                        if len(chunk) + len(word) > 100_000:
                            parts.append(chunk)
                            total += len(chunk)
                            chunk = word + " "
                        else:
                            chunk += word + " "
                    if chunk:
                        parts.append(chunk)
                        total += len(chunk)
                else:
                    parts.append(line)
                    total += len(line)
                if total >= max_bytes:
                    logger.warning("streaming_truncated_at_limit", file=path.name, bytes_read=total)
                    break
    except Exception as exc:
        logger.warning("streaming_load_failed", file=path.name, error=str(exc))
    return "".join(parts)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _FANCY_SPACE_RE.sub(" ", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _detect_language(text: str) -> Optional[str]:
    sample = text[:3000]
    try:
        from lingua import LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_all_languages().build()
        lang = detector.detect_language_of(sample)
        if lang:
            return lang.iso_code_639_1.name.lower()
    except Exception:
        pass
    return None


def _is_rtl(language: Optional[str]) -> bool:
    return (language or "").lower() in {"ar", "he", "fa", "ur", "yi", "dv", "ku", "ps"}


def _redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    if not settings.PII_DETECTION_ENABLED:
        return text, {}
    entity_counts: Dict[str, int] = {}
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        entities = getattr(settings, "PII_ENTITIES", [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
            "LOCATION", "IP_ADDRESS",
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


def _detect_subtype(chunk: str) -> str:
    lines = [l.strip() for l in chunk.split("\n") if l.strip()]
    if not lines:
        return "paragraph"
    first = lines[0]
    if first.startswith("#"):
        return "heading"
    if len(lines) == 1 and len(first.split()) <= 8 and not first.endswith("."):
        return "heading"
    return "paragraph"


def _extract_heading_level(line: str) -> Optional[int]:
    match = re.match(r"^(#{1,3})\s", line)
    if match:
        return len(match.group(1))
    return None


def _extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(top=max_keywords, stopwords=None)
        return [kw for kw, _ in kw_extractor.extract_keywords(text)]
    except ImportError:
        pass
    try:
        from keybert import KeyBERT
        kb = KeyBERT()
        return [kw for kw, _ in kb.extract_keywords(text, top_n=max_keywords)]
    except ImportError:
        pass
    return []


def _readability_score(text: str) -> float:
    try:
        import textstat
        return float(textstat.flesch_reading_ease(text))
    except ImportError:
        pass
    try:
        sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
        words = max(len(text.split()), 1)
        syllables = sum(max(1, sum(1 for ch in w.lower() if ch in "aeiou")) for w in text.split())
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return round(max(0.0, min(score, 100.0)), 2)
    except Exception:
        return 0.0


def _quality_score(chunk: str) -> float:
    length = len(chunk)
    word_count = len(chunk.split())
    if length < settings.CHUNK_MIN_SIZE:
        return 0.1
    if length < 100 or word_count < 10:
        return 0.3
    if length < 300 or word_count < 30:
        return 0.6
    return 1.0


def _detect_section_metadata(chunk: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    m = _NUMBERED_SECTION_RE.search(chunk[:400])
    if m:
        try:
            meta["section_number"] = int(m.group(1))
        except (ValueError, IndexError):
            pass
    meta["is_forward_looking"] = any(w in chunk.lower() for w in _FORWARD_LOOKING_WORDS)
    return meta


def _split_sections(text: str) -> List[Tuple[Optional[str], Optional[str], str]]:
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [(None, None, text)]
    sections: List[Tuple[Optional[str], Optional[str], str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble and len(preamble) >= settings.CHUNK_MIN_SIZE:
        sections.append((None, None, preamble))
    for i, m in enumerate(matches):
        section_id = m.group(1).strip()
        section_title = m.group(2).strip() or None
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        sections.append((section_id, section_title, body))
    return sections


def _looks_like_data_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.search(r"\S(?:  +)\S", line))


def _extract_table_blocks(text: str) -> List[Tuple[int, int, str, List[str]]]:
    lines = text.split("\n")
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    blocks: List[Tuple[int, int, str, List[str]]] = []
    MIN_TABLE_LINES = 4
    i = 0
    while i < len(lines):
        if not lines[i].strip() or _TABLE_SEPARATOR_RE.match(lines[i]):
            i += 1
            continue
        if not _looks_like_data_row(lines[i]):
            i += 1
            continue
        run_start = i
        run_lines: List[int] = [i]
        j = i + 1
        while j < len(lines):
            row = lines[j]
            if _TABLE_SEPARATOR_RE.match(row):
                j += 1
                continue
            if not row.strip():
                break
            if not _looks_like_data_row(row):
                break
            run_lines.append(j)
            j += 1
        if len(run_lines) >= MIN_TABLE_LINES:
            header_idx = run_lines[0]
            data_indices = run_lines[1:]
            header = lines[header_idx]
            data_rows = [lines[k] for k in data_indices]
            start_off = offsets[header_idx]
            end_line_idx = run_lines[-1]
            end_off = offsets[end_line_idx] + len(lines[end_line_idx]) + 1
            blocks.append((start_off, end_off, header, data_rows))
            i = j
            continue
        i += 1
    return blocks


def _is_pipe_row(line: str) -> bool:
    return bool(_PIPE_ROW_RE.match(line)) and line.count("|") >= 2


def _is_pipe_separator_row(line: str) -> bool:
    stripped = line.strip().strip("|")
    cells = [c.strip() for c in stripped.split("|")]
    return bool(cells) and all(bool(_PIPE_SEP_CELL_RE.match(c)) for c in cells if c)


def _normalize_pipe_row(line: str) -> str:
    parts = line.strip().strip("|").split("|")
    non_trivial = [p.strip() for p in parts if p.strip()]
    return " | ".join(non_trivial)


def _extract_pipe_table_blocks(text: str) -> List[Tuple[int, int, str, List[str]]]:
    lines = text.split("\n")
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    blocks: List[Tuple[int, int, str, List[str]]] = []
    MIN_PIPE_ROWS = 3
    i = 0
    while i < len(lines):
        if not _is_pipe_row(lines[i]):
            i += 1
            continue
        run_start = i
        run_indices: List[int] = []
        j = i
        while j < len(lines):
            ln = lines[j]
            if not ln.strip():
                break
            if _is_pipe_row(ln):
                run_indices.append(j)
                j += 1
            else:
                break
        content_rows = [k for k in run_indices if not _is_pipe_separator_row(lines[k])]
        if len(content_rows) >= MIN_PIPE_ROWS:
            look_back_lines: List[str] = []
            for step in range(1, 13):
                lb = run_start - step
                if lb < 0:
                    break
                lb_line = lines[lb].strip()
                if not lb_line:
                    break
                look_back_lines.insert(0, lb_line)
            _year_candidates = look_back_lines + [lines[content_rows[0]].strip()]
            year_matches: List[str] = []
            best_count = 0
            for cand in _year_candidates:
                yrs = list(dict.fromkeys(_YEAR_IN_LINE_RE.findall(cand)))
                if len(yrs) > best_count:
                    best_count = len(yrs)
                    year_matches = yrs
            if best_count < 2:
                year_matches = []
            preceding_heading: Optional[str] = None
            for step in range(1, 9):
                lb = run_start - step
                if lb < 0:
                    break
                lb_line = lines[lb].strip()
                if not lb_line:
                    break
                if _FINANCIAL_HEADING_RE.match(lb_line) or (
                    lb_line.isupper() and 3 <= len(lb_line.split()) <= 10
                ):
                    preceding_heading = lb_line
                    break
            header_norm = _normalize_pipe_row(lines[content_rows[0]])
            header_parts: List[str] = []
            if preceding_heading:
                header_parts.append(preceding_heading)
            if year_matches:
                header_parts.append("Years: " + ", ".join(year_matches))
            header_parts.append(header_norm)
            enriched_header = "\n".join(header_parts)
            data_rows = [_normalize_pipe_row(lines[k]) for k in content_rows[1:]]
            data_rows = [d for d in data_rows if d]
            if data_rows:
                start_off = offsets[run_start]
                end_off = offsets[run_indices[-1]] + len(lines[run_indices[-1]]) + 1
                blocks.append((start_off, end_off, enriched_header, data_rows))
        i = j if j > i else i + 1
    return blocks


def _make_nl_summary(rows: List[str], years: Optional[List[str]] = None) -> str:
    years = years or []
    mapping_lines: List[str] = []
    fallback_labels: List[str] = []
    fallback_numbers: List[str] = []
    seen_labels: set = set()
    for row in rows[:12]:
        parts = [p.strip() for p in row.split("|") if p.strip()]
        label: Optional[str] = None
        for p in parts:
            if p and not re.match(r'^[\$\d,\.%\(\)\-\s]+$', p):
                label = p[:40].strip()
                break
        nums = _TABLE_VALUE_RE.findall(row)
        if label and label not in seen_labels:
            seen_labels.add(label)
            fallback_labels.append(label[:30])
        fallback_numbers.extend(nums[:3])
        if label and years and nums:
            pairs = [f"FY{yr} = {nums[idx]}" for idx, yr in enumerate(years) if idx < len(nums)]
            if pairs:
                mapping_lines.append(f"{label}: " + ", ".join(pairs))
    if mapping_lines:
        return "[Financial data by fiscal year]\n" + "\n".join(mapping_lines) + "\n"
    parts_out: List[str] = []
    if fallback_labels:
        parts_out.append(" | ".join(fallback_labels[:4]))
    if fallback_numbers:
        parts_out.append(" | ".join(fallback_numbers[:6]))
    if parts_out:
        return "[Financial data: " + " — ".join(parts_out) + "]\n"
    return ""


def _chunk_table(header: str, data_rows: List[str], min_size: Optional[int] = None) -> List[str]:
    chunks: List[str] = []
    header_len = len(header) + 1
    _years_m = re.search(r'Years:\s*([0-9,\s]+)', header)
    header_years = re.findall(r'20\d{2}', _years_m.group(1)) if _years_m else []
    pending_rows: List[str] = []
    pending_len = 0
    effective_min = min_size if min_size is not None else settings.CHUNK_MIN_SIZE

    def flush():
        if pending_rows:
            nl_prefix = _make_nl_summary(pending_rows, header_years)
            chunks.append(nl_prefix + header + "\n" + "\n".join(pending_rows))

    for row in data_rows:
        candidate_size = header_len + pending_len + len(row) + 1
        if pending_rows and candidate_size >= effective_min:
            flush()
            pending_rows = [row]
            pending_len = len(row) + 1
        else:
            pending_rows.append(row)
            pending_len += len(row) + 1
    flush()
    return chunks


def _protect_finance_numbers(text: str) -> Tuple[str, Dict[str, str]]:
    placeholder = "\x02COMMA\x03"
    _INNER_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
    protected = _INNER_COMMA_RE.sub(placeholder, text)
    mapping = {placeholder: ","}
    return protected, mapping


def _restore_finance_numbers(chunk: str, mapping: Dict[str, str]) -> str:
    for placeholder, original in mapping.items():
        chunk = chunk.replace(placeholder, original)
    return chunk


def _detect_transcript_format(text: str) -> bool:
    sample = text[:3000]
    return len(_SPEAKER_TURN_RE.findall(sample)) >= 3


def _extract_speaker(chunk: str) -> Optional[str]:
    first_line = chunk.split("\n")[0].strip()
    m = re.match(
        r"^(?P<speaker>[A-Z][A-Z0-9 \-\.']+(?:\s*[-–]\s*[A-Z][A-Z0-9 \-]+)?)[ \t]*:",
        first_line,
    )
    return m.group("speaker").strip() if m else None


def _detect_call_section(chunk: str) -> Optional[str]:
    lower = chunk.lower()
    for section, keywords in _CALL_SECTIONS.items():
        if any(kw in lower for kw in keywords):
            return section
    return None


def _extract_section_headings(text: str) -> List[Tuple[int, str]]:
    headings: List[Tuple[int, str]] = []
    for m in re.finditer(
        r"(?:^|\n)(?P<heading>"
        r"PART\s+[IVX]+[^\n]{0,60}|"
        r"ITEM\s+\d+[A-Z]?\.[^\n]{0,80}|"
        r"(?:[A-Z][A-Z\s\-\&]{10,60})(?=\n))",
        text, re.MULTILINE,
    ):
        h = m.group("heading").strip()
        if h and len(h.split()) <= 12:
            headings.append((m.start("heading"), h))
    return headings


def _chunk_text(text: str, is_transcript: bool = False) -> List[str]:
    table_chunks: List[str] = []
    pipe_blocks = _extract_pipe_table_blocks(text)
    if pipe_blocks:
        non_pipe_text = text
        for start_off, end_off, header, data_rows in reversed(pipe_blocks):
            non_pipe_text = non_pipe_text[:start_off] + non_pipe_text[end_off:]
            table_chunks.extend(_chunk_table(header, data_rows, min_size=600))
        text = non_pipe_text

    fixed_blocks = _extract_table_blocks(text)
    if fixed_blocks:
        non_table_text = text
        for start_off, end_off, header, data_rows in reversed(fixed_blocks):
            non_table_text = non_table_text[:start_off] + non_table_text[end_off:]
            table_chunks.extend(_chunk_table(header, data_rows))
        text = non_table_text

    protected_text, num_mapping = _protect_finance_numbers(text)

    chunks: List[str] = []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        if is_transcript:
            separators = _TRANSCRIPT_SEPARATORS + [
                "\n[DOC-", "\nPART ", "\nITEM ", "\nSECTION ",
                "\n====", "\n----", "\n####", "\n###", "\n##", "\n#",
                "\n\n", "\n", ". ", "! ", "? ", " ", "",
            ]
        else:
            separators = [
                "\n[DOC-", "\nPART ", "\nITEM ", "\nSECTION ",
                "\n====", "\n----", "\n####", "\n###", "\n##", "\n#",
                "\n\n", "\n", ". ", "! ", "? ", " ", "",
            ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=separators,
        )
        chunks = splitter.split_text(protected_text)
        chunks = [_restore_finance_numbers(c.strip(), num_mapping) for c in chunks if c.strip()]
    except Exception:
        chunks = []

    if not chunks:
        size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        step = max(size - overlap, 1)
        for i in range(0, len(protected_text), step):
            ch = _restore_finance_numbers(protected_text[i:i + size].strip(), num_mapping)
            if ch:
                chunks.append(ch)

    if chunks:
        headings = _extract_section_headings(text)
        if headings:
            enriched: List[str] = []
            active_heading: Optional[str] = None
            h_idx = 0
            running_offset = 0
            for chunk in chunks:
                chunk_pos = text.find(chunk[:40], running_offset)
                if chunk_pos < 0:
                    chunk_pos = running_offset
                while h_idx < len(headings) and headings[h_idx][0] <= chunk_pos:
                    active_heading = headings[h_idx][1]
                    h_idx += 1
                if active_heading and not chunk.startswith(active_heading):
                    enriched.append(f"{active_heading}\n{chunk}")
                else:
                    enriched.append(chunk)
                running_offset = chunk_pos + len(chunk)
            chunks = enriched

    return table_chunks + chunks


# ─── Phase 1: TxtIngestor ─────────────────────────────────────────────────────

class TxtIngestor(BaseIngestor):
    """Extracts raw text from TXT/MD/CSV/LOG files → List[RawExtract].

    Phase 1 responsibility: file I/O + normalization + section splitting.
    Does NOT chunk. The chunker (Phase 2) handles splitting.
    """

    async def extract(
        self,
        path: Path,
        metadata: UniversalMetadata,
    ) -> List[RawExtract]:
        file_size = path.stat().st_size

        # Binary check
        is_bin = await asyncio.get_event_loop().run_in_executor(None, _is_binary, path)
        if is_bin:
            raise ValueError(f"BINARY_FILE_DISGUISED_AS_TEXT: {path.name}")

        # Load
        stream_threshold = 5 * 1024 * 1024  # 5 MB
        if file_size > stream_threshold:
            raw_text = await asyncio.get_event_loop().run_in_executor(
                None, _load_text_streaming, path, settings.MAX_FILE_SIZE_TEXT
            )
        else:
            raw_text = await asyncio.get_event_loop().run_in_executor(None, _load_text, path)

        raw_text = _strip_bom(raw_text)
        raw_text, null_count = _strip_null_bytes(raw_text)
        if null_count:
            logger.warning("null_bytes_stripped", count=null_count, file=path.name)

        # Repair
        raw_text, _ = await asyncio.get_event_loop().run_in_executor(None, repair_text, raw_text)

        # Normalize
        text = _normalize_text(raw_text)
        text = re.sub(r"\n={4,}\n", "\n", text)
        text = re.sub(r"\n-{4,}\n", "\n", text)

        if not text or text.isspace():
            raise ValueError("EMPTY_CONTENT_AFTER_NORMALIZE")
        if len(text) < 50:
            raise ValueError("TEXT_TOO_SHORT")

        # Injection guard
        text = self._sanitize(text, surface="txt_ingest")
        if not text or text.isspace():
            raise ValueError("INJECTION_ONLY_CONTENT")

        # PII scrub
        text = self._scrub_pii(text, surface="txt_ingest")

        # Language
        language = await asyncio.get_event_loop().run_in_executor(None, _detect_language, text)

        # Section splitting — [DOC-NNN] or blank-line for large files
        sections = await asyncio.get_event_loop().run_in_executor(None, _split_sections, text)

        # For large files without [DOC-NNN]: additionally split on blank lines
        if len(sections) == 1 and file_size > stream_threshold:
            body = sections[0][2]
            parts = [p.strip() for p in body.split("\n\n") if p.strip()]
            if len(parts) > 1:
                sections = [(None, None, p) for p in parts]

        is_transcript = await asyncio.get_event_loop().run_in_executor(
            None, _detect_transcript_format, text
        )

        extracts: List[RawExtract] = []
        for i, (section_id, section_title, body) in enumerate(sections):
            if is_placeholder(body):
                continue
            cleaned_body, footnotes = strip_footnotes(body)
            cleaned_body, error_markers = detect_error_markers(cleaned_body)
            if not cleaned_body.strip():
                continue

            extract_type = "speaker_turn" if (
                is_transcript and _detect_transcript_format(cleaned_body)
            ) else "prose"

            source_ref = f"file:{path.name}"
            if section_id:
                source_ref += f"|section:{section_id}"
            elif i > 0:
                source_ref += f"|part:{i}"

            extra: Dict[str, Any] = {}
            if section_id:
                extra["section_id"] = section_id
            if section_title:
                extra["section_title"] = section_title
            if footnotes:
                extra["footnotes"] = footnotes
            if error_markers:
                extra["error_markers"] = error_markers
            if is_transcript:
                extra["is_transcript"] = True
            if language:
                extra["language"] = language

            extracts.append(RawExtract(
                text=cleaned_body,
                extract_type=extract_type,
                raw_source_ref=source_ref,
                extra=extra,
            ))

        if not extracts:
            raise ValueError("NO_EXTRACTS_PRODUCED")

        return extracts


# ─── Backward-compat ingest() — full pipeline (extraction + chunking) ─────────

async def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:
    """Backward-compatible entry point.  Router imports this until Phase 8."""

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("EMPTY_FILE")
    if file_size > settings.MAX_FILE_SIZE_TEXT:
        raise ValueError(f"FILE_TOO_LARGE: {file_size} bytes")

    with tracer.start_as_current_span("txt_ingest") as span:
        span.set_attribute("file.name", path.name)
        span.set_attribute("file.size", file_size)
        span.set_attribute("session.id", session_id)
        start = time.time()

        async with _semaphore:
            try:
                logger.info("txt_ingest_start", file=path.name, size=file_size, session_id=session_id)

                # Binary check
                is_bin = await asyncio.get_event_loop().run_in_executor(None, _is_binary, path)
                if is_bin:
                    raise ValueError(f"BINARY_FILE_DISGUISED_AS_TEXT: {path.name}")

                # Load
                stream_threshold = 5 * 1024 * 1024
                if file_size > stream_threshold:
                    raw_text = await asyncio.get_event_loop().run_in_executor(
                        None, _load_text_streaming, path, settings.MAX_FILE_SIZE_TEXT
                    )
                else:
                    raw_text = await asyncio.get_event_loop().run_in_executor(None, _load_text, path)

                raw_text = _strip_bom(raw_text)
                raw_text, null_count = _strip_null_bytes(raw_text)
                if null_count:
                    logger.warning("null_bytes_stripped", count=null_count, file=path.name)

                raw_text, repair_stats = await asyncio.get_event_loop().run_in_executor(
                    None, repair_text, raw_text
                )
                if repair_stats:
                    logger.info("text_repair_file_level", file=path.name, **repair_stats)

                text = _normalize_text(raw_text)
                text = re.sub(r"\n={4,}\n", "\n", text)
                text = re.sub(r"\n-{4,}\n", "\n", text)

                if not text or text.isspace():
                    raise ValueError("EMPTY_CONTENT_AFTER_NORMALIZE")
                if len(text) < 50:
                    raise ValueError("TEXT_TOO_SHORT")

                # Injection guard
                try:
                    from app.guardrails.input_guard import sanitize as _guard_sanitize
                    clean = _guard_sanitize(text, surface="txt_ingest")
                    if clean != text:
                        logger.warning("txt_injection_sanitized", file=path.name,
                                       original_len=len(text), sanitized_len=len(clean))
                        text = clean
                except Exception as _ge:
                    logger.warning("txt_guardrail_failed", file=path.name, error=str(_ge))

                if not text or text.isspace():
                    raise ValueError("INJECTION_ONLY_CONTENT")

                # PII scrub
                try:
                    from app.guardrails.pii import scrub_pii
                    _scrubbed, _changed = scrub_pii(text)
                    if _changed:
                        logger.warning("txt_pii_scrubbed", file=path.name,
                                       original_len=len(text), scrubbed_len=len(_scrubbed))
                        text = _scrubbed
                except Exception as _pe:
                    logger.warning("txt_pii_scrub_failed", file=path.name, error=str(_pe))

                # Metadata
                file_hash = _hash(text[:10000])
                doc_id = str(uuid.uuid4())
                source_name = path.name
                source_path_str = str(path.resolve())
                line_count = text.count("\n") + 1
                word_count = len(text.split())

                language = await asyncio.get_event_loop().run_in_executor(None, _detect_language, text)
                is_rtl = _is_rtl(language)

                text, pii_counts = await asyncio.get_event_loop().run_in_executor(
                    None, _redact_pii, text
                )

                sections = await asyncio.get_event_loop().run_in_executor(None, _split_sections, text)
                is_transcript = await asyncio.get_event_loop().run_in_executor(
                    None, _detect_transcript_format, text
                )
                if is_transcript:
                    logger.info("transcript_format_detected", file=path.name, session_id=session_id)

                # Chunk per section
                chunk_tuples: List[Tuple[Optional[str], Optional[str], str, Dict[str, Any]]] = []
                dropped_sections = 0
                for section_id_val, section_title_val, body in sections:
                    if is_placeholder(body):
                        dropped_sections += 1
                        continue
                    cleaned_body, footnotes = strip_footnotes(body)
                    cleaned_body, error_markers = detect_error_markers(cleaned_body)
                    title_marked_mismatch = any("wrong label" in m.lower() for m in error_markers)
                    section_extras: Dict[str, Any] = {}
                    if footnotes:
                        section_extras["footnotes"] = footnotes
                    if error_markers:
                        section_extras["error_markers"] = error_markers
                    if title_marked_mismatch:
                        section_extras["title_mismatch"] = True
                    version_info = extract_version(section_id_val, section_title_val)
                    if version_info:
                        section_extras["doc_version"] = version_info["version"]
                        section_extras["doc_version_kind"] = version_info["kind"]
                    section_chunks = await asyncio.get_event_loop().run_in_executor(
                        None, _chunk_text, cleaned_body, is_transcript
                    )
                    for ch in section_chunks:
                        if ch and ch.strip():
                            chunk_tuples.append((section_id_val, section_title_val, ch, section_extras))

                if dropped_sections:
                    logger.info("dropped_empty_sections_total", count=dropped_sections, file=path.name)
                if not chunk_tuples:
                    raise ValueError("NO_CHUNKS_PRODUCED")
                if len(chunk_tuples) > settings.MAX_CHUNKS:
                    logger.warning("chunk_limit_applied", original=len(chunk_tuples),
                                   limited=settings.MAX_CHUNKS, file=path.name)
                    chunk_tuples = chunk_tuples[:settings.MAX_CHUNKS]

                total_chunks = len(chunk_tuples)
                documents: List[IngestedDocument] = []
                seen_hashes: set = set()
                seen_simhashes: List[int] = []

                for i, (section_id_val, section_title_val, chunk, section_extras) in enumerate(chunk_tuples):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    chunk_repairs: Dict[str, Any] = {}
                    repaired, ws_fixed = recover_whitespace(chunk)
                    if ws_fixed:
                        chunk = repaired
                        chunk_repairs["whitespace_recovered"] = True
                    repaired, ocr_fixed = normalize_ocr_noise(chunk)
                    if ocr_fixed:
                        chunk = repaired
                        chunk_repairs["ocr_normalized"] = True
                    if len(chunk) < settings.CHUNK_MIN_SIZE:
                        continue
                    exact_h = _hash(chunk)
                    if exact_h in seen_hashes:
                        continue
                    seen_hashes.add(exact_h)
                    sh = _simhash(chunk)
                    if any(_simhash_distance(sh, prev) <= 0 for prev in seen_simhashes):
                        continue
                    seen_simhashes.append(sh)

                    quality = _quality_score(chunk)
                    subtype = _detect_subtype(chunk)
                    keywords = _extract_keywords(chunk)
                    fk_score = _readability_score(chunk)
                    heading_level: Optional[int] = None
                    first_line = chunk.split("\n")[0]
                    heading_level = _extract_heading_level(first_line)
                    title_mismatch = has_title_mismatch(section_title_val, keywords)

                    chunk_extra_metadata: Dict[str, Any] = {
                        "modality_weight": 1.0,
                        "importance_score": quality,
                        "data_quality_score": quality,
                    }
                    if section_extras:
                        chunk_extra_metadata.update(section_extras)
                    if chunk_repairs:
                        chunk_extra_metadata.update(chunk_repairs)
                    if title_mismatch:
                        chunk_extra_metadata["title_mismatch"] = True

                    _sec_meta = _detect_section_metadata(chunk)
                    structure_carry_overs: Dict[str, Any] = {}
                    if section_extras.get("error_markers"):
                        structure_carry_overs["error_markers"] = section_extras["error_markers"]
                    if section_extras.get("doc_version"):
                        structure_carry_overs["doc_version"] = section_extras["doc_version"]
                        structure_carry_overs["doc_version_kind"] = section_extras.get("doc_version_kind")
                    if section_extras.get("footnotes"):
                        structure_carry_overs["footnotes"] = section_extras["footnotes"]
                    if chunk_extra_metadata.get("title_mismatch"):
                        structure_carry_overs["title_mismatch"] = True

                    _speaker = _extract_speaker(chunk) if is_transcript else None
                    _chunk_type = "speaker_turn" if _speaker else (
                        "heading" if subtype == "heading" else "paragraph"
                    )
                    _call_section = _detect_call_section(chunk) if is_transcript else None

                    structure_payload: Dict[str, Any] = {
                        "doc_id": doc_id,
                        "session_id": session_id,
                        "file_hash": file_hash,
                        "source_path": source_path_str,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        "chunk_length": len(chunk),
                        "page_number": None,
                        "total_pages": None,
                        "section_id": section_id_val,
                        "section_title": section_title_val,
                        "ingestion_timestamp": time.time(),
                        "language": "en",
                        "file_size_bytes": file_size,
                        "is_rtl": is_rtl,
                        "heading_level": heading_level,
                        "readability_score": fk_score,
                        "tags": keywords,
                        "pii_redacted": bool(pii_counts),
                        "content_type": "text_chunk",
                        "ingestion_time": time.time(),
                        "section_number": _sec_meta.get("section_number"),
                        "is_forward_looking": _sec_meta.get("is_forward_looking", False),
                        "is_transcript": is_transcript,
                        "speaker": _speaker,
                        "chunk_type": _chunk_type,
                        "call_section": _call_section,
                    }
                    structure_payload.update(structure_carry_overs)

                    doc = IngestedDocument(
                        text=chunk,
                        modality="text",
                        subtype=subtype,
                        source_type="file",
                        source=source_name,
                        chunk_id=i,
                        structure=structure_payload,
                        extra_metadata=chunk_extra_metadata,
                    ).finalize()
                    documents.append(doc)

                if not documents:
                    raise ValueError("NO_VALID_DOCUMENTS_AFTER_FILTERING")

                latency = round(time.time() - start, 2)
                _ingest_duration.labels(status="success").observe(latency)
                span.set_attribute("docs.count", len(documents))
                span.set_attribute("language", language or "unknown")
                span.set_status(Status(StatusCode.OK))
                logger.info("txt_ingest_success", file=path.name, docs=len(documents),
                            total_chunks=total_chunks, language=language, is_rtl=is_rtl,
                            word_count=word_count, line_count=line_count, latency=latency,
                            session_id=session_id)
                return documents

            except Exception as exc:
                latency = round(time.time() - start, 2)
                error_type = type(exc).__name__
                _ingest_duration.labels(status="error").observe(latency)
                _ingest_errors.labels(error_type=error_type).inc()
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                logger.error("txt_ingest_failed", file=path.name, session_id=session_id,
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
