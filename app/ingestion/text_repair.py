from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


# PASS 1 — MOJIBAKE REPAIR
#
# Reverses double-encoded UTF-8 like "Ã¸" -> "ø", "â" -> "—".
# Operates on the whole file right after _load_text so every downstream
# pass sees clean characters. ftfy is the de-facto industry-standard lib
# for this task.

def repair_mojibake(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0
    try:
        import ftfy
        fixed = ftfy.fix_text(text)
        diff  = sum(1 for a, b in zip(text, fixed) if a != b) + abs(len(text) - len(fixed))
        return fixed, diff
    except ImportError:
        logger.debug("ftfy_not_installed_skipping_mojibake_repair")
        return text, 0
    except Exception as exc:
        logger.warning("mojibake_repair_failed", error=str(exc))
        return text, 0


# PASS 2 — NOISE-LINE STRIP
#
# Drops obvious garbage that ends up inside text dumps: log lines, debug
# metadata, HTML/script tags, binary control chars, lines that are mostly
# non-alphanumeric symbols. Conservative: leaves paragraph text alone.

_LOG_LINE_RE   = re.compile(r"^\s*---\s*LOG\s+ENTRY", re.IGNORECASE)
_DEBUG_LINE_RE = re.compile(r"^\s*===\s*DEBUG[:=]", re.IGNORECASE)
_ERROR_LINE_RE = re.compile(r"^\s*ERROR\s*:|^\s*\[ERROR\]|^\s*NULL POINTER", re.IGNORECASE)
_HTML_TAG_RE   = re.compile(r"<[^>]+>")
_BINARY_GARBAGE_RE = re.compile(r"[\x00-\x08\x0E-\x1F]")
_HEX_ESCAPE_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}){3,}")
_SYMBOL_LINE_RE = re.compile(r"^[^A-Za-z0-9\s]{20,}$")


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _LOG_LINE_RE.match(stripped):
        return True
    if _DEBUG_LINE_RE.match(stripped):
        return True
    if _ERROR_LINE_RE.match(stripped):
        return True
    if _SYMBOL_LINE_RE.match(stripped):
        return True
    if _HEX_ESCAPE_RE.search(stripped):
        return True
    # Mostly non-alphanumeric and short — likely separator junk
    if len(stripped) >= 10:
        alnum = sum(1 for c in stripped if c.isalnum() or c.isspace())
        if alnum / len(stripped) < 0.4:
            return True
    return False


def strip_noise_lines(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0

    # Remove HTML/script tags first (preserves inner text)
    try:
        import bleach
        text = bleach.clean(text, tags=[], attributes={}, strip=True)
    except ImportError:
        text = _HTML_TAG_RE.sub(" ", text)
    except Exception:
        text = _HTML_TAG_RE.sub(" ", text)

    # Strip binary control chars
    text = _BINARY_GARBAGE_RE.sub("", text)

    # Drop noise lines
    dropped = 0
    out_lines: List[str] = []
    for line in text.split("\n"):
        if _is_noise_line(line):
            dropped += 1
            continue
        out_lines.append(line)
    return "\n".join(out_lines), dropped


# PASS 3 — WHITESPACE RECOVERY
#
# For chunks where spaces have been stripped (e.g. concatenated walls of
# text from broken PDFs), use wordsegment to re-insert word boundaries.
# Only fires when whitespace_ratio is very low AND the chunk is long
# enough to be meaningful — never on short clean strings.

_MIN_RECOVERY_LEN = 200


def _whitespace_ratio(text: str) -> float:
    if not text:
        return 0.0
    spaces = sum(1 for c in text if c.isspace())
    return spaces / len(text)


def recover_whitespace(text: str) -> Tuple[str, bool]:
    if not getattr(settings, "TEXT_REPAIR_WHITESPACE", True):
        return text, False
    if not text or len(text) < _MIN_RECOVERY_LEN:
        return text, False
    if _whitespace_ratio(text) >= 0.05:
        return text, False
    try:
        from wordsegment import load as _ws_load, segment
        _ws_load()
        # Segment in line-sized chunks to avoid pathological memory use
        rebuilt: List[str] = []
        for line in text.split("\n"):
            if len(line) >= _MIN_RECOVERY_LEN and _whitespace_ratio(line) < 0.05:
                # wordsegment lowercases — keep original casing only for the first letter
                words = segment(line)
                if words:
                    rebuilt.append(" ".join(words))
                    continue
            rebuilt.append(line)
        return "\n".join(rebuilt), True
    except ImportError:
        logger.debug("wordsegment_not_installed_skipping_whitespace_recovery")
        return text, False
    except Exception as exc:
        logger.warning("whitespace_recovery_failed", error=str(exc))
        return text, False


# PASS 4 — OCR NOISE NORMALIZATION
#
# Conservative leet-style substitution applied ONLY when the chunk fails
# a "looks-like-English" gate — otherwise we'd corrupt clean text that
# contains intentional digits/symbols.

_OCR_SUBS = {
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "@": "a",
    "$": "s",
}


def _english_word_ratio(text: str) -> float:
    try:
        from wordfreq import top_n_list
    except ImportError:
        return 1.0  # No way to check — assume clean, skip OCR pass
    except Exception:
        return 1.0
    sample = text.lower()
    tokens = [t for t in re.findall(r"[a-z]{3,}", sample)]
    if not tokens:
        return 0.0
    try:
        common = set(top_n_list("en", 30000))
    except Exception:
        return 1.0
    hits = sum(1 for t in tokens if t in common)
    return hits / max(1, len(tokens))


def normalize_ocr_noise(text: str) -> Tuple[str, bool]:
    if not getattr(settings, "TEXT_REPAIR_OCR", True):
        return text, False
    if not text or len(text) < 60:
        return text, False
    ratio = _english_word_ratio(text)
    if ratio >= 0.30:
        return text, False
    # Substitute char-by-char but only inside tokens that mix letters and digits
    # (avoids damaging pure numeric data like "$0.05/kWh")
    def _fix_token(tok: str) -> str:
        if not tok:
            return tok
        has_letter = any(c.isalpha() for c in tok)
        has_digit  = any(c.isdigit() or c in "@$" for c in tok)
        if not (has_letter and has_digit):
            return tok
        return "".join(_OCR_SUBS.get(c, c) for c in tok)

    rebuilt = re.sub(r"\S+", lambda m: _fix_token(m.group(0)), text)
    # Re-check after substitution
    after = _english_word_ratio(rebuilt)
    if after <= ratio:
        return text, False
    return rebuilt, True


# PASS 5 — FOOTNOTE & EDITOR-NOTE STRIP
#
# Drops trailing footnote / editor-note lines but preserves them as
# metadata so we don't silently destroy provenance.

_SUPERSCRIPT_RE = re.compile(r"^\s*[¹²³⁴⁵⁶⁷⁸⁹⁰]\s+")
_NUMBERED_FOOTNOTE_RE = re.compile(r"^\s*\d{1,2}\s+[A-Z]")  # e.g. "1 See..."
_EDITOR_NOTE_RE = re.compile(r"^\s*\[EDITOR\s+NOTE", re.IGNORECASE)
_REFERENCE_MISSING_RE = re.compile(r"^\s*\[REFERENCE\s+MISSING", re.IGNORECASE)


def strip_footnotes(text: str) -> Tuple[str, List[str]]:
    if not getattr(settings, "TEXT_REPAIR_FOOTNOTES", True):
        return text, []
    if not text:
        return text, []
    lines = text.split("\n")
    notes: List[str] = []
    cleaned: List[str] = []
    for line in lines:
        if _SUPERSCRIPT_RE.match(line) or _EDITOR_NOTE_RE.match(line) or _REFERENCE_MISSING_RE.match(line):
            notes.append(line.strip())
            continue
        cleaned.append(line)
    return "\n".join(cleaned), notes


# PASS 6 — EMPTY / PLACEHOLDER DETECTION
#
# True when a section is effectively content-free. Used by the ingester
# to skip the section entirely with a `dropped_empty_section` log.

_PLACEHOLDER_PATTERNS = [
    re.compile(r"^\s*\.+\s*$"),
    re.compile(r"^\s*\[PAGE\s+INTENTIONALLY\s+LEFT\s+BLANK\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(?content\s+removed", re.IGNORECASE),
    re.compile(r"^\s*\[REST\s+OF\s+DOCUMENT\s+MISSING", re.IGNORECASE),
]


def is_placeholder(text: str) -> bool:
    if not getattr(settings, "TEXT_REPAIR_PLACEHOLDERS", True):
        # Still drop strictly empty content even when the placeholder
        # heuristic is disabled — the downstream validator would reject it.
        return not text or not text.strip()
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < settings.CHUNK_MIN_SIZE:
        return True
    for pat in _PLACEHOLDER_PATTERNS:
        if pat.match(stripped):
            return True
    return False


# PASS 7 — TITLE / CONTENT MISMATCH FLAG
#
# Compares section_title tokens to the chunk's top keywords. If overlap is
# zero we flag the chunk so downstream consumers can warn — but we never
# overwrite the title, because content-based embedding still retrieves
# correctly regardless of label.

def has_title_mismatch(section_title: Optional[str], keywords: List[str]) -> bool:
    if not getattr(settings, "TEXT_REPAIR_TITLE_MISMATCH", True):
        return False
    if not section_title or not keywords:
        return False
    title_tokens = set(re.findall(r"[a-z]{3,}", section_title.lower()))
    kw_tokens    = set()
    for kw in keywords[:5]:
        kw_tokens.update(re.findall(r"[a-z]{3,}", str(kw).lower()))
    if not title_tokens or not kw_tokens:
        return False
    return title_tokens.isdisjoint(kw_tokens)


# PASS 7B — ERROR MARKER DETECTION
#
# Catches in-corpus self-flagging that the strip_footnotes pass would miss:
#   - "NOTE: \"ada-003\" does not exist — this is an intentional error..."
#   - "→ WRONG LABEL (actual content: ...)"
#   - "[CONFIRMED]", "[UPDATED: ...]" version-tag annotations
#   - "this claim is disputed" / "disputed in some literature"
#
# Strips the marker text from the body (so it doesn't pollute the embedding)
# and returns the markers as a list for the chunk's extra_metadata. The
# prompt builder's general branch knows how to consume an ERROR_MARKERS
# header and tell the LLM to treat the chunk's claim as suspect.

_ERROR_MARKER_PATTERNS = [
    # "intentional error" / "this is an intentional error" / "for testing"
    re.compile(
        r"(?:NOTE\s*:?\s*)?[\"']?[^\"'\n]{0,40}[\"']?\s*"
        r"(?:does\s+not\s+exist|is\s+an?\s+intentional\s+error|"
        r"for\s+testing|fake|hallucinated)[^.\n]*\.?",
        re.IGNORECASE,
    ),
    # "→ WRONG LABEL (actual content: ...)" style mislabel flags
    re.compile(r"[→\->]+\s*WRONG\s+LABEL[^\n]*", re.IGNORECASE),
    # "disputed in some literature" / "this claim is disputed"
    re.compile(r"(?:this\s+claim\s+is\s+)?disputed(?:\s+in\s+some\s+literature)?", re.IGNORECASE),
    # version-tag annotations inside body text
    re.compile(r"\[CONFIRMED\]|\[UPDATED:[^\]]*\]|\[UNCHANGED\]", re.IGNORECASE),
]


def detect_error_markers(text: str) -> Tuple[str, List[str]]:
    """
    Scan `text` for in-corpus error/warning self-flags. Returns
    (cleaned_text, markers) where `markers` is a list of canonical short
    labels (e.g. ["intentional error", "does not exist", "WRONG LABEL"]).

    Stripping is conservative: only the matched span is removed so the
    surrounding sentence stays intact for the embedder.
    """
    if not text:
        return text, []

    found: List[str] = []
    cleaned = text

    for pat in _ERROR_MARKER_PATTERNS:
        for match in pat.finditer(cleaned):
            snippet = match.group(0).strip()
            if snippet:
                # Canonicalise: keep it short and lowercase for dedup
                label = re.sub(r"\s+", " ", snippet).strip(" .\"'")
                if len(label) > 80:
                    label = label[:77] + "..."
                if label and label.lower() not in {f.lower() for f in found}:
                    found.append(label)
        cleaned = pat.sub(" ", cleaned)

    if found:
        # Collapse the whitespace the substitutions left behind
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)

    return cleaned, found


# PASS 8 — VERSION TAGGING
#
# Extracts version info from section id/title. Returned as a small dict
# the ingester can splice into extra_metadata.

_VERSION_RE = re.compile(r"-v(\d+)|REVISED|FINAL|DRAFT", re.IGNORECASE)


def extract_version(section_id: Optional[str], section_title: Optional[str]) -> Optional[Dict[str, str]]:
    if not getattr(settings, "TEXT_REPAIR_VERSION_TAG", True):
        return None
    candidates = [section_id or "", section_title or ""]
    for c in candidates:
        m = _VERSION_RE.search(c)
        if m:
            if m.group(1):
                return {"version": f"v{m.group(1)}", "kind": "numeric"}
            label = m.group(0).upper()
            return {"version": label, "kind": "label"}
    return None


# TOP-LEVEL REPAIR ORCHESTRATOR
#
# Applied AT THE FILE LEVEL right after _load_text + BOM/null-byte strip
# but BEFORE _normalize_text and section splitting. Returns the repaired
# text plus a stats dict that the ingester can log.
#
# Each pass is individually toggleable through settings so a clean corpus
# pays near-zero cost (every check is short-circuited by content shape).


def repair_text(raw: str) -> Tuple[str, Dict[str, int]]:
    stats: Dict[str, int] = {}
    if not raw:
        return raw, stats

    if not getattr(settings, "TEXT_REPAIR_ENABLED", True):
        return raw, stats

    text = raw

    if getattr(settings, "TEXT_REPAIR_MOJIBAKE", True):
        text, mojibake_fixes = repair_mojibake(text)
        if mojibake_fixes:
            stats["mojibake_chars_changed"] = mojibake_fixes

    if getattr(settings, "TEXT_REPAIR_NOISE_LINES", True):
        text, noise_dropped = strip_noise_lines(text)
        if noise_dropped:
            stats["noise_lines_dropped"] = noise_dropped

    # Whitespace recovery and OCR repair are applied per-chunk inside the
    # ingester (after sectioning) because the heuristics are length- and
    # content-shape-dependent; calling them on a whole file is too coarse.
    return text, stats
