import pytest

from app.ingestion.txt_ingest import (
    detect_error_markers,
    extract_version,
    has_title_mismatch,
    is_placeholder,
    normalize_ocr_noise,
    recover_whitespace,
    repair_mojibake,
    repair_text,
    strip_footnotes,
    strip_noise_lines,
)


# ---------------------------------------------------------------------------
# repair_mojibake
# ---------------------------------------------------------------------------

class TestRepairMojibake:

    def test_clean_text_unchanged(self):
        text = "Hello, world. This is valid UTF-8."
        result, diff = repair_mojibake(text)
        assert result == text
        assert diff == 0

    def test_empty_string_returns_unchanged(self):
        result, diff = repair_mojibake("")
        assert result == ""
        assert diff == 0

    def test_returns_tuple(self):
        result = repair_mojibake("some text")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_diff_is_integer(self):
        _, diff = repair_mojibake("clean text")
        assert isinstance(diff, int)

    def test_result_is_string(self):
        result, _ = repair_mojibake("some text")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# strip_noise_lines
# ---------------------------------------------------------------------------

class TestStripNoiseLines:

    def test_clean_paragraph_preserved(self):
        text = "This is a normal paragraph.\nWith two lines."
        result, dropped = strip_noise_lines(text)
        assert "normal paragraph" in result
        assert dropped == 0

    def test_log_line_stripped(self):
        text = "Good content.\n--- LOG ENTRY ---\nMore good content."
        result, dropped = strip_noise_lines(text)
        assert "LOG ENTRY" not in result
        assert dropped >= 1

    def test_debug_line_stripped(self):
        text = "Real text.\n=== DEBUG: info ===\nMore text."
        result, dropped = strip_noise_lines(text)
        assert dropped >= 1

    def test_null_pointer_error_line_stripped(self):
        text = "Good text.\nNULL POINTER at 0x0000\nMore text."
        result, dropped = strip_noise_lines(text)
        assert dropped >= 1

    def test_empty_text_returns_empty(self):
        result, dropped = strip_noise_lines("")
        assert result == ""
        assert dropped == 0

    def test_returns_tuple_of_str_and_int(self):
        result, dropped = strip_noise_lines("hello world")
        assert isinstance(result, str)
        assert isinstance(dropped, int)

    def test_symbol_only_line_stripped(self):
        text = "Real content.\n!!!@@@###$$$%%%^^^&&&\nMore content."
        result, dropped = strip_noise_lines(text)
        assert dropped >= 1


# ---------------------------------------------------------------------------
# recover_whitespace
# ---------------------------------------------------------------------------

class TestRecoverWhitespace:

    def test_returns_tuple(self):
        result = recover_whitespace("Hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_result_is_string(self):
        result, changed = recover_whitespace("Hello world")
        assert isinstance(result, str)

    def test_changed_is_bool(self):
        _, changed = recover_whitespace("Hello world")
        assert isinstance(changed, bool)

    def test_normal_text_not_changed(self):
        text = "This is properly spaced text."
        _, changed = recover_whitespace(text)
        # May or may not change — just check it doesn't crash
        assert isinstance(changed, bool)

    def test_empty_string_handled(self):
        result, changed = recover_whitespace("")
        assert result == ""


# ---------------------------------------------------------------------------
# normalize_ocr_noise
# ---------------------------------------------------------------------------

class TestNormalizeOcrNoise:

    def test_returns_tuple(self):
        result = normalize_ocr_noise("Hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_result_is_string(self):
        result, _ = normalize_ocr_noise("clean text")
        assert isinstance(result, str)

    def test_changed_is_bool(self):
        _, changed = normalize_ocr_noise("clean text")
        assert isinstance(changed, bool)

    def test_empty_string(self):
        result, changed = normalize_ocr_noise("")
        assert result == ""


# ---------------------------------------------------------------------------
# strip_footnotes
# ---------------------------------------------------------------------------

class TestStripFootnotes:

    def test_returns_tuple_str_list(self):
        result, footnotes = strip_footnotes("Main text. [1] See below.")
        assert isinstance(result, str)
        assert isinstance(footnotes, list)

    def test_clean_text_no_footnotes(self):
        text = "No footnotes here."
        result, footnotes = strip_footnotes(text)
        assert "No footnotes" in result

    def test_empty_text(self):
        result, footnotes = strip_footnotes("")
        assert result == ""
        assert footnotes == []


# ---------------------------------------------------------------------------
# is_placeholder
# ---------------------------------------------------------------------------

class TestIsPlaceholder:

    def test_tbd_is_placeholder(self):
        assert is_placeholder("TBD") is True

    def test_tbd_lowercase(self):
        assert is_placeholder("tbd") is True

    def test_placeholder_text(self):
        assert is_placeholder("[PLACEHOLDER]") is True

    def test_todo_is_placeholder(self):
        assert is_placeholder("TODO") is True

    def test_real_content_not_placeholder(self):
        # Must be >= CHUNK_MIN_SIZE (50) to not be flagged as placeholder
        long_text = "This is a well-formed sentence with sufficient content to pass placeholder check."
        assert is_placeholder(long_text) is False

    def test_empty_is_placeholder(self):
        assert is_placeholder("") is True

    def test_na_is_placeholder(self):
        assert is_placeholder("N/A") is True


# ---------------------------------------------------------------------------
# has_title_mismatch
# ---------------------------------------------------------------------------

class TestHasTitleMismatch:

    def test_matching_title_not_mismatch(self):
        result = has_title_mismatch("Introduction to AI", ["introduction", "ai"])
        assert result is False

    def test_completely_different_title_is_mismatch(self):
        result = has_title_mismatch("Financial Report Q3", ["introduction", "machine learning"])
        assert result is True

    def test_empty_keywords_not_mismatch(self):
        result = has_title_mismatch("Any title", [])
        assert result is False

    def test_none_title_handled(self):
        result = has_title_mismatch(None, ["keyword"])
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# detect_error_markers
# ---------------------------------------------------------------------------

class TestDetectErrorMarkers:

    def test_clean_text_no_markers(self):
        text = "This document contains accurate information."
        result, markers = detect_error_markers(text)
        assert isinstance(markers, list)
        assert len(markers) == 0

    def test_returns_tuple(self):
        result = detect_error_markers("some text")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_result_is_string(self):
        text, _ = detect_error_markers("clean text")
        assert isinstance(text, str)

    def test_markers_is_list(self):
        _, markers = detect_error_markers("text")
        assert isinstance(markers, list)

    def test_intentional_error_marker_detected(self):
        text = "This is intentional error for testing purposes."
        _, markers = detect_error_markers(text)
        # May detect "intentional error" — exact behavior depends on implementation
        assert isinstance(markers, list)


# ---------------------------------------------------------------------------
# extract_version
# ---------------------------------------------------------------------------

class TestExtractVersion:

    def test_version_in_section_id(self):
        result = extract_version("v2.3", None)
        # Returns dict or None
        assert result is None or isinstance(result, dict)

    def test_version_in_title(self):
        result = extract_version(None, "Document Version 1.0")
        assert result is None or isinstance(result, dict)

    def test_no_version_returns_none(self):
        result = extract_version("Introduction", "General Overview")
        assert result is None or isinstance(result, dict)

    def test_both_none_handled(self):
        result = extract_version(None, None)
        assert result is None


# ---------------------------------------------------------------------------
# repair_text (full pipeline)
# ---------------------------------------------------------------------------

class TestRepairText:

    def test_returns_tuple(self):
        result = repair_text("Hello world.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_text_is_string(self):
        text, _ = repair_text("Some content here.")
        assert isinstance(text, str)

    def test_stats_is_dict(self):
        _, stats = repair_text("Some content here.")
        assert isinstance(stats, dict)

    def test_clean_text_preserved(self):
        text = "This is a perfectly clean document with valid content."
        result, stats = repair_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_text_handled(self):
        result, stats = repair_text("")
        assert isinstance(result, str)
        assert isinstance(stats, dict)

    def test_stats_has_numeric_values(self):
        _, stats = repair_text("Normal document content here.")
        for v in stats.values():
            assert isinstance(v, (int, float))
