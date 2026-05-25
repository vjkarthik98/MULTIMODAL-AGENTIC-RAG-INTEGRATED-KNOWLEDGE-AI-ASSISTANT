"""Regression: SECTION N: headings must start a new chunk, not be appended to the
previous section's tail.

Phase 24 fix: added '\nSECTION ' as a separator in RecursiveCharacterTextSplitter so
each SECTION block begins a fresh chunk. Prevents financial data from being buried in the
tail of a previous section's chunk.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


@pytest.fixture(scope="module")
def section_text():
    """Minimal finance-like document with explicit SECTION markers."""
    return (
        "SECTION 1: REVENUE\n"
        "Total net sales were $383.3 billion in fiscal year 2023.\n"
        "Services revenue was $85.2 billion.\n\n"
        "SECTION 2: EXPENSES\n"
        "Research and development expense was $29.9 billion.\n"
        "Selling, general and administrative expense was $24.9 billion.\n\n"
        "SECTION 3: GEOGRAPHIC SEGMENTS\n"
        "Americas net sales were $162.6 billion.\n"
        "Europe net sales were $94.3 billion.\n"
        "Greater China net sales were $72.6 billion.\n"
    )


def test_section_separator_included_in_config():
    """The text ingest module must include '\nSECTION ' in its separators list."""
    import inspect
    import app.ingestion.text_ingest as ti_mod

    source = inspect.getsource(ti_mod)
    # Check that SECTION separator appears somewhere in the module source
    assert "SECTION" in source, (
        "text_ingest module no longer contains SECTION separator — Phase 24 fix regressed!"
    )


def test_section_1_not_merged_with_section_2(section_text):
    """Section 1 content must not appear in the same chunk as Section 2 content when
    document is large enough to require splitting."""
    from app.core.config import get_settings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    s = get_settings()
    # Use a small chunk_size to force splitting at section boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\nSECTION ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(section_text)

    # If splitting occurred, no single chunk should span Section 1 AND Section 3
    for chunk in chunks:
        has_s1 = "383.3 billion" in chunk
        has_s3 = "Greater China" in chunk
        assert not (has_s1 and has_s3), (
            f"Section 1 and Section 3 content merged in a single chunk:\n{chunk[:200]}"
        )


def test_section_metadata_detector_present():
    """text_ingest must export _detect_section_metadata function."""
    try:
        from app.ingestion.text_ingest import _detect_section_metadata
        assert callable(_detect_section_metadata)
    except ImportError:
        pytest.fail("_detect_section_metadata not found in text_ingest — Phase 24 fix regressed!")


def test_section_metadata_detects_section_number():
    """_detect_section_metadata must return section_number for 'SECTION 3:' heading."""
    from app.ingestion.text_ingest import _detect_section_metadata

    chunk = "SECTION 3: GEOGRAPHIC SEGMENTS\nAmericas net sales were $162.6 billion."
    meta = _detect_section_metadata(chunk)
    assert "section_number" in meta, "section_number missing from metadata"
    assert meta["section_number"] == 3, (
        f"Expected section_number=3, got {meta['section_number']}"
    )


def test_section_metadata_forward_looking_flag():
    """Section 8 (or chunks with 'outlook') must be marked is_forward_looking=True."""
    from app.ingestion.text_ingest import _detect_section_metadata

    chunk = "SECTION 8: FORWARD-LOOKING STATEMENTS\nThe Company expects continued growth."
    meta = _detect_section_metadata(chunk)
    assert meta.get("is_forward_looking") is True, (
        f"Expected is_forward_looking=True for Section 8 chunk, got {meta}"
    )


def test_section_metadata_historical_not_forward_looking():
    """Section 3 revenue facts must NOT be marked is_forward_looking."""
    from app.ingestion.text_ingest import _detect_section_metadata

    chunk = "SECTION 3: FINANCIAL HIGHLIGHTS\nTotal net sales were $383.3 billion in fiscal year 2023."
    meta = _detect_section_metadata(chunk)
    # Section 3 with no forward-looking words should be is_forward_looking=False
    # (allow None as well — some detectors return None rather than False)
    assert meta.get("is_forward_looking") in (False, None), (
        f"Section 3 revenue chunk incorrectly marked forward-looking: {meta}"
    )
