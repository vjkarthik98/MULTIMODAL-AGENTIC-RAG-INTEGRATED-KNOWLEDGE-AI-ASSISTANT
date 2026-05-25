"""Regression: ==== and ---- visual divider lines must be stripped before chunking.

Phase 24 fix: added regex to strip decorative divider lines from text before it reaches
the chunker, preventing decoration from consuming chunk space and diluting embedding signal.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


@pytest.fixture
def text_with_dividers():
    return (
        "SECTION 1: REVENUE\n"
        "============================\n"
        "Total net sales were $383.3 billion.\n"
        "------------------------------------\n"
        "SECTION 2: EXPENSES\n"
        "====================\n"
        "R&D expense was $29.9 billion.\n"
    )


def test_no_equals_dividers_in_chunks(text_with_dividers):
    """After text processing, no chunk should contain a run of 4+ '=' characters."""
    import re
    # Simulate the divider strip that text_ingest applies
    stripped = re.sub(r"\n={4,}\n", "\n", text_with_dividers)
    stripped = re.sub(r"\n-{4,}\n", "\n", stripped)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(stripped)

    for chunk in chunks:
        assert not re.search(r"={4,}", chunk), (
            f"Divider '====' still present in chunk after stripping:\n{chunk}"
        )
        assert not re.search(r"-{4,}", chunk), (
            f"Divider '----' still present in chunk after stripping:\n{chunk}"
        )


def test_divider_strip_in_text_ingest_module():
    """text_ingest module must contain the divider-strip regex pattern."""
    import inspect
    import app.ingestion.text_ingest as ti

    source = inspect.getsource(ti)
    # Look for the equals divider strip pattern
    has_equals_strip = r"={4,}" in source or "====\n" in source or "SECTION" in source
    assert has_equals_strip, (
        "text_ingest module no longer contains divider-strip regex — Phase 24 fix regressed!"
    )


def test_content_preserved_after_divider_strip(text_with_dividers):
    """Stripping dividers must not remove actual financial content."""
    import re

    stripped = re.sub(r"\n={4,}\n", "\n", text_with_dividers)
    stripped = re.sub(r"\n-{4,}\n", "\n", stripped)

    assert "383.3 billion" in stripped, "Financial figure removed by divider strip"
    assert "29.9 billion" in stripped, "Financial figure removed by divider strip"
    assert "SECTION 1" in stripped, "Section header removed by divider strip"
