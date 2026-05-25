"""Regression: temporal boost must demote forward-looking chunks and promote
historical sections when query contains FY20XX / revenue / reported anchors.

Phase 24 fix: _apply_temporal_boost in query_pipeline.py multiplies forward-looking
chunk scores x0.4 and historical chunks x1.3 for temporal queries.
Root cause fixed: Section 8 FY2025 guidance was outranking Section 3 FY2024 actuals,
causing LLM to report guidance figures as reported actuals.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


@pytest.fixture
def mixed_docs() -> List[Dict[str, Any]]:
    """Two docs: one historical (section 3), one forward-looking (section 8)."""
    return [
        {
            "text": "SECTION 8: OUTLOOK. The Company expects revenue of $400 billion in FY2025.",
            "score": 0.75,
            "metadata": {"section_number": 8, "is_forward_looking": True, "modality": "text"},
        },
        {
            "text": "SECTION 3: REVENUE. Total net sales were $383.3 billion in FY2024.",
            "score": 0.70,
            "metadata": {"section_number": 3, "is_forward_looking": False, "modality": "text"},
        },
    ]


def test_temporal_boost_function_exists():
    """_apply_temporal_boost must be importable from query_pipeline."""
    try:
        from app.pipeline.query_pipeline import _apply_temporal_boost, _is_temporal_query
        assert callable(_apply_temporal_boost)
        assert callable(_is_temporal_query)
    except ImportError:
        pytest.fail("_apply_temporal_boost not found in query_pipeline — Phase 24 fix regressed!")


def test_temporal_query_detected():
    """_is_temporal_query must return True for FY2024 / revenue / reported queries."""
    from app.pipeline.query_pipeline import _is_temporal_query

    assert _is_temporal_query("What was Apple's FY2024 revenue?")
    assert _is_temporal_query("What were the reported earnings for fiscal year 2023?")
    assert _is_temporal_query("What is the audited revenue for 2023?")


def test_non_temporal_query_not_detected():
    """_is_temporal_query must return False for non-temporal queries."""
    from app.pipeline.query_pipeline import _is_temporal_query

    assert not _is_temporal_query("What are Apple's main products?")
    assert not _is_temporal_query("Explain what P/E ratio means")


def test_forward_looking_chunk_demoted(mixed_docs):
    """Forward-looking chunks must have lower score after temporal boost than before."""
    from app.pipeline.query_pipeline import _apply_temporal_boost

    original_forward_score = mixed_docs[0]["score"]
    boosted = _apply_temporal_boost(mixed_docs, "What was Apple's FY2024 reported revenue?")

    forward_chunk = next(d for d in boosted if d["metadata"].get("is_forward_looking"))
    assert forward_chunk["score"] < original_forward_score, (
        f"Forward-looking chunk score {forward_chunk['score']} should be < {original_forward_score} after demotion"
    )


def test_historical_chunk_promoted(mixed_docs):
    """Historical chunks must have higher score after temporal boost than before."""
    from app.pipeline.query_pipeline import _apply_temporal_boost

    original_historical_score = mixed_docs[1]["score"]
    boosted = _apply_temporal_boost(mixed_docs, "What was Apple's FY2024 reported revenue?")

    historical_chunk = next(d for d in boosted if not d["metadata"].get("is_forward_looking"))
    assert historical_chunk["score"] >= original_historical_score, (
        f"Historical chunk score {historical_chunk['score']} should be >= {original_historical_score} after boost"
    )


def test_historical_outranks_forward_after_boost(mixed_docs):
    """After boost on a temporal query, Section 3 (historical) must rank above Section 8 (forward)."""
    from app.pipeline.query_pipeline import _apply_temporal_boost

    boosted = _apply_temporal_boost(mixed_docs, "What was Apple's FY2024 reported revenue?")
    boosted_sorted = sorted(boosted, key=lambda d: d["score"], reverse=True)

    top_doc = boosted_sorted[0]
    assert not top_doc["metadata"].get("is_forward_looking"), (
        f"Forward-looking chunk still ranked #1 after temporal boost. "
        f"Top doc section: {top_doc['metadata'].get('section_number')}"
    )


def test_temporal_boost_noop_on_non_temporal_query(mixed_docs):
    """Temporal boost must not change scores for non-temporal queries."""
    from app.pipeline.query_pipeline import _apply_temporal_boost

    original_scores = [d["score"] for d in mixed_docs]
    boosted = _apply_temporal_boost(mixed_docs, "What are Apple's main products?")
    after_scores = [d["score"] for d in boosted]

    assert original_scores == after_scores, (
        "Temporal boost changed scores for a non-temporal query — should be a no-op"
    )
