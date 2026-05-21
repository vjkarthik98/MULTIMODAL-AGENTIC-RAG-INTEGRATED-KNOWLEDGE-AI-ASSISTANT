"""
Gap Test 3 — Multi-Turn Session Memory (MT1-MT6)

Covers: 6-turn conversation over a single session_id using the heliosphere
document. Verifies that prior-turn facts are injected into later answers via
the memory pipeline (Redis history → memory_filter → memory_fusion →
memory_context in prompt).

Turn sequence:
  MT1  "What is Heliosphere Energy Systems?" — baseline fact retrieval
  MT2  "Who is their CEO?"                   — entity from doc
  MT3  "What is their revenue?"              — financial figure from doc
  MT4  "What did you tell me about revenue?" — recall from MT3 turn
  MT5  "Summarise what we have discussed so far." — multi-turn recall
  MT6  "What funding round did I ask about?"  — meta recall (asked nothing
        about funding; answer should say 'you have not asked about funding')

Pass criteria:
  MT1-MT3  standard RAG — answer, confidence > 0, session_id correct
  MT4      memory_injected: True  (history from MT3 available)
  MT5      memory_injected: True  AND answer mentions ≥2 of: CEO, revenue, Heliosphere
  MT6      answer indicates the topic was not raised (guard against hallucinating
           a funding round the user never asked about)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("app.pipeline.query_pipeline")


HELIO_DOC = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "heliosphere_energy_systems.txt"

SESSION_ID = "gap3_mem_test"


# ---------------------------------------------------------------------------
# Fixture: ingest heliosphere doc once, clear memory before the session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_gap3():
    """Ingest the doc and wipe any stale session history before the 6-turn run."""
    from app.pipeline.ingestion_pipeline import process_file

    assert HELIO_DOC.exists(), f"Heliosphere doc not found: {HELIO_DOC}"

    result = process_file(str(HELIO_DOC), session_id=SESSION_ID)
    assert result.get("status") == "success", (
        f"Heliosphere ingestion failed: {result}"
    )

    # Clear stale session memory so tests are deterministic
    try:
        from app.core.infra_registry import infra
        memory = infra.get_memory()
        if memory and hasattr(memory, "clear_history"):
            memory.clear_history(SESSION_ID)
    except Exception:
        pass

    time.sleep(1)
    yield


# ---------------------------------------------------------------------------
# Helper — runs one turn and returns full response dict
# ---------------------------------------------------------------------------

def _run(query: str) -> Dict[str, Any]:
    from app.pipeline.query_pipeline import query_pipeline
    return query_pipeline(query, session_id=SESSION_ID)


def _memory_injected(resp: Dict[str, Any]) -> bool:
    return bool(
        resp.get("metadata", {}).get("memory_injected")
        or resp.get("memory_injected")
    )


# ---------------------------------------------------------------------------
# The 6 ordered turns — must run in sequence (no parallelism)
# ---------------------------------------------------------------------------

class TestGap3MultiTurn:

    # Shared state across turns in the class
    _responses: List[Dict[str, Any]] = []

    def test_mt1_baseline_what_is_heliosphere(self):
        """MT1 — Baseline RAG retrieval, first turn, no memory yet."""
        resp = _run("What is Heliosphere Energy Systems?")
        TestGap3MultiTurn._responses.append(resp)

        assert resp.get("confidence", 0) > 0.0
        answer = resp.get("answer", "").lower()
        assert "heliosphere" in answer or "solar" in answer or "energy" in answer, (
            f"MT1: Expected Heliosphere context, got: {answer[:200]}"
        )
        assert resp["session_id"] == SESSION_ID

    def test_mt2_ceo_query(self):
        """MT2 — CEO query, heliosphere doc context."""
        resp = _run("Who is the CEO of Heliosphere Energy Systems?")
        TestGap3MultiTurn._responses.append(resp)

        assert resp.get("confidence", 0) > 0.0
        answer = resp.get("answer", "").lower()
        # CEO is Dr. Sophia Becker
        assert "becker" in answer or "sophia" in answer or "ceo" in answer, (
            f"MT2: Expected CEO Sophia Becker, got: {answer[:200]}"
        )

    def test_mt3_revenue_query(self):
        """MT3 — Revenue query, stores financial fact in session memory."""
        resp = _run("What is Heliosphere Energy Systems' annual revenue?")
        TestGap3MultiTurn._responses.append(resp)

        assert resp.get("confidence", 0) > 0.0
        answer = resp.get("answer", "").lower()
        has_revenue = "480" in answer or "revenue" in answer or "eur" in answer
        assert has_revenue, (
            f"MT3: Expected EUR 480M revenue figure, got: {answer[:200]}"
        )

    def test_mt4_memory_recall_revenue(self):
        """MT4 — Ask about prior turn; memory_injected must be True."""
        # Allow Redis to persist MT3 interaction
        time.sleep(0.5)

        resp = _run("What did you tell me about Heliosphere's revenue just now?")
        TestGap3MultiTurn._responses.append(resp)

        assert resp.get("confidence", 0) >= 0.0
        # memory_injected is in metadata
        injected = _memory_injected(resp)
        assert injected, (
            "MT4: memory_injected should be True after 3 prior turns. "
            f"Response metadata: {resp.get('metadata')}"
        )

    def test_mt5_multi_turn_summary(self):
        """MT5 — Multi-turn summary; memory_injected True; answer references multiple topics."""
        time.sleep(0.5)

        resp = _run(
            "Can you summarise everything we have discussed in this conversation?"
        )
        TestGap3MultiTurn._responses.append(resp)

        injected = _memory_injected(resp)
        assert injected, (
            "MT5: memory_injected should be True after 4 prior turns. "
            f"Response metadata: {resp.get('metadata')}"
        )

        answer = resp.get("answer", "").lower()
        # Should mention at least 2 of the topics raised in MT1-MT4
        topics_found = sum([
            "heliosphere" in answer,
            "ceo" in answer or "becker" in answer or "sophia" in answer,
            "revenue" in answer or "480" in answer,
        ])
        assert topics_found >= 2, (
            f"MT5: Summary should cover at least 2 discussed topics, found {topics_found}. "
            f"Answer: {answer[:300]}"
        )

    def test_mt6_no_hallucinated_funding_recall(self):
        """MT6 — User never asked about funding; answer must not fabricate a prior funding discussion."""
        time.sleep(0.5)

        resp = _run(
            "What funding round did I ask you about in this conversation?"
        )
        TestGap3MultiTurn._responses.append(resp)

        answer = resp.get("answer", "").lower()

        # The answer should indicate no funding question was raised, OR give
        # a hedged response. It must NOT confidently name a specific round.
        confabulated = (
            "series a" in answer or "series b" in answer or
            "series c" in answer or "series d" in answer or
            "series e" in answer
        ) and (
            "you asked" in answer or "you mentioned" in answer or
            "we discussed" in answer
        )
        assert not confabulated, (
            "MT6: Pipeline hallucinated a funding discussion that never happened. "
            f"Answer: {answer[:300]}"
        )

    def test_mt_all_responses_have_required_fields(self):
        """Structural check: all 6 turn responses must have required fields."""
        required = ["answer", "confidence", "decision", "session_id", "latency", "sources"]

        # Re-run a single query to capture a fresh response if _responses is empty
        # (in case tests ran out of order)
        if not TestGap3MultiTurn._responses:
            TestGap3MultiTurn._responses.append(_run("Tell me about Heliosphere."))

        for i, resp in enumerate(TestGap3MultiTurn._responses):
            for field in required:
                assert field in resp, (
                    f"Turn MT{i+1}: missing required field '{field}'"
                )
            assert isinstance(resp["answer"], str) and resp["answer"].strip(), (
                f"Turn MT{i+1}: answer is empty"
            )
            assert 0.0 <= resp["confidence"] <= 1.0, (
                f"Turn MT{i+1}: confidence out of range"
            )
            assert resp["session_id"] == SESSION_ID, (
                f"Turn MT{i+1}: session_id mismatch"
            )
