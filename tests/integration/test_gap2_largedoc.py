"""
Gap Test 2 — Large-Document Buried Needle (LD1-LD5)

Covers: a ~1,200-word multi-section report (meridian_grid_report.txt) where
the key facts are buried deep in section 2.3 (Karachi project).

The "needle" facts that must be retrievable:
  - Project: Karachi Grid East Rehabilitation
  - Capital invested: USD 127.4 million
  - Fault-clearance improvement: 86.2%

These are not in the document summary or section headers — they live midway
through section 2.3.  If chunking + retrieval works correctly at production
depth (top_k=8 for hybrid, top_k=5 for rag), these facts surface.

Pass criteria:
  LD1  return USD 127.4M for Karachi project capital
  LD2  return 86.2% fault-clearance improvement
  LD3  correctly name HTLS conductors rated at 220 kV for the Pakistan upgrade
  LD4  return FY2024 revenue USD 318M and EBITDA USD 97M (Section 3 facts)
  LD5  return the Vietnam BESS spec: 120 MW / 240 MWh LFP from CATL
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("app.pipeline.query_pipeline")

# See test_gap1_multidoc.py — query_pipeline() blocks on a real llama-server
# that may hang past pytest's own timeout without one running. Skip cleanly.
from conftest import requires_llama_server  # noqa: E402

pytestmark = requires_llama_server


MERIDIAN_DOC = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "meridian_grid_report.txt"

SESSION_ID = "gap2_largedoc_test"


# ---------------------------------------------------------------------------
# Fixture: ingest meridian doc once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ingest_meridian():
    from app.pipeline.ingestion_pipeline import process_file

    assert MERIDIAN_DOC.exists(), f"Test document not found: {MERIDIAN_DOC}"

    result = process_file(str(MERIDIAN_DOC), session_id=SESSION_ID)
    assert result.get("status") == "success", (
        f"Meridian doc ingestion failed: {result}"
    )
    time.sleep(1)
    yield


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(query: str) -> Dict[str, Any]:
    from app.pipeline.query_pipeline import query_pipeline
    return query_pipeline(query, session_id=SESSION_ID)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGap2LargeDoc:

    def test_ld1_karachi_capital_buried_needle(self):
        """LD1 — Buried needle: Karachi project capital must be USD 127.4 million."""
        resp = _run(
            "How much capital did Meridian Infrastructure Partners invest in the "
            "Karachi Grid East Rehabilitation project?"
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        has_amount = "127" in answer or "127.4" in answer
        assert has_amount, (
            f"Expected USD 127.4M for Karachi project, got: {answer[:300]}"
        )

    def test_ld2_fault_clearance_improvement(self):
        """LD2 — Buried needle: fault-clearance improvement must be 86.2%."""
        resp = _run(
            "By what percentage did the Karachi grid project improve fault-clearance performance?"
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        has_pct = "86" in answer or "86.2" in answer
        assert has_pct, (
            f"Expected 86.2% fault-clearance figure, got: {answer[:300]}"
        )

    def test_ld3_htls_conductor_spec(self):
        """LD3 — Pakistan upgrade conductor type: HTLS at 220 kV."""
        resp = _run(
            "What type of transmission line conductors were installed in the "
            "Pakistan Karachi grid rehabilitation project?"
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        has_htls = "htls" in answer or "high-temperature" in answer or "220" in answer
        assert has_htls, (
            f"Expected HTLS or 220 kV conductor info, got: {answer[:300]}"
        )

    def test_ld4_financials_surface_section(self):
        """LD4 — Section 3 financials: FY2024 revenue USD 318M, EBITDA USD 97M."""
        resp = _run(
            "What was Meridian Infrastructure Partners' revenue and EBITDA for FY2024?"
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        has_revenue = "318" in answer or "revenue" in answer
        has_ebitda = "97" in answer or "ebitda" in answer
        assert has_revenue or has_ebitda, (
            f"Expected FY2024 revenue/EBITDA figures, got: {answer[:300]}"
        )

    def test_ld5_vietnam_bess_spec(self):
        """LD5 — Vietnam BESS: 120 MW / 240 MWh LFP from CATL."""
        resp = _run(
            "What are the specifications of the battery energy storage system "
            "deployed in the Vietnam Mekong Delta project?"
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        has_spec = (
            "120" in answer or "240" in answer or
            "catl" in answer or "lfp" in answer or
            "lithium" in answer or "bess" in answer
        )
        assert has_spec, (
            f"Expected Vietnam BESS spec (120MW/240MWh/LFP/CATL), got: {answer[:300]}"
        )

    def test_ld_no_hallucination_on_missing_fact(self):
        """LD — pipeline must not confabulate a fact not present in the document."""
        resp = _run(
            "What is Meridian Infrastructure Partners' stock ticker symbol?"
        )
        answer = resp.get("answer", "").lower()

        # MIP is private — no ticker. Answer must not invent one.
        assert resp.get("confidence", 0) >= 0.0
        invented_tickers = ["mip", "mifp", "nasdaq", "nyse", "lse"]
        hallucinated = any(
            t in answer and ("ticker" in answer or "listed" in answer or "trading" in answer)
            for t in invented_tickers
        )
        assert not hallucinated, (
            f"Pipeline may have hallucinated a stock ticker for a private company: {answer[:200]}"
        )

    def test_ld_response_structure(self):
        """All LD responses must have required fields."""
        resp = _run("What countries does Meridian Infrastructure Partners operate in?")

        required = ["answer", "confidence", "decision", "session_id", "latency", "sources"]
        for field in required:
            assert field in resp, f"Missing field '{field}'"

        assert isinstance(resp["answer"], str) and resp["answer"].strip()
        assert 0.0 <= resp["confidence"] <= 1.0
        assert isinstance(resp["sources"], list)
        assert resp["session_id"] == SESSION_ID
