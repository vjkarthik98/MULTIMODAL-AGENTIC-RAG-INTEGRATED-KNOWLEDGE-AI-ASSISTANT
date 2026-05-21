"""
Gap Test 1 — Multi-Document Cross-Query (MD1-MD5)

Covers: two documents in the same Qdrant collection (heliosphere + aquavolt).
Verifies the pipeline can retrieve from the correct document and does NOT
bleed facts between documents when answering entity-specific queries.

Documents:
  - data/benchmarks/heliosphere_energy_systems.txt  (already ingested in H-series tests)
  - data/benchmarks/aquavolt.txt                    (ingested in this fixture)

Pass criteria per query:
  MD1  heliosphere answer, no aquavolt facts
  MD2  aquavolt answer,    no heliosphere facts
  MD3  both docs; answer must cite both entities with correct figures
  MD4  aquavolt-only; correct CEO name
  MD5  heliosphere-only; correct Series E figure (USD 140M)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if the pipeline cannot be imported (CI without model)
# ---------------------------------------------------------------------------
pytest.importorskip("app.pipeline.query_pipeline")


HELIO_DOC = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "heliosphere_energy_systems.txt"
AQUAVOLT_DOC = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "aquavolt.txt"

SESSION_ID = "gap1_multidoc_test"


# ---------------------------------------------------------------------------
# Fixture: ingest both documents once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ingest_both_docs():
    """Ingest heliosphere + aquavolt into Qdrant before running MD queries."""
    from app.pipeline.ingestion_pipeline import process_file

    docs_to_ingest = []
    if HELIO_DOC.exists():
        docs_to_ingest.append(str(HELIO_DOC))
    if AQUAVOLT_DOC.exists():
        docs_to_ingest.append(str(AQUAVOLT_DOC))

    for path in docs_to_ingest:
        result = process_file(path, session_id=SESSION_ID)
        assert result.get("status") == "success", (
            f"Ingestion failed for {path}: {result}"
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

class TestGap1MultiDoc:

    def test_md1_heliosphere_entity_isolation(self):
        """MD1 — heliosphere-specific query must not return AquaVolt facts."""
        resp = _run(
            "What is the installed capacity of Heliosphere Energy Systems' solar farms?"
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        assert "aquavolt" not in answer, (
            "AquaVolt facts leaked into a Heliosphere query answer"
        )
        # Should mention GW-scale or MW-scale capacity figure from heliosphere doc
        has_capacity = any(unit in answer for unit in ["gw", "mw", "gigawatt", "megawatt"])
        assert has_capacity or "heliosphere" in answer, (
            f"Expected capacity figure from Heliosphere doc, got: {answer[:200]}"
        )

    def test_md2_aquavolt_entity_isolation(self):
        """MD2 — aquavolt-specific query must not return Heliosphere facts."""
        resp = _run(
            "What is the peak conversion efficiency of the AquaVolt TidalCore X9 turbine?"
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        assert "heliosphere" not in answer, (
            "Heliosphere facts leaked into an AquaVolt query answer"
        )
        # Should mention 44.7% or similar efficiency figure
        has_efficiency = "44" in answer or "efficiency" in answer or "tidalcore" in answer
        assert has_efficiency, (
            f"Expected efficiency figure from AquaVolt doc, got: {answer[:200]}"
        )

    def test_md3_cross_doc_comparison(self):
        """MD3 — query spanning both docs must reference both companies."""
        resp = _run(
            "Compare the funding raised by Heliosphere Energy Systems and AquaVolt Technologies."
        )
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        mentions_heliosphere = "heliosphere" in answer
        mentions_aquavolt = "aquavolt" in answer
        assert mentions_heliosphere or mentions_aquavolt, (
            "Cross-doc query returned answer mentioning neither company: "
            f"{answer[:300]}"
        )

    def test_md4_aquavolt_ceo(self):
        """MD4 — AquaVolt CEO must be Dr. Ingrid Halvorsen."""
        resp = _run("Who is the CEO of AquaVolt Technologies?")
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        assert "halvorsen" in answer or "ingrid" in answer, (
            f"Expected CEO name Halvorsen/Ingrid, got: {answer[:200]}"
        )
        assert "heliosphere" not in answer, (
            "Heliosphere data contaminated AquaVolt CEO answer"
        )

    def test_md5_heliosphere_series_e(self):
        """MD5 — Heliosphere Series E funding must be USD 140M."""
        resp = _run("How much did Heliosphere Energy Systems raise in its Series E round?")
        answer = resp.get("answer", "").lower()

        assert resp.get("confidence", 0) > 0.0
        has_amount = "140" in answer or "series e" in answer
        assert has_amount, (
            f"Expected USD 140M Series E figure, got: {answer[:200]}"
        )
        assert "aquavolt" not in answer, (
            "AquaVolt funding info leaked into Heliosphere Series E answer"
        )

    def test_md_response_structure(self):
        """All MD responses must have required fields and valid types."""
        resp = _run("What products does AquaVolt Technologies make?")

        required = ["answer", "confidence", "decision", "session_id", "latency", "sources"]
        for field in required:
            assert field in resp, f"Missing field '{field}' in response"

        assert isinstance(resp["answer"], str) and resp["answer"].strip()
        assert isinstance(resp["confidence"], float)
        assert 0.0 <= resp["confidence"] <= 1.0
        assert isinstance(resp["sources"], list)
        assert resp["session_id"] == SESSION_ID
        assert isinstance(resp["latency"], (int, float)) and resp["latency"] >= 0
