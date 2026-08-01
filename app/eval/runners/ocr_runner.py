"""OCR suite runner.

Calls app/ingestion/image_ingest.py:ingest() — the same code production runs.
Extracts OCR text from ingested image documents and scores CER/WER/exact-match
against gold_ocr_text in image_gold.jsonl (P1-8 garbled-text measurement).
"""

from __future__ import annotations

import time
from typing import Any

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.metrics.ocr_metrics import ocr_metrics_batch


def _collect_ocr_text(documents: list[Any]) -> str:
    """Collect all OCR text from ingested image documents."""
    parts = []
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        subtype = getattr(doc, "subtype", "") or meta.get("subtype", "")
        content_type = meta.get("content_type", "")
        # Prefer OCR-typed docs; fall back to any text that isn't pure caption
        if "ocr" in subtype or "ocr" in content_type:
            text = getattr(doc, "text", "") or ""
            if text:
                parts.append(text)
    if not parts:
        # Fall back to all text combined if no explicit OCR subtype
        for doc in documents:
            text = getattr(doc, "text", "") or ""
            subtype = getattr(doc, "subtype", "") or ""
            if "caption" not in subtype and text:
                parts.append(text)
    return " ".join(parts).strip()


def run_ocr_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the OCR benchmark against real image_ingest.ingest()."""
    t0 = time.time()
    result = SuiteResult(suite="ocr")

    try:
        import app.ingestion.image_ingest as image_ingest
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    gold_rows = load_gold("image", gold_dir=cfg.gold_dir, include_todos=False)
    if not gold_rows:
        # All rows are TODO — this is expected at v1 before corpus download
        result.add(
            MetricResult.empty(
                "ocr_cer", "no curated image gold rows; run download_eval_corpus.sh first"
            )
        )
        result.add(MetricResult.empty("ocr_wer", "no curated image gold rows"))
        result.add(MetricResult.empty("ocr_exact_match", "no curated image gold rows"))
        result.duration_sec = time.time() - t0
        return result

    hypotheses: list[str] = []
    gold_ocr_texts: list[str] = []

    raw_corpus_dir = cfg.raw_corpus_dir / "image"

    # Multiple gold rows commonly test the SAME image (different OCR
    # assertions against one chart/screenshot). Without this cache, every
    # row re-ran full ingestion — captioning, OCR, chunking — from scratch;
    # confirmed live: the same file ingested 14 times in one suite run, with
    # the first cold-start call alone costing 155s against a 10s SLO. Ingest
    # each distinct source_file once per suite run and reuse its OCR text.
    _ingest_cache: dict[str, str | None] = {}

    for row in gold_rows:
        gold_ocr = row.get("gold_ocr_text", "")
        if not gold_ocr or gold_ocr == "TODO_fill_after_ocr":
            continue

        source_file = row.get("source_file", "")
        image_path = raw_corpus_dir / source_file
        if not image_path.exists():
            result.breached[f"missing_file_{row['id']}"] = str(image_path)
            continue

        if source_file in _ingest_cache:
            ocr_text = _ingest_cache[source_file]
            if ocr_text is None:
                continue  # this file already failed to ingest earlier this run
        else:
            session_id = f"{cfg.session_prefix}_ocr_{row['id']}"
            try:
                from app.utils.paths import reset_current_user, set_current_user

                _token = set_current_user(cfg.user_id)
                try:
                    docs = image_ingest.ingest(
                        file_path=str(image_path),
                        session_id=session_id,
                    )
                finally:
                    reset_current_user(_token)
            except Exception as exc:
                result.breached[f"ingest_error_{row['id']}"] = str(exc)
                _ingest_cache[source_file] = None
                continue

            ocr_text = _collect_ocr_text(docs)
            _ingest_cache[source_file] = ocr_text

        hypotheses.append(ocr_text)
        gold_ocr_texts.append(gold_ocr)

    if not hypotheses:
        result.add(MetricResult.empty("ocr_cer", "no completed OCR hypothesis/reference pairs"))
        result.add(MetricResult.empty("ocr_wer", "no completed OCR pairs"))
        result.add(MetricResult.empty("ocr_exact_match", "no completed OCR pairs"))
        result.duration_sec = time.time() - t0
        return result

    metrics = ocr_metrics_batch(hypotheses, gold_ocr_texts)
    for m in metrics.values():
        result.add(m)

    result.duration_sec = time.time() - t0
    return result
