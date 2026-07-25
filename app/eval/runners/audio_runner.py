"""Audio suite runner.

Calls app/ingestion/audio_ingest.py:ingest() — the same code production runs.
Extracts Whisper transcript from ingested audio documents and scores WER
against gold_transcript_excerpt in audio_gold.jsonl.
"""

from __future__ import annotations

import time
from typing import Any

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.audio_metrics import audio_wer_batch
from app.eval.metrics.base import MetricResult, SuiteResult


def _collect_transcript(documents: list[Any]) -> str:
    """Collect all speech transcript text from ingested audio documents."""
    parts = []
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        subtype = getattr(doc, "subtype", "") or meta.get("subtype", "")
        content_type = meta.get("content_type", "")
        if subtype == "speech" or "speech" in content_type or "transcript" in content_type:
            text = getattr(doc, "text", "") or ""
            if text:
                parts.append(text)
    if not parts:
        for doc in documents:
            text = getattr(doc, "text", "") or ""
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def run_audio_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the audio benchmark against real audio_ingest.ingest()."""
    t0 = time.time()
    result = SuiteResult(suite="audio")

    try:
        import app.ingestion.audio_ingest as audio_ingest
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    gold_rows = load_gold("audio", gold_dir=cfg.gold_dir, include_todos=False)
    if not gold_rows:
        result.add(
            MetricResult.empty(
                "audio_wer", "no curated audio gold rows; run download_eval_corpus.sh first"
            )
        )
        result.duration_sec = time.time() - t0
        return result

    transcripts: list[str] = []
    gold_transcripts: list[str] = []

    raw_corpus_dir = cfg.raw_corpus_dir / "audio"

    for row in gold_rows:
        gold_transcript = row.get("gold_transcript_excerpt", "")
        if not gold_transcript or gold_transcript in ("TODO_fill_from_official_transcript", "TODO"):
            continue

        source_file = row.get("source_file", "")
        audio_path = raw_corpus_dir / source_file
        if not audio_path.exists():
            result.breached[f"missing_file_{row['id']}"] = str(audio_path)
            continue

        session_id = f"{cfg.session_prefix}_audio_{row['id']}"
        try:
            from app.utils.paths import reset_current_user, set_current_user

            _token = set_current_user(cfg.user_id)
            try:
                docs = audio_ingest.ingest(
                    file_path=str(audio_path),
                    session_id=session_id,
                )
            finally:
                reset_current_user(_token)
        except Exception as exc:
            result.breached[f"ingest_error_{row['id']}"] = str(exc)
            continue

        transcript = _collect_transcript(docs)
        transcripts.append(transcript)
        gold_transcripts.append(gold_transcript)

    if not transcripts:
        result.add(MetricResult.empty("audio_wer", "no completed transcript/reference pairs"))
        result.duration_sec = time.time() - t0
        return result

    result.add(audio_wer_batch(transcripts, gold_transcripts))

    result.duration_sec = time.time() - t0
    return result
