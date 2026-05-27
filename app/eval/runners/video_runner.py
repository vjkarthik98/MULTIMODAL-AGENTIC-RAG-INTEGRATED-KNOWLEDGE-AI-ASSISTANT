"""Video suite runner.

Calls app/ingestion/video_ingest.py:ingest() — the same code production runs.
Extracts frame captions and transcript from ingested video documents and scores:
  - frame_caption_recall: BLEU-1 match of generated captions vs gold (P1-9 surface)
  - caption_repetition_rate: BLIP repetition loop detection (P1-9 measurement)
  - audio_wer: WER of Whisper transcript vs gold (when gold transcript available)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.audio_metrics import audio_wer_batch
from app.eval.metrics.base import MetricResult, SuiteResult
from app.eval.metrics.video_metrics import caption_repetition_rate, frame_caption_recall


def _split_by_subtype(documents: List[Any]) -> Tuple[List[str], List[str]]:
    """Return (frame_captions, transcript_parts) from ingested video documents."""
    captions: List[str] = []
    transcript_parts: List[str] = []
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        subtype = getattr(doc, "subtype", "") or meta.get("subtype", "")
        content_type = meta.get("content_type", "")
        text = getattr(doc, "text", "") or ""
        if not text:
            continue
        if subtype == "frame" or "frame" in content_type:
            captions.append(text)
        elif subtype == "speech" or "speech" in content_type or "transcript" in content_type:
            transcript_parts.append(text)
    return captions, transcript_parts


def run_video_suite(cfg: EvalConfig) -> SuiteResult:
    """Run the video benchmark against real video_ingest.ingest()."""
    t0 = time.time()
    result = SuiteResult(suite="video")

    try:
        import app.ingestion.video_ingest as video_ingest
    except ImportError as e:
        result.breached["import_error"] = str(e)
        return result

    gold_rows = load_gold("video", gold_dir=cfg.gold_dir, include_todos=False)
    if not gold_rows:
        result.add(MetricResult.empty("frame_caption_recall", "no curated video gold rows; run download_eval_corpus.sh first"))
        result.add(MetricResult.empty("caption_repetition_rate", "no curated video gold rows"))
        result.duration_sec = time.time() - t0
        return result

    all_generated_captions: List[str] = []
    all_gold_captions: List[str] = []
    all_generated_transcripts: List[str] = []
    all_gold_transcripts: List[str] = []

    raw_corpus_dir = cfg.raw_corpus_dir / "video"

    for row in gold_rows:
        gold_frame_caps = row.get("gold_frame_captions", [])
        gold_transcript = row.get("gold_transcript_excerpt", "")
        has_gold_caps = bool(gold_frame_caps) and gold_frame_caps != ["TODO_fill_after_processing"]
        has_gold_transcript = bool(gold_transcript) and gold_transcript not in ("TODO_fill_after_processing", "TODO")

        if not has_gold_caps and not has_gold_transcript:
            continue

        source_file = row.get("source_file", "")
        video_path = raw_corpus_dir / source_file
        if not video_path.exists():
            result.breached[f"missing_file_{row['id']}"] = str(video_path)
            continue

        session_id = f"{cfg.session_prefix}_video_{row['id']}"
        try:
            from app.utils.paths import set_current_user, reset_current_user
            _token = set_current_user(cfg.user_id)
            try:
                docs = video_ingest.ingest(
                    file_path=str(video_path),
                    session_id=session_id,
                )
            finally:
                reset_current_user(_token)
        except Exception as exc:
            result.breached[f"ingest_error_{row['id']}"] = str(exc)
            continue

        captions, transcript_parts = _split_by_subtype(docs)

        if has_gold_caps and captions:
            # Pair generated captions with gold captions (by position, padded with empty)
            for i, gold_cap in enumerate(gold_frame_caps):
                gen_cap = captions[i] if i < len(captions) else ""
                all_generated_captions.append(gen_cap)
                all_gold_captions.append(gold_cap)

        if has_gold_transcript and transcript_parts:
            all_generated_transcripts.append(" ".join(transcript_parts))
            all_gold_transcripts.append(gold_transcript)

    # Frame caption recall (BLEU-1)
    if all_generated_captions:
        result.add(frame_caption_recall(all_generated_captions, all_gold_captions))
    else:
        result.add(MetricResult.empty("frame_caption_recall", "no frame caption pairs available"))

    # BLIP repetition detection (P1-9) — run on ALL captions regardless of gold
    all_captions_flat: List[str] = []
    for row in gold_rows:
        source_file = row.get("source_file", "")
        video_path = raw_corpus_dir / source_file
        if video_path.exists():
            session_id = f"{cfg.session_prefix}_video_rep_{row['id']}"
            try:
                from app.utils.paths import set_current_user, reset_current_user
                _tok = set_current_user(cfg.user_id)
                try:
                    docs = video_ingest.ingest(str(video_path), session_id=session_id)
                finally:
                    reset_current_user(_tok)
                caps, _ = _split_by_subtype(docs)
                all_captions_flat.extend(caps)
            except Exception:
                pass

    if all_captions_flat:
        result.add(caption_repetition_rate(all_captions_flat))
    else:
        result.add(MetricResult.empty("caption_repetition_rate", "no video captions to check"))

    # ASR WER (when gold transcript available)
    if all_generated_transcripts:
        wer_result = audio_wer_batch(all_generated_transcripts, all_gold_transcripts)
        # Rename to distinguish from standalone audio suite
        wer_result = type(wer_result)(
            name="video_transcript_wer",
            value=wer_result.value,
            n=wer_result.n,
            notes=wer_result.notes,
        )
        result.add(wer_result)

    result.duration_sec = time.time() - t0
    return result
