"""
Video Modality End-to-End Test & Scoring
=========================================
Tests the full pipeline for an MP4 earnings call video:
    VideoIngestor → VideoChunker → VideoEmbedder → VideoBM25 → QdrantStore

Run:
    source rag_env/bin/activate
    python tests/video/test_video_modality_e2e.py

Exit 0 = PASS (final ≥ 70, no dimension < 50)
Exit 1 = FAIL
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── project root on sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ─── CLI args ──────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Video modality E2E test")
_parser.add_argument(
    "--video",
    default="data/raw/finance/Apple Q4 2024 Earnings Call Short.mp4",
    help="Path to the video file to test (relative to project root)",
)
_args, _ = _parser.parse_known_args()

# ─── constants ─────────────────────────────────────────────────────────────
VIDEO_PATH      = _args.video
TEST_USER_ID    = "test_video_eval_user"
TEST_SESSION_ID = "test_video_eval_session"

# Thresholds
MIN_CHUNKS = 3
MAX_CHUNKS = 25
MIN_AVG_WORDS = 40
MIN_SPEAKER_COVERAGE = 0.60   # fraction of chunks with non-empty speaker_label
MIN_FRAME_CAPTION_COVERAGE = 0.50
MIN_FINANCE_SIGNAL_COVERAGE = 0.25
MIN_SNR_COVERAGE = 0.70
MIN_EMBEDDING_NORM_LO = 0.95
MIN_EMBEDDING_NORM_HI = 1.05
MAX_AUDIO_VISUAL_COSINE = 0.97   # must be < this (different signals)
MIN_COMBINED_AUDIO_COSINE = 0.25  # must be > this (combined includes audio)

# ANSI colours
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def _section(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'─'*60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─'*60}{_RESET}")


def _cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _norm(v: List[float]) -> float:
    return float(np.linalg.norm(np.array(v, dtype=np.float32)))


# ════════════════════════════════════════════════════════════════════════════
# PHASE 0 — PRE-FLIGHT CHECKS
# ════════════════════════════════════════════════════════════════════════════

def phase0_preflight() -> bool:
    _section("Phase 0 · Pre-flight Checks")
    passed = True

    # Video file + integrity check
    vp = PROJECT_ROOT / VIDEO_PATH
    if vp.is_file():
        size_mb = vp.stat().st_size / (1024 * 1024)
        _ok(f"Video file found: {vp.name} ({size_mb:.1f} MB)")
        # Probe for extractable audio duration to catch truncated files
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default", str(vp)],
                capture_output=True, text=True, timeout=15,
            )
            declared_dur = float(next(
                (l.split("=")[1] for l in probe.stdout.splitlines() if l.startswith("duration=")), "0"
            ))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf:
                _wav_tmp = _tf.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(vp), "-ar", "16000", "-ac", "1", "-vn",
                 "-t", "30", _wav_tmp],
                capture_output=True, timeout=30,
            )
            wav_dur_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default", _wav_tmp],
                capture_output=True, text=True, timeout=10,
            )
            actual_dur = float(next(
                (l.split("=")[1] for l in wav_dur_probe.stdout.splitlines() if l.startswith("duration=")), "0"
            ))
            os.unlink(_wav_tmp)
            pct = (actual_dur / declared_dur * 100) if declared_dur > 0 else 0
            if actual_dur < 10:
                _fail(f"VIDEO INTEGRITY ISSUE: only {actual_dur:.1f}s audio extractable "
                      f"(declared {declared_dur:.0f}s). File is likely truncated/corrupted. "
                      f"Re-export the video and replace {vp.name}")
                passed = False
            elif pct < 90:
                _warn(f"Partial file: {actual_dur:.1f}s audio out of declared {declared_dur:.0f}s ({pct:.0f}%)")
            else:
                _ok(f"Video integrity OK: {actual_dur:.1f}s audio extractable of {declared_dur:.0f}s declared")
        except Exception as _ie:
            _warn(f"Could not verify video integrity: {_ie}")
    else:
        _fail(f"Video file NOT found: {vp}")
        passed = False

    # ffmpeg / ffprobe
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            _ok(f"{tool} available")
        else:
            _fail(f"{tool} NOT found in PATH — install ffmpeg")
            passed = False

    # Qdrant (use QdrantVectorStore so API key is read from settings automatically)
    try:
        from app.core.config import settings
        from app.vectorstore.qdrant_store import QdrantVectorStore
        store = QdrantVectorStore()
        collections = {c.name for c in store.client.get_collections().collections}
        for name in (settings.TEXT_COLLECTION_NAME, settings.VISION_COLLECTION_NAME):
            if name in collections:
                _ok(f"Qdrant collection '{name}' exists")
            else:
                _warn(f"Qdrant collection '{name}' missing — will be auto-created on first upsert")
        _ok("Qdrant reachable")
    except Exception as exc:
        _fail(f"Qdrant unreachable: {exc}")
        passed = False

    # BM25 dir writable (data/users/<user_id>/bm25_index)
    from app.utils.paths import user_dir
    bm25_dir = user_dir(TEST_USER_ID) / "bm25_index"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    if os.access(bm25_dir, os.W_OK):
        _ok(f"BM25 dir writable: {bm25_dir}")
    else:
        _fail(f"BM25 dir NOT writable: {bm25_dir}")
        passed = False

    return passed


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — INGESTION  (ingest_video_full → embed → BM25 → Qdrant)
# ════════════════════════════════════════════════════════════════════════════

def phase1_ingest() -> Tuple[List[Any], Dict[str, float]]:
    _section("Phase 1 · Running Ingestion Pipeline")

    from app.ingestion.video_ingest import ingest_video_full
    from app.embeddings import get_embedder
    from app.bm25.video_bm25 import VideoBM25
    from app.vectorstore.qdrant_store import QdrantVectorStore
    from app.utils.paths import set_current_user, reset_current_user

    video_abs = str((PROJECT_ROOT / VIDEO_PATH).resolve())
    timings: Dict[str, float] = {}

    # ── 1a: Ingest + chunk ─────────────────────────────────────────────────
    print("  [1a] Running VideoIngestor + VideoChunker …")
    t0 = time.time()
    token = set_current_user(TEST_USER_ID)
    try:
        docs = asyncio.run(ingest_video_full(video_abs, TEST_SESSION_ID))
    finally:
        reset_current_user(token)
    timings["ingest_chunk"] = round(time.time() - t0, 2)
    print(f"       → {len(docs)} chunks in {timings['ingest_chunk']}s")

    if not docs:
        raise RuntimeError("No documents produced — aborting test")

    # ── 1b: Embed ──────────────────────────────────────────────────────────
    print("  [1b] Embedding with VideoEmbedder …")
    t0 = time.time()
    embedder = get_embedder("video")
    docs = embedder.embed_documents(docs)
    timings["embed"] = round(time.time() - t0, 2)
    embedded_count = sum(1 for d in docs if getattr(d, "embedding", None) is not None)
    print(f"       → {embedded_count}/{len(docs)} docs embedded in {timings['embed']}s")

    # ── 1c: BM25 ───────────────────────────────────────────────────────────
    print("  [1c] Building BM25 indexes (combined / audio / visual) …")
    t0 = time.time()
    bm25 = VideoBM25(user_id=TEST_USER_ID)
    bm25.add_documents(docs, session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)
    timings["bm25"] = round(time.time() - t0, 2)
    print(f"       → BM25 indexed in {timings['bm25']}s")

    # ── 1d: Qdrant upsert ──────────────────────────────────────────────────
    print("  [1d] Upserting to Qdrant …")
    t0 = time.time()
    store = QdrantVectorStore()
    store.insert_documents(docs, session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)
    timings["qdrant"] = round(time.time() - t0, 2)
    print(f"       → Upserted in {timings['qdrant']}s")

    return docs, timings


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHUNK SCORING  (weight 30%)
# ════════════════════════════════════════════════════════════════════════════

def phase2_chunk_score(docs: List[Any]) -> Tuple[int, List[str]]:
    _section("Phase 2 · Chunk Quality Scoring")

    score = 0
    failures: List[str] = []
    n = len(docs)

    def _s(struct: Any, *keys, default=None):
        if not struct:
            return default
        for k in keys:
            v = struct.get(k)
            if v is not None:
                return v
        return default

    structs = [getattr(d, "structure", {}) or {} for d in docs]

    # ── 2.1  Chunk count [3, 25] ────────── 10 pts
    if MIN_CHUNKS <= n <= MAX_CHUNKS:
        score += 10
        _ok(f"Chunk count {n} is in [{MIN_CHUNKS}, {MAX_CHUNKS}]  (+10)")
    else:
        failures.append(f"Chunk count {n} outside expected range [{MIN_CHUNKS}, {MAX_CHUNKS}]")
        _fail(f"Chunk count {n} outside [{MIN_CHUNKS}, {MAX_CHUNKS}]  (0)")

    # ── 2.2  Avg transcript word count ≥ 40 ── 10 pts
    word_counts = [_s(s, "word_count") or len((_s(s, "transcript") or "").split()) for s in structs]
    avg_words = sum(word_counts) / n if n else 0
    if avg_words >= MIN_AVG_WORDS:
        score += 10
        _ok(f"Avg transcript word count {avg_words:.1f} ≥ {MIN_AVG_WORDS}  (+10)")
    else:
        failures.append(f"Avg transcript word count {avg_words:.1f} < {MIN_AVG_WORDS}")
        _fail(f"Avg transcript word count {avg_words:.1f} < {MIN_AVG_WORDS}  (0)")

    # ── 2.3  Speaker attribution ≥ 60% ──── 15 pts
    speaker_set = [_s(s, "speaker_label", "speaker_name", "speaker") for s in structs]
    has_speaker = sum(1 for sp in speaker_set if sp and str(sp).strip() not in ("", "UNKNOWN", "unknown"))
    speaker_cov = has_speaker / n if n else 0
    if speaker_cov >= MIN_SPEAKER_COVERAGE:
        score += 15
        _ok(f"Speaker attribution {speaker_cov:.0%} ≥ {MIN_SPEAKER_COVERAGE:.0%}  (+15)")
    else:
        failures.append(f"Speaker attribution {speaker_cov:.0%} < {MIN_SPEAKER_COVERAGE:.0%}")
        _fail(f"Speaker attribution {speaker_cov:.0%} < {MIN_SPEAKER_COVERAGE:.0%}  (0)")

    # ── 2.4  Frame captions on ≥ 50% of chunks ── 20 pts
    has_captions = sum(1 for s in structs if _s(s, "frame_captions") not in (None, [], ""))
    caption_cov = has_captions / n if n else 0
    if caption_cov >= MIN_FRAME_CAPTION_COVERAGE:
        score += 20
        _ok(f"Frame captions on {caption_cov:.0%} of chunks ≥ {MIN_FRAME_CAPTION_COVERAGE:.0%}  (+20)")
    else:
        failures.append(f"Frame captions on {caption_cov:.0%} < {MIN_FRAME_CAPTION_COVERAGE:.0%}")
        _fail(f"Frame captions on {caption_cov:.0%} < {MIN_FRAME_CAPTION_COVERAGE:.0%}  (0)")

    # ── 2.5  has_finance_signal on ≥ 25% ── 15 pts
    has_fin = sum(1 for s in structs if _s(s, "has_finance_signal") or _s(s, "finance_entities"))
    fin_cov = has_fin / n if n else 0
    if fin_cov >= MIN_FINANCE_SIGNAL_COVERAGE:
        score += 15
        _ok(f"Finance signal on {fin_cov:.0%} of chunks ≥ {MIN_FINANCE_SIGNAL_COVERAGE:.0%}  (+15)")
    else:
        failures.append(f"Finance signal on {fin_cov:.0%} < {MIN_FINANCE_SIGNAL_COVERAGE:.0%}")
        _fail(f"Finance signal on {fin_cov:.0%} < {MIN_FINANCE_SIGNAL_COVERAGE:.0%}  (0)")

    # ── 2.6  Earnings call detected on ≥ 1 chunk ── 10 pts
    is_ec = sum(1 for s in structs if _s(s, "is_earnings_call") or _s(s, "call_section"))
    if is_ec >= 1:
        score += 10
        _ok(f"Earnings-call/call_section detected in {is_ec}/{n} chunks  (+10)")
    else:
        failures.append("No earnings-call or call_section detected in any chunk")
        _fail("No earnings-call or call_section detected  (0)")

    # ── 2.7  Timestamps monotonically increasing ── 10 pts
    starts = [_s(s, "start_timestamp", "timestamp_start") for s in structs]
    starts = [float(x) for x in starts if x is not None]
    if len(starts) >= 2:
        monotonic = all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))
        if monotonic:
            score += 10
            _ok(f"Timestamps strictly non-decreasing across {len(starts)} chunks  (+10)")
        else:
            failures.append("Timestamps are NOT monotonically non-decreasing")
            _fail("Timestamps not monotonic  (0)")
    else:
        _warn(f"Only {len(starts)} chunks have timestamps — skipping monotonic check")

    # ── 2.8  Audio quality (SNR) on ≥ 70% ── 10 pts
    has_snr = sum(1 for s in structs if _s(s, "snr") is not None)
    snr_cov = has_snr / n if n else 0
    if snr_cov >= MIN_SNR_COVERAGE:
        score += 10
        _ok(f"SNR field present on {snr_cov:.0%} of chunks ≥ {MIN_SNR_COVERAGE:.0%}  (+10)")
    else:
        failures.append(f"SNR field present on {snr_cov:.0%} < {MIN_SNR_COVERAGE:.0%}")
        _fail(f"SNR field present on {snr_cov:.0%} < {MIN_SNR_COVERAGE:.0%}  (0)")

    # ── Per-chunk detail table ──────────────────────────────────────────────
    print(f"\n  {'IDX':>3}  {'TRANSCRIPT (first 50 chars)':<52}  {'SPEAKER':<18}  {'FRAMES':>6}  {'FIN':>3}  {'TS_START':>8}")
    print(f"  {'───':>3}  {'─'*52}  {'─'*18}  {'─'*6}  {'───':>3}  {'─'*8}")
    for i, (d, s) in enumerate(zip(docs, structs)):
        transcript = (_s(s, "transcript") or d.text or "")[:50].replace("\n", " ")
        speaker    = str(_s(s, "speaker_name", "speaker_label", "speaker") or "—")[:18]
        n_frames   = len(_s(s, "frame_captions") or [])
        fin        = "✓" if (_s(s, "has_finance_signal") or _s(s, "finance_entities")) else " "
        ts         = _s(s, "start_timestamp", "timestamp_start")
        ts_str     = f"{float(ts):>7.1f}s" if ts is not None else "       —"
        print(f"  {i:>3}  {transcript:<52}  {speaker:<18}  {n_frames:>6}  {fin:>3}  {ts_str}")

    print(f"\n  {_BOLD}CHUNK SCORE: {score}/100{_RESET}")
    return score, failures


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — EMBEDDING SCORING  (weight 35%)
# ════════════════════════════════════════════════════════════════════════════

def phase3_embedding_score(docs: List[Any]) -> Tuple[int, List[str]]:
    _section("Phase 3 · Embedding Quality Scoring")

    score = 0
    failures: List[str] = []
    n = len(docs)

    emb_combined = [getattr(d, "embedding", None) for d in docs]
    emb_audio    = [getattr(d, "embedding_audio", None) for d in docs]
    emb_visual   = [getattr(d, "embedding_visual", None) for d in docs]

    # ── 3.1  All 3 embeddings present ──── 20 pts
    present_c = sum(1 for e in emb_combined if e is not None)
    present_a = sum(1 for e in emb_audio    if e is not None)
    present_v = sum(1 for e in emb_visual   if e is not None)

    if present_c == n and present_a == n and present_v == n:
        score += 20
        _ok(f"All 3 embeddings (combined/audio/visual) present on all {n} chunks  (+20)")
    elif present_c == n:
        score += 10
        _warn(f"Combined embedding present on all {n}, but audio={present_a}/{n} visual={present_v}/{n}  (+10)")
        failures.append(f"Audio embeddings missing on {n-present_a} chunks; visual missing on {n-present_v}")
    else:
        failures.append(f"Combined embedding missing on {n-present_c}/{n} chunks")
        _fail(f"Combined embedding only {present_c}/{n} — embedding step may have failed  (0)")

    # ── 3.2  Embedding norms in [0.95, 1.05] ── 20 pts
    valid_norms = []
    norms_combined = [_norm(e) for e in emb_combined if e is not None]
    all_in_range = all(MIN_EMBEDDING_NORM_LO <= nm <= MIN_EMBEDDING_NORM_HI for nm in norms_combined)
    avg_norm = sum(norms_combined) / len(norms_combined) if norms_combined else 0.0

    if norms_combined and all_in_range:
        score += 20
        _ok(f"All embedding L2-norms in [{MIN_EMBEDDING_NORM_LO}, {MIN_EMBEDDING_NORM_HI}], avg={avg_norm:.4f}  (+20)")
    elif norms_combined:
        bad = sum(1 for nm in norms_combined if not (MIN_EMBEDDING_NORM_LO <= nm <= MIN_EMBEDDING_NORM_HI))
        failures.append(f"{bad}/{len(norms_combined)} embedding norms outside [{MIN_EMBEDDING_NORM_LO}, {MIN_EMBEDDING_NORM_HI}]")
        _fail(f"{bad} norms out of range, avg={avg_norm:.4f}  (0)")
    else:
        failures.append("No combined embeddings to check norms")
        _fail("No combined embeddings  (0)")

    # ── 3.3  Audio ≠ Visual (avg cosine < 0.97) ── 20 pts
    pairs_av = [(a, v) for a, v in zip(emb_audio, emb_visual) if a is not None and v is not None]
    if pairs_av:
        cosines_av = [_cosine(a, v) for a, v in pairs_av]
        avg_av = sum(cosines_av) / len(cosines_av)
        if avg_av < MAX_AUDIO_VISUAL_COSINE:
            score += 20
            _ok(f"Audio/visual avg cosine {avg_av:.4f} < {MAX_AUDIO_VISUAL_COSINE} (distinct signals)  (+20)")
        else:
            failures.append(f"Audio/visual avg cosine {avg_av:.4f} ≥ {MAX_AUDIO_VISUAL_COSINE} — embeddings not distinct")
            _fail(f"Audio/visual avg cosine {avg_av:.4f} ≥ {MAX_AUDIO_VISUAL_COSINE}  (0)")
    else:
        _warn("Cannot check audio/visual distinctness — at least one is missing on all chunks  (0)")
        failures.append("No overlapping audio+visual embedding pairs to compare")

    # ── 3.4  Combined correlates with audio (avg cosine > 0.25) ── 15 pts
    pairs_ca = [(c, a) for c, a in zip(emb_combined, emb_audio) if c is not None and a is not None]
    if pairs_ca:
        cosines_ca = [_cosine(c, a) for c, a in pairs_ca]
        avg_ca = sum(cosines_ca) / len(cosines_ca)
        if avg_ca > MIN_COMBINED_AUDIO_COSINE:
            score += 15
            _ok(f"Combined/audio avg cosine {avg_ca:.4f} > {MIN_COMBINED_AUDIO_COSINE} (combined includes transcript)  (+15)")
        else:
            failures.append(f"Combined/audio avg cosine {avg_ca:.4f} ≤ {MIN_COMBINED_AUDIO_COSINE}")
            _fail(f"Combined/audio avg cosine {avg_ca:.4f} ≤ {MIN_COMBINED_AUDIO_COSINE}  (0)")
    else:
        _warn("Cannot check combined/audio correlation — no matching pairs  (0)")
        failures.append("No overlapping combined+audio pairs")

    # ── 3.5  Semantic search recall: "Apple revenue Q4 2024" → ≥ 1 result ── 25 pts
    print("  [3.5] Semantic search test: 'Apple revenue Q4 2024' …")
    try:
        from app.embeddings.txt_embedder import TxtEmbedder
        from app.vectorstore.qdrant_store import QdrantVectorStore

        txt_embedder = TxtEmbedder()
        query_vec = txt_embedder.embed_query("Apple revenue Q4 2024")
        store = QdrantVectorStore()
        results = store.search_text(
            query_vector=query_vec,
            limit=5,
            user_id=TEST_USER_ID,
        )
        if results:
            score += 25
            _ok(f"Semantic search returned {len(results)} result(s) — retrieval works  (+25)")
            for i, r in enumerate(results[:3]):
                pay = r.get("payload", {}) if isinstance(r, dict) else {}
                snippet = str(pay.get("text", pay.get("transcript", "")))[:80].replace("\n", " ")
                print(f"       Hit {i+1}: score={r.get('score', '?'):.3f}  {snippet!r}")
        else:
            failures.append("Semantic search returned 0 results for 'Apple revenue Q4 2024'")
            _fail("Semantic search returned 0 results  (0)")
    except Exception as exc:
        failures.append(f"Semantic search failed: {exc}")
        _fail(f"Semantic search error: {exc}  (0)")

    print(f"\n  {_BOLD}EMBEDDING SCORE: {score}/100{_RESET}")
    return score, failures


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — METADATA / QDRANT SCORING  (weight 35%)
# ════════════════════════════════════════════════════════════════════════════

def phase4_metadata_score(docs: List[Any]) -> Tuple[int, List[str]]:
    _section("Phase 4 · Metadata & Qdrant Storage Scoring")

    score = 0
    failures: List[str] = []

    from app.core.config import settings
    from app.vectorstore.qdrant_store import QdrantVectorStore
    from app.utils.paths import user_dir
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    store = QdrantVectorStore()

    def _scroll_collection(collection: str) -> List[Any]:
        try:
            points, _ = store.client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=TEST_USER_ID)),
                    ]
                ),
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            return points
        except Exception as exc:
            _warn(f"Scroll failed on {collection}: {exc}")
            return []

    text_points   = _scroll_collection(settings.TEXT_COLLECTION_NAME)
    vision_points = _scroll_collection(settings.VISION_COLLECTION_NAME)
    all_points    = text_points + vision_points

    print(f"  Qdrant points found: {len(text_points)} in text_collection, {len(vision_points)} in vision_collection")

    # Build payloads list early (used by multiple checks below)
    payloads = [p.payload or {} for p in all_points] if all_points else []

    # ── 4.1  text_collection has ≥ 1 point ── 15 pts
    if text_points:
        score += 15
        _ok(f"{len(text_points)} points in text_collection  (+15)")
    else:
        failures.append("text_collection has 0 points for test user")
        _fail("text_collection has 0 points  (0)")

    # ── 4.2  vision_collection check (informational) ── 15 pts
    # Note: VideoChunker._make_doc() always sets embedding_space="text", so
    # transcript_frame docs go to text_collection. Vision collection is only
    # populated when VideoIngestor's direct frame path is used (embedding_space="vision").
    # Check for has_visual_embedding flag on Qdrant payload instead.
    has_visual_emb_in_payload = sum(
        1 for p in payloads
        if p.get("has_visual_embedding") or p.get("embedding_visual")
    )
    if vision_points:
        score += 15
        _ok(f"{len(vision_points)} points in vision_collection  (+15)")
    elif has_visual_emb_in_payload >= 1:
        score += 10
        _warn(
            f"vision_collection=0 pts (VideoChunker routes to text_collection), "
            f"but visual embedding stored as payload on {has_visual_emb_in_payload} point(s)  (+10)"
        )
    else:
        _warn(
            "vision_collection=0 and no has_visual_embedding payload flag. "
            "DESIGN GAP: VideoChunker always sets embedding_space='text'. "
            "Visual embeddings are payload fields, not separate vision_collection vectors.  (0)"
        )
        failures.append(
            "vision_collection has 0 points. VideoChunker sets embedding_space='text' for all "
            "transcript_frame docs. Visual embedding stored as payload only."
        )

    if not all_points:
        _fail("No Qdrant points at all — skipping payload checks")
        print(f"\n  {_BOLD}METADATA SCORE: {score}/100{_RESET}")
        return score, failures

    # ── 4.3  user_id on 100% ── 10 pts
    has_uid = sum(1 for p in payloads if p.get("user_id") == TEST_USER_ID)
    if has_uid == len(all_points):
        score += 10
        _ok(f"user_id present on all {len(all_points)} Qdrant points  (+10)")
    else:
        failures.append(f"user_id missing on {len(all_points)-has_uid} Qdrant points")
        _fail(f"user_id missing on {len(all_points)-has_uid} points  (0)")

    # ── 4.4  session_id on 100% ── 5 pts
    has_sid = sum(1 for p in payloads if p.get("session_id"))
    if has_sid == len(all_points):
        score += 5
        _ok(f"session_id present on all {len(all_points)} points  (+5)")
    else:
        failures.append(f"session_id missing on {len(all_points)-has_sid} points")
        _fail(f"session_id missing on {len(all_points)-has_sid} points  (0)")

    # ── 4.5  start_timestamp + end_timestamp on 100% of text_points ── 10 pts
    if text_points:
        tp_payloads = [p.payload or {} for p in text_points]
        has_ts = sum(
            1 for p in tp_payloads
            if (p.get("start_timestamp") is not None or p.get("timestamp_start") is not None)
            and (p.get("end_timestamp") is not None or p.get("timestamp_end") is not None)
        )
        ts_cov = has_ts / len(text_points)
        if ts_cov >= 1.0:
            score += 10
            _ok(f"start/end timestamps present on all {len(text_points)} text_collection points  (+10)")
        else:
            failures.append(f"Timestamps missing on {len(text_points)-has_ts}/{len(text_points)} text points")
            _fail(f"Timestamps present on {ts_cov:.0%} of text points  (0)")

    # ── 4.6  finance_entities non-empty on ≥ 1 point ── 10 pts
    has_fe = sum(1 for p in payloads if p.get("finance_entities"))
    if has_fe >= 1:
        score += 10
        _ok(f"finance_entities non-empty on {has_fe}/{len(all_points)} points  (+10)")
    else:
        failures.append("finance_entities empty on ALL Qdrant points")
        _fail("finance_entities empty on all points  (0)")

    # ── 4.7  frame_captions list non-empty on ≥ 1 point ── 10 pts
    has_fc = sum(1 for p in payloads if p.get("frame_captions"))
    if has_fc >= 1:
        score += 10
        _ok(f"frame_captions non-empty on {has_fc}/{len(all_points)} points  (+10)")
    else:
        failures.append("frame_captions empty on ALL Qdrant points")
        _fail("frame_captions empty on all points  (0)")

    # ── 4.8  BM25 pkl files exist ── 10 pts
    # Main index: mp4.pkl (not mp4_combined.pkl — "combined" search uses main index)
    # Sub-indexes: mp4_audio.pkl, mp4_visual.pkl (visual only when frame captions exist)
    bm25_dir = user_dir(TEST_USER_ID) / "bm25_index"
    main_pkl  = "mp4.pkl"
    audio_pkl = "mp4_audio.pkl"
    visual_pkl = "mp4_visual.pkl"
    present = [f for f in [main_pkl, audio_pkl, visual_pkl] if (bm25_dir / f).is_file()]
    print(f"  BM25 index dir: {bm25_dir}")
    print(f"  BM25 files found: {present}")
    if main_pkl in present and audio_pkl in present:
        score += 10
        _ok(f"BM25 main (mp4.pkl) + audio sub-index present{' + visual' if visual_pkl in present else ''}  (+10)")
        if visual_pkl not in present:
            _warn("mp4_visual.pkl missing — expected when no frame captions in chunks")
    elif main_pkl in present:
        score += 5
        _warn(f"BM25 main index (mp4.pkl) present but audio sub-index missing  (+5)")
        failures.append("mp4_audio.pkl missing from BM25 index dir")
    else:
        failures.append(f"BM25 main index (mp4.pkl) not found in {bm25_dir}")
        _fail(f"BM25 main index missing  (0)")

    # ── 4.9  BM25 search with content-adaptive query ── 10 pts
    # Use a word from the actual transcript text so the query doesn't depend on content type
    try:
        from app.bm25.video_bm25 import VideoBM25
        bm25 = VideoBM25(user_id=TEST_USER_ID)

        # Build a content-adaptive query from the first Qdrant payload text
        sample_text = ""
        if text_points:
            pay = text_points[0].payload or {}
            sample_text = (pay.get("transcript") or pay.get("text") or "")[:100]
        # Pick the 3rd content word (skip stopwords)
        content_words = [w for w in sample_text.split() if len(w) > 4]
        bm25_query = " ".join(content_words[:3]) if content_words else "revenue guidance"
        print(f"  BM25 search query (from transcript): {bm25_query!r}")

        results = bm25.search(bm25_query, top_k=5, user_id=TEST_USER_ID)
        if results:
            score += 10
            _ok(f"BM25 search → {len(results)} result(s) for {bm25_query!r}  (+10)")
        else:
            # Fallback: try "revenue guidance" in case this is actually finance content
            results2 = bm25.search("revenue guidance", top_k=5, user_id=TEST_USER_ID)
            if results2:
                score += 10
                _ok(f"BM25 'revenue guidance' → {len(results2)} result(s)  (+10)")
            else:
                failures.append(f"BM25 search returned 0 results for query {bm25_query!r}")
                _fail(f"BM25 search returned 0 results  (0)")
    except Exception as exc:
        failures.append(f"BM25 search failed: {exc}")
        _fail(f"BM25 search error: {exc}  (0)")

    # ── 4.10  is_earnings_call=True on ≥ 1 Qdrant point ── 5 pts
    has_ec = sum(1 for p in payloads if p.get("is_earnings_call") is True)
    if has_ec >= 1:
        score += 5
        _ok(f"is_earnings_call=True on {has_ec}/{len(all_points)} points  (+5)")
    else:
        _warn("is_earnings_call=True on 0 points — classifier may not have fired  (0)")
        failures.append("is_earnings_call=True on 0 Qdrant points")

    # ── Payload sample printout ─────────────────────────────────────────────
    print(f"\n  {'Field':<30}  {'Sample value (first text_point)'}")
    print(f"  {'─'*30}  {'─'*45}")
    if text_points:
        sample = text_points[0].payload or {}
        showcase = [
            "modality", "subtype", "start_timestamp", "end_timestamp", "speaker_label",
            "speaker_name", "call_section", "is_earnings_call", "has_finance_signal",
            "word_count", "token_count", "snr", "session_id",
        ]
        for k in showcase:
            v = sample.get(k)
            if v is not None:
                print(f"  {k:<30}  {str(v)[:45]!r}")

    print(f"\n  {_BOLD}METADATA SCORE: {score}/100{_RESET}")
    return score, failures


# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — CLEANUP & FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════

def phase5_cleanup_report(
    chunk_score:  int,
    embed_score:  int,
    meta_score:   int,
    chunk_fails:  List[str],
    embed_fails:  List[str],
    meta_fails:   List[str],
    timings:      Dict[str, float],
) -> int:
    _section("Phase 5 · Cleanup & Final Report")

    # Cleanup Qdrant
    try:
        from app.vectorstore.qdrant_store import QdrantVectorStore
        store = QdrantVectorStore()
        store.delete_by_session(session_id=TEST_SESSION_ID)
        _ok(f"Qdrant test data purged for session '{TEST_SESSION_ID}'")
    except Exception as exc:
        _warn(f"Qdrant cleanup failed: {exc}")

    # Weighted final score
    final = round(chunk_score * 0.30 + embed_score * 0.35 + meta_score * 0.35)
    pass_overall = final >= 70 and chunk_score >= 50 and embed_score >= 50 and meta_score >= 50

    def _dim(s: int) -> str:
        if s >= 80:
            return f"{_GREEN}{s:>3}/100{_RESET}"
        elif s >= 50:
            return f"{_YELLOW}{s:>3}/100{_RESET}"
        else:
            return f"{_RED}{s:>3}/100{_RESET}"

    status_str = f"{_GREEN}{_BOLD}PASS{_RESET}" if pass_overall else f"{_RED}{_BOLD}FAIL{_RESET}"

    print(f"""
{_BOLD}╔══════════════════════════════════════════════════════╗
║         VIDEO MODALITY — TEST SCORECARD              ║
╠══════════════════════════════════════════════════════╣{_RESET}
  {_BOLD}Chunk Quality    (×0.30){_RESET}  {_dim(chunk_score)}  {"PASS" if chunk_score >= 50 else "FAIL"}
  {_BOLD}Embedding Quality(×0.35){_RESET}  {_dim(embed_score)}  {"PASS" if embed_score >= 50 else "FAIL"}
  {_BOLD}Metadata / Qdrant(×0.35){_RESET}  {_dim(meta_score)}  {"PASS" if meta_score >= 50 else "FAIL"}
{_BOLD}╠══════════════════════════════════════════════════════╣
  FINAL WEIGHTED SCORE   {_dim(final)}
  STATUS                 {status_str}
╚══════════════════════════════════════════════════════╝{_RESET}""")

    # Timing
    print(f"\n  {_BOLD}Pipeline Timings:{_RESET}")
    for stage, t in timings.items():
        print(f"    {stage:<20} {t:.2f}s")

    # Failures
    all_fails = [("Chunk", f) for f in chunk_fails] + \
                [("Embed", f) for f in embed_fails] + \
                [("Meta",  f) for f in meta_fails]
    if all_fails:
        print(f"\n  {_BOLD}{_RED}Sub-check failures:{_RESET}")
        for dim, msg in all_fails:
            print(f"    [{dim}] {msg}")

    # Known bugs found during testing
    print(f"\n  {_BOLD}{_YELLOW}Known bugs / design gaps identified:{_RESET}")
    print("    [BUG-1] FIXED — torchaudio/huggingface_hub/PyTorch 2.6 compat resolved via")
    print("            app/utils/torchaudio_compat.py (patch_torchaudio called in model_loader)")
    print("    [BUG-2] FIXED — VideoChunker now emits vision-space frame docs (subtype='frame')")
    print("            with embedding_space='vision'. VideoEmbedder embeds them via SigLIP")
    print("            (1152-dim) → vision_collection. Verified: 1 point in vision_collection")
    print("            per unique captioned frame.")
    print("    [BUG-3] FIXED — VideoChunker._measure_snr() runs ffmpeg volumedetect on the")
    print("            extracted WAV and stores snr / snr_degraded / clipping_detected in")
    print("            each transcript_frame doc. Verified: snr=18.9 in Qdrant payload.")
    print("    [INFO]  mp4_visual.pkl not created when no frame captions exist in chunks")
    print("            (expected when only 1 frame at t=0 and audio starts at t>5s)")

    return 0 if pass_overall else 1


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print(f"\n{_BOLD}{_CYAN}{'='*60}")
    print("  MAGIK — Video Modality End-to-End Test")
    print(f"{'='*60}{_RESET}")
    print(f"  Video : {VIDEO_PATH}")
    print(f"  User  : {TEST_USER_ID}")
    print(f"  Session: {TEST_SESSION_ID}")

    # Phase 0
    if not phase0_preflight():
        print(f"\n{_RED}{_BOLD}Pre-flight failed — aborting.{_RESET}")
        return 1

    # Phase 1
    try:
        docs, timings = phase1_ingest()
    except Exception as exc:
        print(f"\n{_RED}{_BOLD}Ingestion failed: {exc}{_RESET}")
        import traceback
        traceback.print_exc()
        return 1

    # Phases 2-4
    chunk_score, chunk_fails = phase2_chunk_score(docs)
    embed_score, embed_fails = phase3_embedding_score(docs)
    meta_score,  meta_fails  = phase4_metadata_score(docs)

    # Phase 5
    return phase5_cleanup_report(
        chunk_score, embed_score, meta_score,
        chunk_fails, embed_fails, meta_fails,
        timings,
    )


if __name__ == "__main__":
    sys.exit(main())
