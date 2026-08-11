"""video_chunker.py

Finance-grade Video chunker — audio pipeline + frame captioning.

Frame-extraction code (FrameMetadata, extract_frames, PySceneDetect/OpenCV
path) lives in app/ingestion/video_ingest.py because frame extraction is
called from within the ingestor before RawExtract objects exist.
"""

from __future__ import annotations

import bisect
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image as PILImage
from prometheus_client import Counter

from app.chunking.av_shared import (
    _assemble_chunks,
    _map_speaker_roles,
    _merge_fragmented_hosts,
)
from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import deterministic_chunk_id, extract_finance_entities
from app.core.config import settings
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_video_chunks_total",
    "Total chunks produced by video chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_video_chunk_errors_total",
    "Total errors in video chunker",
)

_FRAME_WINDOW_S = 5.0  # attach frames within ±5s of audio chunk
_FINANCIAL_TRIGGER_RE = re.compile(r"[$%]|\bbillion\b|\brevenue\b|\bearnings\b", re.I)

# Qwen2-VL-2B INT8 = ~2.2 GB; semaphore limits concurrent inference calls.
_QWEN2VL_SEMAPHORE = threading.Semaphore(settings.VIDEO_CAPTION_CONCURRENCY)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO EXTRACTION HELPER
# ══════════════════════════════════════════════════════════════════════════════


def _extract_audio(video_path: str, wav_path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
            capture_output=True,
            timeout=600,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.warning(event="video_ffmpeg_audio_failed", error=str(exc))
        return False


def _measure_snr(wav_path: str) -> dict:
    """Measure audio quality from an extracted WAV using ffmpeg volumedetect.

    Returns snr (dynamic range in dB), snr_degraded flag, clipping_detected flag.
    SNR here is estimated as peak - mean volume, a proxy for dynamic range.
    """
    out = {"snr": None, "snr_degraded": False, "clipping_detected": False}
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", wav_path, "-af", "volumedetect", "-f", "null", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stderr = r.stderr or ""
        mean_m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
        max_m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
        if mean_m and max_m:
            mean_vol = float(mean_m.group(1))
            max_vol = float(max_m.group(1))
            out["snr"] = round(max_vol - mean_vol, 1)
            out["snr_degraded"] = mean_vol < -30.0
            out["clipping_detected"] = max_vol > -1.0
    except Exception as exc:
        logger.warning(event="snr_measurement_failed", error=str(exc))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO ASR — earnings-webcast tuned transcription (video-scoped)
# ══════════════════════════════════════════════════════════════════════════════

# Video-specific Whisper priming prompt. The shared audio prompt in
# audio_chunker.py is FOMC/Powell-tuned (Federal Reserve press conference); an
# investor/earnings webcast needs company-executive priming instead so
# faster-whisper keeps correct capitalization and gets the proper nouns
# (executive names, product names, financial metrics) right. Owned by the
# video pipeline; must NOT be shared back into the audio prompt.
_VIDEO_WHISPER_PROMPT = (
    "The following is a corporate quarterly earnings conference call and investor "
    "webcast, transcribed with correct capitalization and punctuation. Company "
    "executives — the CEO, the CFO, and the head of Investor Relations — deliver "
    "prepared remarks on revenue, diluted EPS, gross margin, Services, iPhone, and "
    "year-over-year growth, then take questions from sell-side analysts who "
    "introduce themselves and their firm. Speakers include Tim Cook, Kevan Parekh, "
    "and Suhasini Chandramouli. Figures such as $102.5 billion in revenue, $1.85 "
    "EPS, 8% growth, and all-time records are stated precisely."
)

# Match the audio pipeline's proven 10-minute transcription window: a single
# faster-whisper call over an hour-scale recording degrades in the later portion
# (dropped capitalization, garbled proper nouns), which is exactly the Q&A half
# of an earnings call. 600 s windows keep each call short enough that the casing
# prompt stays effective throughout.
_VIDEO_TRANSCRIBE_SEGMENT_SEC = 600


def _run_whisper_video(wav_path: str) -> list[dict]:
    """Transcribe with faster-whisper using the earnings-webcast prompt.

    Mirrors audio_chunker._run_whisper but swaps in the video-domain priming
    prompt. Returns a list of word dicts ({"word","start","end"}).
    """
    try:
        from app.core.model_loader import model_loader as loader

        model = loader.get_whisper()
        # faster-whisper's transcribe() returns a lazy generator — the actual
        # CUDA decode happens during iteration, so materializing it to a list
        # must stay inside the lock too, not just the transcribe() call itself.
        with loader.get_whisper_lock():
            segments, _ = model.transcribe(
                wav_path,
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=_VIDEO_WHISPER_PROMPT,
                beam_size=5,
            )
            segments = list(segments)
        words: list[dict] = []
        for seg in segments:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
            else:
                words.append({"word": seg.text, "start": seg.start, "end": seg.end})
        return words
    except Exception as exc:
        logger.warning(event="video_whisper_failed", error=str(exc))
        return []


def _transcribe_video_audio(wav_path: str, duration_sec: float) -> list[dict]:
    """Segmented transcription for video audio (>10 min → 600 s windows).

    Video previously called _run_whisper directly (a single call over the whole
    track), inheriting the exact long-audio quality degradation the audio
    pipeline already solved with segmentation. This mirrors
    audio_chunker._transcribe_long_audio but uses the video Whisper prompt and
    keeps the logic inside the video-owned file.
    """
    if duration_sec <= 0 or duration_sec <= _VIDEO_TRANSCRIBE_SEGMENT_SEC:
        return _run_whisper_video(wav_path)

    import math as _math

    from pydub import AudioSegment

    audio = AudioSegment.from_wav(wav_path)
    chunk_sec = _VIDEO_TRANSCRIBE_SEGMENT_SEC
    n_segments = _math.ceil(duration_sec / chunk_sec)

    segment_paths: list[tuple[str, float]] = []
    for i in range(n_segments):
        start_ms = int(i * chunk_sec * 1000)
        end_ms = int(min((i + 1) * chunk_sec, duration_sec) * 1000)
        seg = audio[start_ms:end_ms]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        seg.export(tmp.name, format="wav")
        tmp.close()
        segment_paths.append((tmp.name, i * chunk_sec))

    words: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=settings.AUDIO_TRANSCRIPTION_WORKERS) as pool:
            futures = {pool.submit(_run_whisper_video, p): off for p, off in segment_paths}
            results: list[tuple[float, list[dict]]] = []
            for fut, off in futures.items():
                try:
                    seg_words = fut.result()
                    for w in seg_words:
                        w["start"] += off
                        w["end"] += off
                    results.append((off, seg_words))
                except Exception as exc:
                    logger.warning(
                        event="video_segment_transcribe_failed", offset=off, error=str(exc)
                    )
        results.sort(key=lambda r: r[0])
        for _, seg_words in results:
            words.extend(seg_words)
    finally:
        for p, _off in segment_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
    return words


# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

# Concise, retrieval-oriented prompt. Verbatim on-screen text (ticker bars,
# headline crawls) is captured separately by TrOCR, so the VLM only needs a
# short factual summary of the visual — this keeps generation short (fast) and
# focused on what a finance query actually retrieves on: chart, prices, and any
# highlighted headline/metric. Bounded by VIDEO_CAPTION_MAX_TOKENS.
_VIDEO_FRAME_PROMPT = (
    "This is a frame from a financial earnings webcast (chart, ticker, or slide). "
    "In 2-3 sentences, state concisely: the chart or slide title; the asset/ticker "
    "shown and its visible price or level; and any headline, metric, or number "
    "visible on screen (revenue, EPS, percentages) — copy numbers exactly, never "
    "round. Be brief and factual."
)


def caption_frame(image: PILImage.Image, prompt: str | None = None) -> str:
    """Caption a single video frame using Qwen2-VL-2B-Instruct. Returns '' on failure."""
    try:
        from app.core.model_loader import model_loader

        # Video-specific (smaller) captioner — see get_qwen2_vl_video. Keeps a
        # 1-hour ingest within VRAM alongside Whisper/pyannote/SigLIP + llama-server.
        processor, model, device = model_loader.get_qwen2_vl_video()
    except Exception as exc:
        logger.warning(event="qwen2vl_unavailable", error=str(exc))
        return ""
    try:
        import torch as _torch

        prompt_text = prompt or _VIDEO_FRAME_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        with _torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=settings.VIDEO_CAPTION_MAX_TOKENS)
        generated_ids = [o[len(i) :] for i, o in zip(inputs.input_ids, out, strict=False)]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        logger.warning(event="qwen2vl_caption_failed", error=str(exc))
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# FRAME CAPTIONING HELPER
# ══════════════════════════════════════════════════════════════════════════════


def _caption_and_ocr_frame(frame_dict: dict) -> dict:
    """Run Qwen2-VL + TrOCR on a frame given its FrameMetadata dict."""
    result = {
        "frame_timestamp": frame_dict["timestamp_start"],
        "frame_path": frame_dict.get("path", ""),
        "scene_change": frame_dict["is_scene_boundary"],
        "frame_caption": "",
        "ocr_text": "",
        "slide_number": None,
    }
    frame_path = frame_dict.get("path", "")
    if not frame_path or not Path(frame_path).exists():
        return result

    try:
        from PIL import Image

        img = Image.open(frame_path).convert("RGB")
    except Exception:
        return result

    try:
        result["frame_caption"] = caption_frame(img)
    except Exception as exc:
        logger.warning(event="qwen2vl_caption_failed", error=str(exc))

    try:
        from app.chunking.image_chunker import ocr as _ocr

        result["ocr_text"] = _ocr(img)
    except Exception as exc:
        logger.warning(event="trocr_frame_failed", error=str(exc))

    # Detect slide number from OCR first, then fall back to caption
    ocr_text = result["ocr_text"]
    caption = result["frame_caption"]
    m = re.search(r"slide\s*(\d+)", ocr_text, re.I) or re.search(r"slide\s*(\d+)", caption, re.I)
    result["slide_number"] = int(m.group(1)) if m else None
    return result


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO CHUNKER
# ══════════════════════════════════════════════════════════════════════════════


class VideoChunker(BaseChunker):
    """Finance-grade chunker for video files (investor days, earnings webcasts, CNBC segments).

    Pipeline:
      ffmpeg audio → Whisper → pyannote diarization → audio chunks
      PySceneDetect/OpenCV frame extraction → Qwen2-VL+TrOCR per frame
      audio-visual sync alignment (±5 s window) → IngestedDocuments.
    """

    def chunk(
        self,
        extracts: list[RawExtract],
        meta: UniversalMetadata,
    ) -> list[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.mp4"
        surface = "video_chunker"
        modality_var.set("video")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="video", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="video", source=source)
            return []
        try:
            docs: list[IngestedDocument] = []

            for ext in extracts:
                if ext.extract_type != "video_raw":
                    continue

                video_path = ext.extra.get("file_path", "") or meta.source_path
                if not video_path or not Path(video_path).exists():
                    logger.warning(event="video_chunker_missing_file", source=source)
                    continue

                with tempfile.TemporaryDirectory() as tmpdir:
                    wav_path = os.path.join(tmpdir, "audio.wav")

                    # 1. Extract audio track.
                    if not _extract_audio(video_path, wav_path):
                        logger.warning(event="video_no_audio", source=source)
                        continue

                    # 2. Whisper transcription — segmented (600 s windows) so an
                    #    hour-scale earnings call does not degrade in its Q&A half.
                    _duration_sec = float((ext.extra or {}).get("duration_seconds") or 0.0)
                    words = _transcribe_video_audio(wav_path, _duration_sec)

                    # 3. Diarization (+ host-fragment merge, matching audio).
                    diarization: list[tuple[float, float, str]] = []
                    try:
                        from app.chunking.audio_chunker import diarize as _diarize

                        diarization = _diarize(wav_path)
                        diarization = _merge_fragmented_hosts(diarization)
                    except Exception:
                        pass

                    full_transcript = " ".join(w["word"] for w in words)
                    # _map_speaker_roles expects the word-dict list (it anchors
                    # names to word timestamps) — passing the joined string here
                    # crashed ingestion whenever diarization returned segments.
                    role_map = _map_speaker_roles(diarization, words)
                    # Video-only finer transcript chunking (vision frames separate).
                    audio_chunks = _assemble_chunks(
                        words,
                        diarization,
                        role_map,
                        min_words=settings.VIDEO_CHUNK_MIN_WORDS,
                        max_words=settings.VIDEO_CHUNK_MAX_WORDS,
                    )

                    # Detect earnings call from full transcript
                    _ft_lower = full_transcript.lower()
                    is_earnings_call = any(
                        kw in _ft_lower
                        for kw in (
                            "earnings call",
                            "quarterly results",
                            "conference call",
                            "revenue",
                            "earnings per share",
                            "fiscal year",
                        )
                    )

                    # Measure audio quality from the extracted WAV (BUG-3 fix).
                    # Falls back to values from ingestor if measurement fails.
                    ext_extra = ext.extra or {}
                    _aq = _measure_snr(wav_path)
                    _snr = _aq["snr"] if _aq["snr"] is not None else ext_extra.get("snr")
                    _snr_degraded = _aq["snr_degraded"] or ext_extra.get("snr_degraded", False)
                    _clipping_detected = _aq["clipping_detected"] or ext_extra.get(
                        "clipping_detected", False
                    )

                    # 4. Frame extraction (import from video_ingest where it lives).
                    try:
                        from app.ingestion.video_ingest import extract_frames

                        frame_dicts = extract_frames(
                            video_path=video_path,
                            interval_sec=settings.VIDEO_FRAME_INTERVAL_SEC,
                            session_id=meta.custom_fields.get("session_id", "internal"),
                        )
                    except Exception as exc:
                        logger.warning(event="frame_extraction_error", error=str(exc))
                        frame_dicts = []

                    # 5. Caption + OCR each frame — concurrent, VRAM-bounded.
                    def _caption_safe(fd: dict) -> tuple[float, dict]:
                        with _QWEN2VL_SEMAPHORE:
                            return fd["timestamp_start"], _caption_and_ocr_frame(fd)

                    captioned_frames: dict[float, dict] = {}
                    with ThreadPoolExecutor(
                        max_workers=settings.VIDEO_CAPTION_CONCURRENCY
                    ) as _caption_pool:
                        for ts, result in _caption_pool.map(_caption_safe, frame_dicts):
                            captioned_frames[ts] = result

                    # Release the VLM's transient activations/KV-cache before the
                    # embedding stage (SigLIP frames + BGE text) so a long ingest
                    # doesn't stack peaks and OOM on a shared GPU.
                    try:
                        import torch as _torch

                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass

                    # 6. Build IngestedDocuments (one per audio chunk).
                    # Pre-sort frame timestamps once for O(log f) bisect window
                    # lookup instead of O(f) linear dict scan per chunk.
                    _sorted_ts: list[float] = sorted(captioned_frames)

                    for chunk_idx, ch in enumerate(audio_chunks):
                        transcript = ch.get("transcript", "")
                        if not transcript.strip():
                            continue

                        t_start = ch["start"]
                        t_end = ch["end"]

                        _lo = bisect.bisect_left(_sorted_ts, t_start - _FRAME_WINDOW_S)
                        _hi = bisect.bisect_right(_sorted_ts, t_end + _FRAME_WINDOW_S)
                        chunk_frames = [captioned_frames[ts] for ts in _sorted_ts[_lo:_hi]]

                        visual_ctx = ""
                        slide_bullets: list[str] = []
                        for cf in chunk_frames:
                            if cf.get("frame_caption"):
                                visual_ctx += f"\n[VISUAL AT {cf['frame_timestamp']:.1f}s]: {cf['frame_caption']}"
                            if cf.get("ocr_text"):
                                visual_ctx += f"\n[ON-SCREEN]: {cf['ocr_text']}"
                            if cf.get("slide_number") is not None:
                                slide_bullets.append(f"Slide {cf['slide_number']}")

                        combined_text = transcript
                        if visual_ctx:
                            combined_text += visual_ctx

                        fin_entities = extract_finance_entities(combined_text)
                        has_finance = bool(_FINANCIAL_TRIGGER_RE.search(transcript))
                        chunk_hash = deterministic_chunk_id(source, f"v_{t_start:.1f}", chunk_idx)

                        # Extract slide numbers from slide_bullets like ["Slide 3", "Slide 4"] (MD 1.7)
                        slide_numbers_covered: list[int] = []
                        for sb in slide_bullets:
                            m = re.search(r"\bslide\s*(\d+)\b", sb, re.IGNORECASE)
                            if m:
                                slide_numbers_covered.append(int(m.group(1)))

                        _words_in_chunk = transcript.split()
                        structure = {
                            "chunk_hash_id": chunk_hash,
                            "source_file": source,
                            "chunk_index": chunk_idx,
                            "start_timestamp": round(t_start, 3),
                            "end_timestamp": round(t_end, 3),
                            "duration_seconds": round(t_end - t_start, 3),
                            # _assemble_chunks returns "speaker", "name", "role" keys
                            "speaker_label": ch.get("speaker"),
                            "speaker_name": ch.get("name"),
                            "speaker_role": ch.get("role"),
                            "topic_section": ch.get("topic_section"),
                            "call_section": ch.get("call_section"),
                            "transcript": transcript,
                            "frame_captions": chunk_frames,
                            "combined_text": combined_text,
                            "slide_bullets": slide_bullets,
                            "has_slide_content": bool(slide_bullets),
                            "slide_numbers_covered": slide_numbers_covered,
                            "finance_entities": fin_entities,
                            "has_finance_signal": has_finance,
                            "is_question": ch.get("is_question", False),
                            "is_answer": ch.get("is_answer", False),
                            "is_earnings_call": is_earnings_call,
                            "word_count": len(_words_in_chunk),
                            "token_count": len(_words_in_chunk),  # approx: 1 word ≈ 1 token
                            # Audio quality signals from VideoIngestor via RawExtract.extra
                            "snr": _snr,
                            "snr_degraded": _snr_degraded,
                            "clipping_detected": _clipping_detected,
                        }

                        doc = self._make_doc(
                            text=combined_text,
                            modality="mp4",
                            subtype="transcript_frame",
                            source=source,
                            page=None,
                            chunk_idx=chunk_idx,
                            structure=structure,
                            meta=meta,
                            surface=surface,
                        )
                        if doc:
                            docs.append(doc)

                    # BUG-2 fix: emit one vision-space frame doc per unique captioned
                    # frame so it lands in vision_collection via SigLIP embedding.
                    # Iterating sorted(captioned_frames) guarantees exactly one doc
                    # per physical frame — no duplicates from the ±5 s audio window.
                    _vis_base = len(audio_chunks) * 10 + 1
                    for _vis_idx, (ts, cf) in enumerate(sorted(captioned_frames.items())):
                        fp = cf.get("frame_path", "")
                        cap = cf.get("frame_caption", "").strip()
                        ocr = cf.get("ocr_text", "").strip()
                        if not fp or not Path(fp).exists() or (not cap and not ocr):
                            continue
                        vis_text = f"{cap}\n[ON-SCREEN]: {ocr}".strip() if ocr else cap
                        _vis_chunk_id = _vis_base + _vis_idx
                        vis_doc = self._make_doc(
                            text=vis_text,
                            modality="mp4",
                            subtype="frame",
                            source=source,
                            page=None,
                            chunk_idx=_vis_chunk_id,
                            structure={
                                "source_file": source,
                                "chunk_index": _vis_chunk_id,
                                "frame_timestamp": round(ts, 3),
                                "start_timestamp": round(ts, 3),
                                "end_timestamp": round(ts, 3),
                                "is_earnings_call": is_earnings_call,
                                "finance_entities": extract_finance_entities(vis_text),
                                "has_finance_signal": bool(_FINANCIAL_TRIGGER_RE.search(vis_text)),
                                "asset_path": fp,
                                "slide_number": cf.get("slide_number"),
                                "scene_change": cf.get("scene_change", False),
                                "embedding_space": "vision",
                            },
                            meta=meta,
                            surface=surface,
                        )
                        if vis_doc:
                            docs.append(vis_doc)

            logger.info(event="video_chunking_done", source=source, chunks=len(docs))
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="video", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "video",
            "status": "ok",
            "class": self.__class__.__name__,
        }
