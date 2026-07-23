from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.ingestion.schema import (
    EmptyFileError,
    FileTooLargeError,
    IngestedDocument,
    Modality,
    UnsupportedMimeError,
    build_universal_metadata,
    normalize_text,
    redact_pii,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# SUPPORTED FORMATS
SUPPORTED_AUDIO_FORMATS = {
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".opus",
    ".wma",
    ".aiff",
}

# MAGIC BYTES MAP
_AUDIO_MAGIC: dict[bytes, str] = {
    b"ID3": "audio/mpeg",
    b"fLaC": "audio/flac",
    b"OggS": "audio/ogg",
    b"RIFF": "audio/wav",
}

# FILLER WORD PATTERN
_FILLER_PATTERN: re.Pattern | None = None

# YAMNET MODEL SINGLETON
_yamnet_model: Any | None = None


def _get_yamnet() -> Any:
    global _yamnet_model
    if _yamnet_model is None:
        import tensorflow_hub as hub

        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    return _yamnet_model


def _get_filler_pattern() -> re.Pattern:
    global _FILLER_PATTERN
    if _FILLER_PATTERN is None:
        words = [re.escape(w) for w in settings.WHISPER_FILLER_WORDS]
        if words:
            _FILLER_PATTERN = re.compile(
                r"\b(" + "|".join(words) + r")\b[,.]?\s*",
                re.IGNORECASE,
            )
        else:
            _FILLER_PATTERN = re.compile(r"(?!x)x")  # NEVER MATCHES
    return _FILLER_PATTERN


# SHA256 DEDUP


def _sha256_check(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# MD5 FILE HASH


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# MAGIC BYTE MIME DETECTION


def _detect_mime(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        for magic, mime in _AUDIO_MAGIC.items():
            if header.startswith(magic):
                if magic == b"RIFF" and header[8:12] == b"WAVE":
                    return "audio/wav"
                if magic != b"RIFF":
                    return mime
        suffix = path.suffix.lower()
        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
            ".opus": "audio/opus",
            ".wma": "audio/x-ms-wma",
            ".aiff": "audio/aiff",
        }
        return mime_map.get(suffix, "application/octet-stream")
    except Exception:
        return "application/octet-stream"


# DISK SPACE GUARD


def _check_disk_space(path: Path) -> None:
    try:
        usage = shutil.disk_usage(path.parent)
        min_bytes = settings.MIN_FREE_DISK_MB * 1024 * 1024
        if usage.free < min_bytes:
            logger.warning(
                event="low_disk_space",
                free_mb=round(usage.free / 1024 / 1024, 1),
                required_mb=settings.MIN_FREE_DISK_MB,
            )
    except OSError as exc:
        logger.warning(event="disk_space_check_failed", error=str(exc))


# PATH TRAVERSAL GUARD


def _safe_resolve(file_path: str) -> Path:
    path = Path(file_path).expanduser().resolve()
    chroot = settings.CHROOT_BASE.resolve()
    try:
        path.relative_to(chroot)
    except ValueError:
        raise ValueError(f"PATH_TRAVERSAL_BLOCKED: {path} is outside chroot {chroot}")
    return path


# VALIDATION


def _validate_audio_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {path}")

    if path.stat().st_size == 0:
        raise EmptyFileError(str(path))

    if path.stat().st_size > settings.MAX_FILE_SIZE_AUDIO:
        raise FileTooLargeError(
            f"FILE_TOO_LARGE: {path.stat().st_size} bytes exceeds "
            f"{settings.MAX_FILE_SIZE_AUDIO} bytes"
        )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise UnsupportedMimeError(f"UNSUPPORTED_AUDIO_FORMAT: {suffix}")

    if settings.MAGIC_BYTE_MIME_DETECTION:
        mime = _detect_mime(path)
        if mime == "application/octet-stream":
            raise UnsupportedMimeError(f"MAGIC_BYTE_MIME_UNRECOGNIZED: {path.name}")


# DRM DETECTION


def _is_drm_protected(path: Path) -> bool:
    try:
        suffix = path.suffix.lower()
        if suffix not in {".m4a", ".aac", ".wma"}:
            return False
        with open(path, "rb") as f:
            header = f.read(512)
        drm_markers = [b"drms", b"DRM", b"encrypted", b"iTunes_CDDB"]
        return any(marker in header for marker in drm_markers)
    except Exception:
        return False


# AUDIO LOAD VIA PYDUB


def _load_audio(file_path: str) -> Any:
    from pydub import AudioSegment

    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as exc:
        # ATTEMPT FFMPEG REPAIR ON CORRUPT HEADER
        logger.warning(
            event="audio_corrupt_repair_attempt",
            file=os.path.basename(file_path),
            error=str(exc),
        )
        try:
            import subprocess

            repaired = file_path + "_repaired.wav"
            result = subprocess.run(
                [
                    settings.FFMPEG_PATH,
                    "-err_detect",
                    "ignore_err",
                    "-i",
                    file_path,
                    "-y",
                    repaired,
                ],
                capture_output=True,
                timeout=settings.FFMPEG_TIMEOUT_SEC,
            )
            if result.returncode == 0 and os.path.exists(repaired):
                audio = AudioSegment.from_file(repaired)
                os.unlink(repaired)
                logger.info(event="audio_repair_success", file=os.path.basename(file_path))
                return audio
        except Exception as repair_exc:
            logger.error(
                event="audio_repair_failed",
                file=os.path.basename(file_path),
                error=str(repair_exc),
            )
        raise ValueError(f"CORRUPTED_FILE: {exc}")
    return audio


# CHANNEL MIXDOWN TO MONO


def _to_mono(audio: Any) -> Any:
    if audio.channels > 1:
        return audio.set_channels(1)
    return audio


# RESAMPLE TO 16KHZ


def _resample(audio: Any) -> Any:
    target = settings.AUDIO_SAMPLE_RATE
    if audio.frame_rate != target:
        logger.debug(
            event="audio_resampled",
            from_rate=audio.frame_rate,
            to_rate=target,
        )
        return audio.set_frame_rate(target)
    return audio


# SNR ESTIMATION


def _estimate_snr(audio: Any) -> float:
    try:
        dbfs = audio.dBFS
        if dbfs == float("-inf"):
            return 0.0
        return round(max(0.0, dbfs + 60.0), 2)
    except Exception:
        return 0.0


# CLIPPING DETECTION VIA LIBROSA


def _detect_clipping(file_path: str) -> bool:
    try:
        import librosa
        import numpy as np

        y, _ = librosa.load(file_path, sr=None, mono=True, duration=30)
        peak = float(np.abs(y).max())
        return (
            peak >= settings.AUDIO_CLIPPING_THRESHOLD
            if hasattr(settings, "AUDIO_CLIPPING_THRESHOLD")
            else peak >= 0.99
        )
    except Exception:
        return False


# SILENCE DETECTION


def _detect_silent_ranges(audio: Any) -> list[tuple[int, int]]:
    try:
        from pydub import silence as pydub_silence

        return pydub_silence.detect_silence(
            audio,
            min_silence_len=settings.AUDIO_SILENCE_GAP_MS,
            silence_thresh=-40,
        )
    except Exception as exc:
        logger.warning(event="silence_detection_failed", error=str(exc))
        return []


# SEGMENT IN SILENT RANGE


def _is_inaudible(start_sec: float, silent_ranges: list[tuple[int, int]]) -> bool:
    start_ms = start_sec * 1000
    return any(s <= start_ms <= e for s, e in silent_ranges)


# CONFIDENCE FROM LOGPROB


def _compute_confidence(avg_logprob: float | None) -> float:
    if avg_logprob is None:
        return 1.0
    return round(max(0.0, min(1.0, 1.0 + avg_logprob)), 4)


# HALLUCINATION RISK


def _hallucination_risk(confidence: float, no_speech_prob: float | None) -> str:
    nsp = no_speech_prob or 0.0
    if confidence < 0.4 or nsp > 0.8:
        return "high"
    if confidence < 0.65 or nsp > 0.5:
        return "medium"
    return "low"


# SPEECH RATE FLAG


def _speed_flag(text: str, duration: float) -> bool:
    words = len(text.split())
    if duration <= 0:
        return False
    wpm = (words / duration) * 60
    return wpm > 250 or wpm < 60


# FILLER WORD REMOVAL


def _strip_fillers(text: str) -> str:
    pattern = _get_filler_pattern()
    return pattern.sub("", text).strip()


# DOMAIN VOCAB CORRECTION


def _apply_domain_vocab(text: str) -> str:
    if not settings.WHISPER_DOMAIN_VOCAB:
        return text
    lower = text.lower()
    corrected = text
    for term in settings.WHISPER_DOMAIN_VOCAB:
        if term.lower() in lower:
            corrected = re.sub(
                re.escape(term),
                term,
                corrected,
                flags=re.IGNORECASE,
            )
    return corrected


# NORMALIZE PUNCTUATION


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ID3 METADATA VIA MUTAGEN


def _extract_id3(file_path: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        from mutagen import File as MutaFile

        mf = MutaFile(file_path)
        if mf is None:
            return meta
        tag_map = {
            "TIT2": "title",
            "TPE1": "artist",
            "TALB": "album",
            "TDRC": "year",
            "TCON": "genre",
            "TRCK": "track",
        }
        for tag_key, friendly in tag_map.items():
            if tag_key in mf:
                meta[friendly] = (
                    str(mf[tag_key].text[0]) if hasattr(mf[tag_key], "text") else str(mf[tag_key])
                )
    except ImportError:
        logger.debug(event="mutagen_not_installed")
    except Exception as exc:
        logger.debug(event="id3_extraction_failed", error=str(exc))
    return meta


# NOISE CLASSIFICATION VIA YAMNET

_YAMNET_SPEECH_CLASSES = {0, 1, 2, 3, 4}
_YAMNET_MUSIC_CLASSES = {137, 138, 139, 140}
_YAMNET_CROWD_CLASSES = {70, 71, 72}


def _classify_noise(file_path: str) -> str | None:
    try:
        import librosa
        import numpy as np
        import tensorflow as tf

        model = _get_yamnet()
        y, _ = librosa.load(file_path, sr=16000, mono=True, duration=10)
        scores, _, _ = model(y)
        class_scores = tf.reduce_mean(scores, axis=0).numpy()
        top_idx = int(np.argmax(class_scores))

        if top_idx in _YAMNET_SPEECH_CLASSES:
            return "speech"
        if top_idx in _YAMNET_MUSIC_CLASSES:
            return "music"
        if top_idx in _YAMNET_CROWD_CLASSES:
            return "crowd"
        return "noise"
    except Exception:
        return None


# FINANCE TOPIC TRANSITION PHRASES
_TOPIC_TRANSITIONS: list[str] = [
    "moving to",
    "turning to",
    "let me now",
    "on the balance sheet",
    "cash flow",
    "guidance",
    "next question",
    "question and answer",
    "q&a",
    "opening remarks",
    "prepared remarks",
    "i'll turn it over",
    "income statement",
    "operating expenses",
    "earnings per share",
    "capital expenditure",
    "free cash flow",
    "outlook",
    "fiscal year",
]

# EARNINGS CALL SECTION MARKERS
_PREPARED_REMARKS_KW = [
    "good morning",
    "good afternoon",
    "good evening",
    "thank you for joining",
    "welcome to",
    "let me walk you through",
    "our results",
    "quarterly results",
]
_QA_KW = [
    "question and answer",
    "q&a session",
    "open up for questions",
    "first question",
    "next question",
    "your question please",
    "please go ahead",
]
_OPERATOR_KW = ["operator:", "operator,", "this is the operator"]

# SPEAKER ROLE KEYWORD MAP
_SPEAKER_ROLE_MAP: dict[str, str] = {
    "ceo": "CEO",
    "chief executive": "CEO",
    "cfo": "CFO",
    "chief financial": "CFO",
    "cto": "CTO",
    "chief technology": "CTO",
    "coo": "COO",
    "chief operating": "COO",
    "analyst": "analyst",
    "research": "analyst",
    "operator": "operator",
    "moderator": "moderator",
}

# FINANCE ENTITY REGEXES
_FIN_AMOUNT_RE = re.compile(
    r'\$[\d,]+\.?\d*\s?[BMKTbmkt]?|\b\d[\d,.]*\s?(?:billion|million|thousand)\b',
    re.IGNORECASE,
)
_FIN_PCT_RE = re.compile(r'[\d.]+\s?(?:percent|%)')
_FIN_TICKER_RE = re.compile(r'\b[A-Z]{2,5}\b')
_FIN_QUARTER_RE = re.compile(r'Q[1-4]\s?(?:FY)?\s?\d{2,4}|H[12]\s?\d{4}', re.IGNORECASE)


def _infer_speaker_role(speaker_name: str | None) -> str:
    if not speaker_name:
        return "unknown"
    lower = speaker_name.lower()
    for keyword, role in _SPEAKER_ROLE_MAP.items():
        if keyword in lower:
            return role
    return "executive"


def _extract_finance_entities(text: str) -> dict[str, list[str]]:
    return {
        "amounts": _FIN_AMOUNT_RE.findall(text)[:20],
        "percentages": _FIN_PCT_RE.findall(text)[:20],
        "tickers": list({t for t in _FIN_TICKER_RE.findall(text) if len(t) >= 2})[:20],
        "dates": _FIN_QUARTER_RE.findall(text)[:10],
    }


def _detect_call_section(text: str, prior_section: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in _OPERATOR_KW):
        return "operator"
    if any(kw in lower for kw in _QA_KW):
        return "qa_session"
    if any(kw in lower for kw in _PREPARED_REMARKS_KW):
        return "prepared_remarks"
    return prior_section


def _detect_topic_section(text: str) -> str | None:
    lower = text.lower()
    for phrase in _TOPIC_TRANSITIONS:
        if phrase in lower:
            return phrase.replace(" ", "_")
    return None


def _audio_is_earnings_call(segments: list[dict[str, Any]]) -> bool:
    sample_text = " ".join(s["text"] for s in segments[:30]).lower()
    operator_turns = sum(1 for s in segments if "operator" in (s.get("text") or "").lower())
    return (
        operator_turns >= 5
        or "earnings call" in sample_text
        or "quarterly results" in sample_text
        or "conference call" in sample_text
    )


# DIARIZATION VIA PYANNOTE


def _diarize(file_path: str, session_id: str) -> dict[tuple[float, float], str]:
    speaker_map: dict[tuple[float, float], str] = {}
    if not settings.DIARIZATION_ENABLED:
        return speaker_map
    try:
        if not settings.HF_TOKEN:
            logger.warning(event="diarization_skipped_no_hf_token", session_id=session_id)
            return speaker_map
        from app.core.model_loader import model_loader

        pipeline = model_loader.get_diarizer()
        if pipeline is None:
            return speaker_map
        diarization = pipeline(file_path)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_map[(round(turn.start, 2), round(turn.end, 2))] = speaker
    except Exception as exc:
        logger.warning(event="diarization_failed", error=str(exc), session_id=session_id)
    return speaker_map


def _get_speaker(
    seg_start: float,
    seg_end: float,
    speaker_map: dict[tuple[float, float], str],
) -> str | None:
    for (start, end), speaker in speaker_map.items():
        if seg_start >= start and seg_end <= end:
            return speaker
    return None


# LONG AUDIO CHUNKING — SPLIT INTO 30-MIN SEGMENTS


def _chunk_audio_file(file_path: str, preloaded: Any | None = None) -> list[str]:
    from pydub import AudioSegment

    chunk_duration_ms = settings.AUDIO_CHUNK_DURATION_SEC * 1000
    audio = preloaded if preloaded is not None else AudioSegment.from_file(file_path)
    total_ms = len(audio)

    if total_ms <= chunk_duration_ms:
        return [file_path]

    from app.utils.paths import resolved_temp_dir

    tmp_dir = resolved_temp_dir() / f"audio_chunks_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []

    for i, start_ms in enumerate(range(0, total_ms, chunk_duration_ms)):
        segment = audio[start_ms : start_ms + chunk_duration_ms]
        chunk_path = str(tmp_dir / f"chunk_{i}.wav")
        segment.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    logger.info(
        event="audio_long_file_chunked",
        total_chunks=len(chunks),
        total_duration_sec=round(total_ms / 1000, 1),
        file=os.path.basename(file_path),
    )
    return chunks


# TRANSCRIBE SINGLE FILE


def _transcribe_file(
    file_path: str,
    session_id: str,
) -> tuple[Any, Any, float]:
    from app.core.model_loader import model_loader

    whisper = model_loader.get_whisper()
    t_start = time.time()
    segments_iter, info = whisper.transcribe(
        file_path,
        language=None,
        beam_size=2,
        word_timestamps=False,
        vad_filter=True,  # Silero VAD skips silence — 20-40% faster on speech+pauses
        condition_on_previous_text=False,  # prevents hallucination loops on long recordings
    )
    latency = round(time.time() - t_start, 2)
    return segments_iter, info, latency


def _transcribe_chunk_eager(
    chunk_file: str,
    session_id: str,
) -> tuple[list[Any], Any, float, float]:
    """Materialize all segments within the calling thread (GPU GIL released during CTranslate2 ops)."""
    segments_iter, info, latency = _transcribe_file(chunk_file, session_id)
    chunk_duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments = list(segments_iter)
    return segments, info, latency, chunk_duration


# MAIN INGEST


def ingest(file_path: str, session_id: str) -> list[IngestedDocument]:
    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    try:
        path = _safe_resolve(file_path)
    except ValueError as exc:
        raise ValueError(str(exc))

    _validate_audio_file(path)
    _check_disk_space(path)

    # DRM CHECK
    if _is_drm_protected(path):
        raise ValueError(f"DRM_PROTECTED_AUDIO: {path.name}")

    ext = path.suffix.lower()
    file_size = path.stat().st_size
    start_time = time.time()
    source_name = path.name
    source_path = str(path)
    doc_id = str(uuid.uuid4())
    file_hash = _sha256_check(path)
    mime_type = _detect_mime(path)

    logger.info(
        event="audio_ingest_start",
        file=source_name,
        ext=ext,
        size=file_size,
        mime=mime_type,
        session_id=session_id,
    )

    temp_chunk_paths: list[str] = []

    try:
        # LOAD AND VALIDATE AUDIO
        audio = _load_audio(file_path)

        if audio.duration_seconds <= 0:
            raise ValueError("INVALID_AUDIO_DURATION")
        if audio.frame_rate <= 0:
            raise ValueError("INVALID_SAMPLE_RATE")

        # DURATION TOO SHORT — SKIP TRANSCRIPTION
        if audio.duration_seconds < 1.0:
            raise ValueError(
                f"AUDIO_TOO_SHORT: duration {round(audio.duration_seconds, 2)}s is less than 1 second"
            )

        duration_total = audio.duration_seconds
        channels = audio.channels
        frame_rate = audio.frame_rate
        original_frame_rate = frame_rate

        # SILENT AUDIO CHECK
        snr = _estimate_snr(audio)
        snr_degraded = snr < settings.AUDIO_SNR_THRESHOLD_DB

        if audio.dBFS < -60.0:
            raise ValueError("EMPTY_CONTENT: silent audio below -60 dBFS threshold")

        if snr_degraded:
            logger.warning(
                event="audio_low_snr",
                snr=snr,
                threshold=settings.AUDIO_SNR_THRESHOLD_DB,
                file=source_name,
                session_id=session_id,
            )

        # CLIPPING DETECTION
        clipping_detected = _detect_clipping(file_path)
        if clipping_detected:
            logger.warning(
                event="audio_clipping_detected",
                file=source_name,
                session_id=session_id,
            )

        # MIXDOWN AND RESAMPLE
        audio = _to_mono(audio)
        audio = _resample(audio)

        # SILENCE RANGES
        silent_ranges = _detect_silent_ranges(audio)

        # ID3 METADATA
        id3_meta = _extract_id3(file_path)
        # Sanitize id3 metadata text fields (untrusted data from audio file tags)
        try:
            from app.guardrails.input_guard import sanitize as _gs

            id3_meta = {k: _gs(str(v), surface="audio_id3_ingest") for k, v in id3_meta.items()}
        except Exception:
            pass

        # NOISE CLASSIFICATION
        noise_class = _classify_noise(file_path)

        # MUSIC-ONLY SKIP DIARIZATION
        is_music = noise_class == "music"

        # LONG AUDIO CHUNKING (> CHUNK_DURATION_SEC) — pass preloaded audio to avoid re-reading original
        chunk_files = _chunk_audio_file(file_path, preloaded=audio)
        temp_chunk_paths = [c for c in chunk_files if c != file_path]
        is_chunked = len(chunk_files) > 1

        # DIARIZATION + TRANSCRIPTION — run concurrently (both are independent on the same file).
        # Diarization runs in its own thread while transcription threads process audio chunks.
        # This makes diarization time effectively free on top of transcription time.
        speaker_map: dict[tuple[float, float], str] = {}
        diarize_future = None
        diarize_pool = None
        if not is_music and settings.DIARIZATION_ENABLED:
            diarize_pool = ThreadPoolExecutor(max_workers=1)
            diarize_future = diarize_pool.submit(_diarize, file_path, session_id)

        # UNIVERSAL METADATA
        metadata = build_universal_metadata(
            str(path),
            Modality.AUDIO,
            mime_type,
            file_size_bytes=file_size,
            checksum_sha256=file_hash,
            language="unknown",
            chunk_count=0,
            tags=[],
            custom_fields={
                "duration_seconds": round(duration_total, 2),
                "channels": channels,
                "original_frame_rate": original_frame_rate,
                "target_frame_rate": settings.AUDIO_SAMPLE_RATE,
                "snr": snr,
                "snr_degraded": snr_degraded,
                "clipping_detected": clipping_detected,
                "noise_class": noise_class,
                "is_music": is_music,
                "diarization_enabled": settings.DIARIZATION_ENABLED and not is_music,
                "speaker_count": len(set(speaker_map.values())),
                "id3": id3_meta,
                "checksum_sha256": file_hash,
            },
        )

        # TRANSCRIPTION — PARALLEL ACROSS CHUNKS
        all_segments: list[dict[str, Any]] = []

        # Submit all chunks concurrently; CTranslate2 releases GIL during CUDA ops,
        # so two Whisper large-v3 instances (×1.55 GB each) fit safely on A10G 24 GB.
        chunk_results: list[tuple[int, list[Any], Any, float, float]] = []
        with ThreadPoolExecutor(max_workers=settings.AUDIO_TRANSCRIPTION_WORKERS) as pool:
            futures = {
                pool.submit(_transcribe_chunk_eager, chunk_file, session_id): chunk_idx
                for chunk_idx, chunk_file in enumerate(chunk_files)
            }
            for fut, chunk_idx in futures.items():
                try:
                    segs, info, transcribe_latency, chunk_duration = fut.result()
                    chunk_results.append(
                        (chunk_idx, segs, info, transcribe_latency, chunk_duration)
                    )
                except Exception as exc:
                    logger.error(
                        event="audio_chunk_transcription_failed",
                        chunk=chunk_idx,
                        file=source_name,
                        error=str(exc),
                        session_id=session_id,
                    )

        # Collect diarization result now that transcription is done.
        if diarize_future is not None:
            try:
                speaker_map = diarize_future.result()
            except Exception as exc:
                logger.warning(event="diarization_failed", error=str(exc), session_id=session_id)
            finally:
                if diarize_pool is not None:
                    diarize_pool.shutdown(wait=False)

        # Restore chronological order; global_offset must be computed in chunk order.
        chunk_results.sort(key=lambda x: x[0])
        global_offset = 0.0
        for chunk_idx, raw_segs, info, transcribe_latency, chunk_duration in chunk_results:
            language = getattr(info, "language", None)

            rtf = transcribe_latency / max(chunk_duration, 1e-6)
            if rtf > settings.LATENCY_TARGET_AUDIO_RTF:
                logger.warning(
                    event="audio_rtf_exceeded",
                    rtf=round(rtf, 3),
                    target=settings.LATENCY_TARGET_AUDIO_RTF,
                    chunk=chunk_idx,
                    file=source_name,
                    session_id=session_id,
                )

            seg_count = 0
            for seg in raw_segs:
                if seg_count >= settings.MAX_AUDIO_SEGMENTS:
                    logger.warning(
                        event="audio_segment_limit_reached",
                        chunk=chunk_idx,
                        session_id=session_id,
                    )
                    break

                raw_text = (getattr(seg, "text", "") or "").strip()
                seg_start = float(getattr(seg, "start", 0.0)) + global_offset
                seg_end = float(getattr(seg, "end", seg_start)) + global_offset
                avg_logprob = getattr(seg, "avg_logprob", None)
                no_speech_prob = getattr(seg, "no_speech_prob", None)

                if not raw_text or seg_end <= seg_start:
                    continue

                if no_speech_prob is not None and no_speech_prob > 0.8:
                    logger.debug(
                        event="audio_segment_skipped_no_speech",
                        no_speech_prob=no_speech_prob,
                        chunk=chunk_idx,
                    )
                    continue

                all_segments.append(
                    {
                        "text": raw_text,
                        "start": seg_start,
                        "end": seg_end,
                        "avg_logprob": avg_logprob,
                        "no_speech_prob": no_speech_prob,
                        "language": language,
                        "chunk_idx": chunk_idx,
                    }
                )
                seg_count += 1

            global_offset += chunk_duration

        if not all_segments:
            raise ValueError("NO_VALID_AUDIO_SEGMENTS")

        # DETECT FINAL LANGUAGE FROM FIRST SEGMENT
        final_language = all_segments[0].get("language") if all_segments else None

        # EARNINGS CALL METADATA — document-level, computed once
        is_earnings_call = _audio_is_earnings_call(all_segments)
        current_call_section: str = "prepared_remarks"

        # BUILD INGESTED DOCUMENTS
        documents: list[IngestedDocument] = []
        global_idx = 0

        for seg_data in all_segments:
            raw_text = seg_data["text"]
            seg_start = seg_data["start"]
            seg_end = seg_data["end"]
            avg_logprob = seg_data["avg_logprob"]
            no_speech_prob = seg_data["no_speech_prob"]
            language = seg_data.get("language") or final_language

            duration = seg_end - seg_start
            inaudible = _is_inaudible(seg_start, silent_ranges)
            confidence = _compute_confidence(avg_logprob)
            risk = _hallucination_risk(confidence, no_speech_prob)
            speed_bad = _speed_flag(raw_text, duration)

            if inaudible:
                text = "[INAUDIBLE]"
            else:
                text = _apply_domain_vocab(raw_text)
                text = _strip_fillers(text)
                text = _normalize_punctuation(text)
                _text_norm = redact_pii(normalize_text(text))
                try:
                    from app.guardrails.input_guard import sanitize as _gs

                    text = _gs(_text_norm, surface="audio_ingest")
                except Exception as e:
                    logger.warning(event="sanitize_failed", surface="audio_ingest", error=str(e))
                    text = _text_norm

            if not text or len(text.strip()) < 2:
                continue

            speaker = _get_speaker(seg_start, seg_end, speaker_map)
            # Sanitize speaker name (diarization model output is untrusted)
            if speaker:
                try:
                    from app.guardrails.input_guard import sanitize as _gs

                    speaker = _gs(speaker, surface="audio_speaker_ingest")
                except Exception:
                    pass
            speaker_role = _infer_speaker_role(speaker)
            finance_ents = _extract_finance_entities(text) if not inaudible else {}
            current_call_section = _detect_call_section(text, current_call_section)
            topic_section = _detect_topic_section(text)

            doc = IngestedDocument(
                text=text,
                modality="audio",
                subtype="speech",
                source_type="audio",
                source=source_name,
                chunk_id=global_idx,
                metadata=metadata,
                structure={
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "file_hash": file_hash,
                    "checksum_sha256": file_hash,
                    "source_path": source_path,
                    "segment_index": global_idx,
                    "start_time": round(seg_start, 3),
                    "end_time": round(seg_end, 3),
                    "duration_sec": round(duration, 3),
                    "timestamp_start": round(seg_start, 2),
                    "timestamp_end": round(seg_end, 2),
                    "duration": round(duration, 2),
                    "total_duration": round(duration_total, 2),
                    "language": language,
                    "confidence": confidence,
                    "no_speech_prob": no_speech_prob,
                    "avg_logprob": avg_logprob,
                    "hallucination_risk": risk,
                    "speed_corrupted": speed_bad,
                    "inaudible": inaudible,
                    "snr": snr,
                    "snr_degraded": snr_degraded,
                    "clipping_detected": clipping_detected,
                    "channels": channels,
                    "frame_rate": frame_rate,
                    "speaker": speaker,
                    "speaker_role": speaker_role,
                    "noise_class": noise_class,
                    "is_earnings_call": is_earnings_call,
                    "call_section": current_call_section,
                    "topic_section": topic_section,
                    "finance_entities": finance_ents,
                    "content_type": "audio_speech_segment",
                    "embedding_space": "text",
                    "ingestion_time": time.time(),
                    "id3": id3_meta,
                },
                extra_metadata={
                    "modality_weight": 1.1,
                    "importance_score": confidence,
                    "data_quality_score": confidence,
                    "pii_redacted": settings.PII_DETECTION_ENABLED,
                    "prompt_injection_sanitized": True,
                },
            ).finalize()

            documents.append(doc)
            global_idx += 1

        if not documents:
            raise ValueError("NO_VALID_AUDIO_DOCUMENTS_AFTER_FILTERING")

        metadata.chunk_count = len(documents)
        latency = round(time.time() - start_time, 2)

        if latency > settings.SLO_AUDIO_P95_MS / 1000:
            logger.warning(
                event="audio_slo_exceeded",
                latency_sec=latency,
                target_sec=settings.SLO_AUDIO_P95_MS / 1000,
                file=source_name,
                session_id=session_id,
            )

        logger.info(
            event="audio_ingest_success",
            file=source_name,
            segments=len(documents),
            language=final_language,
            duration=round(duration_total, 2),
            snr=snr,
            clipping=clipping_detected,
            speaker_count=len(set(speaker_map.values())),
            noise_class=noise_class,
            chunked=is_chunked,
            latency=latency,
            session_id=session_id,
        )

        return documents

    except (EmptyFileError, FileTooLargeError, UnsupportedMimeError):
        raise
    except Exception as exc:
        logger.error(
            event="audio_ingest_failed",
            file=source_name,
            session_id=session_id,
            error=str(exc),
            latency=round(time.time() - start_time, 2),
        )
        raise
    finally:
        # CLEANUP TEMP CHUNK FILES
        for tmp_path in temp_chunk_paths:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    parent = os.path.dirname(tmp_path)
                    if os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
            except Exception as cleanup_exc:
                logger.warning(
                    event="audio_chunk_cleanup_failed",
                    path=tmp_path,
                    error=str(cleanup_exc),
                )


# ASYNC WRAPPER


async def async_ingest(file_path: str, session_id: str) -> list[IngestedDocument]:
    return await asyncio.to_thread(ingest, file_path, session_id)


# ─── Phase 1: AudioIngestor ────────────────────────────────────────────────────

from prometheus_client import Counter

from app.ingestion.base_ingest import BaseIngestor
from app.ingestion.schema import RawExtract, UniversalMetadata

_EXTRACTS_TOTAL = Counter("magik_audio_extracts_total", "Total extracts produced by audio ingestor")
_EXTRACT_ERRORS = Counter("magik_audio_extract_errors_total", "Errors in audio ingestor")


class AudioIngestor(BaseIngestor):
    """Validates and loads audio files → List[RawExtract].

    Phase 1 responsibility: DRM check, pydub load, ffmpeg repair, duration check.
    Does NOT transcribe or diarize. The chunker (Phase 2) runs Whisper + pyannote.
    """

    def health_check(self) -> dict:
        return {
            "modality": "audio",
            "status": "ok",
            "class": self.__class__.__name__,
        }

    async def extract(
        self,
        path: Path,
        metadata: UniversalMetadata,
    ) -> list[RawExtract]:
        source = path.name
        suffix = path.suffix.lower()
        logger.info(
            event="extraction_start", modality="audio", file=str(path), size=path.stat().st_size
        )
        try:
            if suffix not in SUPPORTED_AUDIO_FORMATS:
                raise UnsupportedMimeError(f"UNSUPPORTED_AUDIO_FORMAT: {suffix}")

            file_size = path.stat().st_size
            if file_size == 0:
                raise EmptyFileError(str(path))
            if file_size > settings.MAX_FILE_SIZE_AUDIO:
                raise FileTooLargeError(f"FILE_TOO_LARGE: {file_size}")

            _check_disk_space(path)

            # DRM / encryption check (can't process DRM audio)
            try:
                from pydub import AudioSegment

                audio = AudioSegment.from_file(str(path))
                duration_s = len(audio) / 1000.0
            except Exception as exc:
                if "DRM" in str(exc).upper() or "encrypted" in str(exc).lower():
                    raise ValueError(f"DRM_PROTECTED_AUDIO: {path.name}")
                # Try ffmpeg repair path
                try:
                    import subprocess
                    import tempfile

                    tmp = tempfile.mktemp(suffix=".wav")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", tmp],
                        capture_output=True,
                        timeout=120,
                    )
                    if not Path(tmp).exists() or Path(tmp).stat().st_size == 0:
                        raise ValueError(f"AUDIO_REPAIR_FAILED: {path.name}")
                    from pydub import AudioSegment

                    audio = AudioSegment.from_wav(tmp)
                    duration_s = len(audio) / 1000.0
                except Exception as repair_exc:
                    raise ValueError(f"AUDIO_LOAD_FAILED: {repair_exc}")

            if duration_s < 0.5:
                raise ValueError(f"AUDIO_TOO_SHORT: {duration_s:.2f}s")
            if duration_s > settings.MAX_AUDIO_DURATION_SEC:
                raise ValueError(f"AUDIO_TOO_LONG: {duration_s:.0f}s")

            # Quality signals — run on the raw audio before mono/resample conversion.
            snr = _estimate_snr(audio)
            snr_degraded = snr < settings.AUDIO_SNR_THRESHOLD_DB
            if audio.dBFS < -60.0:
                raise ValueError("EMPTY_CONTENT: silent audio below -60 dBFS threshold")
            if snr_degraded:
                logger.warning(
                    event="audio_low_snr",
                    snr=snr,
                    threshold=settings.AUDIO_SNR_THRESHOLD_DB,
                    file=source,
                )
            clipping_detected = _detect_clipping(str(path))
            if clipping_detected:
                logger.warning(event="audio_clipping_detected", file=source)

            # Export as 16kHz mono WAV bytes for chunker
            import io as _io

            wav_buf = _io.BytesIO()
            audio.set_frame_rate(16000).set_channels(1).export(wav_buf, format="wav")
            audio_bytes = wav_buf.getvalue()

            _EXTRACTS_TOTAL.inc(1)
            logger.info(event="extraction_complete", modality="audio", file=str(path), extracts=1)
            return [
                RawExtract(
                    text="",
                    extract_type="audio_raw",
                    timestamp_start=0.0,
                    timestamp_end=duration_s,
                    raw_source_ref=f"audio:{path.name}",
                    raw_bytes=audio_bytes,
                    extra={
                        "duration_seconds": duration_s,
                        "file_size": file_size,
                        "format": suffix.lstrip("."),
                        "sample_rate": 16000,
                        "channels": 1,
                        "snr": snr,
                        "snr_degraded": snr_degraded,
                        "clipping_detected": clipping_detected,
                    },
                )
            ]
        except Exception as _exc:
            _EXTRACT_ERRORS.inc()
            logger.error(
                event="extraction_failed", modality="audio", source=source, error=str(_exc)
            )
            raise


async def ingest_audio_full(file_path: str, session_id: str) -> list[IngestedDocument]:
    """Production audio ingestion: AudioIngestor.extract() → AudioChunker.chunk().

    Replaces the backward-compat ingest() as the INGESTION_HANDLERS entry.
    Produces modality='mp3' docs with full Phase 1.6/2.6/3.6 metadata:
    start_timestamp, end_timestamp, duration_seconds, speaker_label, speaker_name,
    speaker_role, call_section, topic_section, transcript, finance_entities,
    word_count, token_count, is_question, is_answer.
    """
    from app.chunking import chunk_raw_extracts
    from app.ingestion.schema import UniversalMetadata

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError("EMPTY_FILE")
    if file_size > settings.MAX_FILE_SIZE_AUDIO:
        raise ValueError(f"FILE_TOO_LARGE: {file_size}")

    meta = UniversalMetadata(
        source_path=str(path.resolve()),
        modality="audio",
        file_size_bytes=file_size,
        custom_fields={"session_id": session_id},
    )

    ingestor = AudioIngestor()
    extracts = await ingestor.extract(path, meta)

    if not extracts:
        raise ValueError("NO_EXTRACTS_PRODUCED")

    docs = chunk_raw_extracts(extracts, meta, "audio")

    for doc in docs:
        struct = getattr(doc, "structure", None)
        if struct is not None:
            if struct.get("session_id") in (None, "default"):
                struct["session_id"] = session_id

    logger.info(
        event="ingest_audio_full_complete",
        file=path.name,
        extracts=len(extracts),
        docs=len(docs),
        session_id=session_id,
    )
    return docs
