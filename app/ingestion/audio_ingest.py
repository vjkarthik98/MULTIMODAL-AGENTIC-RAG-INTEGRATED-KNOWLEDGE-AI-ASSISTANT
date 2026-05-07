import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydub import AudioSegment
from pydub import silence as pydub_silence

from app.core.config import settings
from app.core.model_loader import model_loader
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


# SUPPORTED FORMATS

SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


# HASH

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# AUDIO VALIDATION

def _validate_audio(file_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(file_path)

    if audio.duration_seconds <= 0:
        raise ValueError("INVALID_AUDIO_DURATION")

    if audio.frame_rate <= 0:
        raise ValueError("INVALID_SAMPLE_RATE")

    return audio


# SNR ESTIMATION

def _estimate_snr(audio: AudioSegment) -> float:
    try:
        dbfs = audio.dBFS
        if dbfs == float("-inf"):
            return 0.0
        # Rough SNR proxy: dBFS offset from silence floor (-60 dBFS)
        snr = max(0.0, dbfs + 60.0)
        return round(snr, 2)
    except Exception:
        return 0.0


# SILENCE DETECTION

def _detect_silent_ranges(audio: AudioSegment) -> List[Tuple[int, int]]:
    try:
        return pydub_silence.detect_silence(
            audio,
            min_silence_len=settings.AUDIO_SILENCE_GAP_MS,
            silence_thresh=-40,
        )
    except Exception as e:
        logger.warning(event="silence_detection_failed", error=str(e))
        return []


# DOMAIN VOCAB CORRECTION

def _apply_domain_vocab(text: str) -> str:
    if not settings.WHISPER_DOMAIN_VOCAB:
        return text

    lower = text.lower()
    corrected = text

    for term in settings.WHISPER_DOMAIN_VOCAB:
        if term.lower() in lower:
            import re
            corrected = re.sub(re.escape(term), term, corrected, flags=re.IGNORECASE)

    return corrected


# CONFIDENCE FROM LOGPROB

def _compute_confidence(avg_logprob: Optional[float]) -> float:
    if avg_logprob is None:
        return 1.0
    return round(max(0.0, min(1.0, 1.0 + avg_logprob)), 4)


# HALLUCINATION RISK

def _hallucination_risk(confidence: float, no_speech_prob: Optional[float]) -> str:
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


# SEGMENT IN SILENT RANGE

def _is_inaudible(start_sec: float, silent_ranges: List[Tuple[int, int]]) -> bool:
    start_ms = start_sec * 1000
    return any(s <= start_ms <= e for s, e in silent_ranges)


# MAIN

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"UNSUPPORTED_AUDIO_FORMAT: {ext}")

    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError("EMPTY_FILE")

    if file_size > settings.MAX_FILE_SIZE_AUDIO:
        raise ValueError(
            f"FILE_TOO_LARGE: {file_size} bytes exceeds {settings.MAX_FILE_SIZE_AUDIO} bytes"
        )

    start_time  = time.time()
    source_name = path.name
    source_path = str(path.resolve())
    doc_id      = str(uuid.uuid4())
    file_hash   = _file_hash(file_path)

    logger.info(
        event="audio_ingest_start",
        file=source_name,
        size=file_size,
        session_id=session_id,
    )

    try:
        # VALIDATE
        audio          = _validate_audio(file_path)
        duration_total = audio.duration_seconds
        channels       = audio.channels
        frame_rate     = audio.frame_rate

        # SNR
        snr = _estimate_snr(audio)
        snr_degraded = snr < settings.AUDIO_SNR_THRESHOLD_DB

        if snr_degraded:
            logger.warning(
                event="audio_low_snr",
                snr=snr,
                threshold=settings.AUDIO_SNR_THRESHOLD_DB,
                file=source_name,
            )

        # SILENCE DETECTION
        silent_ranges = _detect_silent_ranges(audio)

        # TRANSCRIPTION
        whisper                = model_loader.get_whisper()
        t_transcribe           = time.time()
        segments_iter, info    = whisper.transcribe(
            file_path,
            language=None,
            beam_size=5,
            word_timestamps=False,
        )
        transcribe_latency     = round(time.time() - t_transcribe, 2)
        rtf                    = transcribe_latency / max(duration_total, 1e-6)

        language = getattr(info, "language", None)

        if rtf > settings.LATENCY_TARGET_AUDIO_RTF:
            logger.warning(
                event="audio_rtf_exceeded",
                rtf=round(rtf, 3),
                target=settings.LATENCY_TARGET_AUDIO_RTF,
                file=source_name,
            )

        documents: List[IngestedDocument] = []

        for idx, segment in enumerate(segments_iter):

            if idx >= settings.MAX_AUDIO_SEGMENTS:
                logger.warning(event="audio_segment_limit_reached", session_id=session_id)
                break

            raw_text       = (getattr(segment, "text", "") or "").strip()
            seg_start      = float(getattr(segment, "start", 0.0))
            seg_end        = float(getattr(segment, "end", seg_start))
            avg_logprob    = getattr(segment, "avg_logprob", None)
            no_speech_prob = getattr(segment, "no_speech_prob", None)

            if not raw_text or seg_end <= seg_start:
                continue

            # SKIP HIGH NO-SPEECH PROBABILITY SEGMENTS
            if no_speech_prob is not None and no_speech_prob > 0.8:
                logger.warning(
                    event="audio_segment_skipped_no_speech",
                    idx=idx,
                    no_speech_prob=no_speech_prob,
                )
                continue

            duration   = seg_end - seg_start
            inaudible  = _is_inaudible(seg_start, silent_ranges)
            text       = "[INAUDIBLE]" if inaudible else _apply_domain_vocab(raw_text)
            confidence = _compute_confidence(avg_logprob)
            risk       = _hallucination_risk(confidence, no_speech_prob)
            speed_bad  = _speed_flag(text, duration)

            doc = IngestedDocument(
                text=text,
                modality="audio",
                subtype="speech",
                source_type="audio",
                source=source_name,
                chunk_id=idx,
                structure={
                    "doc_id":           doc_id,
                    "session_id":       session_id,
                    "file_hash":        file_hash,
                    "source_path":      source_path,
                    "segment_index":    idx,
                    "timestamp_start":  round(seg_start, 2),
                    "timestamp_end":    round(seg_end, 2),
                    "duration":         round(duration, 2),
                    "total_duration":   round(duration_total, 2),
                    "language":         language,
                    "confidence":       confidence,
                    "no_speech_prob":   no_speech_prob,
                    "avg_logprob":      avg_logprob,
                    "hallucination_risk": risk,
                    "speed_corrupted":  speed_bad,
                    "inaudible":        inaudible,
                    "snr":              snr,
                    "snr_degraded":     snr_degraded,
                    "channels":         channels,
                    "frame_rate":       frame_rate,
                    "content_type":     "audio_speech_segment",
                    "ingestion_time":   time.time(),
                },
                extra_metadata={
                    "modality_weight":    1.1,
                    "importance_score":   confidence,
                    "data_quality_score": confidence,
                },
            ).finalize()

            documents.append(doc)

        if not documents:
            raise ValueError("NO_VALID_AUDIO_SEGMENTS")

        latency = round(time.time() - start_time, 2)

        logger.info(
            event="audio_ingest_success",
            file=source_name,
            segments=len(documents),
            language=language,
            duration=round(duration_total, 2),
            snr=snr,
            rtf=round(rtf, 3),
            latency=latency,
            session_id=session_id,
        )

        return documents

    except Exception as e:
        logger.error(
            event="audio_ingest_failed",
            file=source_name,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start_time, 2),
        )
        raise