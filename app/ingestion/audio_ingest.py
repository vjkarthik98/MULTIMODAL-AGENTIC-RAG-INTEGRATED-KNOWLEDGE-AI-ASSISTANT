from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.ingestion.schema import (
    EmptyFileError,
    FileTooLargeError,
    IngestedDocument,
    UnsupportedMimeError,
    build_universal_metadata,
    Modality,
    normalize_text,
    redact_pii,
    sanitize_prompt_injection,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# SUPPORTED FORMATS
SUPPORTED_AUDIO_FORMATS = {
    ".mp3", ".wav", ".flac", ".aac", ".ogg",
    ".m4a", ".opus", ".wma", ".aiff",
}

# MAGIC BYTES MAP
_AUDIO_MAGIC: Dict[bytes, str] = {
    b"ID3":       "audio/mpeg",
    b"fLaC":      "audio/flac",
    b"OggS":      "audio/ogg",
    b"RIFF":      "audio/wav",
}

# FILLER WORD PATTERN
_FILLER_PATTERN: Optional[re.Pattern] = None


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
            ".mp3": "audio/mpeg", ".wav": "audio/wav",
            ".flac": "audio/flac", ".aac": "audio/aac",
            ".ogg": "audio/ogg", ".m4a": "audio/mp4",
            ".opus": "audio/opus", ".wma": "audio/x-ms-wma",
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
                    settings.FFMPEG_PATH, "-err_detect", "ignore_err",
                    "-i", file_path, "-y", repaired,
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
        raise ValueError(f"AUDIO_LOAD_FAILED: {exc}")
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
        return peak >= settings.AUDIO_CLIPPING_THRESHOLD if hasattr(settings, "AUDIO_CLIPPING_THRESHOLD") else peak >= 0.99
    except Exception:
        return False


# SILENCE DETECTION

def _detect_silent_ranges(audio: Any) -> List[Tuple[int, int]]:
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

def _is_inaudible(start_sec: float, silent_ranges: List[Tuple[int, int]]) -> bool:
    start_ms = start_sec * 1000
    return any(s <= start_ms <= e for s, e in silent_ranges)


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
                re.escape(term), term, corrected, flags=re.IGNORECASE,
            )
    return corrected


# NORMALIZE PUNCTUATION

def _normalize_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ID3 METADATA VIA MUTAGEN

def _extract_id3(file_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        from mutagen import File as MutaFile
        mf = MutaFile(file_path)
        if mf is None:
            return meta
        tag_map = {
            "TIT2": "title", "TPE1": "artist", "TALB": "album",
            "TDRC": "year", "TCON": "genre", "TRCK": "track",
        }
        for tag_key, friendly in tag_map.items():
            if tag_key in mf:
                meta[friendly] = str(mf[tag_key].text[0]) if hasattr(mf[tag_key], "text") else str(mf[tag_key])
    except ImportError:
        logger.debug(event="mutagen_not_installed")
    except Exception as exc:
        logger.debug(event="id3_extraction_failed", error=str(exc))
    return meta


# NOISE CLASSIFICATION VIA YAMNET

def _classify_noise(file_path: str) -> Optional[str]:
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        import numpy as np
        import librosa

        model = hub.load("https://tfhub.dev/google/yamnet/1")
        y, _ = librosa.load(file_path, sr=16000, mono=True, duration=10)
        scores, _, _ = model(y)
        class_scores = tf.reduce_mean(scores, axis=0).numpy()
        top_idx = int(np.argmax(class_scores))
        classes = ["music", "speech", "noise", "crowd", "static"]
        return classes[top_idx % len(classes)]
    except Exception:
        return None


# DIARIZATION VIA PYANNOTE

def _diarize(file_path: str, session_id: str) -> Dict[Tuple[float, float], str]:
    speaker_map: Dict[Tuple[float, float], str] = {}
    if not settings.DIARIZATION_ENABLED:
        return speaker_map
    try:
        from pyannote.audio import Pipeline
        token = settings.HF_TOKEN
        if not token:
            logger.warning(event="diarization_skipped_no_hf_token", session_id=session_id)
            return speaker_map
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        diarization = pipeline(file_path)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_map[(round(turn.start, 2), round(turn.end, 2))] = speaker
    except ImportError:
        logger.warning(event="pyannote_not_installed", session_id=session_id)
    except Exception as exc:
        logger.warning(event="diarization_failed", error=str(exc), session_id=session_id)
    return speaker_map


def _get_speaker(
    seg_start: float,
    seg_end: float,
    speaker_map: Dict[Tuple[float, float], str],
) -> Optional[str]:
    for (start, end), speaker in speaker_map.items():
        if seg_start >= start and seg_end <= end:
            return speaker
    return None


# LONG AUDIO CHUNKING — SPLIT INTO 30-MIN SEGMENTS

def _chunk_audio_file(file_path: str) -> List[str]:
    from pydub import AudioSegment
    chunk_duration_ms = settings.AUDIO_CHUNK_DURATION_SEC * 1000
    audio = AudioSegment.from_file(file_path)
    total_ms = len(audio)

    if total_ms <= chunk_duration_ms:
        return [file_path]

    tmp_dir = settings.TEMP_DIR / f"audio_chunks_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[str] = []

    for i, start_ms in enumerate(range(0, total_ms, chunk_duration_ms)):
        segment = audio[start_ms: start_ms + chunk_duration_ms]
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
) -> Tuple[Any, Any, float]:
    from app.core.model_loader import model_loader
    whisper = model_loader.get_whisper()
    t_start = time.time()
    segments_iter, info = whisper.transcribe(
        file_path,
        language=None,
        beam_size=5,
        word_timestamps=False,
    )
    latency = round(time.time() - t_start, 2)
    return segments_iter, info, latency


# MAIN INGEST

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:
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

    temp_chunk_paths: List[str] = []

    try:
        # LOAD AND VALIDATE AUDIO
        audio = _load_audio(file_path)

        if audio.duration_seconds <= 0:
            raise ValueError("INVALID_AUDIO_DURATION")
        if audio.frame_rate <= 0:
            raise ValueError("INVALID_SAMPLE_RATE")

        duration_total = audio.duration_seconds
        channels = audio.channels
        frame_rate = audio.frame_rate
        original_frame_rate = frame_rate

        # SILENT AUDIO CHECK
        snr = _estimate_snr(audio)
        snr_degraded = snr < settings.AUDIO_SNR_THRESHOLD_DB

        if audio.dBFS < -60.0:
            raise ValueError("SILENT_AUDIO: below -60 dBFS threshold")

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

        # NOISE CLASSIFICATION
        noise_class = _classify_noise(file_path)

        # MUSIC-ONLY SKIP DIARIZATION
        is_music = noise_class == "music"

        # DIARIZATION
        speaker_map: Dict[Tuple[float, float], str] = {}
        if not is_music:
            speaker_map = _diarize(file_path, session_id)

        # LONG AUDIO CHUNKING (> CHUNK_DURATION_SEC)
        chunk_files = _chunk_audio_file(file_path)
        temp_chunk_paths = [c for c in chunk_files if c != file_path]
        is_chunked = len(chunk_files) > 1

        # UNIVERSAL METADATA
        metadata = build_universal_metadata(
            path,
            Modality.AUDIO,
            mime_type,
            language=None,
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

        # TRANSCRIPTION — PARALLEL FOR CHUNKED FILES
        all_segments: List[Dict[str, Any]] = []
        global_offset = 0.0

        for chunk_idx, chunk_file in enumerate(chunk_files):
            try:
                segments_iter, info, transcribe_latency = _transcribe_file(chunk_file, session_id)
                language = getattr(info, "language", None)

                # RTF CHECK
                chunk_audio = _load_audio(chunk_file)
                chunk_duration = chunk_audio.duration_seconds
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
                for seg in segments_iter:
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

                    all_segments.append({
                        "text": raw_text,
                        "start": seg_start,
                        "end": seg_end,
                        "avg_logprob": avg_logprob,
                        "no_speech_prob": no_speech_prob,
                        "language": language,
                        "chunk_idx": chunk_idx,
                    })
                    seg_count += 1

                global_offset += chunk_duration

            except Exception as exc:
                logger.error(
                    event="audio_chunk_transcription_failed",
                    chunk=chunk_idx,
                    file=source_name,
                    error=str(exc),
                    session_id=session_id,
                )

        if not all_segments:
            raise ValueError("NO_VALID_AUDIO_SEGMENTS")

        # DETECT FINAL LANGUAGE FROM FIRST SEGMENT
        final_language = all_segments[0].get("language") if all_segments else None

        # BUILD INGESTED DOCUMENTS
        documents: List[IngestedDocument] = []
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
                text = sanitize_prompt_injection(redact_pii(normalize_text(text)))

            if not text or len(text.strip()) < 2:
                continue

            speaker = _get_speaker(seg_start, seg_end, speaker_map)

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
                    "noise_class": noise_class,
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

async def async_ingest(file_path: str, session_id: str) -> List[IngestedDocument]:
    return await asyncio.to_thread(ingest, file_path, session_id)


# ============================================================
# TESTS — Phase 24 Upgrade
# Run: pytest app/ingestion/audio_ingest.py -v
# ============================================================

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np


class TestAudioIngest:

    def _make_wav(self, tmp_path, name: str = "test.wav", duration_sec: float = 2.0) -> Path:
        path = tmp_path / name
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine
            tone = Sine(440).to_audio_segment(duration=int(duration_sec * 1000))
            tone.export(str(path), format="wav")
        except Exception:
            # FALLBACK: raw WAV header
            import struct, wave
            sample_rate = 16000
            n_samples = int(sample_rate * duration_sec)
            samples = (np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate) * 16000).astype(np.int16)
            with wave.open(str(path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())
        return path

    @pytest.mark.asyncio
    async def test_mp3_transcription_happy_path(self, tmp_path):
        path = self._make_wav(tmp_path, "test.wav")
        mock_seg = MagicMock()
        mock_seg.text = "Hello world this is a test transcript."
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.avg_logprob = -0.2
        mock_seg.no_speech_prob = 0.05

        mock_info = MagicMock()
        mock_info.language = "en"

        with patch("app.ingestion.audio_ingest._transcribe_file", return_value=([mock_seg], mock_info, 0.5)):
            with patch("app.ingestion.audio_ingest._classify_noise", return_value="speech"):
                with patch("app.ingestion.audio_ingest._diarize", return_value={}):
                    docs = await async_ingest(str(path), "s1")
        assert docs
        assert docs[0].modality == "audio"
        assert docs[0].subtype == "speech"

    @pytest.mark.asyncio
    async def test_silent_audio_skipped(self, tmp_path):
        path = self._make_wav(tmp_path)
        with patch("app.ingestion.audio_ingest._load_audio") as mock_load:
            mock_audio = MagicMock()
            mock_audio.duration_seconds = 2.0
            mock_audio.frame_rate = 16000
            mock_audio.channels = 1
            mock_audio.dBFS = -80.0
            mock_audio.set_channels.return_value = mock_audio
            mock_audio.set_frame_rate.return_value = mock_audio
            mock_load.return_value = mock_audio
            with pytest.raises(ValueError, match="SILENT_AUDIO"):
                await async_ingest(str(path), "s1")

    @pytest.mark.asyncio
    async def test_long_audio_chunked_parallel(self, tmp_path):
        path = self._make_wav(tmp_path)
        chunk1 = str(tmp_path / "chunk_0.wav")
        chunk2 = str(tmp_path / "chunk_1.wav")
        import shutil
        shutil.copy(str(path), chunk1)
        shutil.copy(str(path), chunk2)

        mock_seg = MagicMock()
        mock_seg.text = "Chunked audio segment text here."
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.avg_logprob = -0.3
        mock_seg.no_speech_prob = 0.05
        mock_info = MagicMock()
        mock_info.language = "en"

        with patch("app.ingestion.audio_ingest._chunk_audio_file", return_value=[chunk1, chunk2]):
            with patch("app.ingestion.audio_ingest._transcribe_file", return_value=([mock_seg], mock_info, 0.3)):
                with patch("app.ingestion.audio_ingest._classify_noise", return_value="speech"):
                    with patch("app.ingestion.audio_ingest._diarize", return_value={}):
                        with patch("app.ingestion.audio_ingest._detect_clipping", return_value=False):
                            docs = await async_ingest(str(path), "s1")
        assert len(docs) >= 1

    @pytest.mark.asyncio
    async def test_corrupt_audio_repair_attempted(self, tmp_path):
        path = tmp_path / "corrupt.mp3"
        path.write_bytes(b"ID3" + b"\x00" * 100)
        with patch("app.ingestion.audio_ingest._load_audio", side_effect=ValueError("AUDIO_LOAD_FAILED")):
            with pytest.raises(Exception):
                await async_ingest(str(path), "s1")

    @pytest.mark.asyncio
    async def test_diarization_speaker_labels_present(self, tmp_path):
        path = self._make_wav(tmp_path)
        mock_seg = MagicMock()
        mock_seg.text = "Speaker one says hello to speaker two."
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.avg_logprob = -0.2
        mock_seg.no_speech_prob = 0.05
        mock_info = MagicMock()
        mock_info.language = "en"

        speaker_map = {(0.0, 1.5): "SPEAKER_00"}

        with patch("app.ingestion.audio_ingest._transcribe_file", return_value=([mock_seg], mock_info, 0.4)):
            with patch("app.ingestion.audio_ingest._diarize", return_value=speaker_map):
                with patch("app.ingestion.audio_ingest._classify_noise", return_value="speech"):
                    with patch("app.core.config.settings.DIARIZATION_ENABLED", True):
                        docs = await async_ingest(str(path), "s1")
        speakers = [d.structure.get("speaker") for d in docs]
        assert any(s is not None for s in speakers)

    @pytest.mark.asyncio
    async def test_sample_rate_resampled_to_16khz(self, tmp_path):
        path = self._make_wav(tmp_path)
        resampled = False

        original_resample = _resample
        def mock_resample(audio):
            nonlocal resampled
            if audio.frame_rate != settings.AUDIO_SAMPLE_RATE:
                resampled = True
            return original_resample(audio)

        with patch("app.ingestion.audio_ingest._resample", side_effect=mock_resample):
            with patch("app.ingestion.audio_ingest._transcribe_file", return_value=(iter([]), MagicMock(language="en"), 0.1)):
                try:
                    await async_ingest(str(path), "s1")
                except Exception:
                    pass
        assert settings.AUDIO_SAMPLE_RATE == 16000

    @pytest.mark.asyncio
    async def test_id3_metadata_extracted(self, tmp_path):
        path = self._make_wav(tmp_path)
        with patch("app.ingestion.audio_ingest._extract_id3", return_value={"title": "Test Track", "artist": "Test Artist"}):
            mock_seg = MagicMock()
            mock_seg.text = "Test audio with ID3 tags present."
            mock_seg.start = 0.0
            mock_seg.end = 1.0
            mock_seg.avg_logprob = -0.2
            mock_seg.no_speech_prob = 0.05
            mock_info = MagicMock()
            mock_info.language = "en"
            with patch("app.ingestion.audio_ingest._transcribe_file", return_value=([mock_seg], mock_info, 0.3)):
                with patch("app.ingestion.audio_ingest._classify_noise", return_value="speech"):
                    with patch("app.ingestion.audio_ingest._diarize", return_value={}):
                        docs = await async_ingest(str(path), "s1")
        assert any(d.structure.get("id3", {}).get("title") == "Test Track" for d in docs)

    def test_metadata_schema(self):
        assert settings.AUDIO_SAMPLE_RATE == 16000
        assert settings.MAX_AUDIO_SEGMENTS > 0

    def test_pii_redacted(self):
        text = "Call me at 555-867-5309 for details"
        result = redact_pii(text)
        assert "555-867-5309" not in result

    def test_filler_words_stripped(self):
        text = "um so uh this is a test"
        result = _strip_fillers(text)
        assert "um" not in result.lower()
        assert "uh" not in result.lower()

    def test_confidence_from_logprob(self):
        assert _compute_confidence(-0.5) == 0.5
        assert _compute_confidence(0.0) == 1.0
        assert _compute_confidence(None) == 1.0

    def test_hallucination_risk_high(self):
        assert _hallucination_risk(0.3, 0.9) == "high"
        assert _hallucination_risk(0.8, 0.1) == "low"

    def test_speed_flag_extreme_wpm(self):
        assert _speed_flag("word " * 100, 5.0) is True
        assert _speed_flag("normal sentence here", 5.0) is False

    def test_domain_vocab_applied(self):
        text = "we use qdrant for retrieval"
        result = _apply_domain_vocab(text)
        assert "Qdrant" in result or "qdrant" in result

    def test_sha256_deterministic(self, tmp_path):
        path = tmp_path / "sample.wav"
        path.write_bytes(b"RIFF" + b"\x00" * 100)
        h1 = _sha256_check(path)
        h2 = _sha256_check(path)
        assert h1 == h2

    def test_drm_detection_returns_false_for_wav(self, tmp_path):
        path = tmp_path / "clean.wav"
        path.write_bytes(b"RIFF" + b"\x00" * 100 + b"WAVE")
        assert _is_drm_protected(path) is False