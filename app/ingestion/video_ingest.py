from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytesseract
from PIL import Image

from app.core.config import settings
from app.ingestion.schema import (
    EmptyFileError,
    FileTooLargeError,
    IngestedDocument,
    UniversalMetadata,
    UnsupportedMimeError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imagehash
    from PIL import Image as _PILImage
    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False

try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector
    _SCENEDETECT_AVAILABLE = True
except ImportError:
    _SCENEDETECT_AVAILABLE = False

try:
    import numpy as _np
except ImportError:
    _np = None


def _sanitize(text: str, surface: str, **log_kw) -> str:
    """Apply Phase-26 injection sanitization. Returns original on guardrail error."""
    try:
        from app.guardrails.input_guard import sanitize as _g
        clean = _g(text, surface=surface)
        if clean != text:
            logger.warning("injection_sanitized", surface=surface,
                           original_len=len(text), sanitized_len=len(clean), **log_kw)
        return clean
    except Exception as exc:
        logger.warning("guardrail_skipped", surface=surface, error=str(exc))
        return text


def _scrub_pii(text: str, surface: str) -> str:
    """Apply Phase-26 PII scrubbing. Returns original on error."""
    try:
        from app.guardrails.pii import scrub_pii
        clean, changed = scrub_pii(text)
        if changed:
            logger.warning("pii_scrubbed", surface=surface,
                           original_len=len(text), scrubbed_len=len(clean))
        return clean
    except Exception as exc:
        logger.warning("pii_scrub_skipped", surface=surface, error=str(exc))
        return text


# SUPPORTED FORMATS

SUPPORTED_VIDEO_FORMATS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".ts",
}

# MAGIC BYTES FOR VIDEO MIME DETECTION
MAGIC_BYTES: Dict[bytes, str] = {
    b"\x00\x00\x00\x18ftyp": "video/mp4",
    b"\x00\x00\x00\x1cftyp": "video/mp4",
    b"\x1aE\xdf\xa3":        "video/webm",
    b"RIFF":                  "video/avi",
    b"FLV":                   "video/x-flv",
}

# DRM DETECTION SIGNATURES
DRM_SIGNATURES = [b"drm", b"DRM", b"encrypted", b"ENCRYPTED", b"protect"]

# EASYOCR SINGLETON — loaded once, reused across all calls
_easyocr_reader: Optional[Any] = None


def _get_easyocr_reader() -> Any:
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


# SLIDE NUMBER PATTERN — matches "Slide 4", "4 / 12", "4 of 12"
_SLIDE_NUM_RE = re.compile(
    r'\bslide\s+(\d+)\b|\b(\d+)\s*/\s*\d+\b|\b(\d+)\s+of\s+\d+\b',
    re.IGNORECASE,
)

# FINANCE-NUMERIC CAPTION KEYWORDS
_FIN_NUMERIC_KW = {"$", "%", "billion", "million", "revenue", "earnings", "ebitda", "eps"}


def _extract_slide_number(caption: str) -> Optional[int]:
    m = _SLIDE_NUM_RE.search(caption)
    if m:
        for g in m.groups():
            if g is not None:
                return int(g)
    return None


def _is_numeric_frame(caption: str) -> bool:
    lower = caption.lower()
    return any(kw in lower for kw in _FIN_NUMERIC_KW)


def _build_combined_text(speech_text: str, nearby_frames: List[Dict[str, Any]]) -> str:
    parts = [speech_text]
    for frame in nearby_frames:
        ts  = frame.get("timestamp", 0.0)
        cap = frame.get("caption", "")
        ocr = frame.get("ocr_text", "")
        if cap:
            parts.append(f"\n\n[VISUAL AT {ts:.1f}s]: {cap}")
            if _is_numeric_frame(cap):
                parts.append(f"\n[VISUAL AT {ts:.1f}s]: {cap}")
        if ocr:
            parts.append(f"\n[ON-SCREEN TEXT]: {ocr}")
    return "".join(parts)


# SHA-256 FILE HASH

def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# MD5 HASH OF FRAME FILE FOR DUPLICATE DETECTION

def _frame_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# MAGIC-BYTE MIME DETECTION — SECTION 2.3

def _detect_mime(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        for magic, mime in MAGIC_BYTES.items():
            if header.startswith(magic):
                return mime
        ext = Path(file_path).suffix.lower()
        fallback = {
            ".mp4": "video/mp4",
            ".avi": "video/avi",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".flv": "video/x-flv",
            ".wmv": "video/x-ms-wmv",
            ".ts":  "video/mp2t",
        }
        return fallback.get(ext, "application/octet-stream")
    except Exception:
        return "application/octet-stream"


# DRM DETECTION — SECTION 2.3

def _is_drm_protected(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(4096)
        return any(sig in header for sig in DRM_SIGNATURES)
    except Exception:
        return False


# DISK SPACE GUARD — SECTION 2.3

def _check_disk_space(path: str) -> None:
    try:
        stat = shutil.disk_usage(Path(path).parent)
        free_mb = stat.free / (1024 * 1024)
        if free_mb < settings.MIN_FREE_DISK_MB:
            raise OSError(
                f"INSUFFICIENT_DISK_SPACE: {free_mb:.0f}MB free, "
                f"need {settings.MIN_FREE_DISK_MB}MB"
            )
    except OSError:
        raise
    except Exception as e:
        logger.warning(event="disk_check_failed", error=str(e))


# FFMPEG / FFPROBE RESOLVER

_ffmpeg_cache: Optional[str] = None
_ffprobe_cache: Optional[str] = None


def _test_binary(path: str) -> bool:
    """Return True only if the binary actually starts (catches DLL/arch errors on Windows)."""
    try:
        r = subprocess.run([path, "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _resolve_ffmpeg() -> str:
    global _ffmpeg_cache
    if _ffmpeg_cache is not None:
        return _ffmpeg_cache

    candidates: list = []

    configured = Path(settings.FFMPEG_PATH)
    if configured.exists():
        candidates.append(str(configured))

    discovered = shutil.which("ffmpeg")
    if discovered and discovered not in candidates:
        candidates.append(discovered)

    # imageio-ffmpeg bundles a static Windows binary — works even when conda ffmpeg has DLL issues
    try:
        import imageio_ffmpeg  # type: ignore
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and bundled not in candidates:
            candidates.append(bundled)
    except Exception:
        pass

    for c in candidates:
        if _test_binary(c):
            logger.info(event="ffmpeg_resolved", path=c)
            _ffmpeg_cache = c
            return c
        logger.warning(event="ffmpeg_candidate_failed", path=c)

    raise FileNotFoundError(
        "FFMPEG_NOT_FOUND: no working ffmpeg binary found. "
        "Install imageio-ffmpeg: pip install imageio-ffmpeg"
    )


def _resolve_ffprobe() -> str:
    global _ffprobe_cache
    if _ffprobe_cache is not None:
        return _ffprobe_cache

    candidates: list = []

    # Sibling of resolved ffmpeg binary (handles manual installs on Windows)
    try:
        ffmpeg_path = _resolve_ffmpeg()
        for name in ("ffprobe", "ffprobe.exe", "ffprobe.EXE"):
            candidate = Path(ffmpeg_path).parent / name
            if candidate.exists():
                candidates.append(str(candidate))
    except Exception:
        pass

    discovered = shutil.which("ffprobe")
    if discovered and discovered not in candidates:
        candidates.append(discovered)

    for c in candidates:
        if _test_binary(c):
            logger.info(event="ffprobe_resolved", path=c)
            _ffprobe_cache = c
            return c
        logger.warning(event="ffprobe_candidate_failed", path=c)

    raise FileNotFoundError("FFPROBE_NOT_FOUND: no working ffprobe binary found")


# PYAV METADATA FALLBACK — used when ffprobe subprocess is unavailable/broken

def _probe_with_pyav(file_path: str) -> Dict[str, Any]:
    """Metadata extraction via PyAV (uses bundled libav DLLs, no subprocess)."""
    try:
        import av  # type: ignore
        container = av.open(file_path)

        duration = None
        if container.duration:
            duration = container.duration / 1_000_000  # microseconds → seconds

        video_stream = next((s for s in container.streams if s.type == "video"), None)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)

        fps = None
        width = 0
        height = 0
        codec = None
        if video_stream:
            try:
                if video_stream.average_rate:
                    fps = float(video_stream.average_rate)
                ctx = video_stream.codec_context
                width  = ctx.width  or 0
                height = ctx.height or 0
                codec  = ctx.name
            except Exception:
                pass

        audio_codec    = None
        audio_channels = 0
        if audio_stream:
            try:
                ctx = audio_stream.codec_context
                audio_codec    = ctx.name
                audio_channels = getattr(ctx, "channels", 0) or 0
            except Exception:
                pass

        result = {
            "duration":         duration,
            "size":             0,
            "bitrate":          container.bit_rate or 0,
            "format_name":      container.format.long_name if container.format else "",
            "has_video":        video_stream is not None,
            "has_audio":        audio_stream is not None,
            "codec":            codec,
            "fps":              fps,
            "width":            width,
            "height":           height,
            "color_space":      None,
            "is_hdr":           False,
            "audio_codec":      audio_codec,
            "audio_channels":   audio_channels,
            "subtitle_streams": [],
            "chapter_streams":  [],
            "streams":          [],
        }
        container.close()
        logger.info(event="pyav_probe_success", duration=duration, has_video=result["has_video"], file=file_path)
        return result
    except Exception as e:
        logger.warning(event="pyav_probe_failed", error=str(e), file=file_path)
        return {}


# FFPROBE FULL METADATA — SECTION 4.1

def _probe_metadata(file_path: str) -> Dict[str, Any]:
    # Try ffprobe subprocess first (richer metadata: chapters, subtitles, HDR)
    try:
        ffprobe = _resolve_ffprobe()
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [])

            video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

            duration = None
            try:
                duration = float(fmt.get("duration") or 0)
            except (ValueError, TypeError):
                pass

            return {
                "duration":      duration,
                "size":          int(fmt.get("size", 0)),
                "bitrate":       int(fmt.get("bit_rate", 0)),
                "format_name":   fmt.get("format_name", ""),
                "has_video":     video_stream is not None,
                "has_audio":     audio_stream is not None,
                "codec":         video_stream.get("codec_name") if video_stream else None,
                "fps":           _parse_fps(video_stream.get("r_frame_rate", "0/1")) if video_stream else None,
                "width":         int(video_stream.get("width", 0)) if video_stream else 0,
                "height":        int(video_stream.get("height", 0)) if video_stream else 0,
                "color_space":   video_stream.get("color_space") if video_stream else None,
                "is_hdr":        _detect_hdr(video_stream) if video_stream else False,
                "audio_codec":   audio_stream.get("codec_name") if audio_stream else None,
                "audio_channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
                "subtitle_streams": [s for s in streams if s.get("codec_type") == "subtitle"],
                "chapter_streams":  data.get("chapters", []),
                "streams":          streams,
            }
        logger.warning(
            event="ffprobe_nonzero",
            returncode=result.returncode,
            stderr=result.stderr[:500] if result.stderr else "",
            file=file_path,
        )
    except FileNotFoundError:
        logger.warning(event="ffprobe_unavailable", file=file_path)
    except Exception as e:
        logger.warning(event="ffprobe_failed", error=str(e))

    # PyAV fallback — uses bundled libav DLLs, works even when system ffprobe is broken
    logger.info(event="probe_pyav_fallback", file=file_path)
    return _probe_with_pyav(file_path)


def _parse_fps(rate_str: str) -> Optional[float]:
    try:
        num, den = rate_str.split("/")
        return float(num) / float(den) if float(den) != 0 else None
    except Exception:
        return None


def _detect_hdr(stream: Dict) -> bool:
    transfer = stream.get("color_transfer", "")
    return transfer in ("smpte2084", "arib-std-b67", "smpte428")


# SUBTITLE EXTRACTION — SECTION 4.1

def _extract_subtitles(
    file_path: str,
    subtitle_streams: List[Dict],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    ffmpeg = _resolve_ffmpeg()

    for idx, stream in enumerate(subtitle_streams[:3]):
        codec = stream.get("codec_name", "")
        lang = stream.get("tags", {}).get("language", f"track_{idx}")
        out_file = output_dir / f"subtitle_{idx}_{lang}.srt"

        try:
            cmd = [
                ffmpeg, "-y", "-i", file_path,
                "-map", f"0:s:{idx}",
                "-f", "srt",
                str(out_file),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and out_file.exists():
                text = out_file.read_text(encoding="utf-8", errors="replace")
                cleaned = _clean_subtitle_text(text)
                if cleaned:
                    results.append({
                        "language":   lang,
                        "codec":      codec,
                        "text":       cleaned,
                        "track_idx":  idx,
                    })
        except Exception as e:
            logger.warning(event="subtitle_extract_failed", track=idx, error=str(e))

    return results


def _clean_subtitle_text(raw: str) -> str:
    import re
    # STRIP SRT TIMESTAMPS AND INDICES
    text = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n", "", raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()


# CHAPTER EXTRACTION — SECTION 4.1

def _extract_chapters(chapters: List[Dict]) -> List[Dict[str, Any]]:
    result = []
    for ch in chapters:
        try:
            start = float(ch.get("start_time", 0))
            end   = float(ch.get("end_time", 0))
            title = ch.get("tags", {}).get("title", f"Chapter_{ch.get('id', 0)}")
            result.append({"title": title, "start": start, "end": end})
        except Exception:
            continue
    return result


# HDR TONE-MAP TO SDR — SECTION 4.1

def _tonemap_frame(frame_path: str, output_path: str) -> bool:
    try:
        ffmpeg = _resolve_ffmpeg()
        cmd = [
            ffmpeg, "-y", "-i", frame_path,
            "-vf", "zscale=transfer=linear,tonemap=hable,zscale=transfer=bt709",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


# DEINTERLACE CHECK — SECTION 4.1

def _needs_deinterlace(stream: Optional[Dict]) -> bool:
    if not stream:
        return False
    return stream.get("field_order", "progressive") not in ("progressive", "unknown", "")


# AUDIO EXTRACTION WITH HDR HANDLING

def _extract_audio(file_path: str, audio_path: str) -> None:
    ffmpeg = _resolve_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", file_path,
        "-vn",
        "-ar", str(settings.AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-f", "wav",
        audio_path,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.FFMPEG_TIMEOUT_SEC,
    )
    if result.returncode != 0:
        logger.error(
            event="ffmpeg_audio_extract_failed",
            stderr=(result.stderr or "")[-500:],
        )
        raise RuntimeError("AUDIO_EXTRACTION_FAILED")


# FRAME OCR WITH EASYOCR ENSEMBLE — SECTION 4.1

def _extract_frame_ocr(image_path: str) -> str:
    results = []

    # TESSERACT
    try:
        img = Image.open(image_path).convert("RGB")
        text = (pytesseract.image_to_string(img) or "").strip()
        if len(text) > 10:
            results.append(text)
    except Exception as e:
        logger.debug(event="tesseract_frame_ocr_failed", error=str(e))

    # EASYOCR ENSEMBLE
    try:
        reader = _get_easyocr_reader()
        detections = reader.readtext(image_path, detail=0)
        if detections:
            results.append(" ".join(detections))
    except Exception as e:
        logger.debug(event="easyocr_frame_ocr_failed", error=str(e))

    if not results:
        return ""

    # MERGE AND DEDUPLICATE
    merged = " ".join(results)
    return unicodedata.normalize("NFC", merged)[:2000]


# BLUR SCORE

def _blur_score(image_path: str) -> float:
    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1.0
        laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
        return float(min(laplacian / 100.0, 1.0))
    except Exception:
        return 1.0


# COSINE SIMILARITY FOR DUPLICATE FRAME DETECTION — SECTION 4.1

def _frame_cosine_similarity(path1: str, path2: str) -> float:
    try:
        import cv2
        import numpy as np
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0
        img1 = cv2.resize(img1, (64, 64)).flatten().astype(float)
        img2 = cv2.resize(img2, (64, 64)).flatten().astype(float)
        norm1 = np.linalg.norm(img1)
        norm2 = np.linalg.norm(img2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(img1, img2) / (norm1 * norm2))
    except Exception:
        return 0.0


# SPEECH ALIGNMENT

def _link_speech(timestamp: float, speech_segments: List[Dict]) -> Optional[Dict]:
    for seg in speech_segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg
    return None


def _alignment_score(caption: str, speech: Optional[Dict]) -> float:
    if not speech or not caption:
        return 0.0
    speech_text = speech.get("text", "")
    if not speech_text:
        return 0.0
    caption_words = set(caption.lower().split())
    speech_words  = set(speech_text.lower().split())
    if not speech_words:
        return 0.0
    return round(len(caption_words & speech_words) / len(speech_words), 3)


# BASE STRUCTURE

def _base_structure(
    doc_id: str,
    session_id: str,
    file_hash: str,
    source_path: str,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "doc_id":      doc_id,
        "session_id":  session_id,
        "file_hash":   file_hash,
        "source_path": source_path,
        **extra,
    }


# THUMBNAIL GRID — SECTION 4.1

def _build_thumbnail_grid(
    frame_paths: List[str],
    output_path: Path,
    rows: int = 3,
    cols: int = 3,
) -> Optional[str]:
    try:
        from PIL import Image as PILImage
        selected = frame_paths[: rows * cols]
        if not selected:
            return None
        imgs = [PILImage.open(p).convert("RGB").resize((320, 180)) for p in selected]
        w, h = 320, 180
        grid = PILImage.new("RGB", (cols * w, rows * h))
        for idx, img in enumerate(imgs):
            r, c = divmod(idx, cols)
            grid.paste(img, (c * w, r * h))
        grid.save(str(output_path))
        return str(output_path)
    except Exception as e:
        logger.warning(event="thumbnail_grid_failed", error=str(e))
        return None


# CORRUPT VIDEO RECOVERY — SECTION 4.1

def _attempt_recovery(file_path: str, output_path: str) -> bool:
    try:
        ffmpeg = _resolve_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-err_detect", "ignore_err",
            "-i", file_path,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and Path(output_path).stat().st_size > 0
    except Exception as e:
        logger.warning(event="video_recovery_failed", error=str(e))
        return False


# MAIN ASYNC INGEST

async def ingest_async(file_path: str, session_id: str) -> List[IngestedDocument]:
    import contextvars
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(None, ctx.run, ingest, file_path, session_id)


# MAIN SYNC INGEST

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {file_path}")

    # MAGIC-BYTE MIME DETECTION — SECTION 2.3
    mime_type = _detect_mime(file_path)

    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_FORMATS:
        raise UnsupportedMimeError(f"UNSUPPORTED_VIDEO_FORMAT: {ext} mime={mime_type}")

    file_size = path.stat().st_size

    # ZERO-BYTE GUARD — SECTION 2.3
    if file_size == 0:
        raise EmptyFileError("EMPTY_FILE")

    # SIZE GUARD — SECTION 2.3
    if file_size > settings.MAX_FILE_SIZE_VIDEO:
        raise FileTooLargeError(
            f"VIDEO_TOO_LARGE: {file_size} bytes exceeds {settings.MAX_FILE_SIZE_VIDEO} bytes"
        )

    # DRM DETECTION — SECTION 4.1
    if _is_drm_protected(file_path):
        raise PermissionError("DRM_PROTECTED_VIDEO: cannot process DRM-protected content")

    # DISK SPACE GUARD — SECTION 2.3
    _check_disk_space(file_path)

    start       = time.time()
    doc_id      = str(uuid.uuid4())
    file_hash   = _sha256(file_path)
    source_name = path.name
    source_path = str(path.resolve())

    # UNIVERSAL METADATA — SECTION 2.2
    universal_meta = UniversalMetadata(
        source_path=source_path,
        modality="video",
        mime_type=mime_type,
        file_size_bytes=file_size,
        checksum_sha256=file_hash,
        encoding="binary",
    )

    audio_path:       Optional[str]  = None
    frame_temp_dir:   Optional[Path] = None
    frame_staging_dir: Optional[Path] = None
    recovered_path:   Optional[str]  = None
    from app.utils.paths import resolved_staging_dir
    staging = resolved_staging_dir()

    # FFMPEG AVAILABILITY CHECK — fail fast with clear message
    try:
        _resolve_ffmpeg()
    except FileNotFoundError:
        raise ValueError("FFMPEG_NOT_FOUND: ffmpeg binary not found, please install ffmpeg")

    logger.info(
        event="video_ingest_start",
        file=source_name,
        size=file_size,
        mime=mime_type,
        session_id=session_id,
    )

    try:
        # FFPROBE METADATA — SECTION 4.1
        probe = _probe_metadata(file_path)

        # Both ffprobe and PyAV returned empty — file is likely corrupt or unreadable
        if not probe:
            raise ValueError("CORRUPTED_FILE: video probe failed with all methods (ffprobe + PyAV). File may be corrupt or an unsupported format.")

        video_duration = probe.get("duration")

        # DURATION GUARD
        if video_duration is not None and video_duration > settings.MAX_VIDEO_DURATION_SEC:
            raise ValueError(
                f"VIDEO_TOO_LONG: {video_duration:.1f}s exceeds {settings.MAX_VIDEO_DURATION_SEC}s"
            )

        # NO VIDEO TRACK → ROUTE TO AUDIO PIPELINE — SECTION 4.1
        has_video = probe.get("has_video", True)
        has_audio = probe.get("has_audio", False)

        universal_meta.duration_s = video_duration
        universal_meta.custom_fields.update({
            "codec":          probe.get("codec"),
            "fps":            probe.get("fps"),
            "width":          probe.get("width"),
            "height":         probe.get("height"),
            "bitrate":        probe.get("bitrate"),
            "is_hdr":         probe.get("is_hdr", False),
            "color_space":    probe.get("color_space"),
            "audio_codec":    probe.get("audio_codec"),
            "audio_channels": probe.get("audio_channels"),
            "audio_available": has_audio,
            "has_video":      has_video,
            "format_name":    probe.get("format_name"),
        })

        documents: List[IngestedDocument] = []

        # SHORT VIDEO — SINGLE CHUNK; SKIP SCENE DETECTION — SECTION 4.1
        skip_scene_detection = (video_duration is not None and video_duration < 2.0)

        # AUDIO EXTRACTION + INGESTION — SECTION 4.1
        speech_segments: List[Dict] = []

        if has_audio:
            try:
                fd, audio_path = tempfile.mkstemp(suffix=".wav", dir=str(staging))
                os.close(fd)
                _extract_audio(file_path, audio_path)

                # LAZY IMPORT TO AVOID CIRCULAR
                from app.ingestion.audio_ingest import ingest as audio_ingest_fn
                audio_docs = audio_ingest_fn(audio_path, session_id)

                for i, doc in enumerate(audio_docs):
                    s       = doc.structure or {}
                    start_t = s.get("timestamp_start")
                    end_t   = s.get("timestamp_end")

                    if start_t is None or end_t is None or end_t <= start_t:
                        continue

                    _speech_clean = _sanitize(doc.text, surface="video_speech_ingest",
                                              file=source_name)
                    _speech_clean = _scrub_pii(_speech_clean, surface="video_speech_ingest")
                    speech_segments.append({
                        "index":      i,
                        "start":      start_t,
                        "end":        end_t,
                        "text":       _speech_clean,
                        "confidence": s.get("confidence", 1.0),
                        "language":   s.get("language"),
                    })

                    documents.append(
                        IngestedDocument(
                            text=_speech_clean,
                            modality="video",
                            subtype="speech",
                            source_type="video",
                            source=source_name,
                            chunk_id=i,
                            structure=_base_structure(
                                doc_id, session_id, file_hash, source_path,
                                chunk_type="audio_segment",
                                start_time=round(start_t, 3),
                                end_time=round(end_t, 3),
                                timestamp_start=start_t,
                                timestamp_end=end_t,
                                confidence=s.get("confidence"),
                                language=s.get("language"),
                                hallucination_risk=s.get("hallucination_risk", "low"),
                                snr=s.get("snr"),
                                snr_degraded=s.get("snr_degraded", False),
                                content_type="video_speech",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "importance_score":   s.get("confidence", 1.0),
                                "modality_weight":    1.2,
                                "data_quality_score": s.get("confidence", 1.0),
                            },
                        ).finalize()
                    )

                if speech_segments:
                    lang = speech_segments[0].get("language", "unknown")
                    universal_meta.language = lang or "unknown"

            except Exception as e:
                logger.warning(
                    event="audio_demux_failed",
                    file=source_name,
                    error=str(e),
                    session_id=session_id,
                )
                universal_meta.add_error(f"AUDIO_DEMUX_FAILED: {e}")
        else:
            # NO AUDIO TRACK — VISUAL ONLY — SECTION 4.1
            universal_meta.custom_fields["audio_available"] = False
            universal_meta.add_error("no_audio_track_found")
            logger.warning(
                event="video_no_audio_track",
                file=source_name,
                warning="no_audio_track_found",
                session_id=session_id,
            )

        # SUBTITLE EXTRACTION — SECTION 4.1
        subtitle_streams = probe.get("subtitle_streams", [])
        if subtitle_streams and settings.VIDEO_SUBTITLE_EXTRACTION:
            sub_dir = staging / f"subs_{doc_id}"
            sub_dir.mkdir(exist_ok=True)
            try:
                subtitles = _extract_subtitles(file_path, subtitle_streams, sub_dir)
                for sub in subtitles:
                    if not sub.get("text"):
                        continue
                    sub["text"] = _sanitize(sub["text"], surface="video_subtitle_ingest",
                                            file=source_name)
                    sub["text"] = _scrub_pii(sub["text"], surface="video_subtitle_ingest")
                    documents.append(
                        IngestedDocument(
                            text=sub["text"],
                            modality="video",
                            subtype="ocr",
                            source_type="video",
                            source=source_name,
                            structure=_base_structure(
                                doc_id, session_id, file_hash, source_path,
                                content_type="video_subtitle",
                                subtitle_language=sub["language"],
                                subtitle_codec=sub["codec"],
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "importance_score":   0.8,
                                "modality_weight":    1.0,
                                "data_quality_score": 0.8,
                            },
                        ).finalize()
                    )
            except Exception as e:
                logger.warning(event="subtitle_extraction_failed", error=str(e))
                universal_meta.add_error(f"SUBTITLE_EXTRACTION_FAILED: {e}")
            finally:
                shutil.rmtree(sub_dir, ignore_errors=True)

        # CHAPTER EXTRACTION — SECTION 4.1
        chapters = _extract_chapters(probe.get("chapter_streams", []))
        if chapters:
            universal_meta.custom_fields["chapters"] = chapters

        # FRAME EXTRACTION — SECTION 4.1
        frames: List[Dict] = []
        if has_video:
            try:
                frames = extract_frames(
                    file_path,
                    settings.VIDEO_FRAME_INTERVAL_SEC,
                    session_id,
                    skip_scene_detection=skip_scene_detection,
                )
            except Exception as e:
                logger.warning(
                    event="frame_extract_failed",
                    file=source_name,
                    error=str(e),
                    session_id=session_id,
                )
                universal_meta.add_error(f"FRAME_EXTRACTION_FAILED: {e}")

        if frames:
            frame_temp_dir = Path(frames[0]["path"]).parent

        # THUMBNAIL GRID — SECTION 4.1
        frame_paths_list = [f["path"] for f in frames if Path(f["path"]).exists()]
        if frame_paths_list:
            grid_path = staging / f"thumbnail_grid_{doc_id}.jpg"
            grid_result = _build_thumbnail_grid(
                frame_paths_list,
                grid_path,
                rows=settings.VIDEO_THUMBNAIL_GRID_ROWS,
                cols=settings.VIDEO_THUMBNAIL_GRID_COLS,
            )
            if grid_result:
                universal_meta.custom_fields["thumbnail_grid"] = grid_result

        # FRAME PROCESSING — DUPLICATE SKIP; HDR TONEMAPPING — SECTION 4.1
        is_hdr = probe.get("is_hdr", False)
        seen_frame_hashes: set = set()
        total_frames_count = len(frames)
        # Collect frame captions + OCR for combined_text linking to speech segments
        frame_manifest: List[Dict[str, Any]] = []

        if total_frames_count > settings.MAX_VIDEO_FRAMES:
            logger.warning(
                event="video_frames_capped",
                original=total_frames_count,
                capped=settings.MAX_VIDEO_FRAMES,
                file=source_name,
                session_id=session_id,
            )
            universal_meta.add_error(
                f"VIDEO_FRAMES_CAPPED: extracted {total_frames_count}, capped at {settings.MAX_VIDEO_FRAMES}"
            )

        for frame in frames[:settings.MAX_VIDEO_FRAMES]:
            try:
                ts     = frame["timestamp_start"]
                f_path = frame["path"]

                if not Path(f_path).exists():
                    continue

                # DUPLICATE FRAME SKIP VIA IMAGE HASH — SECTION 4.1
                fhash = _frame_hash(f_path)
                if fhash in seen_frame_hashes:
                    logger.debug(event="duplicate_frame_skipped_hash", ts=ts)
                    continue
                seen_frame_hashes.add(fhash)

                # HDR TONE-MAP TO SDR — SECTION 4.1
                processing_path = f_path
                if is_hdr and settings.VIDEO_HDR_TONEMAPPING:
                    sdr_path = f_path.replace(".jpg", "_sdr.jpg")
                    if _tonemap_frame(f_path, sdr_path):
                        processing_path = sdr_path

                # COPY FRAME TO PERSISTENT STAGING SO SIGLIP EMBEDDING CAN READ IT
                # (frame_temp_dir is cleaned in finally before pipeline embeds)
                if frame_staging_dir is None:
                    frame_staging_dir = staging / f"frames_{doc_id}"
                    frame_staging_dir.mkdir(parents=True, exist_ok=True)
                persistent_frame_path = str(
                    frame_staging_dir / Path(processing_path).name
                )
                shutil.copy2(processing_path, persistent_frame_path)

                # FRAME CAPTION — ML inference is the chunker's job; store path for VideoChunker
                caption = f"Scene at {ts:.1f}s"

                # FRAME OCR
                ocr_text = _extract_frame_ocr(persistent_frame_path)
                if ocr_text:
                    ocr_text = _sanitize(ocr_text, surface="video_ocr_ingest", file=source_name)
                    ocr_text = _scrub_pii(ocr_text, surface="video_ocr_ingest")

                blur         = _blur_score(persistent_frame_path)
                linked       = _link_speech(ts, speech_segments)
                align        = _alignment_score(caption, linked)
                # Raised threshold to 0.2 — finance audio/slides often phrase the same
                # number differently (e.g. "four billion" vs "$4B"), so 0.1 caused too many
                # false conflict flags on financial presentation videos.
                conflict_flag = bool(linked and ocr_text and align < 0.2)
                slide_number  = _extract_slide_number(caption)

                # Collect for combined_text linking in the post-frame pass
                frame_manifest.append({
                    "timestamp": ts,
                    "caption":   caption,
                    "ocr_text":  ocr_text or "",
                })

                documents.append(
                    IngestedDocument(
                        text=caption,
                        modality="video",
                        subtype="frame",
                        source_type="video",
                        source=source_name,
                        chunk_id=frame["frame_index"],
                        structure=_base_structure(
                            doc_id, session_id, file_hash, source_path,
                            chunk_type="frame",
                            timestamp_sec=round(ts, 3),
                            timestamp_start=ts,
                            timestamp_end=frame.get("timestamp_end", ts),
                            frame_index=frame["frame_index"],
                            total_frames=total_frames_count,
                            caption=caption,
                            slide_number=slide_number,
                            asset_path=persistent_frame_path,
                            linked_speech=linked,
                            conflict_flag=conflict_flag,
                            alignment_score=align,
                            blur_score=blur,
                            fps=frame.get("fps"),
                            video_duration=video_duration,
                            is_hdr=is_hdr,
                            content_type="video_frame",
                            embedding_space="vision",
                            ingestion_time=time.time(),
                        ),
                        extra_metadata={
                            "importance_score":   blur,
                            "modality_weight":    1.0,
                            "data_quality_score": blur,
                        },
                    ).finalize()
                )

                # FRAME OCR DOCUMENT
                if ocr_text:
                    documents.append(
                        IngestedDocument(
                            text=ocr_text,
                            modality="video",
                            subtype="ocr",
                            source_type="video",
                            source=source_name,
                            structure=_base_structure(
                                doc_id, session_id, file_hash, source_path,
                                timestamp_start=ts,
                                frame_index=frame["frame_index"],
                                content_type="video_ocr",
                                ingestion_time=time.time(),
                            ),
                            extra_metadata={
                                "importance_score":   0.7,
                                "modality_weight":    0.9,
                                "data_quality_score": 0.7,
                            },
                        ).finalize()
                    )

            except Exception as e:
                logger.warning(
                    event="frame_process_error",
                    frame_index=frame.get("frame_index"),
                    error=str(e),
                    session_id=session_id,
                )
                universal_meta.add_error(f"FRAME_PROCESS_ERROR at ts={frame.get('timestamp_start')}: {e}")

        # POST-FRAME PASS — attach combined_text to speech docs
        if frame_manifest:
            _WINDOW = 5.0
            for doc in documents:
                if not doc.structure or doc.structure.get("chunk_type") != "audio_segment":
                    continue
                st = doc.structure.get("start_time", 0.0)
                et = doc.structure.get("end_time", st)
                nearby = [
                    f for f in frame_manifest
                    if f["timestamp"] >= st - _WINDOW and f["timestamp"] <= et + _WINDOW
                ]
                if nearby:
                    doc.structure["combined_text"] = _build_combined_text(doc.text, nearby)

        # GUARD — ATTEMPT RECOVERY ON EMPTY RESULT — SECTION 4.1
        if not documents:
            logger.warning(
                event="video_no_content_attempting_recovery",
                file=source_name,
                session_id=session_id,
            )
            recovered_path = str(staging / f"recovered_{doc_id}.mp4")
            if _attempt_recovery(file_path, recovered_path):
                logger.info(event="video_recovery_success", file=source_name)
                universal_meta.add_error("PARTIAL_CONTENT_RECOVERED")
            else:
                raise ValueError("NO_VIDEO_CONTENT_EXTRACTED")

        # FINALIZE UNIVERSAL METADATA
        universal_meta.chunk_count = len(documents)
        universal_meta.embedding_model = settings.EMBEDDING_MODEL
        universal_meta.status = "success"

        latency = round(time.time() - start, 2)

        logger.info(
            event="video_ingest_success",
            file=source_name,
            docs=len(documents),
            speech_segments=len(speech_segments),
            frames_extracted=len(frames),
            video_duration=video_duration,
            has_audio=has_audio,
            has_video=has_video,
            subtitles=len(subtitle_streams),
            chapters=len(chapters),
            latency=latency,
            session_id=session_id,
        )

        return documents

    except (EmptyFileError, FileTooLargeError, UnsupportedMimeError, PermissionError):
        raise

    except Exception as e:
        universal_meta.status = "failed"
        universal_meta.add_error(str(e))
        logger.error(
            event="video_ingest_failed",
            file=source_name,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 2),
        )
        raise

    finally:
        # TEMP FILE CLEANUP 
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        if frame_temp_dir and frame_temp_dir.exists():
            shutil.rmtree(frame_temp_dir, ignore_errors=True)

        if recovered_path and os.path.exists(recovered_path):
            try:
                os.remove(recovered_path)
            except Exception:
                pass


# ─── Phase 1: VideoIngestor ────────────────────────────────────────────────────

from app.ingestion.base_ingest import BaseIngestor
from app.ingestion.schema import RawExtract


class VideoIngestor(BaseIngestor):
    """Validates video files → List[RawExtract].

    Phase 1 responsibility: container validation, DRM check, duration/resolution metadata.
    Does NOT extract frames, caption, or transcribe — the chunker (Phase 2) drives ffmpeg.
    """

    async def extract(
        self,
        path: Path,
        metadata: UniversalMetadata,
    ) -> List[RawExtract]:
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_VIDEO_FORMATS:
            raise UnsupportedMimeError(f"UNSUPPORTED_VIDEO_FORMAT: {suffix}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise EmptyFileError(str(path))
        if file_size > settings.MAX_FILE_SIZE_VIDEO:
            raise FileTooLargeError(f"VIDEO_TOO_LARGE: {file_size}")

        # Probe duration + resolution via ffprobe
        duration_s: Optional[float] = None
        width: Optional[int] = None
        height: Optional[int] = None
        has_audio = False
        has_video = False

        try:
            probe_result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-print_format", "json",
                    "-show_streams", "-show_format",
                    str(path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            import json as _json
            probe_data = _json.loads(probe_result.stdout or "{}")
            fmt = probe_data.get("format", {})
            duration_s = float(fmt.get("duration", 0) or 0) or None
            for stream in probe_data.get("streams", []):
                codec_type = stream.get("codec_type", "")
                if codec_type == "video" and not has_video:
                    has_video = True
                    width = stream.get("width")
                    height = stream.get("height")
                elif codec_type == "audio":
                    has_audio = True
        except Exception as exc:
            logger.warning("video_probe_failed", file=path.name, error=str(exc))

        if duration_s is not None and duration_s > settings.MAX_VIDEO_DURATION_SEC:
            raise ValueError(f"VIDEO_TOO_LONG: {duration_s:.0f}s")

        # DRM / container sanity via magic bytes
        try:
            with open(path, "rb") as f:
                header = f.read(16)
            # Simple check: known bad magic → raise
            if header.startswith(b"\x00\x00\x00\x00"):
                raise ValueError(f"VIDEO_CONTAINER_INVALID: zero-byte header in {path.name}")
        except (ValueError, FileTooLargeError, EmptyFileError):
            raise
        except Exception:
            pass

        return [
            RawExtract(
                text="",
                extract_type="video_raw",
                timestamp_start=0.0,
                timestamp_end=duration_s,
                raw_source_ref=f"video:{path.name}",
                raw_bytes=None,  # file path stored in metadata; chunker reads directly
                extra={
                    "file_path": str(path.resolve()),
                    "file_size": file_size,
                    "duration_seconds": duration_s,
                    "width": width,
                    "height": height,
                    "has_audio": has_audio,
                    "has_video": has_video,
                    "format": suffix.lstrip("."),
                },
            )
        ]


# ══════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION  (moved from video_chunker.py — ingestion concern)
# ══════════════════════════════════════════════════════════════════════════════

def _vf_make_metrics():
    if not settings.PROMETHEUS_ENABLED:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def set(self, *a, **kw): pass
            def labels(self, **kw): return self
        noop = _Noop()
        return noop, noop, noop, noop
    try:
        from prometheus_client import Counter, Gauge, Histogram
        frames_extracted = Counter("video_frames_extracted_total", "Total frames extracted", ["session_id"])
        frames_skipped   = Counter("video_frames_skipped_total",   "Total frames skipped",  ["reason"])
        extract_duration = Histogram("video_frame_extraction_duration_seconds", "Frame extraction time")
        active           = Gauge("video_frame_active_extractions", "In-progress extractions")
        return frames_extracted, frames_skipped, extract_duration, active
    except Exception:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def set(self, *a, **kw): pass
            def labels(self, **kw): return self
        noop = _Noop()
        return noop, noop, noop, noop


_VF_FRAMES_EXTRACTED, _VF_FRAMES_SKIPPED, _VF_EXTRACT_DURATION, _VF_ACTIVE = _vf_make_metrics()

_VF_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _vf_get_semaphore() -> asyncio.Semaphore:
    global _VF_SEMAPHORE
    if _VF_SEMAPHORE is None:
        _VF_SEMAPHORE = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)
    return _VF_SEMAPHORE


class FrameMetadata:
    """Structured metadata for a single extracted video frame."""

    def __init__(
        self,
        frame_index: int,
        timestamp_start: float,
        timestamp_end: float,
        path: str,
        frame_width: int,
        frame_height: int,
        fps: float,
        video_duration: float,
        video_id: str,
        blur_score: float,
        brightness_mean: float,
        phash: Optional[str],
        is_scene_boundary: bool,
        scene_index: int,
        shot_type: str,
        hdr_detected: bool,
        interlaced: bool,
        aspect_ratio: float,
        extraction_method: str,
        session_id: str,
    ) -> None:
        self.frame_index       = frame_index
        self.timestamp_start   = timestamp_start
        self.timestamp_end     = timestamp_end
        self.path              = path
        self.frame_width       = frame_width
        self.frame_height      = frame_height
        self.fps               = fps
        self.video_duration    = video_duration
        self.video_id          = video_id
        self.blur_score        = blur_score
        self.brightness_mean   = brightness_mean
        self.phash             = phash
        self.is_scene_boundary = is_scene_boundary
        self.scene_index       = scene_index
        self.shot_type         = shot_type
        self.hdr_detected      = hdr_detected
        self.interlaced        = interlaced
        self.aspect_ratio      = aspect_ratio
        self.extraction_method = extraction_method
        self.session_id        = session_id

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "frame_index", "timestamp_start", "timestamp_end", "path",
            "frame_width", "frame_height", "fps", "video_duration", "video_id",
            "blur_score", "brightness_mean", "phash", "is_scene_boundary",
            "scene_index", "shot_type", "hdr_detected", "interlaced",
            "aspect_ratio", "extraction_method", "session_id",
        )}


class VideoFrameError(Exception):
    """Base exception for video frame extraction errors."""

class VideoOpenError(VideoFrameError):
    """Raised when the video file cannot be opened."""

class VideoTooShortError(VideoFrameError):
    """Raised when the video is too short for scene detection."""

class NoFramesExtractedError(VideoFrameError):
    """Raised when zero frames survive all filters."""

class DiskSpaceError(VideoFrameError):
    """Raised when insufficient disk space is available."""


def _probe_video(video_path: str) -> Dict[str, Any]:
    import json
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,color_transfer,color_space,field_order",
        "-show_entries", "format=duration,size",
        "-of", "json", video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data   = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        fmt    = data.get("format", {})
        fps    = 0.0
        try:
            num, den = stream.get("r_frame_rate", "0/1").split("/")
            fps = float(num) / max(float(den), 1e-6)
        except Exception:
            pass
        hdr         = stream.get("color_transfer", "") in ("smpte2084", "arib-std-b67", "smpte428")
        field_order = stream.get("field_order", "progressive")
        interlaced  = field_order not in ("progressive", "unknown", "")
        duration    = 0.0
        try:
            duration = float(fmt.get("duration", 0))
        except Exception:
            pass
        return {
            "width": int(stream.get("width", 0)), "height": int(stream.get("height", 0)),
            "fps": fps, "codec": stream.get("codec_name", "unknown"),
            "hdr": hdr, "interlaced": interlaced, "duration": duration,
        }
    except Exception as exc:
        logger.warning(event="ffprobe_failed", error=str(exc))
        return {"width": 0, "height": 0, "fps": 0.0, "codec": "unknown",
                "hdr": False, "interlaced": False, "duration": 0.0}


def _detect_scenes_pyscenedetect(video_path: str, threshold: float) -> List[float]:
    if not _SCENEDETECT_AVAILABLE:
        return []
    try:
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector(adaptive_threshold=threshold))
        sm.detect_scenes(video)
        return [round(sc[0].get_seconds(), 3) for sc in sm.get_scene_list()]
    except Exception as exc:
        logger.warning(event="pyscenedetect_failed", error=str(exc))
        return []


def _cv_blur_score(frame: Any) -> float:
    """Compute blur score from a cv2 numpy frame."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(min(cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0, 1.0))
    except Exception:
        return 1.0


def _cv_brightness_mean(frame: Any) -> float:
    try:
        import numpy as np
        return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    except Exception:
        return 128.0


def _cv_is_too_dark(brightness: float) -> bool:
    return brightness < settings.FRAME_DARKNESS_THRESHOLD


def _cv_compute_phash(image_path: str) -> Optional[str]:
    if not _IMAGEHASH_AVAILABLE:
        return None
    try:
        with _PILImage.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def _cv_phash_distance(h1: str, h2: str) -> int:
    try:
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
    except Exception:
        return 999


def _cv_tonemap_frame(frame: Any) -> Any:
    try:
        tonemap = cv2.createTonemapReinhard(gamma=2.2)
        f32     = frame.astype(_np.float32) / frame.max()
        return (tonemap.process(f32) * 255).clip(0, 255).astype(_np.uint8)
    except Exception:
        return frame


def _cv_deinterlace_frame(frame: Any) -> Any:
    try:
        return cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)
    except Exception:
        return frame


def _cv_classify_shot(frame: Any) -> str:
    try:
        density = float(_np.mean(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150) > 0))
        if density < 0.03: return "wide"
        if density < 0.10: return "medium"
        return "close"
    except Exception:
        return "unknown"


def _cv_resize_if_needed(frame: Any, max_dim: int) -> Any:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)


def _cv_save_frame(frame: Any, path: str, quality: int = 95) -> bool:
    try:
        return bool(cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]))
    except Exception:
        return False


def _extract_frames_opencv(
    video_path: str, interval_sec: int, max_frames: int,
    max_dim: int, max_duration: float, scene_thresh: float,
    session_id: str, temp_dir: Path, probe: Dict[str, Any],
) -> List[FrameMetadata]:
    fps = max(probe.get("fps") or 25.0, 0.01)
    duration     = probe.get("duration") or 0.0
    hdr_detected = probe.get("hdr", False)
    interlaced   = probe.get("interlaced", False)
    video_id     = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError(f"VIDEO_OPEN_FAILED: {video_path}")

    import numpy as np
    interval_frames = max(int(fps * interval_sec), 1)
    frames: List[FrameMetadata] = []
    seen_phashes: List[str] = []
    prev_frame = None
    frame_idx = 0
    saved = 0
    scene_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps
            if timestamp > max_duration:
                break

            take_frame = (frame_idx % interval_frames == 0)
            diff = 0.0
            if prev_frame is not None:
                _p = cv2.resize(prev_frame, (frame.shape[1], frame.shape[0])) if prev_frame.shape != frame.shape else prev_frame
                diff = float(np.mean(cv2.absdiff(_p, frame)))
                if diff > scene_thresh:
                    take_frame = True
                    scene_idx += 1

            if take_frame and saved < max_frames:
                h, w = frame.shape[:2]
                if h < 32 or w < 32:
                    _VF_FRAMES_SKIPPED.labels(reason="too_small").inc()
                else:
                    brightness = _cv_brightness_mean(frame)
                    if _cv_is_too_dark(brightness):
                        _VF_FRAMES_SKIPPED.labels(reason="too_dark").inc()
                    else:
                        if hdr_detected and settings.VIDEO_HDR_TONEMAPPING:
                            frame = _cv_tonemap_frame(frame)
                        if interlaced and settings.VIDEO_DEINTERLACE:
                            frame = _cv_deinterlace_frame(frame)
                        frame = _cv_resize_if_needed(frame, max_dim)
                        h_r, w_r = frame.shape[:2]
                        frame_path = str(temp_dir / f"frame_{saved:06d}.jpg")
                        if _cv_save_frame(frame, frame_path):
                            ph = _cv_compute_phash(frame_path)
                            if ph and seen_phashes and min(_cv_phash_distance(ph, e) for e in seen_phashes) < 8:
                                Path(frame_path).unlink(missing_ok=True)
                                _VF_FRAMES_SKIPPED.labels(reason="duplicate_phash").inc()
                            else:
                                if ph:
                                    seen_phashes.append(ph)
                                ts_end = min(round((frame_idx + interval_frames) / fps, 3), duration)
                                frames.append(FrameMetadata(
                                    frame_index=saved, timestamp_start=round(timestamp, 3),
                                    timestamp_end=ts_end, path=frame_path,
                                    frame_width=w_r, frame_height=h_r, fps=fps,
                                    video_duration=round(duration, 3), video_id=video_id,
                                    blur_score=round(_cv_blur_score(frame), 4),
                                    brightness_mean=round(brightness, 2), phash=ph,
                                    is_scene_boundary=(prev_frame is not None and diff > scene_thresh),
                                    scene_index=scene_idx, shot_type=_cv_classify_shot(frame),
                                    hdr_detected=hdr_detected, interlaced=interlaced,
                                    aspect_ratio=round(w_r / max(h_r, 1), 3),
                                    extraction_method="opencv_fallback", session_id=session_id,
                                ))
                                saved += 1
            prev_frame = frame
            frame_idx += 1
    finally:
        cap.release()
    return frames


def _extract_frames_pyscenedetect(
    video_path: str, max_frames: int, max_dim: int,
    max_duration: float, threshold: float, session_id: str,
    temp_dir: Path, probe: Dict[str, Any],
) -> List[FrameMetadata]:
    if not _SCENEDETECT_AVAILABLE or cv2 is None:
        raise VideoFrameError("PYSCENEDETECT_OR_CV2_UNAVAILABLE")

    fps          = max(probe.get("fps") or 25.0, 0.01)
    duration     = probe.get("duration") or 0.0
    hdr_detected = probe.get("hdr", False)
    interlaced   = probe.get("interlaced", False)
    video_id     = os.path.basename(video_path)

    scene_timestamps = _detect_scenes_pyscenedetect(video_path, threshold)
    all_timestamps   = sorted(set([0.0] + [ts for ts in scene_timestamps if ts > 0.1]))
    if len(all_timestamps) > max_frames:
        step = max(len(all_timestamps) // max_frames, 1)
        all_timestamps = all_timestamps[::step][:max_frames]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError(f"VIDEO_OPEN_FAILED: {video_path}")

    frames: List[FrameMetadata] = []
    seen_phashes: List[str] = []
    saved = 0

    try:
        for scene_idx, ts in enumerate(all_timestamps):
            if saved >= max_frames or ts > max_duration:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            if h < 32 or w < 32:
                _VF_FRAMES_SKIPPED.labels(reason="too_small").inc()
                continue
            brightness = _cv_brightness_mean(frame)
            if _cv_is_too_dark(brightness):
                _VF_FRAMES_SKIPPED.labels(reason="too_dark").inc()
                continue
            if hdr_detected and settings.VIDEO_HDR_TONEMAPPING:
                frame = _cv_tonemap_frame(frame)
            if interlaced and settings.VIDEO_DEINTERLACE:
                frame = _cv_deinterlace_frame(frame)
            frame = _cv_resize_if_needed(frame, max_dim)
            h_r, w_r = frame.shape[:2]
            next_ts   = all_timestamps[scene_idx + 1] if scene_idx + 1 < len(all_timestamps) else duration
            frame_path = str(temp_dir / f"frame_{saved:06d}.jpg")
            if not _cv_save_frame(frame, frame_path):
                continue
            ph = _cv_compute_phash(frame_path)
            if ph and seen_phashes and min(_cv_phash_distance(ph, e) for e in seen_phashes) < 8:
                Path(frame_path).unlink(missing_ok=True)
                _VF_FRAMES_SKIPPED.labels(reason="duplicate_phash").inc()
                continue
            if ph:
                seen_phashes.append(ph)
            frames.append(FrameMetadata(
                frame_index=saved, timestamp_start=round(ts, 3),
                timestamp_end=round(min(next_ts, duration), 3), path=frame_path,
                frame_width=w_r, frame_height=h_r, fps=fps,
                video_duration=round(duration, 3), video_id=video_id,
                blur_score=round(_cv_blur_score(frame), 4),
                brightness_mean=round(brightness, 2), phash=ph,
                is_scene_boundary=(scene_idx > 0), scene_index=scene_idx,
                shot_type=_cv_classify_shot(frame), hdr_detected=hdr_detected,
                interlaced=interlaced, aspect_ratio=round(w_r / max(h_r, 1), 3),
                extraction_method="pyscenedetect", session_id=session_id,
            ))
            saved += 1
    finally:
        cap.release()
    return frames


def extract_frames(
    video_path: str,
    interval_sec: int,
    session_id: str,
    skip_scene_detection: bool = False,
) -> List[Dict[str, Any]]:
    """Extract keyframes from a video file.

    Primary path: PySceneDetect (semantic scene boundaries).
    Fallback path: OpenCV interval + scene-diff.

    Returns List[FrameMetadata.to_dict()].
    """
    import math as _math
    if cv2 is None:
        raise ImportError("OPENCV_REQUIRED_FOR_FRAME_EXTRACTION")
    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"VIDEO_NOT_FOUND: {video_path}")
    if Path(video_path).stat().st_size == 0:
        raise ValueError("EMPTY_VIDEO_FILE")

    interval_sec = max(1, min(int(interval_sec), 300))
    max_frames   = settings.MAX_VIDEO_FRAMES
    max_dim      = settings.MAX_IMAGE_DIM
    max_duration = float(settings.MAX_VIDEO_DURATION_SEC)
    scene_thresh = settings.SCENE_CHANGE_THRESHOLD

    start = time.time()
    _VF_ACTIVE.set(1)

    try:
        from app.utils.paths import resolved_temp_frames_dir
        temp_root = resolved_temp_frames_dir()
    except Exception:
        import tempfile as _tf
        temp_root = Path(_tf.gettempdir())

    _check_disk_space(str(temp_root))
    import tempfile as _tf
    temp_dir = Path(_tf.mkdtemp(prefix="frames_", dir=str(temp_root)))

    try:
        probe    = _probe_video(video_path)
        fps      = probe.get("fps") or 25.0
        if fps <= 0 or _math.isnan(fps):
            fps = 25.0
        duration = probe.get("duration") or 0.0
        is_short = (0 < duration < 2.0)

        frames: List[FrameMetadata]
        if _SCENEDETECT_AVAILABLE and not is_short and duration >= 2.0 and not skip_scene_detection:
            try:
                frames = _extract_frames_pyscenedetect(
                    video_path=video_path, max_frames=max_frames, max_dim=max_dim,
                    max_duration=max_duration, threshold=scene_thresh,
                    session_id=session_id, temp_dir=temp_dir, probe=probe,
                )
                if not frames:
                    raise NoFramesExtractedError("PYSCENEDETECT_PRODUCED_NO_FRAMES")
            except (VideoFrameError, Exception) as exc:
                logger.warning(event="pyscenedetect_fallback_to_opencv", error=str(exc))
                frames = _extract_frames_opencv(
                    video_path=video_path, interval_sec=interval_sec, max_frames=max_frames,
                    max_dim=max_dim, max_duration=max_duration, scene_thresh=scene_thresh,
                    session_id=session_id, temp_dir=temp_dir, probe=probe,
                )
        else:
            frames = _extract_frames_opencv(
                video_path=video_path, interval_sec=interval_sec, max_frames=max_frames,
                max_dim=max_dim, max_duration=max_duration, scene_thresh=scene_thresh,
                session_id=session_id, temp_dir=temp_dir, probe=probe,
            )

        if not frames:
            raise NoFramesExtractedError(f"NO_FRAMES_EXTRACTED from {os.path.basename(video_path)}")

        _VF_FRAMES_EXTRACTED.labels(session_id=session_id).inc(len(frames))
        logger.info(
            event="frame_extraction_success",
            video=os.path.basename(video_path),
            frames=len(frames),
            latency=round(time.time() - start, 2),
        )
        return [f.to_dict() for f in frames]

    except (NoFramesExtractedError, VideoOpenError, DiskSpaceError):
        raise
    except Exception as exc:
        logger.error(event="frame_extraction_failed", error=str(exc))
        raise
    finally:
        _VF_ACTIVE.set(0)


async def extract_frames_async(
    video_path: str,
    session_id: str,
    interval_sec: Optional[int] = None,
) -> List[Dict[str, Any]]:
    async with _vf_get_semaphore():
        import contextvars
        loop = asyncio.get_event_loop()
        ctx  = contextvars.copy_context()
        return await loop.run_in_executor(
            None,
            ctx.run,
            lambda: extract_frames(
                video_path=video_path,
                interval_sec=interval_sec or settings.VIDEO_FRAME_INTERVAL_SEC,
                session_id=session_id,
            ),
        )


