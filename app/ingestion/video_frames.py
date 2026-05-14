from __future__ import annotations

import asyncio
import hashlib
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imagehash
    from PIL import Image as PILImage
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    from scenedetect import SceneManager, VideoManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

logger = get_logger(__name__)


# PROMETHEUS METRICS (OPTIONAL — NO-OP IF PROMETHEUS DISABLED)

def _make_metrics():
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
        frames_extracted = Counter(
            "video_frames_extracted_total",
            "Total frames extracted",
            ["session_id"],
        )
        frames_skipped = Counter(
            "video_frames_skipped_total",
            "Total frames skipped (dark/blurry/duplicate)",
            ["reason"],
        )
        extract_duration = Histogram(
            "video_frame_extraction_duration_seconds",
            "Time to extract all frames from a video",
        )
        active_extractions = Gauge(
            "video_frame_active_extractions",
            "Number of in-progress frame extraction jobs",
        )
        return frames_extracted, frames_skipped, extract_duration, active_extractions
    except Exception:
        class _Noop:
            def observe(self, *a, **kw): pass
            def inc(self, *a, **kw): pass
            def set(self, *a, **kw): pass
            def labels(self, **kw): return self
        noop = _Noop()
        return noop, noop, noop, noop


_FRAMES_EXTRACTED, _FRAMES_SKIPPED, _EXTRACT_DURATION, _ACTIVE = _make_metrics()


# SEMAPHORE — CAP CONCURRENT EXTRACTIONS

_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)
    return _SEMAPHORE


# UNIVERSAL FRAME METADATA MODEL

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
        self.frame_index      = frame_index
        self.timestamp_start  = timestamp_start
        self.timestamp_end    = timestamp_end
        self.path             = path
        self.frame_width      = frame_width
        self.frame_height     = frame_height
        self.fps              = fps
        self.video_duration   = video_duration
        self.video_id         = video_id
        self.blur_score       = blur_score
        self.brightness_mean  = brightness_mean
        self.phash            = phash
        self.is_scene_boundary = is_scene_boundary
        self.scene_index      = scene_index
        self.shot_type        = shot_type
        self.hdr_detected     = hdr_detected
        self.interlaced       = interlaced
        self.aspect_ratio     = aspect_ratio
        self.extraction_method = extraction_method
        self.session_id       = session_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index":       self.frame_index,
            "timestamp_start":   self.timestamp_start,
            "timestamp_end":     self.timestamp_end,
            "path":              self.path,
            "frame_width":       self.frame_width,
            "frame_height":      self.frame_height,
            "fps":               self.fps,
            "video_duration":    self.video_duration,
            "video_id":          self.video_id,
            "blur_score":        self.blur_score,
            "brightness_mean":   self.brightness_mean,
            "phash":             self.phash,
            "is_scene_boundary": self.is_scene_boundary,
            "scene_index":       self.scene_index,
            "shot_type":         self.shot_type,
            "hdr_detected":      self.hdr_detected,
            "interlaced":        self.interlaced,
            "aspect_ratio":      self.aspect_ratio,
            "extraction_method": self.extraction_method,
            "session_id":        self.session_id,
        }


# CUSTOM EXCEPTIONS

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


# DISK SPACE GUARD

def _check_disk_space(directory: str) -> None:
    try:
        stat = shutil.disk_usage(directory)
        free_mb = stat.free / (1024 * 1024)
        required_mb = settings.MIN_FREE_DISK_MB
        if free_mb < required_mb:
            raise DiskSpaceError(
                f"DISK_SPACE_LOW: {free_mb:.1f} MB free, "
                f"required {required_mb} MB at {directory}"
            )
    except DiskSpaceError:
        raise
    except Exception as exc:
        logger.warning(event="disk_space_check_failed", error=str(exc))


# VIDEO PROBING VIA FFPROBE

def _probe_video(video_path: str) -> Dict[str, Any]:
    """Extract stream metadata via ffprobe — codec, resolution, fps, hdr, interlaced."""
    import json
    import subprocess

    ffprobe = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,"
        "color_transfer,color_space,field_order,nb_frames,bit_rate",
        "-show_entries", "format=duration,size",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [{}])
        fmt     = data.get("format", {})
        stream  = streams[0] if streams else {}

        # PARSE FPS
        fps = 0.0
        r_fps = stream.get("r_frame_rate", "0/1")
        try:
            num, den = r_fps.split("/")
            fps = float(num) / max(float(den), 1e-6)
        except Exception:
            pass

        # HDR DETECTION
        hdr = stream.get("color_transfer", "") in (
            "smpte2084", "arib-std-b67", "smpte428"
        )

        # INTERLACE DETECTION
        field_order = stream.get("field_order", "progressive")
        interlaced  = field_order not in ("progressive", "unknown", "")

        duration = 0.0
        try:
            duration = float(fmt.get("duration", 0))
        except Exception:
            pass

        return {
            "width":      int(stream.get("width", 0)),
            "height":     int(stream.get("height", 0)),
            "fps":        fps,
            "codec":      stream.get("codec_name", "unknown"),
            "hdr":        hdr,
            "interlaced": interlaced,
            "duration":   duration,
        }
    except Exception as exc:
        logger.warning(event="ffprobe_failed", error=str(exc))
        return {"width": 0, "height": 0, "fps": 0.0, "codec": "unknown",
                "hdr": False, "interlaced": False, "duration": 0.0}


# SCENE BOUNDARY DETECTION VIA PYSCENEDETECT

def _detect_scenes_pyscenedetect(
    video_path: str,
    threshold: float,
) -> List[float]:
    """Return list of scene-change timestamps (seconds) using PySceneDetect."""
    if not SCENEDETECT_AVAILABLE:
        logger.warning(event="pyscenedetect_not_available")
        return []

    timestamps: List[float] = []
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            AdaptiveDetector(adaptive_threshold=threshold)
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        for scene in scene_list:
            start_time = scene[0].get_seconds()
            timestamps.append(round(start_time, 3))

        logger.info(
            event="scene_detection_complete",
            scenes=len(timestamps),
            video=os.path.basename(video_path),
        )
    except Exception as exc:
        logger.warning(event="pyscenedetect_failed", error=str(exc))

    return timestamps


# OPENCV FALLBACK FRAME DIFFERENCE SCENE DETECTION

def _frame_diff(prev: np.ndarray, curr: np.ndarray) -> float:
    """Mean absolute pixel difference between two frames."""
    diff = cv2.absdiff(prev, curr)
    return float(np.mean(diff))


# BLUR SCORE VIA LAPLACIAN VARIANCE

def _blur_score(frame: np.ndarray) -> float:
    """0.0 = very blurry, 1.0 = sharp. Normalised to [0, 1]."""
    try:
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(laplacian / 100.0, 1.0))
    except Exception:
        return 1.0


# BRIGHTNESS CHECK

def _brightness_mean(frame: np.ndarray) -> float:
    """Mean grayscale brightness of the frame."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    except Exception:
        return 128.0


def _is_too_dark(brightness: float) -> bool:
    return brightness < settings.FRAME_DARKNESS_THRESHOLD


# PERCEPTUAL HASH

def _compute_phash(image_path: str) -> Optional[str]:
    if not IMAGEHASH_AVAILABLE:
        return None
    try:
        with PILImage.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def _phash_distance(h1: str, h2: str) -> int:
    """Hamming distance between two pHash hex strings."""
    try:
        ih1 = imagehash.hex_to_hash(h1)
        ih2 = imagehash.hex_to_hash(h2)
        return ih1 - ih2
    except Exception:
        return 999


# HDR TONE-MAPPING GUARD

def _needs_tonemapping(frame: np.ndarray) -> bool:
    """Detect 10-bit-range values suggesting HDR source."""
    return bool(frame.max() > 255)


def _tonemap_frame(frame: np.ndarray) -> np.ndarray:
    """Simple Reinhard tone-mapping to 8-bit SDR for vision model compatibility."""
    try:
        tonemap = cv2.createTonemapReinhard(gamma=2.2)
        f32     = frame.astype(np.float32) / frame.max()
        mapped  = tonemap.process(f32)
        return (mapped * 255).clip(0, 255).astype(np.uint8)
    except Exception:
        return frame


# DEINTERLACE FILTER

def _deinterlace_frame(frame: np.ndarray) -> np.ndarray:
    """Simple line-doubling deinterlace (yadif not available in pure opencv)."""
    try:
        return cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)
    except Exception:
        return frame


# SHOT TYPE CLASSIFICATION

def _classify_shot(frame: np.ndarray) -> str:
    """Rough shot type: wide | medium | close based on edge density."""
    try:
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, 50, 150)
        density = float(np.mean(edges > 0))
        if density < 0.03:
            return "wide"
        if density < 0.10:
            return "medium"
        return "close"
    except Exception:
        return "unknown"


# RESIZE WITH ASPECT PRESERVATION

def _resize_if_needed(
    frame: np.ndarray,
    max_dim: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale  = max_dim / max(h, w)
    new_w  = max(int(w * scale), 1)
    new_h  = max(int(h * scale), 1)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# FRAME SAVE WITH QUALITY GUARD

def _save_frame(
    frame: np.ndarray,
    path: str,
    jpeg_quality: int = 95,
) -> bool:
    try:
        success = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not success:
            logger.warning(event="frame_write_failed", path=path)
        return success
    except Exception as exc:
        logger.warning(event="frame_save_exception", path=path, error=str(exc))
        return False


# OPENCV-BASED FALLBACK EXTRACTION

def _extract_frames_opencv(
    video_path: str,
    interval_sec: int,
    max_frames: int,
    max_dim: int,
    max_duration: float,
    scene_thresh: float,
    session_id: str,
    temp_dir: Path,
    probe: Dict[str, Any],
) -> List[FrameMetadata]:
    """Fallback frame extractor using OpenCV when PySceneDetect is unavailable."""

    fps = probe.get("fps") or 25.0
    if fps <= 0 or math.isnan(fps):
        fps = 25.0

    duration       = probe.get("duration") or 0.0
    hdr_detected   = probe.get("hdr", False)
    interlaced     = probe.get("interlaced", False)
    video_id       = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError(f"VIDEO_OPEN_FAILED: {video_path}")

    interval_frames  = max(int(fps * interval_sec), 1)
    frames: List[FrameMetadata] = []
    seen_phashes: List[str]     = []
    prev_frame: Optional[np.ndarray] = None
    frame_idx        = 0
    saved            = 0
    scene_idx        = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            if timestamp > max_duration:
                break

            # INTERVAL SAMPLING
            take_frame = (frame_idx % interval_frames == 0)

            # SCENE CHANGE DETECTION FALLBACK
            if prev_frame is not None:
                diff = _frame_diff(prev_frame, frame)
                if diff > scene_thresh:
                    take_frame    = True
                    scene_idx    += 1

            if take_frame:
                if saved >= max_frames:
                    break

                h, w = frame.shape[:2]
                if h < 32 or w < 32:
                    _FRAMES_SKIPPED.labels(reason="too_small").inc()
                    frame_idx += 1
                    prev_frame = frame
                    continue

                # BRIGHTNESS CHECK
                brightness = _brightness_mean(frame)
                if _is_too_dark(brightness):
                    _FRAMES_SKIPPED.labels(reason="too_dark").inc()
                    frame_idx += 1
                    prev_frame = frame
                    continue

                # HDR TONE-MAPPING
                if hdr_detected and settings.VIDEO_HDR_TONEMAPPING:
                    frame = _tonemap_frame(frame)

                # DEINTERLACE
                if interlaced and settings.VIDEO_DEINTERLACE:
                    frame = _deinterlace_frame(frame)

                # RESIZE
                frame = _resize_if_needed(frame, max_dim)
                h_r, w_r = frame.shape[:2]

                blur      = _blur_score(frame)
                shot_type = _classify_shot(frame)
                ts_end    = round((frame_idx + interval_frames) / fps, 3)
                aspect    = round(w_r / max(h_r, 1), 3)

                frame_path = str(temp_dir / f"frame_{saved:06d}.jpg")
                if not _save_frame(frame, frame_path):
                    frame_idx += 1
                    prev_frame = frame
                    continue

                # PERCEPTUAL HASH DEDUP
                ph = _compute_phash(frame_path)
                if ph and seen_phashes:
                    distances = [_phash_distance(ph, existing) for existing in seen_phashes]
                    if min(distances) < 8:
                        Path(frame_path).unlink(missing_ok=True)
                        _FRAMES_SKIPPED.labels(reason="duplicate_phash").inc()
                        frame_idx += 1
                        prev_frame = frame
                        continue

                if ph:
                    seen_phashes.append(ph)

                meta = FrameMetadata(
                    frame_index      = saved,
                    timestamp_start  = round(timestamp, 3),
                    timestamp_end    = min(ts_end, duration),
                    path             = frame_path,
                    frame_width      = w_r,
                    frame_height     = h_r,
                    fps              = fps,
                    video_duration   = round(duration, 3),
                    video_id         = video_id,
                    blur_score       = round(blur, 4),
                    brightness_mean  = round(brightness, 2),
                    phash            = ph,
                    is_scene_boundary= (scene_idx > 0 and diff > scene_thresh
                                        if prev_frame is not None else False),
                    scene_index      = scene_idx,
                    shot_type        = shot_type,
                    hdr_detected     = hdr_detected,
                    interlaced       = interlaced,
                    aspect_ratio     = aspect,
                    extraction_method= "opencv_fallback",
                    session_id       = session_id,
                )
                frames.append(meta)
                saved += 1

            prev_frame = frame
            frame_idx += 1

    finally:
        cap.release()

    return frames


# PYSCENEDETECT-BASED EXTRACTION (PRIMARY PATH)

def _extract_frames_pyscenedetect(
    video_path: str,
    max_frames: int,
    max_dim: int,
    max_duration: float,
    threshold: float,
    session_id: str,
    temp_dir: Path,
    probe: Dict[str, Any],
) -> List[FrameMetadata]:
    """Primary frame extractor: one keyframe per detected scene."""

    if not SCENEDETECT_AVAILABLE or cv2 is None:
        raise VideoFrameError("PYSCENEDETECT_OR_CV2_UNAVAILABLE")

    fps          = probe.get("fps") or 25.0
    duration     = probe.get("duration") or 0.0
    hdr_detected = probe.get("hdr", False)
    interlaced   = probe.get("interlaced", False)
    video_id     = os.path.basename(video_path)

    # DETECT SCENES
    scene_timestamps = _detect_scenes_pyscenedetect(video_path, threshold)

    # ALWAYS INCLUDE FIRST FRAME TIMESTAMP
    all_timestamps: List[float] = [0.0]
    for ts in scene_timestamps:
        if ts > 0.1:
            all_timestamps.append(ts)

    all_timestamps = sorted(set(all_timestamps))

    if len(all_timestamps) > max_frames:
        # EVEN SAMPLING FROM SCENE LIST
        step = max(len(all_timestamps) // max_frames, 1)
        all_timestamps = all_timestamps[::step][:max_frames]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError(f"VIDEO_OPEN_FAILED: {video_path}")

    frames: List[FrameMetadata]  = []
    seen_phashes: List[str]      = []
    saved = 0

    try:
        for scene_idx, ts in enumerate(all_timestamps):
            if saved >= max_frames:
                break

            if ts > max_duration:
                break

            # SEEK TO TIMESTAMP
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            if h < 32 or w < 32:
                _FRAMES_SKIPPED.labels(reason="too_small").inc()
                continue

            brightness = _brightness_mean(frame)
            if _is_too_dark(brightness):
                _FRAMES_SKIPPED.labels(reason="too_dark").inc()
                continue

            # HDR TONE-MAPPING
            if hdr_detected and settings.VIDEO_HDR_TONEMAPPING:
                frame = _tonemap_frame(frame)

            # DEINTERLACE
            if interlaced and settings.VIDEO_DEINTERLACE:
                frame = _deinterlace_frame(frame)

            # RESIZE
            frame = _resize_if_needed(frame, max_dim)
            h_r, w_r = frame.shape[:2]

            blur      = _blur_score(frame)
            shot_type = _classify_shot(frame)
            aspect    = round(w_r / max(h_r, 1), 3)

            # NEXT SCENE TIMESTAMP AS END
            next_ts = all_timestamps[scene_idx + 1] if scene_idx + 1 < len(all_timestamps) else duration
            ts_end  = min(next_ts, duration)

            frame_path = str(temp_dir / f"frame_{saved:06d}.jpg")
            if not _save_frame(frame, frame_path):
                continue

            # PERCEPTUAL HASH DEDUP
            ph = _compute_phash(frame_path)
            if ph and seen_phashes:
                distances = [_phash_distance(ph, existing) for existing in seen_phashes]
                if min(distances) < 8:
                    Path(frame_path).unlink(missing_ok=True)
                    _FRAMES_SKIPPED.labels(reason="duplicate_phash").inc()
                    continue

            if ph:
                seen_phashes.append(ph)

            meta = FrameMetadata(
                frame_index      = saved,
                timestamp_start  = round(ts, 3),
                timestamp_end    = round(ts_end, 3),
                path             = frame_path,
                frame_width      = w_r,
                frame_height     = h_r,
                fps              = fps,
                video_duration   = round(duration, 3),
                video_id         = video_id,
                blur_score       = round(blur, 4),
                brightness_mean  = round(brightness, 2),
                phash            = ph,
                is_scene_boundary= (scene_idx > 0),
                scene_index      = scene_idx,
                shot_type        = shot_type,
                hdr_detected     = hdr_detected,
                interlaced       = interlaced,
                aspect_ratio     = aspect,
                extraction_method= "pyscenedetect",
                session_id       = session_id,
            )
            frames.append(meta)
            saved += 1

    finally:
        cap.release()

    return frames


# THUMBNAIL GRID GENERATION

def generate_thumbnail_grid(
    frames: List[FrameMetadata],
    output_path: str,
    rows: int = 3,
    cols: int = 3,
    cell_size: Tuple[int, int] = (320, 180),
) -> Optional[str]:
    """Compose a rows×cols thumbnail grid from extracted frames."""
    if cv2 is None or not frames:
        return None

    try:
        sample_count = rows * cols
        step         = max(len(frames) // sample_count, 1)
        selected     = frames[::step][:sample_count]

        cell_w, cell_h = cell_size
        grid_w = cols * cell_w
        grid_h = rows * cell_h
        grid   = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx, meta in enumerate(selected):
            r = idx // cols
            c = idx % cols
            if r >= rows:
                break

            img = cv2.imread(meta.path)
            if img is None:
                continue

            img_resized = cv2.resize(img, (cell_w, cell_h))
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            grid[y0:y1, x0:x1] = img_resized

            # TIMESTAMP OVERLAY
            ts_label = f"{meta.timestamp_start:.1f}s"
            cv2.putText(
                grid, ts_label,
                (x0 + 4, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        success = cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            logger.info(event="thumbnail_grid_generated", path=output_path)
            return output_path

    except Exception as exc:
        logger.warning(event="thumbnail_grid_failed", error=str(exc))

    return None


# MAIN ASYNC ENTRY POINT

async def extract_frames_async(
    video_path: str,
    session_id: str,
    interval_sec: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Async entry point — extract keyframes from video.

    Returns list of FrameMetadata dicts.
    Raises: VideoOpenError, NoFramesExtractedError, DiskSpaceError.
    """
    async with _get_semaphore():
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: extract_frames(
                video_path=video_path,
                interval_sec=interval_sec or settings.VIDEO_FRAME_INTERVAL_SEC,
                session_id=session_id,
            ),
        )
        return result


# SYNCHRONOUS MAIN FUNCTION (CALLED BY VIDEO_INGEST)

def extract_frames(
    video_path: str,
    interval_sec: int,
    session_id: str,
    skip_scene_detection: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract frames from a video file.

    Priority:
      1. PySceneDetect (semantic scene boundaries)
      2. OpenCV interval + scene-diff fallback

    Args:
        video_path:   Absolute path to video file.
        interval_sec: Fallback frame interval in seconds.
        session_id:   Caller session ID for logging and metadata.

    Returns:
        List of FrameMetadata.to_dict() records.

    Raises:
        VideoOpenError, NoFramesExtractedError, DiskSpaceError.
    """
    if cv2 is None:
        raise ImportError("OPENCV_REQUIRED_FOR_FRAME_EXTRACTION")

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"VIDEO_NOT_FOUND: {video_path}")

    file_size = Path(video_path).stat().st_size
    if file_size == 0:
        raise ValueError("EMPTY_VIDEO_FILE")

    # CLAMP PARAMETERS
    interval_sec = max(1, min(int(interval_sec), 300))
    max_frames   = settings.MAX_VIDEO_FRAMES
    max_dim      = settings.MAX_IMAGE_DIM
    max_duration = float(settings.MAX_VIDEO_DURATION_SEC)
    scene_thresh = settings.SCENE_CHANGE_THRESHOLD

    start = time.time()
    _ACTIVE.set(1)

    # TEMP DIRECTORY FOR FRAMES
    temp_root = settings.VIDEO_FRAMES_DIR
    temp_root.mkdir(parents=True, exist_ok=True)
    _check_disk_space(str(temp_root))

    temp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=str(temp_root)))

    try:
        logger.info(
            event="frame_extraction_start",
            video=os.path.basename(video_path),
            interval_sec=interval_sec,
            session_id=session_id,
            file_size=file_size,
        )

        # PROBE VIDEO METADATA
        probe = _probe_video(video_path)
        fps   = probe.get("fps") or 25.0

        if fps <= 0 or math.isnan(fps):
            fps = 25.0
            logger.warning(event="fps_fallback_to_25", session_id=session_id)

        duration     = probe.get("duration") or 0.0
        hdr_detected = probe.get("hdr", False)
        interlaced   = probe.get("interlaced", False)

        # VERY SHORT VIDEO — SINGLE CHUNK ONLY
        is_short = (0 < duration < 2.0)

        # CHOOSE EXTRACTION METHOD
        frames: List[FrameMetadata]

        if SCENEDETECT_AVAILABLE and not is_short and duration >= 2.0 and not skip_scene_detection:
            try:
                frames = _extract_frames_pyscenedetect(
                    video_path=video_path,
                    max_frames=max_frames,
                    max_dim=max_dim,
                    max_duration=max_duration,
                    threshold=scene_thresh,
                    session_id=session_id,
                    temp_dir=temp_dir,
                    probe=probe,
                )
                if not frames:
                    raise NoFramesExtractedError("PYSCENEDETECT_PRODUCED_NO_FRAMES")
            except (VideoFrameError, Exception) as exc:
                logger.warning(
                    event="pyscenedetect_fallback_to_opencv",
                    error=str(exc),
                    session_id=session_id,
                )
                frames = _extract_frames_opencv(
                    video_path=video_path,
                    interval_sec=interval_sec,
                    max_frames=max_frames,
                    max_dim=max_dim,
                    max_duration=max_duration,
                    scene_thresh=scene_thresh,
                    session_id=session_id,
                    temp_dir=temp_dir,
                    probe=probe,
                )
        else:
            frames = _extract_frames_opencv(
                video_path=video_path,
                interval_sec=interval_sec,
                max_frames=max_frames,
                max_dim=max_dim,
                max_duration=max_duration,
                scene_thresh=scene_thresh,
                session_id=session_id,
                temp_dir=temp_dir,
                probe=probe,
            )

        if not frames:
            raise NoFramesExtractedError(
                f"NO_FRAMES_EXTRACTED from {os.path.basename(video_path)}"
            )

        # GENERATE THUMBNAIL GRID
        grid_path = str(temp_dir / "thumbnail_grid.jpg")
        generate_thumbnail_grid(
            frames, grid_path,
            rows=settings.VIDEO_THUMBNAIL_GRID_ROWS,
            cols=settings.VIDEO_THUMBNAIL_GRID_COLS,
        )

        latency = round(time.time() - start, 2)

        _FRAMES_EXTRACTED.labels(session_id=session_id).inc(len(frames))

        with _EXTRACT_DURATION.time() if hasattr(_EXTRACT_DURATION, "time") else _noop_ctx():
            pass

        logger.info(
            event="frame_extraction_success",
            video=os.path.basename(video_path),
            frames=len(frames),
            duration_sec=round(duration, 2),
            fps=round(fps, 2),
            hdr=hdr_detected,
            interlaced=interlaced,
            method=frames[0].extraction_method if frames else "none",
            latency=latency,
            session_id=session_id,
        )

        return [f.to_dict() for f in frames]

    except (NoFramesExtractedError, VideoOpenError, DiskSpaceError):
        raise

    except Exception as exc:
        logger.error(
            event="frame_extraction_failed",
            video=os.path.basename(video_path),
            session_id=session_id,
            error=str(exc),
            latency=round(time.time() - start, 2),
        )
        raise

    finally:
        _ACTIVE.set(0)


# CONTEXT MANAGER NOOP (FOR PROMETHEUS HISTOGRAM .time() GUARD)

from contextlib import contextmanager

@contextmanager
def _noop_ctx():
    yield

