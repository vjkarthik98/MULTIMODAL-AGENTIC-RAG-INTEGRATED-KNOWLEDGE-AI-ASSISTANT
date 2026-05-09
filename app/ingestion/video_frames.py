import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import cv2
except ImportError:
    cv2 = None

logger = get_logger(__name__)


# FRAME DIFFERENCE

def _frame_diff(prev: np.ndarray, curr: np.ndarray) -> float:
    diff = cv2.absdiff(prev, curr)
    return float(np.mean(diff))


# BRIGHTNESS CHECK

def _is_too_dark(frame: np.ndarray, threshold: float = 15.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


# PERCEPTUAL HASH (optional dedup)

def _phash(image_path: str) -> Optional[str]:
    try:
        import imagehash
        from PIL import Image
        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


# MAIN

def extract_frames(
    video_path: str,
    interval_sec: int,
    session_id: str,
) -> List[Dict]:

    if cv2 is None:
        raise ImportError("OPENCV_REQUIRED")

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"VIDEO_NOT_FOUND: {video_path}")

    # CLAMP INTERVAL
    interval_sec = max(1, min(int(interval_sec or settings.VIDEO_FRAME_INTERVAL_SEC), settings.MAX_VIDEO_DURATION_SEC))
    max_frames   = settings.MAX_VIDEO_FRAMES
    max_duration = settings.MAX_VIDEO_DURATION_SEC
    max_dim      = settings.MAX_IMAGE_DIM
    scene_thresh = settings.SCENE_CHANGE_THRESHOLD

    start = time.time()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("VIDEO_OPEN_FAILED")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 25.0
            logger.warning(event="fps_fallback", session_id=session_id)

        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration          = total_frame_count / fps if fps else 0
        interval_frames   = max(int(fps * interval_sec), 1)

        # STAGING DIR
        temp_root = settings.VIDEO_FRAMES_DIR
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir  = Path(tempfile.mkdtemp(prefix="frames_", dir=str(temp_root)))

        frames:     List[Dict]          = []
        seen_hashes: set                = set()
        prev_frame: Optional[np.ndarray] = None
        frame_idx   = 0
        saved       = 0
        frames_read = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_read += 1
            timestamp    = frame_idx / fps

            if timestamp > max_duration:
                break

            take_frame = False

            # INTERVAL SAMPLING
            if frame_idx % interval_frames == 0:
                take_frame = True

            # SCENE CHANGE DETECTION
            if prev_frame is not None:
                diff = _frame_diff(prev_frame, frame)
                if diff > scene_thresh:
                    take_frame = True

            if take_frame:

                if saved >= max_frames:
                    break

                try:
                    h, w = frame.shape[:2]

                    if h < 32 or w < 32:
                        frame_idx += 1
                        continue

                    # BRIGHTNESS GUARD
                    if _is_too_dark(frame):
                        frame_idx += 1
                        continue

                    # RESIZE IF NEEDED
                    if max(h, w) > max_dim:
                        scale = max(h, w) / max_dim
                        frame = cv2.resize(
                            frame,
                            (int(w / scale), int(h / scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                        h, w = frame.shape[:2]

                    frame_path = temp_dir / f"frame_{saved}.jpg"

                    # JPEG QUALITY 95
                    success = cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )

                    if not success:
                        frame_idx += 1
                        continue

                    # PERCEPTUAL DEDUP
                    ph = _phash(str(frame_path))
                    if ph and ph in seen_hashes:
                        frame_path.unlink(missing_ok=True)
                        frame_idx += 1
                        continue

                    if ph:
                        seen_hashes.add(ph)

                    frames.append({
                        "path":            str(frame_path),
                        "timestamp_start": round(timestamp, 2),
                        "timestamp_end":   round(timestamp + (1.0 / fps), 2),
                        "frame_index":     saved,
                        "frame_width":     w,
                        "frame_height":    h,
                        "fps":             fps,
                        "video_duration":  duration,
                        "video_id":        os.path.basename(video_path),
                    })

                    saved += 1

                except Exception as e:
                    logger.warning(
                        event="frame_save_error",
                        frame_idx=frame_idx,
                        error=str(e),
                        session_id=session_id,
                    )

            prev_frame = frame
            frame_idx  += 1

        if not frames:
            raise ValueError("NO_FRAMES_EXTRACTED")

        latency = round(time.time() - start, 2)

        logger.info(
            event="frame_extract_success",
            frames=len(frames),
            frames_read=frames_read,
            duration=round(duration, 2),
            fps=fps,
            latency=latency,
            session_id=session_id,
        )

        return frames

    except Exception as e:
        logger.error(
            event="frame_extract_failed",
            video=os.path.basename(video_path),
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 2),
        )
        raise

    finally:
        cap.release()