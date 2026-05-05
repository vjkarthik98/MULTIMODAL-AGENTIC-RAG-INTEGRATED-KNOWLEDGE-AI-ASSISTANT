import math
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import cv2
except ImportError:
    cv2 = None


logger = get_logger(__name__)


#  FRAME DIFFERENCE 
def _frame_diff(prev, curr) -> float:
    diff = cv2.absdiff(prev, curr)
    return float(np.mean(diff))


#  MAIN 
def extract_frames(
    video_path: str,
    interval_sec: int,
    session_id: str
) -> List[Dict]:

    if cv2 is None:
        raise ImportError("OPENCV_REQUIRED")

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    interval_sec = interval_sec or settings.VIDEO_FRAME_INTERVAL_SEC
    max_frames = settings.MAX_VIDEO_FRAMES
    max_duration = settings.MAX_VIDEO_DURATION_SEC

    start = time.time()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("VIDEO_OPEN_FAILED")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 25.0
            logger.warning(event="fps_fallback")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0

        interval_frames = max(int(fps * interval_sec), 1)

        temp_root = settings.DATA_DIR / "temp_frames"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=temp_root))

        frames = []
        prev_frame = None

        frame_idx = 0
        saved = 0

        max_dim = getattr(settings, "MAX_IMAGE_DIM", 1024)
        scene_threshold = getattr(settings, "SCENE_CHANGE_THRESHOLD", 25.0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            if timestamp > max_duration:
                break

            take_frame = False

            # interval sampling
            if frame_idx % interval_frames == 0:
                take_frame = True

            # scene detection
            if prev_frame is not None:
                diff = _frame_diff(prev_frame, frame)
                if diff > scene_threshold:
                    take_frame = True

            if take_frame:
                if saved >= max_frames:
                    break

                try:
                    h, w = frame.shape[:2]

                    if h < 32 or w < 32:
                        frame_idx += 1
                        continue

                    scale = max(h, w) / max_dim if max(h, w) > max_dim else 1
                    if scale > 1:
                        frame = cv2.resize(frame, (int(w / scale), int(h / scale)))

                    frame_path = temp_dir / f"frame_{saved}.jpg"

                    if not cv2.imwrite(str(frame_path), frame):
                        frame_idx += 1
                        continue

                    frames.append({
                        "path": str(frame_path),
                        "timestamp_start": round(timestamp, 2),
                        "timestamp_end": round(timestamp + (1 / fps), 2),
                        "frame_index": saved,
                        "fps": fps,
                        "video_duration": duration,
                        "video_id": os.path.basename(video_path),
                    })

                    saved += 1

                except Exception as e:
                    logger.warning(event="frame_error", error=str(e))

            prev_frame = frame
            frame_idx += 1

        if not frames:
            raise ValueError("NO_FRAMES_EXTRACTED")

        latency = round(time.time() - start, 2)

        logger.info(
            event="frame_extract_success",
            frames=len(frames),
            latency=latency
        )

        return frames

    except Exception as e:
        logger.error(event="frame_extract_failed", error=str(e))
        raise

    finally:
        cap.release()