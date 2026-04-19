import math
import os
import tempfile
import time
from pathlib import Path

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


logger = get_logger(__name__)


def extract_frames(
    video_path: str,
    interval_sec: int = 2,
    session_id: str = "default",
):
    if cv2 is None:
        raise ImportError("opencv-python is required for video frame extraction")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be greater than 0")
    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} not found")

    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS detected")

    interval_frames = max(int(round(fps * interval_sec)), 1)
    temp_root = settings.DATA_DIR / "temp_frames"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=temp_root))

    frames = []
    frame_count = 0
    saved_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % interval_frames == 0:
                timestamp = round(frame_count / fps, 2)
                frame_path = temp_dir / f"frame_{saved_index}.jpg"
                cv2.imwrite(str(frame_path), frame)

                frames.append(
                    {
                        "path": str(frame_path),
                        "timestamp": timestamp,
                        "frame_index": saved_index,
                        "temp_dir": str(temp_dir),
                    }
                )
                saved_index += 1

            frame_count += 1

        latency = time.time() - start_time
        logger.info(
            "[FrameExtract][SUCCESS] session_id=%s | frames=%s | latency=%.2fs",
            session_id,
            len(frames),
            latency,
        )
        return frames

    except Exception as exc:
        logger.error("[FrameExtract][ERROR] session_id=%s | error=%s", session_id, exc)
        raise

    finally:
        cap.release()
