import math
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict

from app.core.config import settings
from app.utils.logger import get_logger

try:
    import cv2
except ImportError:
    cv2 = None


logger = get_logger(__name__)


# EXTRACT FRAMES FROM VIDEO
def extract_frames(
    video_path: str,
    interval_sec: int = None,
    session_id: str = "default",
) -> List[Dict]:

    # VALIDATE DEPENDENCIES
    if cv2 is None:
        raise ImportError("OPENCV REQUIRED")

    # VALIDATE INPUTS
    if not session_id:
        raise ValueError("SESSION_ID REQUIRED")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} NOT FOUND")

    # LOAD CONFIG
    interval_sec = interval_sec or settings.VIDEO_FRAME_INTERVAL_SEC
    max_frames = settings.MAX_VIDEO_FRAMES
    max_duration = settings.MAX_VIDEO_DURATION_SEC

    if interval_sec <= 0:
        raise ValueError("INTERVAL_SEC MUST BE > 0")

    start = time.time()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("FAILED TO OPEN VIDEO")

    try:
        # READ FPS
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or math.isnan(fps) or fps <= 0:
            logger.warning("[FrameExtract] INVALID FPS -> USING FALLBACK 25")
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0

        if duration > max_duration:
            logger.warning("[FrameExtract] VIDEO DURATION EXCEEDS LIMIT")

        interval_frames = max(int(round(fps * interval_sec)), 1)

        # TEMP DIRECTORY CREATION
        temp_root = settings.DATA_DIR / "temp_frames"
        temp_root.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=temp_root))

        frames = []
        frame_count = 0
        saved = 0

        max_dim = getattr(settings, "MAX_IMAGE_DIM", 1024)

        # FRAME EXTRACTION LOOP
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_count / fps

            # DURATION LIMIT
            if timestamp > max_duration:
                logger.warning("[FrameExtract] DURATION CUTOFF REACHED")
                break

            if frame_count % interval_frames == 0:

                # FRAME LIMIT
                if saved >= max_frames:
                    logger.warning("[FrameExtract] FRAME LIMIT REACHED")
                    break

                try:
                    # RESIZE FRAME
                    h, w = frame.shape[:2]

                    scale = max(h, w) / max_dim if max(h, w) > max_dim else 1

                    if scale > 1:
                        new_w = int(w / scale)
                        new_h = int(h / scale)
                        frame = cv2.resize(frame, (new_w, new_h))

                    frame_path = temp_dir / f"frame_{saved}.jpg"

                    # SAVE FRAME
                    success_write = cv2.imwrite(str(frame_path), frame)

                    if not success_write:
                        logger.warning("[FrameExtract] FAILED TO WRITE FRAME")
                        continue

                    frames.append(
                        {
                            "path": str(frame_path),
                            "timestamp": round(timestamp, 2),
                            "frame_index": saved,
                            "temp_dir": str(temp_dir),
                            "fps": fps,
                            "video_duration": duration,
                        }
                    )

                    saved += 1

                except Exception as e:
                    logger.warning("[FrameExtract][FRAME_ERROR] %s", str(e))

            frame_count += 1

        # FINAL VALIDATION
        if not frames:
            raise ValueError("NO FRAMES EXTRACTED")

        latency = round(time.time() - start, 2)

        logger.info(
            "[FrameExtract][SUCCESS] session_id=%s | frames=%s | latency=%ss",
            session_id,
            len(frames),
            latency
        )

        return frames

    except Exception as e:
        logger.error("[FrameExtract][FAILED] session_id=%s | error=%s", session_id, str(e))
        raise

    finally:
        cap.release()