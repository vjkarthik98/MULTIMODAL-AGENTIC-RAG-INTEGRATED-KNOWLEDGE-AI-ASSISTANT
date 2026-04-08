import cv2
import os
import logging

# Logger
logger = logging.getLogger(__name__)


def extract_frames(video_path: str, output_dir: str = "temp_frames", interval: int = 30):
    """
    Extract frames every N frames

    Args:
        video_path: input video
        output_dir: where frames will be saved
        interval: extract every N frames
    """

    try:
        logger.info(f"[VideoFrames] Starting extraction | video={video_path}")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frames = []
        frame_count = 0
        saved_count = 0

        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % interval == 0:
                timestamp = frame_count / fps if fps else 0
                frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")

                cv2.imwrite(frame_path, frame)

                frames.append({
                    "path": frame_path,
                    "timestamp": timestamp
                })

                saved_count += 1

            frame_count += 1

        cap.release()

        logger.info(
            f"[VideoFrames] Extraction completed | video={video_path} | frames_saved={saved_count}"
        )

        return frames

    except Exception as e:
        logger.error(f"[VideoFrames] Failed | video={video_path} | error={str(e)}")
        raise