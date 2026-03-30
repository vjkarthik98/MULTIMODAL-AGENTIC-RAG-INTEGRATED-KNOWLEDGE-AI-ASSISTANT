import cv2
import os

def extract_frames(video_path: str, output_dir: str = "temp_frames", interval: int = 30):
    """
    Extract frames every N frames
    
    Args:
        Video_path: input video
        output_dir: where frames will be saved
        interval: extract every N frames
        """
    
    os.makedirs(output_dir, exist_ok= True)

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
            timestamp = frame_count / fps
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")

            cv2.imwrite(frame_path, frame)

            frames.append({
                "path": frame_path,
                "timestamp": timestamp
            })

            saved_count += 1

        frame_count += 1

    cap.release()

    return frames