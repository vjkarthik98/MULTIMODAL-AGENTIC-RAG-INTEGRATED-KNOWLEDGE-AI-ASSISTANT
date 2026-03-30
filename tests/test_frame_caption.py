from app.ingestion.video_frames import extract_frames
from app.ingestion.frame_captioner import generate_caption

frames = extract_frames("sample.mp4", interval=10)

for f in frames[:5]:
    print(f["timestamp"], generate_caption(f["path"]))