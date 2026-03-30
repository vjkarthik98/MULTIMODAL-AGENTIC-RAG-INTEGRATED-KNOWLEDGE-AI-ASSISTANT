from app.ingestion.video_frames import extract_frames

frames = extract_frames("sample.mp4")

print(frames[:3])

