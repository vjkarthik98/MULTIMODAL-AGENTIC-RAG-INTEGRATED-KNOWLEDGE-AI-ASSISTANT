from app.ingestion.router import detect_modality

print(detect_modality("test.jpg"))
print(detect_modality("filet.pdf"))
print(detect_modality("aduio.mp3"))
print(detect_modality("video.mp4"))