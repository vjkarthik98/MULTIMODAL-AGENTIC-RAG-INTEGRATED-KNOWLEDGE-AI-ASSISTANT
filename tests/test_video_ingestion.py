from app.ingestion.video_ingest import ingest

doc = ingest("sample.mp4")

print(doc.text)
print(doc.metadata)
