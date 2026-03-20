from app.ingestion.audio_ingest import ingest

doc = ingest("sample.mp3")


print(doc.text)
print(doc.metadata)