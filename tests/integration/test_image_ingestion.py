from app.ingestion.image_ingest import ingest


doc = ingest("sample.jpg")


print(doc.text)
print(doc.metadata)