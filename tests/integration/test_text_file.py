from app.ingestion.text_ingest import ingest

doc = ingest("sample.txt")


print(doc.text)
print(doc.metadata)