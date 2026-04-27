from app.ingestion.pipeline import process_file

doc = process_file("sample.mp4")

print(doc.text)
print(doc.metadata)