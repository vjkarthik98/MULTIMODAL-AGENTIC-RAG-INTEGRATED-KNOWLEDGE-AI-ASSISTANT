from app.ingestion.text_ingest import ingest_pipeline

count = ingest_pipeline("data/raw/sample.txt")

print(f"Inserted {count} chunks into Qdrant")