import os 
import pdfplumber
import docx
import pandas as pd 

from datetime import datetime
from app.ingestion.schema import IngestedDocument
from app.utils.chunking import chunk_text



def ingest(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    text = ""

    try:
        # PDF
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        # Word
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"

        # Excel
        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                df = pd.read_excel(file_path)

            text = df.astype(str).to_string()

        else:
            raise ValueError("Unsupported document type")
        
    except Exception as e:
        raise ValueError(f"Document parsing failed: {str(e)}")
    
    # Extra Safety
    if not text.strip():
        raise ValueError("No readable content extracted from document")
    
    metadata = {
        "source": os.path.basename(file_path),
        "modality": "document",
        "ingestion_time": datetime.utcnow().isoformat()
    }

    # Chunking
    chunks = chunk_text(text)

    return [
        IngestedDocument(
            text=chunk,
            metadata={**metadata, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    print(file_path) 