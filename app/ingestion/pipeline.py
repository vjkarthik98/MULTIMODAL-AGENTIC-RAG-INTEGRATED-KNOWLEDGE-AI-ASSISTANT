from app.ingestion.router import detect_modality

from app.ingestion.text_ingest import ingest as text_ingest
from app.ingestion.image_ingest import ingest as image_ingest
from app.ingestion.audio_ingest import ingest as audio_ingest
from app.ingestion.video_ingest import ingest as video_ingest

from app.ingestion.schema import IngestedDocument


def process_file(file_path: str) -> list[IngestedDocument]:
    modality = detect_modality(file_path)

    if modality == "text":
        return text_ingest(file_path)
    
    elif modality == "image": 
        return image_ingest(file_path)
    
    elif modality == "audio":
        return audio_ingest(file_path)
    
    elif modality == "video":
        return video_ingest(file_path)
    
    else:
        raise ValueError(f"Unsupported modality: {modality}")