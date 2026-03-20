from moviepy.editor import VideoFileClip
import os
from datetime import datetime
from app.ingestion.schema import IngestedDocument
from app.ingestion.audio_ingest import ingest as audio_ingest

def ingest(file_path: str) -> IngestedDocument:
    temp_audio = "temp_audio.mp3"

    clip = VideoFileClip(file_path)
    clip.audio.write_audiofile(temp_audio)

    audio_doc = audio_ingest(temp_audio)

    os.remove(temp_audio)

    metadata = {
        "source": os.path.basename(file_path),
        "modality": "video",
        "ingestion_time": datetime.utcnow().isoformat(),
        "derived_from": "audio"
    }

    return IngestedDocument(
        text = audio_doc.text,
        metadata= {**audio_doc.metadata, **metadata}
    )


    

    # Debug check
    if not os.path.exists(temp_audio):
        print("FFmpeg Error:", result.stderr.decode())
        print("FFmpeg stdout:", result.stdout.decode())
        raise RuntimeError("Audio extraction failed")
    
    text = audio_ingest(temp_audio)
    os.remove(temp_audio)

    return text.strip()