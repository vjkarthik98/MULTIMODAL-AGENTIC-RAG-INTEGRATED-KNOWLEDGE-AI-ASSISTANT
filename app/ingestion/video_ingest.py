import os
import subprocess

FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg.exe")
from datetime import datetime

from app.ingestion.schema import IngestedDocument
from app.ingestion.video_frames import extract_frames
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.audio_ingest import ingest as audio_ingest

def ingest(file_path: str):
    try:
        documents = []

        # Part 1: Extract Audio from Video
        audio_path = os.path.splitext(file_path)[0] + ".wav"

        result = subprocess.run(
            [
                FFMPEG_PATH,
                "-i", file_path,
                "-ar", "16000",
                "-ac", "1",
                audio_path
            ],
            capture_output=True,
            text=True
        )

        if result.returncode !=0:
            print("FFmpeg Error:", result.stderr)
            raise RuntimeError("Audio extraction failed")
        
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio file not created: {audio_path}")
        
        # Part 2: Audio + Transcription
        audio_docs = audio_ingest(audio_path)
        print("\n=== AUDIO DOCS ===")
        for doc in audio_docs[:5]:
            print(doc.text, doc.metadata)
        print("==============\n")

        for doc in audio_docs:
            doc.metadata["modality"] = "video_audio"
            doc.metadata["source"] = os.path.basename(file_path)
            documents.append(doc)

            # cleanup audio
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # Part 3: Frames -> Captions
        frames = extract_frames(file_path, interval=10)

        for i, frame in enumerate(frames):
            caption = generate_caption(frame["path"])
            
            documents.append(
                IngestedDocument(
                    text=caption,
                    metadata={
                        "source": os.path.basename(file_path),
                        "modality": "video_frame",
                        "chunk_id": i,
                        "timestamp": frame["timestamp"],
                        "ingestion_time": datetime.utcnow().isoformat()
                    }
                )
            )
        # Cleanup
        import shutil
        shutil.rmtree("temp_frames", ignore_errors=True)

        return documents
    
    except Exception as e:
        raise RuntimeError(f"Video ingestion failed: {str(e)}")