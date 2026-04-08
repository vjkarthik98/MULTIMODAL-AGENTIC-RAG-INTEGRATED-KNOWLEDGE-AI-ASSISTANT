import os
import subprocess
import shutil
import logging

FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg.exe")

from app.ingestion.schema import IngestedDocument
from app.ingestion.video_frames import extract_frames
from app.ingestion.frame_captioner import generate_caption
from app.ingestion.audio_ingest import ingest as audio_ingest
from datetime import datetime

# ✅ Logger
logger = logging.getLogger(__name__)


def ingest(file_path: str):
    try:
        logger.info(f"[VideoIngest] Starting ingestion | file={file_path}")

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

        if result.returncode != 0:
            logger.error(f"[VideoIngest] FFmpeg error | {result.stderr}")
            raise RuntimeError("Audio extraction failed")

        if not os.path.exists(audio_path):
            logger.error(f"[VideoIngest] Audio file not created | path={audio_path}")
            raise RuntimeError(f"Audio file not created: {audio_path}")

        # Part 2: Audio + Transcription
        audio_docs = audio_ingest(audio_path)

        logger.info(f"[VideoIngest] Audio segments extracted | count={len(audio_docs)}")

        for doc in audio_docs:
            doc.metadata["modality"] = "video_audio"
            doc.metadata["source"] = os.path.basename(file_path)
            documents.append(doc)

        # cleanup audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Part 3: Frames -> Captions
        frames = extract_frames(file_path, interval=10)

        logger.info(f"[VideoIngest] Frames extracted | count={len(frames)}")

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
        shutil.rmtree("temp_frames", ignore_errors=True)

        if not documents:
            logger.error(f"[VideoIngest] No content extracted | file={file_path}")
            raise ValueError("No content extracted from video")

        logger.info(
            f"[VideoIngest] Completed | file={file_path} | total_docs={len(documents)}"
        )

        return documents

    except Exception as e:
        logger.error(f"[VideoIngest] Failed | file={file_path} | error={str(e)}")
        raise RuntimeError(f"Video ingestion failed: {str(e)}")