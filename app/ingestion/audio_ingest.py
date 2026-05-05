import hashlib
import os
import time
import uuid
from typing import List

from pydub import AudioSegment, silence

from app.core.config import settings
from app.core.model_loader import model_loader
from app.ingestion.schema import IngestedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  HASH 
def _generate_file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


#  VALIDATION 
def _validate_audio(file_path: str):
    audio = AudioSegment.from_file(file_path)

    if audio.duration_seconds <= 0:
        raise ValueError("INVALID_AUDIO_DURATION")

    if audio.frame_rate <= 0:
        raise ValueError("INVALID_SAMPLE_RATE")

    return audio


#  MAIN 
def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise ValueError("FILE_TOO_LARGE")

    start_time = time.time()

    source_name = os.path.basename(file_path)
    source_path = os.path.abspath(file_path)

    doc_id = str(uuid.uuid4())
    file_hash = _generate_file_hash(file_path)

    logger.info(event="audio_ingest_start", file=file_path)

    try:
        #  VALIDATE 
        audio = _validate_audio(file_path)

        duration_total = audio.duration_seconds

        #  SILENCE 
        silent_ranges = silence.detect_silence(
            audio,
            min_silence_len=200,
            silence_thresh=-40
        )

        #  MODEL 
        whisper = model_loader.get_whisper()
        segments_iter, info = whisper.transcribe(file_path)

        documents: List[IngestedDocument] = []

        language = getattr(info, "language", None)

        max_segments = getattr(settings, "MAX_AUDIO_SEGMENTS", 500)

        for idx, segment in enumerate(segments_iter):

            if idx >= max_segments:
                logger.warning(event="audio_segment_limit")
                break

            text = (getattr(segment, "text", "") or "").strip()
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", start))

            if not text or end <= start:
                continue

            duration = end - start

            #  SILENCE CHECK 
            inaudible = any(s <= start * 1000 <= e for s, e in silent_ranges)

            if inaudible:
                text = "[INAUDIBLE]"

            #  CONFIDENCE 
            avg_logprob = getattr(segment, "avg_logprob", None)
            no_speech_prob = getattr(segment, "no_speech_prob", None)

            confidence = 1.0
            if avg_logprob is not None:
                confidence = max(0.0, min(1.0, 1 + avg_logprob))

            #  SPEECH RATE 
            words = len(text.split())
            wpm = (words / duration) * 60 if duration > 0 else 0

            speed_flag = False
            if wpm > 250 or wpm < 60:
                speed_flag = True

            #  QUALITY 
            hallucination_risk = "low"
            if confidence < 0.5:
                hallucination_risk = "high"

            doc = IngestedDocument(
                text=f"{text}",
                modality="audio",
                subtype="speech",
                source_type="audio",
                source=source_name,

                structure={
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "file_hash": file_hash,
                    "source_path": source_path,
                    "segment_index": idx,
                    "timestamp_start": round(start, 2),
                    "timestamp_end": round(end, 2),
                    "duration": round(duration, 2),
                    "language": language,
                    "confidence": confidence,
                    "no_speech_prob": no_speech_prob,
                    "avg_logprob": avg_logprob,
                    "hallucination_risk": hallucination_risk,
                    "speed_corrupted": speed_flag,
                    "content_type": "speech_segment",
                    "ingestion_time": time.time(),
                },

                extra_metadata={
                    "modality_weight": 1.1,
                    "importance_score": confidence,
                    "data_quality_score": confidence,
                },
            ).finalize()

            documents.append(doc)

        if not documents:
            raise ValueError("NO_VALID_AUDIO")

        latency = round(time.time() - start_time, 2)

        logger.info(
            event="audio_ingest_success",
            segments=len(documents),
            latency=latency
        )

        return documents

    except Exception as e:
        logger.error(event="audio_ingest_failed", error=str(e))
        raise