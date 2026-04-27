import pytest
import os
from app.ingestion.audio_ingest import ingest

TEST_SESSION_ID = "test_audio_ingest"



# HELPER: CREATE DUMMY AUDIO (SILENT)

def create_dummy_audio(file_path):
    import wave

    with wave.open(str(file_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)  # 1 sec silence



# TEST: INVALID PATH

def test_invalid_path():
    with pytest.raises(ValueError):
        ingest("non_existent_audio.wav", session_id=TEST_SESSION_ID)



# TEST: MISSING SESSION

def test_missing_session(tmp_path):
    file_path = tmp_path / "test.wav"
    create_dummy_audio(file_path)

    with pytest.raises(ValueError):
        ingest(str(file_path), session_id=None)



# TEST: SILENT AUDIO → SHOULD FAIL

def test_audio_ingest_silent_audio(tmp_path):

    file_path = tmp_path / "test.wav"
    create_dummy_audio(file_path)

    with pytest.raises(RuntimeError):
        ingest(str(file_path), session_id=TEST_SESSION_ID)



# TEST: EMPTY AUDIO CASE

def test_audio_ingest_empty_audio(tmp_path):
    file_path = tmp_path / "empty.wav"
    create_dummy_audio(file_path)

    with pytest.raises(RuntimeError):
        ingest(str(file_path), session_id=TEST_SESSION_ID)



