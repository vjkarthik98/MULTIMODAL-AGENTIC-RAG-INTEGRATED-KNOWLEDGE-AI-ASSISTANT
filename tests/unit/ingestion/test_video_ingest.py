import pytest
import os
from unittest.mock import patch, MagicMock

from app.ingestion.video_ingest import ingest
from app.ingestion.schema import IngestedDocument

TEST_SESSION_ID = "test_video_ingest"



# HELPER: CREATE DUMMY VIDEO FILE

def create_dummy_video(file_path):
    # Minimal dummy file (not real video, but enough for path validation)
    with open(file_path, "wb") as f:
        f.write(b"\x00\x00\x00\x00")



# TEST: INVALID PATH

def test_invalid_path():
    with pytest.raises(ValueError):
        ingest("non_existent_video.mp4", session_id=TEST_SESSION_ID)



# TEST: MISSING SESSION

def test_missing_session(tmp_path):
    file_path = tmp_path / "test.mp4"
    create_dummy_video(file_path)

    with pytest.raises(ValueError):
        ingest(str(file_path), session_id=None)



# TEST: FFMPEG FAILURE

@patch("app.ingestion.video_ingest.subprocess.run")
def test_ffmpeg_failure(mock_run, tmp_path):
    file_path = tmp_path / "test.mp4"
    create_dummy_video(file_path)

    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "ffmpeg error"

    with pytest.raises(RuntimeError):
        ingest(str(file_path), session_id=TEST_SESSION_ID)



# TEST: SUCCESSFUL INGEST (MOCKED PIPELINE)

@patch("app.ingestion.video_ingest.extract_frames")
@patch("app.ingestion.video_ingest.generate_caption")
@patch("app.ingestion.video_ingest.audio_ingest")
@patch("app.ingestion.video_ingest.subprocess.run")
def test_video_ingest_success(
    mock_run,
    mock_audio_ingest,
    mock_caption,
    mock_frames,
    tmp_path
):
    file_path = tmp_path / "test.mp4"
    create_dummy_video(file_path)

    # Mock FFmpeg success
    mock_run.return_value.returncode = 0

    # Mock audio ingestion output
    mock_audio_ingest.return_value = [
        IngestedDocument(
            text="hello world",
            modality="audio",
            subtype="speech",
            source_type="audio",
            source="audio.wav",
            page=None,
            chunk_id=0,
            structure={
                "start_time": 0,
                "end_time": 2
            }
        )
    ]

    # Mock frames
    mock_frames.return_value = [
        {"path": "frame1.jpg", "timestamp": 2}
    ]

    # Mock caption
    mock_caption.return_value = "a man speaking"

    docs = ingest(str(file_path), session_id=TEST_SESSION_ID)

    assert isinstance(docs, list)
    assert len(docs) > 0

    for doc in docs:
        assert doc.modality == "video"
        assert doc.text is not None
        assert len(doc.text) > 0

        # Structure validation
        structure = doc.structure
        assert "doc_id" in structure
        assert "session_id" in structure
        assert "file_hash" in structure

        # Either audio or frame metadata must exist
        assert (
            "segment_index" in structure or
            "frame_index" in structure
        )



# TEST: NO CONTENT EXTRACTED

@patch("app.ingestion.video_ingest.extract_frames")
@patch("app.ingestion.video_ingest.audio_ingest")
@patch("app.ingestion.video_ingest.subprocess.run")
def test_no_content(mock_run, mock_audio, mock_frames, tmp_path):
    file_path = tmp_path / "test.mp4"
    create_dummy_video(file_path)

    mock_run.return_value.returncode = 0
    mock_audio.return_value = []
    mock_frames.return_value = []

    with pytest.raises(RuntimeError):
        ingest(str(file_path), session_id=TEST_SESSION_ID)