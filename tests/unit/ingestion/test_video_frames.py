"""
Unit tests for pure helper functions in app/ingestion/video_frames.py.
Functions that require OpenCV, ffmpeg, or actual video files are not tested here.
"""
import pytest

from app.ingestion.video_frames import (
    FrameMetadata,
    _is_too_dark,
    _phash_distance,
)


# ---------------------------------------------------------------------------
# _is_too_dark
# ---------------------------------------------------------------------------

class TestIsToooDark:

    def test_very_dark_frame_is_too_dark(self):
        assert _is_too_dark(5.0) is True

    def test_bright_frame_not_too_dark(self):
        assert _is_too_dark(200.0) is False

    def test_returns_bool(self):
        assert isinstance(_is_too_dark(100.0), bool)


# ---------------------------------------------------------------------------
# _phash_distance
# ---------------------------------------------------------------------------

class TestPhashDistance:

    def test_identical_hashes_zero_distance(self):
        h = "f0f0f0f0f0f0f0f0"
        dist = _phash_distance(h, h)
        assert dist == 0

    def test_different_hashes_nonzero_distance(self):
        h1 = "0000000000000000"
        h2 = "ffffffffffffffff"
        dist = _phash_distance(h1, h2)
        assert dist > 0

    def test_invalid_hash_returns_999(self):
        dist = _phash_distance("not_valid", "also_not_valid")
        assert dist == 999

    def test_returns_numeric(self):
        import numbers
        h = "f0f0f0f0f0f0f0f0"
        assert isinstance(_phash_distance(h, h), numbers.Integral)


# ---------------------------------------------------------------------------
# FrameMetadata
# ---------------------------------------------------------------------------

def _make_frame_metadata(frame_index=0):
    return FrameMetadata(
        frame_index=frame_index,
        timestamp_start=1.0,
        timestamp_end=1.5,
        path="/tmp/frame_0.jpg",
        frame_width=1280,
        frame_height=720,
        fps=25.0,
        video_duration=60.0,
        video_id="vid_001",
        blur_score=0.8,
        brightness_mean=150.0,
        phash=None,
        is_scene_boundary=True,
        scene_index=0,
        shot_type="medium",
        hdr_detected=False,
        interlaced=False,
        aspect_ratio=16 / 9,
        extraction_method="opencv",
        session_id="test_session",
    )


class TestFrameMetadata:

    def test_construction(self):
        fm = _make_frame_metadata(frame_index=0)
        assert fm.frame_index == 0
        assert fm.timestamp_start == 1.0
        assert fm.is_scene_boundary is True

    def test_to_dict_returns_dict(self):
        fm = _make_frame_metadata(frame_index=1)
        d = fm.to_dict()
        assert isinstance(d, dict)
        assert d["frame_index"] == 1
        assert d["shot_type"] == "medium"
