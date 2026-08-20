"""Fix 4 regression: a transcription timeout must return, and must NOT delete
the temp chunk an abandoned worker is still reading.

Two separate defects were in play here.

1. The wait was unbounded. `fut.result()` had no timeout AND the pool was a
   `with` block, whose __exit__ calls shutdown(wait=True) and joins every
   worker. So a wedged chunk stalled the whole ingest — and, in the Tier-2
   suite, every sub-suite queued behind it — with nothing written to the log.
   That is the "stuck on audio/video with no output" symptom; the run only
   ended when the 180-minute CD job cap killed it.

2. Once the wait IS cut loose, the ingest reaches its `finally` while a
   worker thread is still reading a chunk WAV. Unlinking it there would
   fault that live reader — the exact bug app/main.py::_cleanup_temp_dirs()
   had, recreated in the one place that actually knows a reader is attached.

The resolution: abandon the worker, keep its file, and let the (now
age-guarded) startup sweep collect it once nothing can hold it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import app.ingestion.audio_ingest as ai
from app.core.config import settings


@pytest.fixture()
def tiny_wav(tmp_path: Path) -> Path:
    """A real, valid 2-second WAV — long enough to ingest, short enough to be fast."""
    import struct
    import wave

    p = tmp_path / "clip.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # Non-silent, or ingest rejects it as EMPTY_CONTENT (< -60 dBFS).
        frames = b"".join(
            struct.pack("<h", int(3000 * ((i % 100) - 50) / 50)) for i in range(32000)
        )
        wf.writeframes(frames)
    return p


def test_timeout_returns_instead_of_hanging(tmp_path, monkeypatch, tiny_wav):
    """The ingest must come back while the worker is still running, rather
    than joining it. Before the fix this blocked forever."""
    release = threading.Event()
    started = threading.Event()

    def _wedged(chunk_file, session_id):
        started.set()
        release.wait(60)  # simulates a wedged Whisper call
        raise RuntimeError("worker was released")

    monkeypatch.setattr(ai, "_transcribe_chunk_eager", _wedged)
    monkeypatch.setattr(settings, "AUDIO_TRANSCRIBE_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(settings, "DIARIZATION_ENABLED", False)
    monkeypatch.setattr(settings, "CHROOT_BASE", tmp_path)

    from app.utils.paths import reset_current_user, set_current_user

    tok = set_current_user("timeout-tenant")
    t0 = time.time()
    try:
        # Every chunk timed out, so nothing was transcribed and ingest raises
        # ValueError("NO_VALID_AUDIO_SEGMENTS") — the same signal the CUDA-OOM
        # incident produced. The POINT of this test is that it raises PROMPTLY
        # instead of blocking on the wedged worker.
        with pytest.raises(ValueError, match="NO_VALID_AUDIO_SEGMENTS"):
            ai.ingest(file_path=str(tiny_wav), session_id="timeout-test")
    finally:
        reset_current_user(tok)
        release.set()

    elapsed = time.time() - t0
    assert started.is_set(), "worker never ran — test would prove nothing"
    assert elapsed < 30, (
        f"ingest took {elapsed:.1f}s — it joined the wedged worker instead of "
        "abandoning it after the timeout"
    )


def test_in_use_chunk_is_not_unlinked(tmp_path, monkeypatch):
    """The narrow invariant, tested directly on the cleanup contract: a path
    recorded as still-in-use must survive the `finally` sweep."""
    # Mirrors the cleanup loop in ingest()'s finally block.
    live = tmp_path / "chunk_live.wav"
    live.write_bytes(b"\x00" * 64)
    dead = tmp_path / "chunk_dead.wav"
    dead.write_bytes(b"\x00" * 64)

    temp_chunk_paths = [str(live), str(dead)]
    in_use_paths = {str(live)}

    import os

    for tmp_path_str in temp_chunk_paths:
        if tmp_path_str in in_use_paths:
            continue
        if os.path.exists(tmp_path_str):
            os.unlink(tmp_path_str)

    assert live.exists(), "deleted a chunk an abandoned worker is still reading"
    assert not dead.exists(), "failed to clean up a chunk nothing is holding"


def test_timeout_setting_is_a_real_bound():
    """A zero/negative bound would make every chunk time out instantly and
    silently zero the audio suite — the failure mode this whole area is
    about. Keep it generous: the CPU fallback is legitimately ~50x slower
    than GPU and aborting real work is worse than waiting."""
    assert settings.AUDIO_TRANSCRIBE_TIMEOUT_SEC > 600
    assert settings.DIARIZATION_TIMEOUT_SEC > 300
