"""Regression tests for app/main.py::_cleanup_temp_dirs().

The sweep exists to clear orphans left by a CRASHED ingestion. It used to
delete every entry under data/users/*/{temp,temp_frames,staging}
unconditionally, on BOTH startup and shutdown — which also deleted the
working files of ingestions running at that moment, including ones in other
processes sharing the volume.

Reproduced 2026-08-20 against the Tier-2 eval: an unrelated FastAPI
TestClient lifespan fired the sweep and removed a live audio ingest's 30-min
WAV chunks mid-transcription, producing

    [Errno 2] No such file or directory: .../audio_chunks_<hex>/chunk_0.wav

after which the audio suite scored audio_wer=nan — which the gate SILENTLY
skips (check_thresholds ignores NaN), so the run went green having measured
nothing at all. The same sweep discards a real user's in-flight upload on
every deploy.

There were previously NO tests over this function, which is how it shipped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from app.core.config import settings
from app.main import _cleanup_temp_dirs

UID = "age-guard-tenant"


def _age(path: Path, seconds_old: float) -> None:
    """Backdate a path (and its children) to look `seconds_old` seconds old."""
    stamp = time.time() - seconds_old
    targets = [path, *path.rglob("*")] if path.is_dir() else [path]
    for p in sorted(targets, reverse=True):
        os.utime(p, (stamp, stamp))


@pytest.fixture()
def temp_root(tmp_path, monkeypatch):
    """_cleanup_temp_dirs resolves the RELATIVE path data/users against the
    process CWD, so chdir'ing into tmp_path isolates the test completely from
    the real repo's data/ directory."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "users" / UID / "temp"
    root.mkdir(parents=True)
    return root


def test_in_flight_chunk_dir_survives(temp_root):
    """The exact failure: a chunk dir written seconds ago must NOT be swept."""
    live = temp_root / "audio_chunks_live"
    live.mkdir()
    (live / "chunk_0.wav").write_bytes(b"\x00" * 512)

    _cleanup_temp_dirs()

    assert live.exists(), "swept an in-flight ingest's working directory"
    assert (live / "chunk_0.wav").exists()


def test_genuine_orphan_is_removed(temp_root):
    """The sweep must still do its actual job."""
    orphan = temp_root / "audio_chunks_orphan"
    orphan.mkdir()
    (orphan / "chunk_0.wav").write_bytes(b"\x00" * 512)
    _age(orphan, settings.TEMP_ORPHAN_GRACE_SEC + 600)

    _cleanup_temp_dirs()

    assert not orphan.exists(), "orphan from a crashed run was not cleared"


def test_recent_child_protects_an_old_parent_dir(temp_root):
    """A chunk dir is created once and then written into, so the DIRECTORY's
    own mtime can be older than the data inside it. Age must be judged from
    the newest mtime in the subtree, or a long ingest gets swept out from
    under itself partway through."""
    d = temp_root / "audio_chunks_slow"
    d.mkdir()
    (d / "chunk_0.wav").write_bytes(b"\x00" * 512)
    _age(d, settings.TEMP_ORPHAN_GRACE_SEC + 600)
    # ...then a fresh write lands inside it, as a running ingest would do.
    (d / "chunk_1.wav").write_bytes(b"\x00" * 512)

    _cleanup_temp_dirs()

    assert d.exists(), "judged age from the parent dir instead of its contents"
    assert (d / "chunk_1.wav").exists()


def test_loose_files_follow_the_same_rule(temp_root):
    """Not every entry is a directory — thumbnails and demuxed wavs are
    written as loose files into the same temp roots."""
    fresh = temp_root / "thumb_fresh.jpg"
    fresh.write_bytes(b"\xff\xd8\xff")
    stale = temp_root / "thumb_stale.jpg"
    stale.write_bytes(b"\xff\xd8\xff")
    _age(stale, settings.TEMP_ORPHAN_GRACE_SEC + 600)

    _cleanup_temp_dirs()

    assert fresh.exists(), "swept a freshly written temp file"
    assert not stale.exists(), "left a stale temp file behind"


def test_sweeps_all_three_managed_subdirs(tmp_path, monkeypatch):
    """temp, temp_frames and staging are all used by the media pipelines
    (video_ingest stages its demuxed wav in `staging` and its frames in
    `temp_frames`), so the guard has to hold for each."""
    monkeypatch.chdir(tmp_path)
    for sub in ("temp", "temp_frames", "staging"):
        d = tmp_path / "data" / "users" / UID / sub
        d.mkdir(parents=True)
        (d / "live.bin").write_bytes(b"x")
        stale = d / "stale.bin"
        stale.write_bytes(b"x")
        _age(stale, settings.TEMP_ORPHAN_GRACE_SEC + 600)

    _cleanup_temp_dirs()

    for sub in ("temp", "temp_frames", "staging"):
        d = tmp_path / "data" / "users" / UID / sub
        assert (d / "live.bin").exists(), f"{sub}: swept live file"
        assert not (d / "stale.bin").exists(), f"{sub}: kept stale file"


def test_other_tenants_in_flight_work_is_not_collateral(tmp_path, monkeypatch):
    """The sweep walks EVERY user directory, so one tenant's app restart must
    not destroy another tenant's running ingest."""
    monkeypatch.chdir(tmp_path)
    a = tmp_path / "data" / "users" / "tenant-a" / "temp"
    b = tmp_path / "data" / "users" / "tenant-b" / "temp"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "chunk_0.wav").write_bytes(b"x")

    _cleanup_temp_dirs()

    assert (a / "chunk_0.wav").exists()
    assert (b / "chunk_0.wav").exists()


def test_is_idempotent_and_safe_on_missing_root(tmp_path, monkeypatch):
    """Called on both startup and shutdown, and on a box with no data/ yet."""
    monkeypatch.chdir(tmp_path)
    _cleanup_temp_dirs()  # no data/users at all
    _cleanup_temp_dirs()
