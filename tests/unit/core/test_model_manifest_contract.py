"""The contract between app/bin/models/download_all_models.py (which WRITES
download_manifest.json) and app/core/startup_validator.py (which READS it and
refuses to serve traffic without it).

These two files are edited independently and have no import relationship, so
nothing but a test can stop them drifting apart. When they do drift, the
symptom is maximally misleading: the provisioning script prints a green
summary, the container starts, and only then does the app hard-fail with
`Required models not cached` and crash-loop — a failure that looks like a
deploy/infra problem and is nothing of the sort.

That drift shipped to production twice (v0.33.0 and v1.0.0-rc1, both on
Qwen/Qwen2-VL-7B-Instruct) before it was understood. Both directions of it
are pinned here.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from app.core.startup_validator import REQUIRED_MODELS


@pytest.fixture
def dl(tmp_path, monkeypatch):
    """download_all_models with its cache roots pointed at a temp dir.

    The module resolves HF_HOME/TORCH_HOME into globals at import time, so it
    has to be reloaded after the env is patched — and reloaded again on the
    way out, or every later test in the session inherits a module still
    pointing at this test's now-deleted tmp_path.
    """
    real_hf, real_torch = os.environ.get("HF_HOME"), os.environ.get("TORCH_HOME")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))

    import app.bin.models.download_all_models as mod

    importlib.reload(mod)
    assert str(mod._hf_home) == str(tmp_path), "fixture did not take effect"
    yield mod

    for var, val in (("HF_HOME", real_hf), ("TORCH_HOME", real_torch)):
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val
    importlib.reload(mod)


def _all_entries(mod) -> list[dict]:
    """Mirrors main()'s own MODELS + GGUF_MODELS assembly."""
    return mod.MODELS + [{"model_id": g["gguf_repo"], "type": "gguf", **g} for g in mod.GGUF_MODELS]


# ── the contract itself ─────────────────────────────────────────────────────


def test_every_required_model_is_downloadable_and_not_optional(dl):
    """A model the app REQUIRES must be one this script will refuse to finish
    without. `optional: True` means "skip on failure and still exit 0", which
    for a required model means: no manifest entry, green provisioning run,
    guaranteed crash-loop on the next boot.

    Caught the three gated pyannote diarization models marked optional while
    startup_validator required all three (fixed 2026-08-23).
    """
    entries = _all_entries(dl)
    by_id = {m["model_id"]: m for m in entries}

    absent = sorted(REQUIRED_MODELS - by_id.keys())
    assert not absent, (
        f"REQUIRED_MODELS entries with no downloader entry at all: {absent}. "
        "Nothing will ever write a manifest entry for these, so the app can "
        "never start with MODEL_CACHE_REQUIRE_MANIFEST=true."
    )

    optional = sorted(m for m in REQUIRED_MODELS if by_id[m].get("optional"))
    assert not optional, (
        f"required by startup_validator but optional in the downloader: {optional}. "
        "Either drop the optional flag or stop requiring them at startup."
    )


def test_required_models_are_in_the_default_boot_run(dl):
    """`--include-eval-models`-only entries are not fetched on a normal boot,
    so a required model hidden behind that flag would never be cached on a
    production box at all."""
    by_id = {m["model_id"]: m for m in _all_entries(dl)}
    excluded = sorted(m for m in REQUIRED_MODELS if by_id[m].get("startup") is False)
    assert not excluded, f"required but excluded from the default run: {excluded}"


# ── the v1.0.0-rc1 bug: files on disk, no manifest entry ────────────────────


def _stage_cached_model(hf_home: Path, model_id: str, revision: str) -> Path:
    snap = hf_home / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"model_type":"qwen2_vl"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"\x00" * 2048)
    return snap


def _manifest_ids(hf_home: Path) -> set[str]:
    path = hf_home / "download_manifest.json"
    if not path.exists():
        return set()
    return {e["model_id"] for e in json.loads(path.read_text(encoding="utf-8"))}


MODEL = "Qwen/Qwen2-VL-7B-Instruct"
REV = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"  # pragma: allowlist secret


def test_cached_model_with_no_manifest_entry_self_heals(dl, tmp_path):
    """The exact production state of both failed promotions: the model's files
    are present (an earlier run downloaded them, then the container was killed
    by this very check before the manifest was written), so the fast
    already-cached path runs and no download happens. That path must still
    write the manifest entry, or the box can never recover on its own.
    """
    _stage_cached_model(tmp_path, MODEL, REV)
    assert dl._is_hub_cached(MODEL, mtype="qwen2vl") is True
    assert MODEL not in _manifest_ids(tmp_path)

    assert dl._handle_cached(MODEL, "qwen2vl", False, 16.59, REV) == "ok"

    assert MODEL in _manifest_ids(tmp_path), "cached fast path did not heal the manifest"
    entry = next(
        e
        for e in json.loads((tmp_path / "download_manifest.json").read_text(encoding="utf-8"))
        if e["model_id"] == MODEL
    )
    assert entry["sha256"], "healed entry recorded without a checksum"
    assert entry["revision"] == REV


def test_self_heal_does_not_duplicate_on_later_runs(dl, tmp_path):
    """Every subsequent deploy re-runs this script. Healing must be a no-op the
    second time, not an append."""
    _stage_cached_model(tmp_path, MODEL, REV)
    dl._handle_cached(MODEL, "qwen2vl", False, 16.59, REV)
    dl._handle_cached(MODEL, "qwen2vl", False, 16.59, REV)

    entries = json.loads((tmp_path / "download_manifest.json").read_text(encoding="utf-8"))
    assert sum(1 for e in entries if e["model_id"] == MODEL) == 1


def test_self_heal_preserves_existing_entries(dl, tmp_path):
    """Healing one model must not drop the other seventeen."""
    (tmp_path / "download_manifest.json").write_text(
        json.dumps([{"model_id": "BAAI/bge-large-en-v1.5", "sha256": "abc", "type": "st"}]),
        encoding="utf-8",
    )
    _stage_cached_model(tmp_path, MODEL, REV)
    dl._handle_cached(MODEL, "qwen2vl", False, 16.59, REV)

    assert _manifest_ids(tmp_path) == {"BAAI/bge-large-en-v1.5", MODEL}


def test_corruption_is_still_detected_after_healing(dl, tmp_path):
    """The heal writes a checksum; that checksum must still be enforced. A
    self-healing manifest that also stopped catching silent corruption would
    be a worse trade than the bug it fixed."""
    snap = _stage_cached_model(tmp_path, MODEL, REV)
    assert dl._handle_cached(MODEL, "qwen2vl", False, 16.59, REV) == "ok"

    (snap / "model.safetensors").write_bytes(b"\xff" * 2048)
    assert dl._handle_cached(MODEL, "qwen2vl", False, 16.59, REV) == "mismatch_fail"
