"""Regression tests for ModelLoader._oom_guard()'s pressure eviction.

THE BUG (reproduced end-to-end 2026-08-20, same shape as the CD run
31139999120 incident that audio_ingest._transcribe_chunk_eager documents):

`_oom_guard(loading=...)` only ran the TTL sweep, `unload_idle_models()`,
whose default idle window is MODEL_IDLE_TIMEOUT_SEC (300s). But `--suite
full` hands off between modality sub-suites in far less than that — measured,
ocr finished at 03:12:04 and whisper began loading 76s later. Every vision
model (Qwen2-VL, BLIP2, TrOCR) was therefore still "recently used" and
NOTHING was evictable at exactly the moment the VRAM was needed:

    model_load_failed  error='CUDA failed with error out of memory'  model=whisper
    audio_transcription_cuda_oom_cpu_fallback | chunk_0.wav

Whisper then fell back to CPU — correct, but ~50x slower — which is what
turned a ~90-minute suite into a 3h+ run against a 180-minute job cap.

`unload_until_free()`, the recency-independent pressure valve built for
exactly this, existed but was only ever driven by app/core/model_reaper.py —
which runs solely in the SERVER process. The Tier-2 eval runs as a separate
`docker exec` process holding its own ModelLoader singleton, so it had no
pressure valve at all. These tests pin the wiring that fixes it.
"""

from __future__ import annotations

import pytest

from app.core import model_loader as ml_mod
from app.core.config import settings

loader = ml_mod.model_loader


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Make `whisper` the ONLY eviction candidate on the shared singleton.

    model_loader is a process-wide singleton, so whatever earlier tests left
    resident is still there. That matters here specifically because
    unload_until_free() evicts LRU-first and STOPS as soon as free VRAM
    clears the target: with another model also resident, that one is dropped,
    the watermark is satisfied, and whisper is never reached — so the
    assertion below would fail for a reason that has nothing to do with the
    behaviour under test. (Observed exactly that: green in isolation, red in
    the full suite.)

    Clearing every other evictable slot makes the outcome depend only on the
    policy, in any execution order.
    """
    saved: dict[str, tuple] = {}
    for name, attrs in loader._EVICTABLE_MODELS.items():
        saved[name] = tuple(getattr(loader, a, None) for a in attrs)
        if name != "whisper":
            for a in attrs:
                monkeypatch.setattr(loader, a, None, raising=False)
    saved_last_used = dict(loader._last_used)
    loader._last_used.clear()

    # Pretend whisper is resident so it is a candidate for eviction.
    monkeypatch.setattr(loader, "_whisper", object(), raising=False)
    loader._last_used["whisper"] = 0.0

    yield

    # monkeypatch restores the attributes; put the bookkeeping back too so a
    # later test still sees the loader it expects.
    loader._last_used.clear()
    loader._last_used.update(saved_last_used)
    del saved


def _set_free_vram(monkeypatch, values):
    """Feed free_vram_gb() a sequence; the last value repeats."""
    seq = list(values)

    def _fake():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(ml_mod.device_manager, "free_vram_gb", _fake)


def test_evicts_a_recently_used_model_under_pressure(monkeypatch):
    """The core regression: a model used SECONDS ago must still be evicted
    when the card is full, because TTL eviction cannot help a back-to-back
    sub-suite handoff."""
    import time

    loader._last_used["whisper"] = time.time()  # used right now -> TTL-immune
    # Below the watermark, then clear after the eviction.
    _set_free_vram(monkeypatch, [0.02, 99.0])

    loader._oom_guard(loading="qwen2_vl")

    assert loader._whisper is None, (
        "a recently-used model was not evicted under VRAM pressure — this is "
        "exactly the state that OOM'd Whisper into the CPU fallback"
    )


def test_no_eviction_when_vram_is_plentiful(monkeypatch):
    """Must not thrash: evicting models nothing is competing for just pays a
    reload cost (~25s for a VLM) to free memory no one wanted."""
    import time

    loader._last_used["whisper"] = time.time()
    _set_free_vram(monkeypatch, [99.0])

    loader._oom_guard(loading="qwen2_vl")

    assert loader._whisper is not None, "evicted despite ample free VRAM"


def test_never_evicts_on_an_unreadable_driver(monkeypatch):
    """free_vram_gb() returns None when CUDA is absent or the driver query
    fails. Guessing under those conditions would evict blindly on CPU boxes."""
    import time

    loader._last_used["whisper"] = time.time()
    monkeypatch.setattr(ml_mod.device_manager, "free_vram_gb", lambda: None)

    loader._oom_guard(loading="qwen2_vl")

    assert loader._whisper is not None, "evicted on an unreadable VRAM reading"


def test_guard_is_a_noop_without_a_loading_target(monkeypatch):
    """_oom_guard() with no `loading` is the plain cache-flush path used
    between unrelated operations; it must not start evicting."""
    import time

    loader._last_used["whisper"] = time.time()
    _set_free_vram(monkeypatch, [0.02])

    loader._oom_guard()

    assert loader._whisper is not None


def test_watermark_of_zero_disables_pressure_eviction(monkeypatch):
    """MODEL_EVICT_VRAM_WATERMARK_GB=0 is the documented off-switch."""
    import time

    loader._last_used["whisper"] = time.time()
    _set_free_vram(monkeypatch, [0.02, 99.0])
    monkeypatch.setattr(settings, "MODEL_EVICT_VRAM_WATERMARK_GB", 0.0)

    loader._oom_guard(loading="qwen2_vl")

    assert loader._whisper is not None, "evicted although the watermark was disabled"


def test_the_model_being_loaded_is_not_evicted_by_its_own_guard(monkeypatch):
    """_touch(loading) must keep the incoming model out of the LRU firing
    line, or a getter could drop the very weights it is about to return."""
    _set_free_vram(monkeypatch, [0.02, 0.02, 99.0])

    loader._oom_guard(loading="whisper")

    # whisper is the model being loaded here, so it must survive its own guard
    assert loader._whisper is not None
