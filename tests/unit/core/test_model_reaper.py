"""Idle/pressure VRAM eviction — app/core/model_loader.py + model_reaper.py.

No GPU required: the eviction path is pure attribute manipulation plus a
free-VRAM reading, and the latter is monkeypatched here. What these tests
actually pin down is the POLICY, which is the part that can silently regress:
which models are eligible, LRU ordering under pressure, and the guarantee
that the query hot set is never evicted.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core import model_reaper
from app.core.model_loader import ModelLoader


@pytest.fixture
def loader() -> ModelLoader:
    """A fresh loader. __init__ only nulls attributes and builds a thread
    pool — it does not load or touch any model."""
    return ModelLoader()


# ── What is eligible for eviction at all ────────────────────────────────────


def test_query_hot_set_is_never_evictable(loader: ModelLoader) -> None:
    """The pinned set is the whole safety story: a live query must never
    block on a model reload. These must not appear in _EVICTABLE_MODELS.

    siglip/image_embedder/siglip_text_embedder are pinned deliberately —
    SigLIP's TEXT path runs on every query for cross-modal search, so it is
    query-path despite sounding ingestion-only.
    """
    for pinned in (
        "llm",
        "text_embedder",
        "reranker",
        "siglip",
        "image_embedder",
        "siglip_text_embedder",
        "multimodal",
    ):
        assert pinned not in loader._EVICTABLE_MODELS, f"{pinned} must stay resident"


def test_evictable_set_is_ingestion_only(loader: ModelLoader) -> None:
    assert set(loader._EVICTABLE_MODELS) == {
        "whisper",
        "blip",
        "qwen2_vl",
        "qwen2_vl_video",
        "trocr",
        "diarizer",
        "ner",
        "finbert",
    }


# ── Idle (TTL) eviction ─────────────────────────────────────────────────────


def test_idle_model_is_evicted(loader: ModelLoader) -> None:
    loader._whisper = object()
    loader._last_used["whisper"] = time.time() - 600

    assert loader.unload_idle_models(idle_seconds=300) == ["whisper"]
    assert loader._whisper is None
    assert "whisper" not in loader._last_used


def test_recently_used_model_is_kept(loader: ModelLoader) -> None:
    loader._whisper = object()
    loader._last_used["whisper"] = time.time() - 10

    assert loader.unload_idle_models(idle_seconds=300) == []
    assert loader._whisper is not None


def test_unloaded_model_is_not_reported_as_evicted(loader: ModelLoader) -> None:
    """Nothing loaded => nothing to evict, and no phantom entry."""
    assert loader.unload_idle_models(idle_seconds=0) == []


def test_eviction_clears_every_attribute_of_a_multi_attr_model(loader: ModelLoader) -> None:
    """blip/qwen2_vl/trocr each own model+processor+device; leaving the
    processor behind would keep a stale device string paired with a None
    model on the next load."""
    loader._blip_model = object()
    loader._blip_processor = object()
    loader._blip_device = "cuda"
    loader._last_used["blip"] = time.time() - 600

    assert loader.unload_idle_models(idle_seconds=300) == ["blip"]
    assert loader._blip_model is None
    assert loader._blip_processor is None
    assert loader._blip_device is None


# ── Pressure (watermark / LRU) eviction ─────────────────────────────────────


def test_no_pressure_eviction_when_vram_is_ample(loader: ModelLoader, monkeypatch) -> None:
    """Above the watermark, evicting is pure loss — you pay a reload to free
    memory nobody was competing for."""
    from app.core import model_loader as ml

    monkeypatch.setattr(ml.device_manager, "free_vram_gb", lambda: 30.0)

    loader._whisper = object()
    loader._last_used["whisper"] = 0.0  # ancient, but there is no pressure

    assert loader.unload_until_free(6.0) == []
    assert loader._whisper is not None


def test_pressure_evicts_least_recently_used_first(loader: ModelLoader, monkeypatch) -> None:
    """LRU order, and it stops as soon as the target is met rather than
    dumping every loaded model."""
    from app.core import model_loader as ml

    readings = iter([2.0, 8.0])  # below watermark, then satisfied after one evict
    monkeypatch.setattr(ml.device_manager, "free_vram_gb", lambda: next(readings))

    now = time.time()
    loader._whisper = object()
    loader._last_used["whisper"] = now - 10  # most recent
    loader._trocr_model = object()
    loader._last_used["trocr"] = now - 900  # oldest -> evicted first

    assert loader.unload_until_free(6.0) == ["trocr"]
    assert loader._trocr_model is None
    assert loader._whisper is not None, "should stop once the target is met"


def test_pressure_eviction_ignores_recency(loader: ModelLoader, monkeypatch) -> None:
    """Under real pressure a model used seconds ago is still fair game —
    that is the whole difference from the TTL sweep."""
    from app.core import model_loader as ml

    monkeypatch.setattr(ml.device_manager, "free_vram_gb", lambda: 1.0)

    loader._whisper = object()
    loader._last_used["whisper"] = time.time()  # used right now

    assert loader.unload_until_free(6.0) == ["whisper"]
    assert loader._whisper is None


def test_no_eviction_when_free_vram_is_unreadable(loader: ModelLoader, monkeypatch) -> None:
    """No CUDA / driver query failed. Never evict on a guess."""
    from app.core import model_loader as ml

    monkeypatch.setattr(ml.device_manager, "free_vram_gb", lambda: None)

    loader._whisper = object()
    loader._last_used["whisper"] = 0.0

    assert loader.unload_until_free(6.0) == []
    assert loader._whisper is not None


# ── The background loop ─────────────────────────────────────────────────────


async def test_reaper_is_a_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(model_reaper.settings, "MODEL_IDLE_EVICTION_ENABLED", False)
    called = False

    def _boom() -> list[str]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(model_reaper, "_sweep_once", _boom)
    await asyncio.wait_for(model_reaper.run_model_reaper_loop(), timeout=2)
    assert called is False


async def test_reaper_skips_while_gpu_busy(monkeypatch) -> None:
    """Never evict out from under an in-flight ingestion job.

    Counts gpu_busy() calls as well as sweeps: asserting only "did not sweep"
    would pass trivially if the loop never ran an iteration at all, which is
    exactly what happened before _MIN_INTERVAL_SEC was made patchable.
    """
    monkeypatch.setattr(model_reaper, "_MIN_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(model_reaper.settings, "MODEL_IDLE_EVICTION_ENABLED", True)
    monkeypatch.setattr(model_reaper.settings, "MODEL_REAPER_INTERVAL_SEC", 0.01)

    checks = 0

    def _busy() -> bool:
        nonlocal checks
        checks += 1
        return True

    monkeypatch.setattr(model_reaper, "gpu_busy", _busy)

    swept = False

    def _sweep() -> list[str]:
        nonlocal swept
        swept = True
        return []

    monkeypatch.setattr(model_reaper, "_sweep_once", _sweep)

    task = asyncio.create_task(model_reaper.run_model_reaper_loop())
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert checks >= 2, "loop must actually be iterating for this test to mean anything"
    assert swept is False, "reaper must not sweep while a GPU slot is held"


async def test_reaper_survives_a_failing_iteration(monkeypatch) -> None:
    """A dead reaper is an invisible VRAM leak — one bad sweep must not end
    the loop."""
    monkeypatch.setattr(model_reaper, "_MIN_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(model_reaper.settings, "MODEL_IDLE_EVICTION_ENABLED", True)
    monkeypatch.setattr(model_reaper.settings, "MODEL_REAPER_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(model_reaper, "gpu_busy", lambda: False)

    calls = 0

    def _flaky() -> list[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("CUDA hiccup")
        return []

    monkeypatch.setattr(model_reaper, "_sweep_once", _flaky)

    task = asyncio.create_task(model_reaper.run_model_reaper_loop())
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls >= 2, "loop must keep running after an exception"
