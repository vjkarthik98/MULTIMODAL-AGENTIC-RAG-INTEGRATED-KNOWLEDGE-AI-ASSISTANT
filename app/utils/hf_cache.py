"""Repair huggingface_hub cache entries that cannot be resolved offline.

download_all_models.py pins every model to a commit SHA. huggingface_hub only
writes ``refs/<revision>`` when the revision is a branch or tag name, so a
SHA-pinned download leaves ``snapshots/<sha>/`` on disk with NO ``refs/main``.
start_server.py runs the app with HF_HUB_OFFLINE=1, and a load by plain repo id
(``pipeline("ner", model="dslim/bert-base-NER")``) resolves ``main`` through
``refs/main`` — which is missing — so the load fails with "couldn't connect to
huggingface.co and couldn't find them in the cached files" even though the
weights are right there. Live on 2026-10-02/06: ner, blip and diarizer failed
this way on every boot.

heal_hub_refs() writes the missing ``refs/main`` for any model that has exactly
one snapshot, which is unambiguous. It never overwrites an existing ref and
never touches a model with several snapshots (it cannot know which one is meant).
Pure pathlib — safe to call before any torch/transformers import.
"""

from __future__ import annotations

from pathlib import Path


def heal_hub_refs(hub_dir: str | Path) -> list[str]:
    """Write ``refs/main`` for every ``models--*`` dir that lacks one.

    Returns the cache dir names that were repaired. Never raises: a read-only
    cache or an odd directory layout must not stop the app from booting.
    """
    healed: list[str] = []
    hub = Path(hub_dir)
    try:
        model_dirs = [d for d in hub.glob("models--*") if d.is_dir()]
    except OSError:
        return healed

    for model_dir in model_dirs:
        try:
            ref = model_dir / "refs" / "main"
            if ref.exists():
                continue
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue
            snapshots = [s for s in snapshots_dir.iterdir() if s.is_dir()]
            if len(snapshots) != 1:
                continue
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(snapshots[0].name, encoding="utf-8")
            healed.append(model_dir.name)
        except OSError:
            continue
    return healed
