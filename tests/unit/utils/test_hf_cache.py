"""heal_hub_refs: SHA-pinned downloads leave no refs/main, which breaks offline
loads by repo id. The tests resolve through the REAL huggingface_hub cache
lookup, so they prove the repaired layout is what the loader actually needs,
not just that a file got written."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.hf_cache import heal_hub_refs

SHA = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"  # pragma: allowlist secret
REPO = "dslim/bert-base-NER"


def _stage(hub: Path, repo: str = REPO, sha: str = SHA) -> Path:
    model_dir = hub / ("models--" + repo.replace("/", "--"))
    snap = model_dir / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    return model_dir


def test_sha_pinned_cache_is_unresolvable_until_healed(tmp_path):
    from huggingface_hub import try_to_load_from_cache

    _stage(tmp_path)
    assert try_to_load_from_cache(REPO, "config.json", cache_dir=tmp_path) is None

    assert heal_hub_refs(tmp_path) == ["models--dslim--bert-base-NER"]

    found = try_to_load_from_cache(REPO, "config.json", cache_dir=tmp_path)
    assert found is not None and Path(found).name == "config.json"


def test_heal_is_idempotent(tmp_path):
    _stage(tmp_path)
    assert len(heal_hub_refs(tmp_path)) == 1
    assert heal_hub_refs(tmp_path) == []


def test_existing_ref_is_never_overwritten(tmp_path):
    model_dir = _stage(tmp_path)
    (model_dir / "refs").mkdir()
    (model_dir / "refs" / "main").write_text("someothersha", encoding="utf-8")

    assert heal_hub_refs(tmp_path) == []
    assert (model_dir / "refs" / "main").read_text(encoding="utf-8") == "someothersha"


def test_ambiguous_multi_snapshot_model_is_left_alone(tmp_path):
    model_dir = _stage(tmp_path)
    (model_dir / "snapshots" / ("b" * 40)).mkdir()

    assert heal_hub_refs(tmp_path) == []
    assert not (model_dir / "refs" / "main").exists()


def test_missing_hub_dir_and_non_model_entries_are_ignored(tmp_path):
    assert heal_hub_refs(tmp_path / "does-not-exist") == []
    (tmp_path / "datasets--foo--bar").mkdir()
    (tmp_path / "models--empty").mkdir()
    assert heal_hub_refs(tmp_path) == []


@pytest.mark.parametrize(
    "repo",
    ["Salesforce/blip-image-captioning-large", "pyannote/speaker-diarization-3.1"],
)
def test_other_failing_models_from_prod_resolve_after_heal(tmp_path, repo):
    from huggingface_hub import try_to_load_from_cache

    _stage(tmp_path, repo=repo, sha="a" * 40)
    heal_hub_refs(tmp_path)
    assert try_to_load_from_cache(repo, "config.json", cache_dir=tmp_path) is not None
