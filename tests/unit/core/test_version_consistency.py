"""The release version resolves from one place, and the two files agree.

This used to be four independent string literals — `VERSION`,
`pyproject.toml`, `app/core/config.py`'s `APP_VERSION` default (plus that
file's own `TestSettings` assertion), and `app/main.py`'s header comment —
with nothing deriving one from another. `.github/workflows/release.yml` only
ever rewrote the first two, so a release could ship with `GET /version` and
`GET /status` reporting a different build than the git tag that produced the
image: the kind of drift nobody notices until they are trying to work out
what is actually running during an incident.

`APP_VERSION` is now resolved by `config._read_project_version()`, which reads
`pyproject.toml` and falls back to `VERSION`. That leaves exactly one thing a
test still has to enforce — that those two files agree, so the fallback order
can never change the answer — plus the guarantee that the resolver actually
finds a real version rather than silently serving its unknown-version
sentinel.

Same rationale as tests/unit/core/test_model_manifest_contract.py: files that
must agree, with no import relationship to make them agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core import config

ROOT = Path(__file__).resolve().parents[3]

VERSION_FILE = ROOT / "VERSION"
PYPROJECT = ROOT / "pyproject.toml"
MAIN_PY = ROOT / "app" / "main.py"

# X.Y.Z with an optional SemVer pre-release suffix (1.0.0, 1.0.0-rc4).
SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$")


@pytest.fixture(scope="module")
def declared_version() -> str:
    assert VERSION_FILE.is_file(), f"{VERSION_FILE} is missing"
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    assert version, "VERSION is empty"
    return version


def test_version_file_is_valid_semver(declared_version: str):
    assert SEMVER.match(declared_version), (
        f"VERSION must be X.Y.Z or X.Y.Z-prerelease with no leading 'v' — "
        f"got {declared_version!r}"
    )


def test_pyproject_matches_version_file(declared_version: str):
    """The resolver prefers pyproject.toml; VERSION is its fallback and is what
    release.yml and `make release` read. They must not disagree."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match, "could not find the [project] version in pyproject.toml"
    assert (
        match.group(1) == declared_version
    ), f"pyproject.toml says {match.group(1)!r} but VERSION says {declared_version!r}"


def test_resolver_returns_the_declared_version(declared_version: str):
    assert config._read_project_version() == declared_version


def test_resolver_did_not_fall_back_to_the_unknown_sentinel():
    """A missing/unreadable pyproject.toml AND VERSION degrades to
    `_VERSION_FALLBACK`. That is correct behaviour for a broken checkout, but
    it must never be what a real build reports."""
    assert config._read_project_version() != config._VERSION_FALLBACK


def test_app_version_matches_the_resolver(declared_version: str, monkeypatch):
    """`APP_VERSION` stays env-overridable, so assert the resolved default
    rather than whatever this process happens to have exported."""
    monkeypatch.delenv("APP_VERSION", raising=False)
    assert config.Settings().APP_VERSION == declared_version


def test_main_header_carries_no_version_literal():
    """app/main.py's header comment used to embed the version and went stale.
    It must not quietly become a fifth copy again."""
    header = MAIN_PY.read_text(encoding="utf-8")[:600]
    assert not re.search(r"MAGIK FINANCE RAG v\d", header), (
        "app/main.py's header has a hardcoded version again — it is not kept in "
        "sync by anything; leave the version to _read_project_version()"
    )
