#!/usr/bin/env python3
"""
Pre-download all required models to .hf_cache so they survive instance restarts.

Re-running is safe: already-cached models are skipped without any network call.

Usage:
    python app/bin/models/download_all_models.py
    python app/bin/models/download_all_models.py --skip-gated    # skip pyannote (needs license)
    python app/bin/models/download_all_models.py --only gguf     # single model by key
    python app/bin/models/download_all_models.py --only qwen_judge   # judge only

Set HF_TOKEN in .env for gated models (pyannote diarization requires license acceptance
at https://hf.co/pyannote/speaker-diarization-3.1).

qwen_judge (MAGIK's single eval judge — Tier-2 gate, Ragas, and DeepEval all
grade through it, see app/eval/judges/qwen_judge.py) IS included in the
default run — it's a pure disk fetch (no VRAM, no model load), so a one-time
~4.7GB download on the box's first boot is worth it to guarantee Tier-2
always has its real judge instead of silently degrading to a weaker lexical
fallback. It's still marked "optional" so a download hiccup can't fail the
whole provisioning run, and app/eval/judges/qwen_judge.py::ensure_available()
is a second, independent fetch attempt if it's ever still missing when
Tier-2 actually runs. `--include-eval-models` remains for any FUTURE
eval-only model that genuinely should stay opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows' default console codepage (cp1252) can't encode characters like
# "✓" — reconfigure to UTF-8 so status prints below never crash the whole
# download run over a cosmetic character. No-op on Linux/Mac (already UTF-8).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ── env must be set before any HuggingFace import ────────────────────────────
_project_root = Path(__file__).resolve().parents[3]
_hf_home = os.getenv("HF_HOME", str(_project_root / ".hf_cache"))
_gguf_dir = Path(_hf_home) / "gguf"

os.environ["HF_HOME"] = _hf_home
os.environ["HF_HUB_CACHE"] = _hf_home + "/hub"
os.environ["TRANSFORMERS_CACHE"] = _hf_home + "/hub"
os.environ["HF_DATASETS_CACHE"] = _hf_home + "/datasets"

# torch.hub (used by detoxify's Detoxify("original")) is a SEPARATE cache
# mechanism from Hugging Face Hub — it does not read HF_HOME. start_server.py
# already sets TORCH_HOME=.torch_cache for the runtime path; mirrored here so
# this standalone pre-download script's torch.hub fetches land in the same
# place instead of torch's own default (~/.cache/torch), which would just
# move the "first real request downloads it live" problem instead of fixing it.
_torch_home = os.getenv("TORCH_HOME", str(_project_root / ".torch_cache"))
os.environ["TORCH_HOME"] = _torch_home

sys.path.insert(0, str(_project_root))

try:
    from dotenv import load_dotenv

    load_dotenv(str(_project_root / ".env"))
except ImportError:
    pass

HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── model manifest ────────────────────────────────────────────────────────────
# Each entry: key, model_id, type, size_gb, gated, optional, revision (optional).
#
# "revision" pins the exact HF Hub commit/tag downloaded — omit (or leave None)
# to track the repo's default branch as before (current behavior, unchanged).
# Once you know a good commit hash for a model (recorded in
# .hf_cache/download_manifest.json's "revision" field after any download), set
# it here to freeze that model against future upstream changes. This project
# can't respons­ibly fabricate "correct" revision hashes sight-unseen — this is
# the mechanism; filling in specific pins is an operational decision per model.
#
# Pinned 2026-08-21 (MLOps reproducibility pass): every model below that has a
# real HF Hub revision concept now pins the exact commit SHA that was each
# repo's default-branch HEAD at that date (fetched live via
# `GET https://huggingface.co/api/models/{repo_id}` -> `.sha`, not fabricated).
# This freezes CURRENT behavior — it does not change which weights get
# downloaded today, it only stops a future fresh box from silently resolving
# to whatever upstream has moved on to since. To intentionally move to a newer
# upstream revision, re-fetch that repo's current `.sha` and update it here as
# a deliberate, reviewable code change. faster-whisper, detoxify, and easyocr
# remain unpinned below — see each one's own downloader for why.
#
# Checksum verification (the other half of "pinned model file + checksum")
# does NOT need a pre-known hash: _verify_or_record_checksum() records the
# SHA-256 on first download and compares against it on every subsequent run,
# so silent corruption or an unexpected upstream file swap is caught even
# without "revision" ever being set.
MODELS: list[dict] = [
    # ── Text / embedding ──────────────────────────────────────────────────────
    {
        "key": "embedder",
        "model_id": "BAAI/bge-large-en-v1.5",
        "type": "sentence-transformers",
        "size_gb": 1.35,
        "gated": False,
        "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",  # pragma: allowlist secret
    },
    {
        "key": "reranker",
        "model_id": "BAAI/bge-reranker-large",
        "type": "sentence-transformers",
        "size_gb": 1.34,
        "gated": False,
        "revision": "55611d7bca2a7133960a6d3b71e083071bbfc312",  # pragma: allowlist secret
    },
    {
        "key": "ner",
        "model_id": "dslim/bert-base-NER",
        "type": "token-classification",
        "size_gb": 0.43,
        "gated": False,
        "revision": "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",  # pragma: allowlist secret
    },
    {
        "key": "finbert",
        "model_id": "yiyanghkust/finbert-tone",
        "type": "sequence-classification",
        "size_gb": 0.44,
        "gated": False,
        "revision": "4921590d3c0c3832c0efea24c8381ce0bda7844b",  # pragma: allowlist secret
    },
    {
        "key": "keybert",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence-transformers",
        "size_gb": 0.09,
        "gated": False,
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",  # pragma: allowlist secret
    },
    # ── Vision ────────────────────────────────────────────────────────────────
    {
        "key": "siglip",
        "model_id": "google/siglip-so400m-patch14-384",
        "type": "transformers",
        "size_gb": 1.76,
        "gated": False,
        "revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",  # pragma: allowlist secret
    },
    {
        "key": "blip",
        "model_id": "Salesforce/blip-image-captioning-large",
        "type": "blip-captioning",
        "size_gb": 0.90,
        "gated": False,
        "revision": "353689b859fcf0523410b1806dace5fb46ecdf41",  # pragma: allowlist secret
    },
    {
        "key": "qwen2vl",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "qwen2vl",
        "size_gb": 2.20,
        "gated": False,
        "revision": "895c3a49bc3fa70a340399125c650a463535e71c",  # pragma: allowlist secret
    },
    {
        "key": "trocr",
        "model_id": "microsoft/trocr-large-printed",
        "type": "vision-encoder-decoder",
        "size_gb": 0.36,
        "gated": False,
        "revision": "9ff792d8e7c22061f2ee67e1ed2246b1f9ef1e98",  # pragma: allowlist secret
    },
    # ── Audio ─────────────────────────────────────────────────────────────────
    {
        "key": "whisper",
        "model_id": "Systran/faster-whisper-large-v3",
        "type": "faster-whisper",
        "size_gb": 1.55,
        "gated": False,
    },
    {
        "key": "diarizer",
        "model_id": "pyannote/speaker-diarization-3.1",
        "type": "pyannote",
        "size_gb": 0.60,
        "gated": True,
        "optional": True,
        "revision": "84fd25912480287da0247647c3d2b4853cb3ee5d",  # pragma: allowlist secret
    },
    {
        "key": "seg30",
        "model_id": "pyannote/segmentation-3.0",
        "type": "pyannote",
        "size_gb": 0.20,
        "gated": True,
        "optional": True,
        "revision": "e66f3d3b9eb0873085418a7b813d3b369bf160bb",  # pragma: allowlist secret
    },
    {
        "key": "wespeaker",
        "model_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "type": "pyannote",
        "size_gb": 0.10,
        "gated": True,
        "optional": True,
        "revision": "837717ddb9ff5507820346191109dc79c958d614",  # pragma: allowlist secret
    },
    # ── Guardrails ────────────────────────────────────────────────────────────
    {
        "key": "detoxify",
        # Not a real HF repo id — detoxify fetches via torch.hub, not
        # huggingface_hub. Used only as this script's own manifest tracking
        # key (see _write_manifest); TORCH_HOME above controls where the
        # actual checkpoint lands.
        "model_id": "detoxify/original",
        "type": "detoxify",
        "size_gb": 0.42,
        "gated": False,
        # output_guard._get_detoxify() already fails open (falls back to the
        # pattern-based harmful-content heuristic) if Detoxify can't load —
        # matches that same non-critical severity here rather than making a
        # missing toxicity-scoring layer block the whole download run.
        "optional": True,
    },
    # ── LLM (GGUF single-file) — handled via GGUF_MODELS below ───────────────
    # key "gguf" (and any other GGUF entry) is injected into the run loop,
    # not listed here — GGUF needs a (repo, filename) pair, not a bare
    # model_id, so it doesn't fit the MODELS dict shape cleanly.
    {
        # Not an HF Hub or torch.hub model — EasyOCR fetches its detector +
        # recognizer weights from its own JaidedAI-hosted CDN into
        # EASYOCR_MODEL_DIR (app/core/config.py). Added here after a live
        # Tier-2 eval run showed the FIRST real image ingest on a fresh box
        # eating 155s against a 10s SLO (both a live download and the model
        # load happening inline on the request path) — every model that can
        # be pre-fetched at provisioning time should be, same as everything
        # else in this file.
        "key": "easyocr",
        "model_id": "easyocr/en",
        "type": "easyocr",
        "size_gb": 0.06,
        "gated": False,
        # image_ingest.py/image_chunker.py already treat a missing/failed
        # EasyOCR as a soft degradation (falls back to TrOCR-only OCR, logs
        # a warning, never raises) — this provisioning script should not be
        # stricter than the app it's provisioning for. Without this flag, a
        # box where `pip install` hasn't picked up easyocr>=1.7 yet (it IS
        # declared in requirements.txt/pyproject.toml — an environment gap,
        # not a missing dependency) hard-fails the ENTIRE script run over
        # one non-critical model, the same mistake this file's own comments
        # already document fixing for detoxify below.
        "optional": True,
    },
]

# Each entry needs its own repo+filename because a single from_pretrained()-
# style model_id isn't enough to locate one file inside a GGUF repo that may
# host several quantizations. "startup" (default True, omit the key to keep
# it True) controls whether main()'s default run (the one start_server.py
# invokes on every boot) includes this entry — set False only for a model
# that would genuinely cost something recurring on every normal boot (real
# VRAM/compute, not just a one-time disk fetch); see qwen_judge's own
# comment above for why a plain download doesn't qualify. `--only <key>` or
# `--include-eval-models` still force-fetch anything regardless of this flag.
GGUF_MODELS: list[dict] = [
    {
        "key": "gguf",
        "gguf_repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "gguf_filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "size_gb": 9.0,
        "gated": False,
        "revision": "05244aa5d871c661c80082a15d3bce44714d068d",  # pragma: allowlist secret
    },
    {
        "key": "qwen_judge",
        # bartowski's requant, not the official Qwen/Qwen2.5-7B-Instruct-GGUF
        # repo — verified live via the HF Hub API (2026-08-01) that the
        # official repo's Q4_K_M is split across 2 shard files
        # (qwen2.5-7b-instruct-q4_k_m-0000{1,2}-of-00002.gguf), which this
        # single-filename download pattern can't express. bartowski publishes
        # a single-file Q4_K_M for this size class, same convention as the
        # "gguf" entry above. Don't change this filename/repo without
        # re-checking that same API response first — see prometheus_judge's
        # retired comment here for what happens when that step is skipped
        # (a guessed filename 404'd on every real download attempt).
        "gguf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "gguf_filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "gated": False,
        "optional": True,
        "revision": "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",  # pragma: allowlist secret
        # Loaded only by app/eval/judges/qwen_judge.py during a Tier-2
        # eval run (also backs the Ragas report and DeepEval — MAGIK's
        # single eval judge) — never touched during normal request serving,
        # so this was originally "startup": False (excluded from the default
        # boot run) on the theory that it shouldn't cost anything on a normal
        # boot. That theory doesn't actually hold: _dl_gguf() below is a
        # pure disk fetch (hf_hub_download + hardlink) — it never
        # instantiates Llama(...), so it costs zero VRAM regardless of this
        # flag; VRAM is only spent if/when qwen_judge.py's own
        # _load_qwen_judge() runs, which this flag doesn't gate at all. The
        # ONLY real cost was a one-time ~4.7GB download on the box's FIRST
        # boot (every boot after that is a fast cached-file + checksum
        # check, same as every other model here). Given that, defaulting
        # it OFF just meant Tier-2 — an important quality gate — silently
        # graded with a much weaker lexical judge until someone remembered
        # a manual `--only qwen_judge` step (confirmed happening live with
        # the previous judge, prometheus_judge, on 2026-07-31). Left
        # "optional": True so a download hiccup still can't fail the whole
        # provisioning run; qwen_judge.py's own ensure_available() is a
        # second, independent safety net that auto-fetches it inline if
        # it's still somehow missing when Tier-2 actually needs it.
    },
]


# ── cache detection ───────────────────────────────────────────────────────────


def _is_hub_cached(model_id: str, mtype: str | None = None) -> bool:
    # detoxify (torch.hub, under TORCH_HOME) and easyocr (its own flat
    # model_storage_directory, under _easyocr_dir) don't use the standard
    # `models--X/snapshots/` huggingface_hub cache layout the check below
    # assumes — checking that layout for them always returns False, so the
    # "already cached, skip" fast path was never taken and every boot paid
    # the full ~12s/~2s cost of re-instantiating Detoxify()/Reader() (the
    # underlying network fetch itself already no-ops when cached; the model
    # construction/load does not). Confirmed live 2026-08-01: both reported
    # "OK in Ns" — not "Already cached — checksum OK, skipping" — on every
    # single boot despite the files genuinely being present and reused.
    if mtype == "detoxify":
        checkpoints = Path(_torch_home) / "hub" / "checkpoints"
        return checkpoints.is_dir() and any(checkpoints.iterdir())
    if mtype == "easyocr":
        return _easyocr_dir.is_dir() and any(_easyocr_dir.iterdir())
    cache_key = "models--" + model_id.replace("/", "--")
    snapshots = Path(_hf_home) / "hub" / cache_key / "snapshots"
    if not snapshots.exists():
        return False
    return any(
        f.is_file() for snap in snapshots.iterdir() if snap.is_dir() for f in snap.rglob("*")
    )


def _is_gguf_cached(filename: str) -> bool:
    return (_gguf_dir / filename).exists()


# ── per-type downloaders ──────────────────────────────────────────────────────
# Bandit B615 (huggingface_unsafe_download) flags every from_pretrained() call
# below — it can't trace `revision` through the **kw dict these functions
# build, so it doesn't see that _dispatch_download() (below) already threads
# a `revision` through to every one of them. The real remaining gap is
# structural, not a missing kwarg: on the very FIRST download of a model
# there is no revision to pin to yet — that's inherent to this project's
# TOFU (Trust-On-First-Use) checksum model (download_manifest.json), not an
# oversight. Every download after the first is pinned. Reviewed 2026-07-25.


def _dl_transformers(model_id: str, token: str, revision: str | None = None) -> None:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    kw = {"token": token} if token else {}
    if revision:
        kw["revision"] = revision
    try:
        AutoProcessor.from_pretrained(model_id, **kw)
    except Exception:
        try:
            AutoTokenizer.from_pretrained(model_id, **kw)
        except Exception:
            pass
    AutoModel.from_pretrained(model_id, **kw)


def _dl_token_classification(model_id: str, token: str, revision: str | None = None) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    kw = {"token": token} if token else {}
    if revision:
        kw["revision"] = revision
    AutoTokenizer.from_pretrained(model_id, **kw)
    AutoModelForTokenClassification.from_pretrained(model_id, **kw)


def _dl_sequence_classification(model_id: str, token: str, revision: str | None = None) -> None:
    from transformers import BertForSequenceClassification, BertTokenizer

    kw = {"token": token} if token else {}
    if revision:
        kw["revision"] = revision
    BertTokenizer.from_pretrained(model_id, **kw)
    BertForSequenceClassification.from_pretrained(model_id, **kw)


def _dl_blip_captioning(model_id: str, token: str, revision: str | None = None) -> None:
    from transformers import BlipForConditionalGeneration, BlipProcessor

    kw = {"token": token} if token else {}
    if revision:
        kw["revision"] = revision
    BlipProcessor.from_pretrained(model_id, **kw)
    BlipForConditionalGeneration.from_pretrained(model_id, **kw)


def _dl_sentence_transformers(model_id: str, revision: str | None = None) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    kw = {"revision": revision} if revision else {}
    if "reranker" in model_id.lower():
        CrossEncoder(model_id, **kw)
    else:
        SentenceTransformer(model_id, **kw)


def _dl_faster_whisper(model_id: str) -> None:
    # NOTE: no revision pinning here — WhisperModel takes a CTranslate2 size
    # shorthand / download_root, not a raw HF revision kwarg passed through
    # to hf_hub_download internally in a way this script can safely target.
    # Left unpinned rather than guessing at an unsupported argument.
    from faster_whisper import WhisperModel

    size = model_id.split("faster-whisper-", 1)[-1]
    WhisperModel(size, device="cpu", compute_type="int8", download_root=None)


_easyocr_dir = Path(_hf_home) / "easyocr"


def _dl_easyocr() -> None:
    # Reader() downloads on construction unless download_enabled=False.
    # model_storage_directory must match app/core/config.py's
    # EASYOCR_MODEL_DIR default (both derive from the same _hf_home /
    # HF_HOME) so the app finds what this script fetched.
    import easyocr

    _easyocr_dir.mkdir(parents=True, exist_ok=True)
    easyocr.Reader(["en"], gpu=False, verbose=False, model_storage_directory=str(_easyocr_dir))


def _dl_detoxify() -> None:
    # No revision pinning — detoxify's own package version selects the
    # checkpoint via torch.hub, not a raw HF revision this script could pin.
    # torch.hub.load_state_dict_from_url() no-ops on a re-run if the
    # checkpoint is already present under TORCH_HOME, so this is safe and
    # fast to call unconditionally (see the missing _is_detoxify_cached
    # fast-path note in main()'s cache-check).
    from detoxify import Detoxify

    Detoxify("original")


def _dl_vision_encoder_decoder(model_id: str, token: str, revision: str | None = None) -> None:
    import transformers
    from transformers import AutoProcessor, VisionEncoderDecoderModel

    kw = {"token": token} if token else {}
    if revision:
        kw["revision"] = revision
    AutoProcessor.from_pretrained(model_id, **kw)
    # TrOCR: suppress benign MISSING encoder.pooler.* (pooler unused in cross-attention
    # generation) and UNEXPECTED embed_positions._float_tensor (legacy positional buffer).
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        VisionEncoderDecoderModel.from_pretrained(model_id, **kw)
    finally:
        transformers.logging.set_verbosity(prev)


def _dl_qwen2vl(model_id: str, token: str, revision: str | None = None) -> None:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    kw: dict = {"trust_remote_code": True}
    if token:
        kw["token"] = token
    if revision:
        kw["revision"] = revision
    AutoProcessor.from_pretrained(model_id, **kw)
    Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kw)


def _dl_pyannote(model_id: str, token: str, revision: str | None = None) -> None:
    if not token:
        print(f"  SKIP {model_id} — HF_TOKEN not set.")
        print(f"        Accept license: https://hf.co/{model_id}")
        return
    # Use snapshot_download instead of Pipeline.from_pretrained so we never
    # instantiate the model here — avoids torchaudio version mismatches at
    # download time (AudioMetaData removed in torchaudio >= 0.12).
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        cache_dir=str(Path(_hf_home) / "hub"),
        token=token,
        revision=revision,
    )


def _dl_gguf(repo: str, filename: str, revision: str | None = None) -> None:
    from huggingface_hub import hf_hub_download

    _gguf_dir.mkdir(parents=True, exist_ok=True)
    dest = _gguf_dir / filename
    print(f"  Downloading {filename} from {repo} ...")
    print(f"  Destination: {dest}")
    cached = hf_hub_download(
        repo_id=repo,
        filename=filename,
        cache_dir=str(Path(_hf_home) / "hub"),
        token=HF_TOKEN or None,
        revision=revision,
    )
    if dest.is_symlink() and not dest.exists():
        dest.unlink()  # dangling symlink from a previous interrupted run
    if not dest.exists():
        # Hardlink instead of copy — a GGUF is several GB, and this file
        # already lives once in the hub blob store; copying it duplicates
        # that on disk for no reason (bit us directly: a copy of a 9GB file
        # ran the disk out of space mid-write). Same filesystem (.hf_cache),
        # so a hardlink is free (zero extra bytes, instant).
        # hf_hub_download() returns a path INSIDE snapshots/ that is itself a
        # symlink to blobs/<hash> — os.link() on a symlink hard-links the
        # symlink's inode (not its target), producing a second symlink whose
        # relative target text is only valid from the original directory.
        # Resolve to the real blob file first so the hardlink always points
        # at actual file bytes. Falls back to a copy if hardlinking isn't
        # possible (e.g. cross-device .hf_cache mount).
        real_src = Path(cached).resolve()
        try:
            os.link(real_src, dest)
        except OSError:
            shutil.copy2(real_src, dest)
    size_gb = dest.stat().st_size / 1e9
    print(f"  Size on disk: {size_gb:.2f} GB")


# ── checksums ─────────────────────────────────────────────────────────────────
# "Pinned model file + checksum" (System Design v2.0 §2.2) doesn't require
# knowing the correct hash in advance — that's not something this script can
# responsibly fabricate. Instead: record the SHA-256 the first time a model is
# downloaded, then verify against it on every later run (including cache-hit
# runs, so a corrupted or unexpectedly-swapped local file is still caught).


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir(path: Path, only_files: list[str] | None = None) -> tuple[str, list[str]]:
    """Combined hash of files under path, plus the sorted relative-path list
    that was actually hashed. Stable regardless of filesystem traversal
    order, so it's comparable across machines/runs.

    `only_files`, when given, restricts hashing to exactly those relative
    paths (skipping any that no longer exist) instead of re-enumerating the
    directory — see _model_checksum's docstring for why this matters: model
    loading code can write auxiliary files into this same directory after
    download (tokenizer merges, generated indexes, framework-specific
    caches), and re-enumerating "everything currently here" would count
    those as drift even though nothing originally downloaded ever changed.
    """
    if only_files is not None:
        candidates = [path / rel for rel in only_files]
    else:
        candidates = sorted(path.rglob("*"))
    parts = []
    hashed: list[str] = []
    for f in candidates:
        if f.is_file():
            rel = f.relative_to(path).as_posix()
            parts.append(f"{rel}:{_sha256_file(f)}")
            hashed.append(rel)
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest, sorted(hashed)


def _model_checksum(
    model_id: str, mtype: str, gguf_filename: str | None = None, only_files: list[str] | None = None
) -> tuple[str | None, list[str] | None]:
    """Compute the current on-disk checksum for a model, or (None, None) if
    its layout doesn't support a clean single hash (rare loader types).

    Returns (sha256, file_list) — file_list is the sorted relative paths
    that were actually hashed (None for the single-file gguf case, where
    there's nothing to enumerate). Pass `only_files` on a verify call to
    hash exactly the files recorded at download time, ignoring anything
    added to the directory since — see _sha256_dir's docstring.

    "detoxify" always falls through to the generic .hf_cache-shaped path
    below, which won't exist (it's cached under TORCH_HOME instead) — this
    correctly returns (None, None) (no checksum available) rather than a
    wrong hash.
    """
    if mtype == "gguf":
        path = _gguf_dir / gguf_filename
        return (_sha256_file(path), None) if path.exists() else (None, None)
    cache_key = "models--" + model_id.replace("/", "--")
    snapshots = Path(_hf_home) / "hub" / cache_key / "snapshots"
    if not snapshots.exists() or not any(snapshots.iterdir()):
        return None, None
    # Deliberately does NOT try to pick "the one right" revision snapshot.
    # An earlier version picked max(revs, key=mtime), assuming exactly one
    # snapshot dir — but a single from_pretrained() call sequence (tokenizer,
    # then model) can resolve "main" to two different commits moments apart
    # and materialize files across BOTH, e.g. yiyanghkust/finbert-tone: an
    # older revision with pytorch_model.bin plus a newer one with
    # model.safetensors, both real, both on disk. With two mtimes that close
    # together, "latest by mtime" depends on directory-iteration order, which
    # is not guaranteed stable — so a model that verified clean would
    # "mismatch" moments later with nothing actually corrupted (caught
    # 2026-07-30: identical files on disk, no re-download in between).
    # _sha256_dir already recurses (rglob), so hashing `snapshots` itself
    # covers every revision directory with one deterministic, sorted walk —
    # more thorough than picking one, not less, and immune to this class of
    # bug regardless of how many revisions the cache ends up holding.
    return _sha256_dir(snapshots, only_files=only_files)


def _verify_or_record_checksum(
    model_id: str, mtype: str, gguf_filename: str | None = None
) -> tuple[str | None, bool]:
    """Returns (sha256, mismatch). mismatch=True means a file recorded at the
    last download is missing or its content changed — silent corruption or
    an unexpected upstream swap, not assumed-safe. Files that appeared in
    the directory AFTER that download (e.g. written by the model-loading
    code the first time the app actually used it) are never part of this
    comparison — see _sha256_dir's docstring for why re-enumerating
    "whatever's there now" produced permanent false-positive mismatches for
    every HF-hub-cached model that gets touched at load time (confirmed
    live 2026-08-01: embedder/reranker/ner/finbert/keybert/siglip/blip/
    qwen2vl/trocr/whisper flagged MISMATCH on every single boot despite
    loading and working fine at runtime)."""
    manifest_path = Path(_hf_home) / "download_manifest.json"
    recorded_files: list[str] | None = None
    recorded_sha: str | None = None
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
        for e in entries:
            if e.get("model_id") == model_id and e.get("sha256"):
                recorded_sha = e["sha256"]
                recorded_files = e.get("files")  # None for pre-fix manifest entries
                break

    current, _hashed_files = _model_checksum(
        model_id, mtype, gguf_filename=gguf_filename, only_files=recorded_files
    )
    if current is None:
        return None, False
    if recorded_sha is None:
        return current, False
    return current, recorded_sha != current


# ── manifest ──────────────────────────────────────────────────────────────────


def _write_manifest(
    model_id: str,
    size_gb: float,
    mtype: str,
    sha256: str | None,
    revision: str | None,
    gguf_filename: str | None = None,
    files: list[str] | None = None,
) -> None:
    manifest_path = Path(_hf_home) / "download_manifest.json"
    entries: list = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries = [e for e in entries if e.get("model_id") != model_id]
    entry = {
        "model_id": model_id,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "size_gb": size_gb,
        "type": mtype,
        "sha256": sha256,
        "revision": revision,  # None = tracked the default branch, not pinned
    }
    if files is not None:
        # The exact relative-path file list hashed at download time — later
        # verification hashes only these, so files the model-loading code
        # writes into this same directory afterward never register as
        # "drift". See _sha256_dir's docstring.
        entry["files"] = files
    if gguf_filename:
        # Disambiguates which GGUF this entry is once more than one GGUF-type
        # model can be in the manifest (e.g. the main LLM vs. the Prometheus
        # eval judge) — app/core/startup_validator.py's _validate_gguf_checksum
        # needs this to pick the right entry instead of "the first gguf one".
        entry["gguf_filename"] = gguf_filename
    entries.append(entry)
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# ── dispatch ──────────────────────────────────────────────────────────────────


# mtype -> downloader, for every type that shares the (model_id, token, *,
# revision) signature. gguf/faster-whisper/sentence-transformers/pyannote
# don't share it (no token, or need the HF_TOKEN early-return below) and stay
# as explicit branches in _dispatch_download rather than forcing a fake
# shared signature on them.
_TOKEN_GATED_DOWNLOADERS: dict = {
    "qwen2vl": _dl_qwen2vl,
    "vision-encoder-decoder": _dl_vision_encoder_decoder,
    "blip-captioning": _dl_blip_captioning,
    "token-classification": _dl_token_classification,
    "sequence-classification": _dl_sequence_classification,
}


def _dispatch_download(
    mtype: str,
    model_id: str,
    gated: bool,
    revision: str | None,
    gguf_repo: str | None = None,
    gguf_filename: str | None = None,
) -> bool:
    """Run the right downloader for this model type. Returns False if the
    caller should count this as a skip rather than a success (pyannote with
    no HF_TOKEN set — a deliberate, expected skip, not a failure)."""
    if mtype == "gguf":
        _dl_gguf(gguf_repo, gguf_filename, revision=revision)
        return True
    if mtype == "faster-whisper":
        _dl_faster_whisper(model_id)
        return True
    if mtype == "sentence-transformers":
        _dl_sentence_transformers(model_id, revision=revision)
        return True
    if mtype == "pyannote":
        _dl_pyannote(model_id, HF_TOKEN, revision=revision)
        return bool(HF_TOKEN)
    if mtype == "detoxify":
        _dl_detoxify()
        return True
    if mtype == "easyocr":
        _dl_easyocr()
        return True

    downloader = _TOKEN_GATED_DOWNLOADERS.get(mtype, _dl_transformers)
    downloader(model_id, HF_TOKEN if gated else "", revision=revision)
    return True


def _handle_cached(
    model_id: str, mtype: str, optional: bool, gguf_filename: str | None = None
) -> str:
    """Model already on disk — verify its checksum instead of re-downloading.
    Returns "ok", "mismatch_skip" (optional model, don't fail the run), or
    "mismatch_fail" (required model, caller should record it as failed)."""
    _current_sha, mismatch = _verify_or_record_checksum(
        model_id, mtype, gguf_filename=gguf_filename
    )
    if not mismatch:
        print("  Already cached — checksum OK, skipping.\n")
        return "ok"
    print(
        f"  CHECKSUM MISMATCH — {model_id} on disk no longer matches the "
        f"hash recorded at last download. Possible corruption or an "
        f"unexpected file swap; re-download to be safe.\n"
    )
    return "mismatch_skip" if optional else "mismatch_fail"


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all MAGIK models to .hf_cache/")
    parser.add_argument(
        "--skip-gated", action="store_true", help="Skip all gated models (pyannote) without error"
    )
    parser.add_argument(
        "--only", metavar="KEY", help="Download only this model key (e.g. embedder, gguf, diarizer)"
    )
    parser.add_argument(
        "--include-eval-models",
        action="store_true",
        help=(
            "Also download any eval-only model explicitly marked "
            '"startup": False (none currently — qwen_judge is in the '
            "default run since it costs nothing but disk on a normal boot). "
            "Kept for a future judge/eval model that genuinely should stay opt-in."
        ),
    )
    args = parser.parse_args()

    all_entries = MODELS + [
        {
            "model_id": g["gguf_repo"],
            "type": "gguf",
            **g,
        }
        for g in GGUF_MODELS
    ]

    if args.only:
        keys = {m["key"] for m in all_entries}
        if args.only not in keys:
            sys.exit(f"Unknown key '{args.only}'. Valid keys: {', '.join(sorted(keys))}")
        # --only always honors the explicit request, regardless of "startup".
        run_list = [m for m in all_entries if m["key"] == args.only]
    elif args.include_eval_models:
        run_list = all_entries
    else:
        run_list = [m for m in all_entries if m.get("startup", True)]

    total_gb = sum(m["size_gb"] for m in run_list)

    print("\n=== MAGIK Model Downloader ===")
    print(f"HF_HOME  : {_hf_home}")
    print(f"GGUF dir : {_gguf_dir}")
    print(f"HF_TOKEN : {'set ✓' if HF_TOKEN else 'NOT SET — gated models will be skipped'}")
    print(f"Models   : {len(run_list)}  (~{total_gb:.1f} GB total)\n")

    Path(_hf_home).mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    failed: list[str] = []

    for i, m in enumerate(run_list, 1):
        key = m["key"]
        model_id = m["model_id"]
        mtype = m["type"]
        gated = m.get("gated", False)
        optional = m.get("optional", False)
        gguf_repo = m.get("gguf_repo")
        gguf_filename = m.get("gguf_filename")

        revision = m.get("revision")
        print(f"[{i}/{len(run_list)}] {key}  —  {model_id}  (~{m['size_gb']:.2f} GB)")
        if revision:
            print(f"  Pinned revision: {revision}")

        # skip gated if requested
        if gated and args.skip_gated:
            print("  SKIP (--skip-gated)\n")
            skipped += 1
            continue

        # fast local cache check
        cached = (
            _is_gguf_cached(gguf_filename)
            if mtype == "gguf"
            else _is_hub_cached(model_id, mtype=mtype)
        )
        if cached:
            result = _handle_cached(model_id, mtype, optional, gguf_filename=gguf_filename)
            if result == "ok":
                ok += 1
                skipped += 1
            elif result == "mismatch_skip":
                skipped += 1
            else:  # mismatch_fail
                failed.append(key)
            continue

        t0 = time.time()
        try:
            proceeded = _dispatch_download(
                mtype, model_id, gated, revision, gguf_repo=gguf_repo, gguf_filename=gguf_filename
            )
            if not proceeded:
                skipped += 1
                print()
                continue

            sha256, files = _model_checksum(model_id, mtype, gguf_filename=gguf_filename)
            _write_manifest(
                model_id,
                m["size_gb"],
                mtype,
                sha256,
                revision,
                gguf_filename=gguf_filename,
                files=files,
            )
            print(
                f"  OK in {time.time() - t0:.0f}s"
                + (f"  sha256={sha256[:12]}..." if sha256 else "")
            )
            print()
            ok += 1

        except Exception as exc:
            if optional:
                print(f"  SKIP (optional): {exc}\n")
                skipped += 1
            else:
                print(f"  FAILED: {exc}\n")
                failed.append(key)

    # ── summary ───────────────────────────────────────────────────────────────
    required_total = sum(1 for m in run_list if not m.get("optional"))
    print(f"{'='*50}")
    print(f"Done: {ok} ready  ({skipped} already cached / skipped)")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        print("Re-run to retry, or set HF_TOKEN for gated models.")
        sys.exit(1)
    else:
        print(f"All {required_total} required models cached successfully.")


if __name__ == "__main__":
    main()
