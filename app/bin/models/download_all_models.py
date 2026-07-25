#!/usr/bin/env python3
"""
Pre-download all required models to .hf_cache so they survive instance restarts.

Re-running is safe: already-cached models are skipped without any network call.

Usage:
    python app/bin/models/download_all_models.py
    python app/bin/models/download_all_models.py --skip-gated    # skip pyannote (needs license)
    python app/bin/models/download_all_models.py --only gguf     # single model by key

Set HF_TOKEN in .env for gated models (pyannote diarization requires license acceptance
at https://hf.co/pyannote/speaker-diarization-3.1).
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
_gguf_file = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"

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
    },
    {
        "key": "reranker",
        "model_id": "BAAI/bge-reranker-large",
        "type": "sentence-transformers",
        "size_gb": 1.34,
        "gated": False,
    },
    {
        "key": "ner",
        "model_id": "dslim/bert-base-NER",
        "type": "token-classification",
        "size_gb": 0.43,
        "gated": False,
    },
    {
        "key": "finbert",
        "model_id": "yiyanghkust/finbert-tone",
        "type": "sequence-classification",
        "size_gb": 0.44,
        "gated": False,
    },
    {
        "key": "keybert",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence-transformers",
        "size_gb": 0.09,
        "gated": False,
    },
    # ── Vision ────────────────────────────────────────────────────────────────
    {
        "key": "siglip",
        "model_id": "google/siglip-so400m-patch14-384",
        "type": "transformers",
        "size_gb": 1.76,
        "gated": False,
    },
    {
        "key": "blip",
        "model_id": "Salesforce/blip-image-captioning-large",
        "type": "blip-captioning",
        "size_gb": 0.90,
        "gated": False,
    },
    {
        "key": "qwen2vl",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "qwen2vl",
        "size_gb": 2.20,
        "gated": False,
    },
    {
        "key": "trocr",
        "model_id": "microsoft/trocr-large-printed",
        "type": "vision-encoder-decoder",
        "size_gb": 0.36,
        "gated": False,
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
    },
    {
        "key": "seg30",
        "model_id": "pyannote/segmentation-3.0",
        "type": "pyannote",
        "size_gb": 0.20,
        "gated": True,
        "optional": True,
    },
    {
        "key": "wespeaker",
        "model_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "type": "pyannote",
        "size_gb": 0.10,
        "gated": True,
        "optional": True,
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
    # ── LLM (GGUF single-file) — handled separately below ────────────────────
    # key "gguf" is injected into the run loop, not listed here
]

GGUF_REPO = "bartowski/Qwen2.5-14B-Instruct-GGUF"
GGUF_SIZE_GB = 9.0


# ── cache detection ───────────────────────────────────────────────────────────


def _is_hub_cached(model_id: str) -> bool:
    cache_key = "models--" + model_id.replace("/", "--")
    snapshots = Path(_hf_home) / "hub" / cache_key / "snapshots"
    if not snapshots.exists():
        return False
    return any(
        f.is_file() for snap in snapshots.iterdir() if snap.is_dir() for f in snap.rglob("*")
    )


def _is_gguf_cached() -> bool:
    return (_gguf_dir / _gguf_file).exists()


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


def _dl_gguf(revision: str | None = None) -> None:
    from huggingface_hub import hf_hub_download

    _gguf_dir.mkdir(parents=True, exist_ok=True)
    dest = _gguf_dir / _gguf_file
    print(f"  Downloading {_gguf_file} from {GGUF_REPO} ...")
    print(f"  Destination: {dest}")
    cached = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=_gguf_file,
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


def _sha256_dir(path: Path) -> str:
    """Combined hash of every real file under path — stable regardless of
    filesystem traversal order, so it's comparable across machines/runs."""
    parts = []
    for f in sorted(path.rglob("*")):
        if f.is_file():
            rel = f.relative_to(path).as_posix()
            parts.append(f"{rel}:{_sha256_file(f)}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _model_checksum(model_id: str, mtype: str) -> str | None:
    """Compute the current on-disk checksum for a model, or None if its
    layout doesn't support a clean single hash (rare loader types).

    "detoxify" always falls through to the generic .hf_cache-shaped path
    below, which won't exist (it's cached under TORCH_HOME instead) — this
    correctly returns None (no checksum available) rather than a wrong hash.
    """
    if mtype == "gguf":
        path = _gguf_dir / _gguf_file
        return _sha256_file(path) if path.exists() else None
    cache_key = "models--" + model_id.replace("/", "--")
    snapshots = Path(_hf_home) / "hub" / cache_key / "snapshots"
    if not snapshots.exists():
        return None
    revs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not revs:
        return None
    # Hub cache can hold multiple revisions side by side — hash the most
    # recently modified snapshot, which is the one just downloaded/verified.
    latest = max(revs, key=lambda d: d.stat().st_mtime)
    return _sha256_dir(latest)


def _verify_or_record_checksum(model_id: str, mtype: str) -> tuple[str | None, bool]:
    """Returns (sha256, mismatch). mismatch=True means the file(s) on disk no
    longer match what was recorded the last time this model was downloaded —
    silent corruption or an unexpected upstream swap, not assumed-safe."""
    current = _model_checksum(model_id, mtype)
    if current is None:
        return None, False
    manifest_path = Path(_hf_home) / "download_manifest.json"
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
        for e in entries:
            if e.get("model_id") == model_id and e.get("sha256"):
                return current, e["sha256"] != current
    return current, False


# ── manifest ──────────────────────────────────────────────────────────────────


def _write_manifest(
    model_id: str, size_gb: float, mtype: str, sha256: str | None, revision: str | None
) -> None:
    manifest_path = Path(_hf_home) / "download_manifest.json"
    entries: list = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries = [e for e in entries if e.get("model_id") != model_id]
    entries.append(
        {
            "model_id": model_id,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "size_gb": size_gb,
            "type": mtype,
            "sha256": sha256,
            "revision": revision,  # None = tracked the default branch, not pinned
        }
    )
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


def _dispatch_download(mtype: str, model_id: str, gated: bool, revision: str | None) -> bool:
    """Run the right downloader for this model type. Returns False if the
    caller should count this as a skip rather than a success (pyannote with
    no HF_TOKEN set — a deliberate, expected skip, not a failure)."""
    if mtype == "gguf":
        _dl_gguf(revision=revision)
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

    downloader = _TOKEN_GATED_DOWNLOADERS.get(mtype, _dl_transformers)
    downloader(model_id, HF_TOKEN if gated else "", revision=revision)
    return True


def _handle_cached(model_id: str, mtype: str, optional: bool) -> str:
    """Model already on disk — verify its checksum instead of re-downloading.
    Returns "ok", "mismatch_skip" (optional model, don't fail the run), or
    "mismatch_fail" (required model, caller should record it as failed)."""
    _current_sha, mismatch = _verify_or_record_checksum(model_id, mtype)
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
    args = parser.parse_args()

    all_entries = MODELS + [
        {
            "key": "gguf",
            "model_id": GGUF_REPO,
            "type": "gguf",
            "size_gb": GGUF_SIZE_GB,
            "gated": False,
        }
    ]

    if args.only:
        keys = {m["key"] for m in all_entries}
        if args.only not in keys:
            sys.exit(f"Unknown key '{args.only}'. Valid keys: {', '.join(sorted(keys))}")
        run_list = [m for m in all_entries if m["key"] == args.only]
    else:
        run_list = all_entries

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
        # NOTE — "detoxify" always reports uncached here (falls through to
        # _is_hub_cached, whose .hf_cache-shaped path never exists for a
        # torch.hub-fetched model) and always re-attempts _dl_detoxify().
        # That's a fast no-op, not a bug: torch.hub.load_state_dict_from_url
        # skips the actual download itself if the file already exists under
        # TORCH_HOME. Not worth a bespoke torch.hub cache-path check against
        # a directory layout that's a detoxify-package internal, not a
        # documented contract.
        cached = _is_gguf_cached() if mtype == "gguf" else _is_hub_cached(model_id)
        if cached:
            result = _handle_cached(model_id, mtype, optional)
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
            proceeded = _dispatch_download(mtype, model_id, gated, revision)
            if not proceeded:
                skipped += 1
                print()
                continue

            sha256 = _model_checksum(model_id, mtype)
            _write_manifest(model_id, m["size_gb"], mtype, sha256, revision)
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
