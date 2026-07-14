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
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── env must be set before any HuggingFace import ────────────────────────────
_project_root = Path(__file__).resolve().parents[3]
_hf_home      = os.getenv("HF_HOME", str(_project_root / ".hf_cache"))
_gguf_dir     = Path(_hf_home) / "gguf"
_gguf_file    = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"

os.environ["HF_HOME"]            = _hf_home
os.environ["HF_HUB_CACHE"]       = _hf_home + "/hub"
os.environ["TRANSFORMERS_CACHE"]  = _hf_home + "/hub"
os.environ["HF_DATASETS_CACHE"]   = _hf_home + "/datasets"

sys.path.insert(0, str(_project_root))

try:
    from dotenv import load_dotenv
    load_dotenv(str(_project_root / ".env"))
except ImportError:
    pass

HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── model manifest ────────────────────────────────────────────────────────────
# Each entry: key, model_id, type, size_gb, gated, optional
MODELS: list[dict] = [
    # ── Text / embedding ──────────────────────────────────────────────────────
    {"key": "embedder",  "model_id": "BAAI/bge-large-en-v1.5",               "type": "sentence-transformers",    "size_gb": 1.35, "gated": False},
    {"key": "reranker",  "model_id": "BAAI/bge-reranker-large",               "type": "sentence-transformers",    "size_gb": 1.34, "gated": False},
    {"key": "ner",       "model_id": "dslim/bert-base-NER",                   "type": "token-classification",     "size_gb": 0.43, "gated": False},
    {"key": "finbert",   "model_id": "yiyanghkust/finbert-tone",              "type": "sequence-classification",  "size_gb": 0.44, "gated": False},
    {"key": "keybert",   "model_id": "sentence-transformers/all-MiniLM-L6-v2","type": "sentence-transformers",    "size_gb": 0.09, "gated": False},

    # ── Vision ────────────────────────────────────────────────────────────────
    {"key": "siglip",    "model_id": "google/siglip-so400m-patch14-384",      "type": "transformers",             "size_gb": 1.76, "gated": False},
    {"key": "blip",      "model_id": "Salesforce/blip-image-captioning-large","type": "blip-captioning",          "size_gb": 0.90, "gated": False},
    {"key": "qwen2vl",   "model_id": "Qwen/Qwen2-VL-2B-Instruct",            "type": "qwen2vl",                  "size_gb": 2.20, "gated": False},
    {"key": "trocr",     "model_id": "microsoft/trocr-large-printed",         "type": "vision-encoder-decoder",   "size_gb": 0.36, "gated": False},

    # ── Audio ─────────────────────────────────────────────────────────────────
    {"key": "whisper",   "model_id": "Systran/faster-whisper-large-v3",       "type": "faster-whisper",        "size_gb": 1.55, "gated": False},
    {"key": "diarizer",  "model_id": "pyannote/speaker-diarization-3.1",       "type": "pyannote", "size_gb": 0.60, "gated": True, "optional": True},
    {"key": "seg30",     "model_id": "pyannote/segmentation-3.0",              "type": "pyannote", "size_gb": 0.20, "gated": True, "optional": True},
    {"key": "wespeaker", "model_id": "pyannote/wespeaker-voxceleb-resnet34-LM","type": "pyannote", "size_gb": 0.10, "gated": True, "optional": True},

    # ── LLM (GGUF single-file) — handled separately below ────────────────────
    # key "gguf" is injected into the run loop, not listed here
]

GGUF_REPO = "bartowski/Qwen2.5-14B-Instruct-GGUF"
GGUF_SIZE_GB = 9.0


# ── cache detection ───────────────────────────────────────────────────────────

def _is_hub_cached(model_id: str) -> bool:
    cache_key    = "models--" + model_id.replace("/", "--")
    snapshots    = Path(_hf_home) / "hub" / cache_key / "snapshots"
    if not snapshots.exists():
        return False
    return any(
        f.is_file()
        for snap in snapshots.iterdir()
        if snap.is_dir()
        for f in snap.rglob("*")
    )

def _is_gguf_cached() -> bool:
    return (_gguf_dir / _gguf_file).exists()


# ── per-type downloaders ──────────────────────────────────────────────────────

def _dl_transformers(model_id: str, token: str) -> None:
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    kw = {"token": token} if token else {}
    try:
        AutoProcessor.from_pretrained(model_id, **kw)
    except Exception:
        try:
            AutoTokenizer.from_pretrained(model_id, **kw)
        except Exception:
            pass
    AutoModel.from_pretrained(model_id, **kw)


def _dl_token_classification(model_id: str, token: str) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    kw = {"token": token} if token else {}
    AutoTokenizer.from_pretrained(model_id, **kw)
    AutoModelForTokenClassification.from_pretrained(model_id, **kw)


def _dl_sequence_classification(model_id: str, token: str) -> None:
    from transformers import BertForSequenceClassification, BertTokenizer
    kw = {"token": token} if token else {}
    BertTokenizer.from_pretrained(model_id, **kw)
    BertForSequenceClassification.from_pretrained(model_id, **kw)


def _dl_blip_captioning(model_id: str, token: str) -> None:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    kw = {"token": token} if token else {}
    BlipProcessor.from_pretrained(model_id, **kw)
    BlipForConditionalGeneration.from_pretrained(model_id, **kw)


def _dl_sentence_transformers(model_id: str) -> None:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    if "reranker" in model_id.lower():
        CrossEncoder(model_id)
    else:
        SentenceTransformer(model_id)


def _dl_faster_whisper(model_id: str) -> None:
    from faster_whisper import WhisperModel
    size = model_id.split("faster-whisper-", 1)[-1]
    WhisperModel(size, device="cpu", compute_type="int8", download_root=None)


def _dl_vision_encoder_decoder(model_id: str, token: str) -> None:
    import transformers
    from transformers import VisionEncoderDecoderModel, AutoProcessor
    kw = {"token": token} if token else {}
    AutoProcessor.from_pretrained(model_id, **kw)
    # TrOCR: suppress benign MISSING encoder.pooler.* (pooler unused in cross-attention
    # generation) and UNEXPECTED embed_positions._float_tensor (legacy positional buffer).
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        VisionEncoderDecoderModel.from_pretrained(model_id, **kw)
    finally:
        transformers.logging.set_verbosity(prev)


def _dl_qwen2vl(model_id: str, token: str) -> None:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    kw: dict = {"trust_remote_code": True}
    if token:
        kw["token"] = token
    AutoProcessor.from_pretrained(model_id, **kw)
    Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kw)


def _dl_pyannote(model_id: str, token: str) -> None:
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
    )


def _dl_gguf() -> None:
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


# ── manifest ──────────────────────────────────────────────────────────────────

def _write_manifest(model_id: str, size_gb: float, mtype: str) -> None:
    manifest_path = Path(_hf_home) / "download_manifest.json"
    entries: list = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries = [e for e in entries if e.get("model_id") != model_id]
    entries.append({
        "model_id":      model_id,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "size_gb":       size_gb,
        "type":          mtype,
    })
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download all MAGIK models to .hf_cache/")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Skip all gated models (pyannote) without error")
    parser.add_argument("--only", metavar="KEY",
                        help="Download only this model key (e.g. embedder, gguf, diarizer)")
    args = parser.parse_args()

    all_entries = MODELS + [
        {"key": "gguf", "model_id": GGUF_REPO, "type": "gguf",
         "size_gb": GGUF_SIZE_GB, "gated": False}
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

    ok = 0; skipped = 0; failed: list[str] = []

    for i, m in enumerate(run_list, 1):
        key      = m["key"]
        model_id = m["model_id"]
        mtype    = m["type"]
        gated    = m.get("gated", False)
        optional = m.get("optional", False)

        print(f"[{i}/{len(run_list)}] {key}  —  {model_id}  (~{m['size_gb']:.2f} GB)")

        # skip gated if requested
        if gated and args.skip_gated:
            print(f"  SKIP (--skip-gated)\n")
            skipped += 1
            continue

        # fast local cache check
        cached = _is_gguf_cached() if mtype == "gguf" else _is_hub_cached(model_id)
        if cached:
            print(f"  Already cached — skipping.\n")
            ok += 1
            skipped += 1
            continue

        t0 = time.time()
        try:
            if mtype == "gguf":
                _dl_gguf()
            elif mtype == "faster-whisper":
                _dl_faster_whisper(model_id)
            elif mtype == "sentence-transformers":
                _dl_sentence_transformers(model_id)
            elif mtype == "pyannote":
                _dl_pyannote(model_id, HF_TOKEN)
                if not HF_TOKEN:
                    skipped += 1
                    print()
                    continue
            elif mtype == "qwen2vl":
                _dl_qwen2vl(model_id, HF_TOKEN if gated else "")
            elif mtype == "vision-encoder-decoder":
                _dl_vision_encoder_decoder(model_id, HF_TOKEN if gated else "")
            elif mtype == "blip-captioning":
                _dl_blip_captioning(model_id, HF_TOKEN if gated else "")
            elif mtype == "token-classification":
                _dl_token_classification(model_id, HF_TOKEN if gated else "")
            elif mtype == "sequence-classification":
                _dl_sequence_classification(model_id, HF_TOKEN if gated else "")
            else:
                _dl_transformers(model_id, HF_TOKEN if gated else "")

            _write_manifest(model_id, m["size_gb"], mtype)
            print(f"  OK in {time.time() - t0:.0f}s\n")
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
