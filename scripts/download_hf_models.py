"""
Download all HuggingFace models needed by the RAG server to HF_HOME.
Skips models that are already cached. Called by ensure_models.sh on startup.
"""
import argparse
import os
import sys


def is_cached(hf_home: str, model_id: str) -> bool:
    cache_dir = os.path.join(
        hf_home, "hub",
        "models--" + model_id.replace("/", "--"),
        "blobs",
    )
    return os.path.isdir(cache_dir) and bool(os.listdir(cache_dir))


def download_snapshot(model_id: str, hf_home: str, token: str | None) -> None:
    os.environ["HF_HOME"] = hf_home
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, token=token or None)


def download_whisper(size: str, hf_home: str) -> None:
    model_id = f"Systran/faster-whisper-{size}"
    cache_dir = os.path.join(
        hf_home, "hub",
        "models--" + model_id.replace("/", "--"),
        "blobs",
    )
    if os.path.isdir(cache_dir) and os.listdir(cache_dir):
        print(f"[ensure_models] HF model present: {model_id}")
        return
    print(f"[ensure_models] HF model missing: {model_id} — downloading...")
    os.environ["HF_HOME"] = hf_home
    from faster_whisper import WhisperModel
    WhisperModel(size, device="cpu", compute_type="int8")
    print(f"[ensure_models] Downloaded: {model_id}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-home",   required=True)
    p.add_argument("--hf-token",  default="")
    p.add_argument("--embedding", required=True)
    p.add_argument("--siglip",    required=True)
    p.add_argument("--blip",      required=True)
    p.add_argument("--reranker",  required=True)
    p.add_argument("--whisper",   required=True)
    args = p.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    token = args.hf_token or None

    hf_models = [args.embedding, args.siglip, args.blip, args.reranker]
    for model_id in hf_models:
        if is_cached(args.hf_home, model_id):
            print(f"[ensure_models] HF model present: {model_id}")
        else:
            print(f"[ensure_models] HF model missing: {model_id} — downloading...")
            download_snapshot(model_id, args.hf_home, token)
            print(f"[ensure_models] Downloaded: {model_id}")

    download_whisper(args.whisper, args.hf_home)


if __name__ == "__main__":
    main()
