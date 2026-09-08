#!/usr/bin/env python
"""Re-upload the bundled finance corpus into the demo account's knowledge base.

Why this exists: uploaded originals live only on the box, under
data/users/<user_id>/knowledge_base — bind-mounted from /opt/magik/data on the
ROOT EBS volume (see deploy/aws/terraform/ec2.tf). A new instance gets a blank
root volume, so replacing the box — or moving to a different AWS account, as on
2026-09-08 — takes the demo knowledge base with it. The per-user BM25 indexes
(data/users/<user_id>/bm25_index/*.pkl) go the same way. Nothing in the app
deletes them; there is simply no copy anywhere else.

Re-uploading through the normal /rag/upload route is the repair, because that
one path rebuilds all three stores at once: the KB copy on disk, the vectors in
Qdrant, and the BM25 index. Hand-copying files onto the box would restore only
the first and leave retrieval empty.

Usage (from the repo root, so the default paths resolve):

    python -m app.bin.restore_demo_kb --api-base https://<host>
    python -m app.bin.restore_demo_kb --api-base http://localhost:8000 --force

The account must already exist and be flagged is_demo — run
app.bin.seed_demo_account first if login comes back asking for an OTP.

Uploads are synchronous and the media files are large: the MP4 and MP3 can each
take several minutes on a cold GPU. --timeout covers a single file, not the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from app.core.config import settings

DEFAULT_EMAIL = settings.DEMO_ACCOUNT_EMAIL or "magikaiassistant@gmail.com"
DEFAULT_PASSWORD = "Demo@2026"  # pragma: allowlist secret — intentionally public demo credential, not a real secret

# One file per modality, the same corpus the accuracy benchmarks use, so a
# restored KB exercises every ingestion path a walkthrough might touch.
# Repo-relative; data/ is gitignored, so these exist only where the corpus has
# been placed by hand.
DEMO_FILES = [
    "data/raw/finance/pdf/apple_10k.pdf",
    "data/raw/finance/xlsx/ctryprem.xlsx",
    "data/raw/finance/image/aapl-20240928_g2.jpg",
    "data/raw/finance/docx/apple_investment_research_report.docx",
    "data/raw/finance/txt/fomc_dec2024.txt",
    "data/raw/finance/audio/FOMC Press Conference September 18, 2024.mp3",
    "data/raw/finance/video/Q4 2025 Earnings Call.mp4",
]


def _login(api_base: str, email: str, password: str, timeout: int) -> str:
    """Return a bearer access token. Bearer auth also sidesteps the CSRF
    double-submit check, which only applies to cookie-authenticated calls
    (see CSRFMiddleware in app/api/middleware.py)."""
    resp = requests.post(
        f"{api_base}/auth/login",
        json={"email": email, "password": password},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"login failed ({resp.status_code}): {resp.text[:300]}")

    body = resp.json()
    if body.get("otp_required"):
        raise RuntimeError(
            f"{email} is not flagged as the demo account, so login wants an email OTP "
            "this script cannot solve. Run `python -m app.bin.seed_demo_account` "
            "against this environment first."
        )
    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"login returned no access_token: {str(body)[:300]}")
    return str(token)


def _existing_files(api_base: str, token: str, timeout: int) -> set[str]:
    """Filenames already in the KB, so a re-run doesn't re-ingest everything."""
    resp = requests.get(
        f"{api_base}/rag/knowledge-base",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"could not list knowledge base ({resp.status_code}): {resp.text[:300]}")
    return {f.get("filename", "") for f in resp.json().get("files", [])}


def _upload(api_base: str, token: str, path: Path, timeout: int) -> dict:
    with path.open("rb") as fh:
        resp = requests.post(
            f"{api_base}/rag/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (path.name, fh)},
            data={"session_id": "demo-restore"},
            timeout=timeout,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    parser.add_argument("--email", default=DEFAULT_EMAIL, help=f"default: {DEFAULT_EMAIL}")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="default: Demo@2026")
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEMO_FILES,
        help="Files to upload (default: the bundled finance corpus, one per modality)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if a file of the same name is already in the KB",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-file upload timeout in seconds (default: 1800 — media ingestion is slow)",
    )
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")

    # Fail before logging in if the corpus isn't where we expect — a run that
    # authenticates and then skips everything is a confusing way to learn the
    # files are missing.
    paths = [Path(f) for f in args.files]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print("FAILED: these files do not exist (run from the repo root):", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 1

    try:
        token = _login(api_base, args.email, args.password, timeout=60)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Logged in as {args.email} at {api_base}")

    try:
        existing = _existing_files(api_base, token, timeout=60)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    if existing:
        print(f"Knowledge base already holds {len(existing)} file(s).")

    uploaded, skipped, failed = 0, 0, []
    for path in paths:
        if path.name in existing and not args.force:
            print(f"  skip     {path.name} (already in the knowledge base; --force to re-upload)")
            skipped += 1
            continue

        print(f"  upload   {path.name} …", flush=True)
        try:
            result = _upload(api_base, token, path, args.timeout)
        except Exception as exc:
            print(f"  FAILED   {path.name}: {exc}", file=sys.stderr)
            failed.append(path.name)
            continue

        print(
            f"  ok       {path.name} — {result.get('modality', '?')}, "
            f"{result.get('chunks', 0)} chunks, {result.get('stored', 0)} stored, "
            f"{result.get('latency', '?')}s"
        )
        uploaded += 1

    print()
    print(f"Uploaded {uploaded}, skipped {skipped}, failed {len(failed)}.")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
        return 1

    print()
    print("Verify with a real query before calling this done — the KB copy on disk")
    print("is not proof that retrieval works. Ask the demo account something only")
    print("these files can answer and check the citations, e.g.:")
    print("  - What was Apple's total revenue and net income in the 10-K?")
    print("  - What did the Fed chair say about rate policy in the FOMC conference?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
