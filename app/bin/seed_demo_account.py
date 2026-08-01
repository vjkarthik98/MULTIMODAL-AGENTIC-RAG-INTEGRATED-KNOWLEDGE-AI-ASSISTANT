#!/usr/bin/env python
"""Create or reset the fixed demo account used for recruiter/hiring-manager
walkthroughs, since guest mode was removed (see CHANGELOG.md).

Usage (run in an environment with a real MONGO_URI configured, e.g. the
deployed box, not this dev/edit sandbox):

    python -m app.bin.seed_demo_account
    python -m app.bin.seed_demo_account --email you@example.com --password 'Other@123'

This ONLY creates/resets the Mongo user document (is_active=True,
is_demo=True — see /auth/login's OTP bypass in app/auth/router.py). It does
NOT upload files or fabricate example chat history: citations must come from
real retrieval against real embeddings, so populating the demo knowledge
base means logging in as this account through the normal UI/API once and
uploading real files / asking real questions — printed as next steps below.
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_EMAIL = "testuser@ragdev.local"
DEFAULT_PASSWORD = (
    "Test@123"  # pragma: allowlist secret — intentionally public demo credential, not a real secret
)

# Bundled finance benchmark files already used for accuracy testing —
# reasonable defaults to upload for a finance-RAG demo walkthrough.
SUGGESTED_FILES = [
    "data/finance/apple_10k.pdf",
    "data/finance/ctryprem.xlsx",
    "data/finance/aapl-20240928_g2.jpg",
    "data/finance/FOMC Press Conference September 18, 2024.mp3",
    "data/finance/Q4 2025 Earnings Call.mp4",
    "data/finance/fomc_dec2024.txt",
]

SUGGESTED_QUERIES = [
    "What was Apple's total revenue and net income for the period covered in the 10-K?",
    "Summarize the key risk factors Apple discloses in this filing.",
    "What country risk premium does the spreadsheet show for the United States vs. an emerging market?",
    "What did the Fed chair say about interest rate policy in the FOMC press conference?",
    "What were the key takeaways from the Q4 2025 earnings call?",
    "What does the revenue chart in the image show?",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--email", default=DEFAULT_EMAIL, help=f"Demo account email (default: {DEFAULT_EMAIL})"
    )
    parser.add_argument(
        "--password", default=DEFAULT_PASSWORD, help="Demo account password (default: Test@123)"
    )
    args = parser.parse_args()

    from app.auth.service import AuthService

    svc = AuthService()
    try:
        user = svc.seed_demo_user(args.email, args.password)
    except ValueError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"FAILED: {exc} — is MONGO_URI configured for this environment?", file=sys.stderr)
        return 1

    print(f"Demo account ready: {user.email} (user_id={user.user_id}, is_demo={user.is_demo})")
    print()
    print("This account skips OTP on every login (see /auth/login in app/auth/router.py).")
    print("It is a normal, fully-featured account otherwise — nothing about uploads,")
    print("deletes, or chat history is restricted for it.")
    print()
    print("Next steps — populate the knowledge base with REAL data (citations only ever")
    print("come from real retrieval, so this can't be faked/pre-scripted):")
    print("  1. Log in as this account through the running UI or API.")
    print("  2. Upload these bundled finance benchmark files:")
    for f in SUGGESTED_FILES:
        print(f"       - {f}")
    print("  3. Ask a handful of real questions so a genuine chat history with citations")
    print("     is already visible under Recents when a recruiter logs in, e.g.:")
    for q in SUGGESTED_QUERIES:
        print(f"       - {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
