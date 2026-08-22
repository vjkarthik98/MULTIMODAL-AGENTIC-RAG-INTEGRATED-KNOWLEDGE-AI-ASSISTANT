#!/usr/bin/env python
"""Create or reset the one account that lets the post-release quality report
workflow (Ragas/DeepEval — cd.yml + quality-live.yml) log in as the eval
tenant (app.eval.config.EVAL_USER_ID) over a real POST /auth/login call.

Why this needs to exist at all: app/eval/http_client.py's EvalAuth normally
mints a token in-process (app.auth.jwt_handler.issue_tokens), which only
produces a token the server will accept when run inside the server's own
container/secret boundary (JWT_SECRET_KEY matches). The report workflow runs
on a bare GitHub Actions runner against the real deployed URL — outside that
boundary — so it needs a token the server issued itself, via a real login.

Pinned to the EXACT existing EVAL_USER_ID, not a fresh UUID: that tenant
already owns the ingested gold-set corpus in Qdrant/BM25
(app.eval.datasets.build_gold_set --ingest) — a new random tenant would only
ever see an empty knowledge base and fail every retrieval-scoped gold row.

Reuses AuthService.seed_load_test_user() (is_load_test => OTP-skip at login,
the same mechanism app/bin/seed_test_tenants.py's tenants already use for the
identical reason: non-interactive tooling can't solve an email OTP), just
pinned to a specific user_id instead of a fresh one.

Run this ONCE against the environment the report workflow targets (i.e. once
against production's real Mongo). Idempotent: reruns reset the password, they
never create a duplicate account or a second tenant.

Usage:
    python -m app.bin.seed_eval_reporter --password <real-password>
    python -m app.bin.seed_eval_reporter                # generates one, prints it once
"""

from __future__ import annotations

import argparse
import secrets
import string
import sys


def _generate_password() -> str:
    # zxcvbn-strong: mixed case + digit + symbol, 20 chars — same generator
    # seed_test_tenants.py already uses, for the same reason.
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(20))
        if (
            any(c.islower() for c in pw)
            and any(c.isupper() for c in pw)
            and any(c.isdigit() for c in pw)
            and any(c in "!@#$%^&*" for c in pw)
        ):
            return pw


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--email",
        default="eval-reporter@magik-internal.test",
        help="Login email for the account (default: eval-reporter@magik-internal.test)",
    )
    parser.add_argument(
        "--password", default=None, help="Omit to auto-generate a strong one and print it once"
    )
    args = parser.parse_args()

    from app.auth.service import AuthService
    from app.eval.config import EVAL_USER_ID

    if not EVAL_USER_ID:
        print("FAILED: EVAL_USER_ID is not set in this environment.", file=sys.stderr)
        return 1

    password = args.password or _generate_password()
    svc = AuthService()
    try:
        user = svc.seed_load_test_user(args.email, password, user_id=EVAL_USER_ID)
    except RuntimeError as exc:
        print(f"FAILED: {exc} — is MONGO_URI configured for this environment?", file=sys.stderr)
        return 1

    print(f"Eval-reporter account ready: {args.email} (user_id={user.user_id})")
    print()
    print("Store these as GitHub Actions repository secrets (Settings -> Secrets")
    print("and variables -> Actions) — cd.yml and quality-live.yml both read them:")
    print(f"  gh secret set EVAL_REPORTER_EMAIL --body '{args.email}'")
    print(f"  gh secret set EVAL_REPORTER_PASSWORD --body '{password}'")
    print()
    print("Never commit this password. Re-run this script (same --email) to rotate it —")
    print("it always resets the password on the existing account rather than creating a second one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
