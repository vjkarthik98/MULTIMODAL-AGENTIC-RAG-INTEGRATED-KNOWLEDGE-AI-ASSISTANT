#!/usr/bin/env python
"""Create or reset the dedicated test-tenant accounts used by automated quality
tooling (k6 load/stress/multi-user simulation, live-mode Schemathesis, ZAP).

These are NEVER the shared public demo login (testuser@ragdev.local, seeded by
app/bin/seed_demo_account.py) — that account is a single, publicly-known login
real recruiters use, and its rate-limit bucket (RATE_LIMIT_RPM=60/min) and
login-attempt bucket (AUTH_LOGIN_RATE_LIMIT_PER_MIN=5) would be contended by
automated tooling hammering it, degrading the demo for whoever's actually
looking at it. Test tenants are ordinary accounts in every other respect —
same Qdrant/BM25/Redis/Mongo tenant isolation as any real user — see
UserInDB.is_load_test in app/auth/models.py for exactly what's special about
them (nothing except an OTP skip, since these are driven by non-interactive
tooling that can't solve an email code).

Run this ONCE per environment (local docker-compose Mongo, or — only with an
explicit go-ahead — the real deployed Mongo) and reuse the credentials across
runs; it is idempotent (reruns reset the password, they don't create
duplicates). Credentials are written to a local, gitignored JSON file — never
commit them, never print them into CI logs, load them into GitHub Actions /
CI secrets for `quality.yml` / `quality-live.yml` to consume.

Usage:
    python -m app.bin.seed_test_tenants                     # 5 tenants, local Mongo
    python -m app.bin.seed_test_tenants --count 10
    python -m app.bin.seed_test_tenants --out /path/creds.json
"""

from __future__ import annotations

import argparse
import json
import secrets
import string
import sys
from pathlib import Path

EMAIL_DOMAIN = "magik-internal.test"  # .test is IANA-reserved, never a real deliverable domain
DEFAULT_OUT = Path(__file__).resolve().parents[2] / ".magik_test_tenants.json"


def _generate_password() -> str:
    # zxcvbn-strong: mixed case + digit + symbol, 20 chars.
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
    parser.add_argument("--count", type=int, default=5, help="Number of test tenants (default: 5)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write credentials JSON")
    args = parser.parse_args()

    from app.auth.service import AuthService

    svc = AuthService()
    tenants = []
    for i in range(1, args.count + 1):
        email = f"loadtest{i}@{EMAIL_DOMAIN}"
        password = _generate_password()
        try:
            user = svc.seed_load_test_user(email, password)
        except RuntimeError as exc:
            print(f"FAILED: {exc} — is MONGO_URI configured for this environment?", file=sys.stderr)
            return 1
        tenants.append({"email": email, "password": password, "user_id": user.user_id})
        print(f"  tenant {i}: {email} (user_id={user.user_id})")

    args.out.write_text(json.dumps({"tenants": tenants}, indent=2))
    print()
    print(f"Wrote {len(tenants)} credentials to {args.out}")
    print("This file is gitignored — DO NOT commit it. For CI (quality.yml /")
    print("quality-live.yml), load it into GitHub Actions secrets, e.g.:")
    print(f"  gh secret set MAGIK_TEST_TENANTS < {args.out}")
    print()
    print("perf/k6/ scripts read tenant credentials from the MAGIK_TEST_TENANTS")
    print("env var (same JSON shape) — see perf/k6/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
