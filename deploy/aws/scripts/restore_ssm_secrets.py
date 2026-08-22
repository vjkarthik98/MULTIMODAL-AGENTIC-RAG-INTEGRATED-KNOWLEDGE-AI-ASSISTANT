#!/usr/bin/env python3
"""Create or restore every /magik/* SSM SecureString this project depends on,
from ONE authoritative manifest — the single source of truth for what must
exist, replacing the individually hand-typed `aws ssm put-parameter` commands
previously documented in deploy/aws/README.md.

Why this exists: those 13 parameters were never Terraform-managed and never
had a script backing them — only one-off CLI commands, run once during the
account rebuild. One of those very commands is on record (CloudTrail) having
been silently mangled by Git Bash/MSYS2's path rewriting on Windows
(`/magik/jwt_secret_key` became `C:/Program Files/Git/magik/jwt_secret_key`)
before being caught and redone. Then, with no script, no drift check, and no
committed record of intent anywhere, all 13 were later found missing with no
corresponding delete event in CloudTrail — the exact failure mode this file
closes: never again a purely manual, unscripted, unverified population step.
Using boto3 directly (not the AWS CLI) also removes the MSYS2 path-mangling
bug class entirely — there is no shell in the path to mangle an argument.

Two ways to supply the actual secret VALUES, chosen per parameter:

  --from-env   Read from THIS PROCESS's own environment. Use this by running
               the script *inside* whichever container already has the real
               value loaded (`docker exec magik-current ...` for the 10
               app-level secrets, `docker exec magik-grafana ...` for the 2
               monitoring-only ones) — the exact mechanism used to restore
               after the 2026-08-22 incident. Recovers everything a live,
               healthy deployment already has, with no value ever hand-typed
               or hand-copied.

  --from-file  Read KEY=VALUE lines from a local, gitignored file (never
               committed — see .gitignore's `deploy/aws/.secrets.local.env`
               entry). This is the true disaster-recovery path: the one that
               still works even if every container and every AWS resource
               were gone simultaneously, provided a human kept this file
               somewhere durable and offline (a password manager export, an
               encrypted backup) — which is now the ONE place these values
               need to be deliberately kept safe, instead of nowhere.

The two GitHub PATs (ghcr_pat, github_actions_pat) are NOT in this manifest's
--from-env path — they are deploy-time/Lambda-time only, never loaded into
the app or monitoring containers' own environment. They can only be restored
via --from-file, using a freshly generated token from GitHub (Settings ->
Developer settings -> Personal access tokens) if lost — see
deploy/aws/README.md's "App secrets in SSM" section for the required scopes.

Never prints a secret value, before or after writing it — only which
parameter names it did or didn't touch.

Usage:
    # Restore the 10 app-level secrets from the running app container:
    docker exec magik-current python3.12 /app/deploy/aws/scripts/restore_ssm_secrets.py \\
        --from-env --group app

    # Restore the 2 monitoring-only secrets from the running Grafana container:
    docker exec magik-grafana python3 /restore_ssm_secrets.py --from-env --group monitoring

    # Full disaster recovery from an offline backup file (all 13, including PATs):
    python deploy/aws/scripts/restore_ssm_secrets.py --from-file /path/to/secrets.env --group all

    # Just check what's currently missing, write nothing:
    python deploy/aws/scripts/restore_ssm_secrets.py --check-only
"""

from __future__ import annotations

import argparse
import os
import sys

# (ssm_name, env_var_name, group) — the one place this project's SSM secret
# surface is enumerated. Add a row here first for any new secret; nothing
# else should ever hand-type an `aws ssm put-parameter` command again.
MANIFEST: list[tuple[str, str, str]] = [
    ("/magik/jwt_secret_key", "JWT_SECRET_KEY", "app"),
    ("/magik/secret_key", "SECRET_KEY", "app"),
    ("/magik/mongo_uri", "MONGO_URI", "app"),
    ("/magik/qdrant_api_key", "QDRANT_API_KEY", "app"),
    ("/magik/redis_token", "REDIS_TOKEN", "app"),
    ("/magik/hf_token", "HF_TOKEN", "app"),
    ("/magik/tavily_api_key", "TAVILY_API_KEY", "app"),
    ("/magik/google_client_secret", "GOOGLE_CLIENT_SECRET", "app"),
    ("/magik/smtp_password", "SMTP_PASSWORD", "app"),
    ("/magik/grafana_admin_password", "GRAFANA_ADMIN_PASSWORD", "monitoring"),
    ("/magik/ntfy_webhook_url", "NTFY_WEBHOOK_URL", "monitoring"),
    # PAT scopes needed if these are ever regenerated from scratch:
    #   ghcr_pat: write:packages, read:packages
    #   github_actions_pat: Administration: read-only (fine-grained), used
    #     only by the idle-stop Lambda's runner-busy check
    ("/magik/ghcr_pat", "GHCR_PAT", "pat"),
    ("/magik/github_actions_pat", "GITHUB_ACTIONS_PAT", "pat"),
]


def _client(profile: str | None, region: str):
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region) if profile else boto3.Session(
        region_name=region
    )
    return session.client("ssm")


def _check_only(profile: str | None, region: str) -> int:
    client = _client(profile, region)
    names = [name for name, _, _ in MANIFEST]
    missing: list[str] = []
    # get_parameters caps at 10 names per call.
    for i in range(0, len(names), 10):
        batch = names[i : i + 10]
        resp = client.get_parameters(Names=batch, WithDecryption=False)
        missing.extend(resp.get("InvalidParameters", []))

    present = [n for n in names if n not in missing]
    print(f"Present ({len(present)}/{len(names)}):")
    for n in present:
        print(f"  OK      {n}")
    if missing:
        print(f"\nMISSING ({len(missing)}/{len(names)}):")
        for n in missing:
            print(f"  MISSING {n}")
        return 1
    print("\nAll expected /magik/* parameters are present.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--from-env", action="store_true", help="Read values from this process's own environment"
    )
    src.add_argument(
        "--from-file", type=str, default=None, help="Read KEY=VALUE lines from this local file"
    )
    parser.add_argument(
        "--group",
        choices=["app", "monitoring", "pat", "all"],
        default="all",
        help="Restrict to one group of the manifest (default: all)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report which parameters exist/are missing; write nothing",
    )
    parser.add_argument(
        "--profile", default=None, help="Named AWS CLI profile (default: boto3's normal chain)"
    )
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    if args.check_only:
        return _check_only(args.profile, args.region)

    if not args.from_env and not args.from_file:
        print("FAILED: pass --from-env, --from-file <path>, or --check-only.", file=sys.stderr)
        return 2

    file_values: dict[str, str] = {}
    if args.from_file:
        try:
            with open(args.from_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    file_values[key.strip()] = value.strip()
        except OSError as exc:
            print(f"FAILED: could not read {args.from_file}: {exc}", file=sys.stderr)
            return 1

    client = _client(args.profile, args.region)
    written: list[str] = []
    skipped: list[str] = []

    for ssm_name, env_var, group in MANIFEST:
        if args.group != "all" and group != args.group:
            continue

        value = file_values.get(env_var) if args.from_file else os.getenv(env_var)
        if not value:
            skipped.append(f"{ssm_name} (no value for {env_var} in this source)")
            continue

        client.put_parameter(Name=ssm_name, Value=value, Type="SecureString", Overwrite=True)
        written.append(ssm_name)

    print(f"Wrote {len(written)} parameter(s):")
    for n in written:
        print(f"  OK      {n}")
    if skipped:
        print(f"\nSkipped {len(skipped)} (no value available from this source):")
        for n in skipped:
            print(f"  SKIP    {n}")

    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
