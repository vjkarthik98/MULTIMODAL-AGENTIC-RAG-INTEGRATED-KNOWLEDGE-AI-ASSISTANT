# DEPRECATED — do not `terraform apply` this module

This is the Terraform module for AWS account **857194222592**, which was
**decommissioned on 2026-09-12**. It is kept only as the as-built record of
that account. The live module is **`deploy/aws/terraform-new-account/`**
(account `266901698137`).

Nothing here is reachable any more: the account's EC2 fleet, EBS volumes, VPC,
IAM roles and OIDC provider are gone, and the `magik-admin` CLI profile that
addressed it returns `InvalidClientTokenId`. An `apply` from this directory
would either fail on credentials or, if someone re-pointed it at a live
account, fight `terraform-new-account/` for the same resource names.

## Why the account moved

Third account for this fleet (`537557168406` → `857194222592` →
`266901698137`). The move off `857194222592` was forced by its GPU vCPU quota
of 4, which left no room to run production and staging `g6e.xlarge` boxes at
the same time. Full sequence is in `.github/workflows/cd.yml`'s
"ACCOUNT MIGRATION NOTE".

## Known exposure in this directory's history

Four saved plan files — `destroy.tfplan`, `destroy2.tfplan`, `kuma.tfplan`,
`kuma_t4g.tfplan` — were committed to this **public** repository. A `.tfplan`
is a zip containing `tfplan`, `tfstate` and `tfstate-prev` members, and those
members are **not** redacted the way `terraform plan`'s console output is. Each
of these four therefore embedded the full PEM of `tls_private_key.magik`, the
RSA-4096 break-glass SSH key for account `857194222592`.

The `.gitignore` here did list plan files, but only as `tfplan*` — which
matches `tfplan3.out` and misses `destroy.tfplan`. Both spellings are now
listed, in this module and in `terraform-new-account/`.

Two independent guards should have caught this and neither did. The
`.gitignore` pattern was the first. The second was CI's `detect-secrets` gate,
which scans `git ls-files` — these files were tracked, so they were in scope,
but detect-secrets skips binary content and a `.tfplan` is a zip. A committed
archive is therefore a blind spot for that gate by design, not a
misconfiguration of it. The `.gitignore` is the only real control here, which
is why both spellings now appear in both modules.

Assessed impact, 2026-09-12:

- **What leaked:** one RSA private key, `magik-admin-key`, for the now-dead
  account `857194222592`. Every instance it could have opened is deleted.
- **What did NOT leak:** no SSM parameter *values*. The embedded state contains
  no `aws_ssm_parameter` resources at all — `/magik/ghcr_pat` and the app
  secrets appear only as IAM policy ARNs, never as values.
- **The current account is not affected.** `terraform-new-account/` generates
  its own `tls_private_key`; its `magik-admin-key` is a different key pair
  (public-key fingerprints compared directly and differ). Nothing in account
  `266901698137` trusts the leaked key.

The four files are untracked as of this commit, which removes them from the
tip but **not from git history** — they remain fetchable from prior commits of
a public repo. Given the key opens nothing that still exists, history was left
intact rather than force-rewritten. If this module is ever revived against a
live account, generate a fresh key pair first and treat the old one as burned.
