# `deploy/aws` — Phase 30 scale-to-zero infrastructure

Everything needed to run MAGIK's public demo on a GPU box that is **stopped by
default** and wakes on demand.

```
Visitor ──▶ API Gateway HTTP API ──▶ wake-gateway Lambda (always on, ~$0)
              │  stopped → StartInstances + "warming up" page (auto-refresh)
              │  booting → same page
              └▶ healthy → 302 ──▶ EC2 g6e.xlarge (L40S) ──▶ Caddy :443 ──▶ app :8000
                                        ▲                                      │
     EventBridge every 5 min ──▶ idle-stop Lambda ──▶ StopInstances ◀──────────┘
                                  (20m idle, 15m min uptime, skips during deploys/eval)
```

The public entry point is `magik.vk-ai.online` — HTTPS end to end, via Caddy
+ Let's Encrypt on the box. Port 8000 is not exposed to the internet; Caddy
reaches the app over localhost only. `vk-ai.online` (the apex domain) points
at a separate portfolio site, unrelated to this app — one domain, two
independent subdomains, no shared infrastructure between them.

**Note on the front door:** the wake gateway sits behind an **API Gateway HTTP
API**, not a Lambda Function URL. A Function URL was tried first — `auth-type
NONE` plus the documented public-invoke resource policy, both verified correct
via the CLI on two independently created URLs — and it still returned a
persistent `403 Forbidden` in this account for reasons never conclusively
identified. API Gateway's invoke-permission model (`apigateway.amazonaws.com`
principal, `lambda:InvokeFunction` action) worked immediately once attached.
`handler.py` is unchanged either way — both integration types deliver the same
`{statusCode, headers, body}` shape.

Why: an always-on `g6e.xlarge` is roughly **$1,340/month**. Stopped-by-default
plus this gateway is roughly **$12/month fixed** plus a few dollars per active
hour — the difference between a demo that outlives a job search and one that
burns a $200 credit in under a week.

## Layout

| Path | What |
|---|---|
| `lambda/wake_gateway/handler.py` | Public front door: starts the instance, holds the visitor on an interstitial, redirects when `/health` answers |
| `lambda/idle_stop/handler.py` | Scheduled idle check; stops the instance, with guards against killing a warming box, an in-flight deploy, or a running self-hosted eval job |
| `iam/github-oidc-trust-policy.json` | Trust policy for `magik-deploy-role` (**read the comment — it explains the `environment:` subject-claim trap**) |
| `iam/lambda-*-permissions.json` | Least-privilege policies, scoped to the single instance ARN |
| `iam/ec2-instance-profile-permissions.json` | Least-privilege policy for the box's own instance profile — `ssm:GetParameter` on `/magik/ghcr_pat` plus the nine app secrets below, each scoped to its exact ARN. Attach alongside the AWS-managed `AmazonSSMManagedInstanceCore`. Not applied by any script here — attach it by hand (console or `aws iam put-role-policy`) to whatever role the instance profile uses |
| `caddy/Caddyfile` | HTTPS reverse proxy on the box; SSE-safe, long timeouts for model loading. Uses `APP_DOMAIN_PLACEHOLDER` as a template — the box's actual `/etc/caddy/Caddyfile` has `magik.vk-ai.online` substituted in directly |
| `prod.env` | Committed, non-secret deploy config (`GRAFANA_ROOT_URL`, `VRAM_BUDGET_GB`, `QDRANT_URL`, `REDIS_URL`, `GOOGLE_CLIENT_ID`, `OAUTH_REDIRECT_URI`, `FRONTEND_URL`, `SMTP_USER`, `CORS_ORIGINS`, `DEFAULT_DEV_USER_ID`, `EVAL_USER_ID`) — layered into the app container by `cd.yml`'s deploy step, and into the monitoring stack's own compose substitution by `scripts/deploy_monitoring.sh`. Editing this and pushing a tag (for the app) or re-running the monitoring script is the only correct way to change these values; hand-editing `/opt/magik/.env` for them makes this file drift out of sync with what's actually live |
| `scripts/deploy_lambdas.sh` | Idempotent one-shot deploy of both Lambdas, the API Gateway HTTP API, and the schedule |
| `scripts/deploy_monitoring.sh` | Idempotent bring-up of the monitoring stack (`docker compose -f docker-compose.monitoring.yml up -d`), fetching `GRAFANA_ADMIN_PASSWORD`/`NTFY_WEBHOOK_URL` from SSM first — see "App secrets in SSM" below. Run this instead of a bare `docker compose up`, or the two SSM-only secrets never reach the stack |

## Deploy

Run from **AWS CloudShell** (already authenticated — no local AWS CLI needed):

```bash
git clone https://github.com/vjkarthik98/multimodal-rag-assistant.git
cd multimodal-rag-assistant/deploy/aws/scripts

bash deploy_lambdas.sh   # APP_URL defaults to https://magik.vk-ai.online
```

Safe to re-run; it updates in place. The script prints the API Gateway
endpoint and a verification checklist when it finishes.

## Verify

```bash
# 1. Cold start
aws ec2 stop-instances --instance-ids i-02efa81c8876a014e
#    then open the API Gateway endpoint — expect the interstitial, then a redirect

# 2. Idle stop (watch for ~25 min after the box has been up 15+ min)
aws logs tail /aws/lambda/magik-idle-stop --follow
aws ec2 describe-instances --instance-ids i-02efa81c8876a014e \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```

`DRY_RUN=true` on the idle-stop Lambda logs its decision without stopping
anything — useful for confirming the thresholds before trusting it.

## Runner busy-check token (required before enabling Tier-2 eval)

Tier-2 eval (`eval-gate.yml`) runs `[self-hosted, gpu]` — on this same box.
Idle-stop's CloudWatch signal alone can't tell a running eval from an idle box
(a GPU-bound job produces almost no external network traffic), so before
setting the `SELF_HOSTED_GPU_RUNNER` repository variable to `true`, create the
token idle-stop uses to check whether the runner is actually busy:

1. GitHub → Settings → Developer settings → Fine-grained personal access
   token. Repository access: this repo only. Permissions: **Administration —
   Read-only** (that's the scope the "list self-hosted runners" endpoint
   requires — it cannot be narrowed further).
2. Store it in SSM (from CloudShell):
   ```bash
   aws ssm put-parameter --name /magik/github_actions_pat --type SecureString \
     --value "<paste the token>" --region us-east-1
   ```
3. Re-run `deploy_lambdas.sh` — it grants idle-stop's role read access to
   exactly this one parameter (`iam/lambda-idle-stop-permissions.json`) and
   nothing else in Parameter Store.
4. Only then set the `SELF_HOSTED_GPU_RUNNER` repository variable to `true`.
   Without step 2, idle-stop fails **safe** — it treats an unreadable token as
   "runner busy" and never stops the box, which is safe but defeats scale-to-
   zero, so don't skip it.

**Known limitation, not yet closed:** this guard can only see a runner GitHub
still considers connected and busy. If the runner process itself disconnects
mid-job (observed once, 2026-07-30 — root cause was a corrupted auth token
sent on every eval request, not resource exhaustion; see `CHANGELOG.md`),
GitHub reports it as simply offline rather than busy, and idle-stop's guard
has nothing left to object to. It stopped the box in that incident, but only
after the eval had already failed on its own — it did not kill a healthy run.
Tightening this further (e.g. a liveness heartbeat independent of GitHub's own
runner-status reporting) is a real improvement, not yet built.

## App secrets in SSM (Phase 31 — closes the secrets-management gap)

Nine values used to live as plaintext in `/opt/magik/.env` indefinitely:
`GOOGLE_CLIENT_SECRET`, `SMTP_PASSWORD`, `SECRET_KEY`, `JWT_SECRET_KEY`,
`MONGO_URI`, `QDRANT_API_KEY`, `REDIS_TOKEN`, `HF_TOKEN`, `TAVILY_API_KEY`
(the last four added 2026-08-01, after auditing `.env.example` against what
was actually still SSM-eligible). `cd.yml`'s deploy job now fetches all
nine from SSM Parameter Store on every deploy (same `--with-decryption`
pattern already used for `/magik/ghcr_pat`) and injects them via a
freshly-generated, `0600`, never-committed `--env-file` that is deleted
immediately after the container starts — the plaintext window on disk is
this one deploy's runtime, not indefinite.

**One-time setup** (from AWS CloudShell — do this before the next tagged
deploy, or `cd.yml`'s deploy job fails fast on the first missing parameter):

```bash
aws ssm put-parameter --name /magik/google_client_secret --type SecureString \
  --value "<the real value, currently in /opt/magik/.env>" --region us-east-1
aws ssm put-parameter --name /magik/smtp_password --type SecureString \
  --value "<the real value>" --region us-east-1
aws ssm put-parameter --name /magik/secret_key --type SecureString \
  --value "<the real value>" --region us-east-1
aws ssm put-parameter --name /magik/jwt_secret_key --type SecureString \
  --value "<the real value>" --region us-east-1
aws ssm put-parameter --name /magik/mongo_uri --type SecureString \
  --value "<the real value>" --region us-east-1
aws ssm put-parameter --name /magik/qdrant_api_key --type SecureString \
  --value "<the real value>" --region us-east-1
aws ssm put-parameter --name /magik/redis_token --type SecureString \
  --value "<the real value>" --region us-east-1
aws ssm put-parameter --name /magik/hf_token --type SecureString \
  --value "<the real value>" --region us-east-1
aws ssm put-parameter --name /magik/tavily_api_key --type SecureString \
  --value "<the real value>" --region us-east-1
```

Then attach the updated `iam/ec2-instance-profile-permissions.json` to the
EC2 instance profile's role (this is a one-time manual step — unlike the
two Lambda roles, nothing in `scripts/` manages the instance profile's
policy; if the role already has the original 5-parameter version attached,
re-applying this file replaces it with the 9-parameter version).

**After the first successful deploy under this scheme**, remove all nine
values from `/opt/magik/.env` on the box — they are no longer read from
there (the freshly-fetched `--env-file` is layered on top and wins on any
key collision, so leaving stale copies in `.env` is harmless but pointless,
and defeats the point of the migration if never cleaned up).

**Why fetch fresh on every deploy instead of once:** these are static,
long-lived secrets — the SSM parameter value does not change between
deploys, so `JWT_SECRET_KEY` in particular stays identical across releases
and existing sessions are not invalidated by a deploy. Only the *storage*
changed (SSM instead of a standing plaintext file), not the values or their
lifecycle.

**Deliberately not in this list:** `QDRANT_URL`/`REDIS_URL` (endpoint
hostnames, inert without their paired key/token above — not secrets) and
`GOOGLE_CLIENT_ID` (OAuth client IDs are public by design, only
`GOOGLE_CLIENT_SECRET` is confidential) all belong in the committed
`deploy/aws/prod.env` instead, not SSM. `GRAFANA_ADMIN_PASSWORD` and
`NTFY_WEBHOOK_URL` are real secrets too, but are consumed by the
separately-deployed monitoring stack, not this app container — see the next
section for how they're seeded and fetched.

## Monitoring-stack secrets in SSM

Two more values, consumed by `docker-compose.monitoring.yml`'s `grafana`
service specifically, not the app container — `scripts/deploy_monitoring.sh`
fetches them, `cd.yml` never touches them.

**One-time setup** (from AWS CloudShell, same account/region as above):

```bash
aws ssm put-parameter --name /magik/grafana_admin_password --type SecureString \
  --value "<a real, freshly-chosen password>" --region us-east-1
aws ssm put-parameter --name /magik/ntfy_webhook_url --type SecureString \
  --value "https://ntfy.sh/<a private, hard-to-guess topic name>" --region us-east-1
```

The updated `iam/ec2-instance-profile-permissions.json` (now with a second
statement, `ReadMonitoringSecretsAtDeployTime`) grants access to both — it's
the *same* instance-profile role as the app secrets above, since
`deploy_monitoring.sh` runs on this same box, so re-attaching the policy
once covers all nine app-secret ARNs plus these two.

**Bring the stack up** with the wrapper script, not a bare `docker compose
up` (which would leave `GRAFANA_ADMIN_PASSWORD` unset and fail Grafana's own
`:?` guard, and never deliver `NTFY_WEBHOOK_URL` to the container at all):

```bash
bash deploy/aws/scripts/deploy_monitoring.sh
```

**After the first successful run**, remove `GRAFANA_ADMIN_PASSWORD` and
`NTFY_WEBHOOK_URL` from `/opt/magik/.env` on the box, same reasoning as the
app secrets above — the freshly-fetched file is layered on top and wins on
any key collision, so a stale copy left in `.env` is harmless but pointless.

## Domain (done)

`magik.vk-ai.online` → A record → the Elastic IP, HTTPS via Caddy + Let's
Encrypt on the box. Completed 2026-07-30:

1. GoDaddy DNS: A record, name `magik`, value `3.208.159.124`.
2. Ports 80 (ACME challenge) and 443 opened in the security group.
3. Caddy installed on the box (`apt` via Cloudsmith's repo), config at
   `/etc/caddy/Caddyfile` — the site address is written as `https://
   magik.vk-ai.online { ... }` with the scheme explicit. **This matters**: a
   bare hostname with no scheme made Caddy log `"listening only on the HTTP
   port, so no automatic HTTPS will be applied"` and never even attempt the
   Let's Encrypt request — no error, just silently no port 443. Forcing
   `https://` in the site address is what actually triggers automatic HTTPS.
4. `OAUTH_REDIRECT_URI`, `FRONTEND_URL`, `CORS_ORIGINS` in `/opt/magik/.env`
   updated to the HTTPS domain; the new redirect URI added in Google Cloud
   Console (Google rejects bare-IP and non-HTTPS redirect URIs outright, so
   Google Sign-In could not work at all before this).
5. `deploy_lambdas.sh`'s `APP_URL` now defaults to `https://magik.vk-ai.online`
   — the wake gateway health-checks and redirects through Caddy, not the bare
   IP on :8000.

**Recreate the container after any `.env` change** — `docker restart` does
**not** re-read `--env-file`; only a fresh `docker run` does. This bit twice
tonight (OAuth redirect URI, then `DEV_OTP_LOG`) before being caught.

## Operational notes

- **Never** put the model cache on instance-store (`/opt/dlami/nvme`). It is
  wiped on every stop/start, and this design stops the box constantly — the
  ~20GB of weights would re-download on every single wake. They live on the EBS
  root volume via `/opt/magik/.hf_cache`.
- `/opt/magik/{.hf_cache,data,logs}` are **symlinks** into
  `/home/ubuntu/multimodal-rag-assistant/`, not real directories under `/opt`.
  Fragile: deleting that checkout (e.g. to reclaim disk) silently breaks
  production. Real directories directly under `/opt/magik` would be sturdier;
  not yet migrated.
- `CORS_ORIGINS` on the box's `/opt/magik/.env` includes the portfolio's
  Vercel origin, `vk-ai.online`, and `magik.vk-ai.online` — same-origin UI
  traffic doesn't need CORS at all, this only covers the portfolio calling the
  API directly. Update via SSM if it ever needs to change; `cd.yml` does not
  manage the contents of `.env`, only `docker run --env-file` reading whatever
  is already there.
- Idle-stop fails **safe**: if CloudWatch, SSM, or the GitHub runners API
  cannot be read, it treats the instance as busy and does nothing. A missed
  stop costs ~$0.15; a wrong stop kills a live session, a running deploy, or a
  running eval.
- `cd.yml`'s deploy job calls `StartInstances` itself, so a tagged release
  deploys correctly whether the box is running or stopped.
