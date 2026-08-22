# `deploy/aws` — Phase 30 scale-to-zero infrastructure

> **REBUILD NOTE (2026-08-21):** the entire prior EC2 fleet + EBS volumes were
> deleted and rebuilt in a **new AWS account (857194222592, was
> 537557168406)** via `deploy/aws/terraform/` (see that directory's own README
> section in `docs/runbooks/phase-30-aws-deployment.md` Appendix E). Every
> instance ID / Elastic IP example below is now current as of the rebuild
> (production: `i-09831ac06b063d36f`, EIP `184.73.239.9`) — but if this ever
> gets rebuilt again, re-run `terraform output` and update these examples
> again rather than trusting them blindly. The architecture, Lambda behavior,
> and DNS/monitoring design described here are otherwise unchanged.

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
| `prod.env` | Committed, non-secret deploy config (`GRAFANA_ROOT_URL`, `VRAM_BUDGET_GB`, `QDRANT_URL`, `REDIS_URL`, `GOOGLE_CLIENT_ID`, `OAUTH_REDIRECT_URI`, `FRONTEND_URL`, `SMTP_USER`, `CORS_ORIGINS`, `DEFAULT_DEV_USER_ID`, `EVAL_USER_ID`) — layered into the app container by `cd.yml`'s `promote-production` step, and into the monitoring stack's own compose substitution by `scripts/deploy_monitoring.sh`. Editing this and pushing a tag (for the app) or re-running the monitoring script is the only correct way to change these values; hand-editing `/opt/magik/.env` for them makes this file drift out of sync with what's actually live |
| `staging.env` | Same idea as `prod.env`, but for the private staging box `cd.yml`'s `deploy-staging` step targets — no `GRAFANA_ROOT_URL`/`CORS_ORIGINS`/`FRONTEND_URL` (staging has no monitoring stack and is never reachable from outside the box), same `QDRANT_URL`/`REDIS_URL`/`VRAM_BUDGET_GB`/`EVAL_USER_ID` as prod (staging shares prod's external services, isolated by `EVAL_USER_ID` tenant scoping). See "Staging gate" below |
| `scripts/deploy_lambdas.sh` | Idempotent one-shot deploy of both Lambdas, the API Gateway HTTP API, and the schedule |
| `scripts/deploy_monitoring.sh` | Idempotent bring-up of the monitoring stack (`docker compose -f docker-compose.monitoring.yml up -d`), fetching `GRAFANA_ADMIN_PASSWORD`/`NTFY_WEBHOOK_URL` from SSM first — see "Monitoring-stack secrets in SSM" below. Run this instead of a bare `docker compose up`, or the two SSM-only secrets never reach the stack |

## Deploy

Run from **AWS CloudShell** (already authenticated — no local AWS CLI needed):

```bash
git clone https://github.com/vjkarthik98/MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT.git
cd MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT/deploy/aws/scripts

bash deploy_lambdas.sh   # APP_URL defaults to https://magik.vk-ai.online
```

Safe to re-run; it updates in place. The script prints the API Gateway
endpoint and a verification checklist when it finishes.

## Custom domain for the wake gateway (`launch.vk-ai.online`)

The raw API Gateway endpoint (`https://<api-id>.execute-api.us-east-1.amazonaws.com`)
works fine on its own, but a `vk-ai.online`-branded link is what's actually
pasted into the portfolio site. This is a manual, one-time setup per rebuild
(not covered by `deploy_lambdas.sh` or Terraform — same category as the
`magik.vk-ai.online` A record below: AWS-side pieces are scriptable, the
GoDaddy DNS side isn't).

**Why `launch.` and not `magik.` itself:** `magik.vk-ai.online` is the FINAL
destination (`APP_URL`) the gateway redirects to once the app is healthy —
pointing the gateway at that same hostname would have it redirect to itself.
The gateway needs its own, different hostname.

1. **Request an ACM certificate** for the subdomain (must be in the SAME
   region as the API Gateway — `us-east-1` here):
   ```bash
   aws acm request-certificate --region us-east-1 \
     --domain-name launch.vk-ai.online --validation-method DNS
   ```
2. **Add the DNS validation CNAME** ACM gives you
   (`aws acm describe-certificate ... DomainValidationOptions[0].ResourceRecord`)
   in GoDaddy. **Keep this record permanently** — the certificate silently
   auto-renews using it every ~13 months; deleting it breaks renewal later,
   not immediately.
3. **Wait for validation**: `aws acm wait certificate-validated --certificate-arn <arn>`.
4. **Create the API Gateway custom domain**, pointing at the now-validated cert:
   ```bash
   aws apigatewayv2 create-domain-name --region us-east-1 \
     --domain-name launch.vk-ai.online \
     --domain-name-configurations CertificateArn=<cert-arn>,EndpointType=REGIONAL
   ```
   This returns a target hostname (`ApiGatewayDomainName`, e.g.
   `d-xxxxxxxxxx.execute-api.us-east-1.amazonaws.com`) — different from the
   certificate validation target in step 2.
5. **Map the wake-gateway API to it**:
   ```bash
   aws apigatewayv2 create-api-mapping --region us-east-1 \
     --domain-name launch.vk-ai.online \
     --api-id <the wake-gateway API's ApiId> --stage '$default'
   ```
6. **Add the routing CNAME** in GoDaddy: `launch` → the `ApiGatewayDomainName`
   from step 4. This is the record that actually makes the domain resolve
   anywhere — separate from, and in addition to, the validation CNAME from
   step 2.

## Live status page (redesigned 2026-08-21)

The wake gateway no longer serves a bare `<meta http-equiv="refresh">` page
that flashes and re-renders identically on every reload. The first hit
renders a full page with an inlined JS/CSS shell; every update after that is
driven by that page's own `fetch('?check=1')` poll against this same Lambda,
which returns a small JSON status object instead of HTML — the DOM updates
in place (3-step progress indicator, message text) with no page flash, and a
client-side `location.replace()` fires the moment status is `"ready"`. See
`lambda/wake_gateway/handler.py`'s own header comment and `_compute_status()`
for the full state machine (`waking` / `loading` / `stuck` / `capacity` /
`error` / `ready`) — `capacity` (AWS's `InsufficientInstanceCapacity`) is a
distinct, clearly-worded state now instead of a generic failure, which
matters more than it might look: this genuinely happens for `g6e.xlarge` in
`us-east-1a` from time to time, confirmed live during this rebuild.

## Verify

```bash
# 1. Cold start
aws ec2 stop-instances --instance-ids i-09831ac06b063d36f
#    then open the API Gateway endpoint — expect the interstitial, then a redirect

# 2. Idle stop (watch for ~25 min after the box has been up 15+ min)
aws logs tail /aws/lambda/magik-idle-stop --follow
aws ec2 describe-instances --instance-ids i-09831ac06b063d36f \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```

`DRY_RUN=true` on the idle-stop Lambda logs its decision without stopping
anything — useful for confirming the thresholds before trusting it.

**If a visitor reports the wake page never redirects even after several
minutes:** the instance is reaching `running` but `/health` is never
answering (past `STUCK_MINUTES`, default 6, the page itself now says so
explicitly instead of repeating the generic "starting up" copy forever — see
`deploy/aws/lambda/wake_gateway/handler.py`). That page change only makes the
symptom visible; it can't fix an app that won't come up. Diagnose from here:

```bash
# What the gateway itself saw on each check (state, elapsed time, stuck warnings)
aws logs tail /aws/lambda/magik-wake-gateway --follow

# What the app itself is doing once EC2 is up — SSM into the box, no SSH key needed
aws ssm start-session --target i-09831ac06b063d36f
sudo journalctl -u magik -n 200 --no-pager   # or: docker compose logs --tail 200, per how it's actually run
curl -s localhost:8000/health               # bypasses Caddy — isolates "app is fine" vs "Caddy/TLS is the problem"
curl -sI https://magik.vk-ai.online/health  # the exact request the Lambda makes, from the box itself
```

`app/main.py`'s `/health` is a trivial, synchronous, no-dependency handler
(returns `{"status": "ok", ...}` immediately, doesn't touch Qdrant/Redis/
Mongo) — so if `curl localhost:8000/health` on the box succeeds but the
Lambda still can't reach `https://magik.vk-ai.online/health`, the app itself
is fine and the problem is in front of it: most likely Caddy failed to start
or its Let's Encrypt certificate renewal failed (the wake gateway's health
check is a real HTTPS request from outside AWS — an expired/invalid cert
fails it exactly like a genuinely dead app would). If `localhost:8000/health`
itself doesn't answer, look for a crash on startup — a bad deploy, or
`app/core/startup_validator.py` raising because the model manifest is
incomplete.

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

## Verifying / restoring SSM secrets

`deploy/aws/scripts/restore_ssm_secrets.py` is the one authoritative manifest
of every `/magik/*` SSM parameter this project depends on (all 13 — the 9
app secrets and 4 more below, in one place). Use it instead of hand-typing
`aws ssm put-parameter` commands, which is exactly what the two sections
below still show as historical reference:

```bash
# Check what's actually present right now, write nothing:
pip install boto3   # not a project dependency — this script's only need
python deploy/aws/scripts/restore_ssm_secrets.py --check-only --profile magik-admin

# Restore the 10 app-level secrets from a live, healthy app container
# (reads its own already-loaded environment — no value ever hand-typed):
docker exec magik-current env | grep -q JWT_SECRET_KEY  # sanity check first
python deploy/aws/scripts/restore_ssm_secrets.py --from-env --group app --profile magik-admin

# True disaster recovery — no running container at all, from an offline
# backup file (KEY=VALUE lines, kept in a password manager, never committed):
python deploy/aws/scripts/restore_ssm_secrets.py --from-file secrets.env --group all --profile magik-admin
```

Uses boto3 directly, deliberately not the AWS CLI — no shell in the path
means no Git-Bash/MSYS2 argument mangling (the exact bug that corrupted the
very first attempt at these `put-parameter` commands on Windows, turning
`/magik/jwt_secret_key` into `C:/Program Files/Git/magik/jwt_secret_key`,
caught and redone at the time — see the script's own docstring).

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
fetches them; `cd.yml`'s app-container deploy step never touches them (its
separate monitoring-sync step below runs the same script, so it does).

**One-time setup** (from AWS CloudShell, same account/region as above):

```bash
aws ssm put-parameter --name /magik/grafana_admin_password --type SecureString \
  --value "<a real, freshly-chosen password>" --region us-east-1
aws ssm put-parameter --name /magik/ntfy_webhook_url --type SecureString \
  --value "https://ntfy.sh/<a private, hard-to-guess topic name>" --region us-east-1
```

**A dedicated on-call escalation contact point (PagerDuty, then Opsgenie)
was tried and removed** — every free real phone/SMS escalation path hit a
hard external blocker not fixable from this codebase: PagerDuty's trial
signup rejects consumer email domains (a GoDaddy-forwarded business-style
address hit the same wall), Opsgenie stopped accepting new customers as of
2025-06-04, and Better Stack's free tier is email/Slack only — real
escalation there needs a paid seat. `NTFY_WEBHOOK_URL` (push notification,
no retry/escalation) is the only alert-delivery mechanism again. See
`monitoring/alerts/contact-points.yml`'s comment and `monitoring/slo.md`'s
Security signal SLO section for the current state.

The `iam/ec2-instance-profile-permissions.json` (`ReadMonitoring
SecretsAtDeployTime` statement) grants access to both — it's the
*same* instance-profile role as the app secrets above, since
`deploy_monitoring.sh` runs on this same box, so re-attaching the policy
once covers all nine app-secret ARNs plus these two.

**Automated as of the `promote-production` job's "Sync + redeploy monitoring
stack" step**: every tag that promotes to production also brings the box's
persistent repo checkout at `/home/ubuntu/MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT` (see
"Operational notes" below) to that exact tag via `git fetch` + `git reset
--hard`, then re-runs `deploy_monitoring.sh` over SSM. This step is
`continue-on-error: true` and gated on the app deploy having already
succeeded — a monitoring-stack hiccup can never fail or roll back the app
promotion, and gets its own separate Slack failure notification so it isn't
mistaken for a production-app incident. Config changes under `monitoring/`
or to `docker-compose.monitoring.yml` now reach production automatically on
the next tag push; no manual step needed for that case.

**Manually re-running the wrapper script is now only needed for two cases**:
one-off/out-of-band changes that don't come with a new tag (e.g. rotating
`GRAFANA_ADMIN_PASSWORD`/`NTFY_WEBHOOK_URL` in SSM on their own), or
recovering a box where the automated step failed:

```bash
bash deploy/aws/scripts/deploy_monitoring.sh
```

(Not a bare `docker compose up` — that would leave `GRAFANA_ADMIN_PASSWORD`
unset and fail Grafana's own `:?` guard, and never deliver
`NTFY_WEBHOOK_URL` to the container at all.)

**After the first successful run**, remove `GRAFANA_ADMIN_PASSWORD` and
`NTFY_WEBHOOK_URL` from `/opt/magik/.env` on the box, same reasoning as the
app secrets above — the freshly-fetched file is layered on top and wins on
any key collision, so a stale copy left in `.env` is harmless but pointless.

## Staging gate (private box, no HTTP, no Caddy)

A tag no longer deploys straight to production. `cd.yml` deploys first to a
**second** EC2 box (`magik-staging`), runs the full Tier-2 RAG-quality suite
against it, and only promotes the exact same image to production if that
passes — see `docs/runbooks/ci-cd.md`'s "Staging gate" section for the full
design, prerequisite list, and manual validation steps.

This box is deliberately **not** wired into the wake-gateway/Caddy/domain
setup above — that's the public-demo front door for production traffic,
which staging never receives. It:

- has **no security-group ingress on 80/443/8000** (SSM is the only way in,
  exactly like management access to the prod box today),
- runs the app container bound to `127.0.0.1:8000` only, never `0.0.0.0`,
- lives under `/opt/magik-staging/{.env,.hf_cache,data,logs}` — parallel to,
  and never colliding with, prod's `/opt/magik/...` — with its own
  `magik-staging-current`/`magik-staging-previous` container pair,
- reuses `iam/ec2-instance-profile-permissions.json` unchanged (attach the
  same policy to its own instance profile — nothing in that file is
  prod-specific) and shares prod's SSM app secrets and external services
  (Qdrant/Redis/Mongo), isolated by `EVAL_USER_ID` tenant scoping,
- needs its own self-hosted GitHub Actions runner registered on it, labelled
  `staging-gpu` (see `SELF_HOSTED_STAGING_GPU_RUNNER` repo variable).

## Domain (done)

`magik.vk-ai.online` → A record → the Elastic IP, HTTPS via Caddy + Let's
Encrypt on the box. Completed 2026-07-30:

1. GoDaddy DNS: A record, name `magik`, value `184.73.239.9`.
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
  `/home/ubuntu/MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT/`, not real directories under `/opt`.
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
