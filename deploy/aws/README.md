# `deploy/aws` — Phase 30 scale-to-zero infrastructure

Everything needed to run MAGIK's public demo on a GPU box that is **stopped by
default** and wakes on demand.

```
Visitor ──▶ API Gateway HTTP API ──▶ wake-gateway Lambda (always on, ~$0)
              │  stopped → StartInstances + "warming up" page (auto-refresh)
              │  booting → same page
              └▶ healthy → 302 ──▶ EC2 g6e.xlarge (L40S) ──▶ app :8000 (plain HTTP)
                                        ▲                                      │
     EventBridge every 5 min ──▶ idle-stop Lambda ──▶ StopInstances ◀──────────┘
                                  (20m idle, 15m min uptime, skips during deploys)
```

There is no custom domain for AWS, and none is needed: the public link is a
portfolio site (hosted elsewhere, e.g. Vercel), which links directly to the
wake gateway's own HTTPS API Gateway endpoint — that URL is already a fully
valid public `https://` address on its own. The app itself is reached over
plain HTTP on `:8000` once woken.

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
| `scripts/deploy_lambdas.sh` | Idempotent one-shot deploy of both Lambdas, the API Gateway HTTP API, and the schedule |

## Deploy

Run from **AWS CloudShell** (already authenticated — no local AWS CLI needed):

```bash
git clone https://github.com/vjkarthik98/multimodal-rag-assistant.git
cd multimodal-rag-assistant/deploy/aws/scripts

APP_URL="http://3.208.159.124:8000" bash deploy_lambdas.sh
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

## Operational notes

- **Never** put the model cache on instance-store (`/opt/dlami/nvme`). It is
  wiped on every stop/start, and this design stops the box constantly — the
  ~20GB of weights would re-download on every single wake. They live on the EBS
  root volume via `/opt/magik/.hf_cache`.
- The wake gateway health-checks `APP_URL` **from outside AWS**, so port 8000
  must stay open in the security group for that check to succeed.
- `CORS_ORIGINS` on the box's `/opt/magik/.env` should include the portfolio
  origin alongside the box's own address — same-origin UI traffic doesn't need
  CORS at all, this only covers the portfolio calling the API directly. Update
  via SSM if it ever needs to change; `cd.yml` does not manage the contents of
  `.env`, only `docker run --env-file` reading whatever is already there.
- Idle-stop fails **safe**: if CloudWatch, SSM, or the GitHub runners API
  cannot be read, it treats the instance as busy and does nothing. A missed
  stop costs ~$0.15; a wrong stop kills a live session, a running deploy, or a
  running eval.
- `cd.yml`'s deploy job calls `StartInstances` itself, so a tagged release
  deploys correctly whether the box is running or stopped.
