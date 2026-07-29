# `deploy/aws` — Phase 30 scale-to-zero infrastructure

Everything needed to run MAGIK's public demo on a GPU box that is **stopped by
default** and wakes on demand.

```
Visitor ──▶ Lambda Function URL (always on, ~$0)
              │  stopped → StartInstances + "warming up" page (auto-refresh)
              │  booting → same page
              └▶ healthy → 302 ──▶ EC2 g6e.xlarge (L40S) ──▶ Caddy :443 ──▶ app :8000
                                        ▲                                      │
     EventBridge every 5 min ──▶ idle-stop Lambda ──▶ StopInstances ◀──────────┘
                                  (20m idle, 15m min uptime, skips during deploys)
```

Why: an always-on `g6e.xlarge` is roughly **$1,340/month**. Stopped-by-default
plus this gateway is roughly **$12/month fixed** plus a few dollars per active
hour — the difference between a demo that outlives a job search and one that
burns a $200 credit in under a week.

## Layout

| Path | What |
|---|---|
| `lambda/wake_gateway/handler.py` | Public front door: starts the instance, holds the visitor on an interstitial, redirects when `/health` answers |
| `lambda/idle_stop/handler.py` | Scheduled idle check; stops the instance, with guards against killing a warming box or an in-flight deploy |
| `iam/github-oidc-trust-policy.json` | Trust policy for `magik-deploy-role` (**read the comment — it explains the `environment:` subject-claim trap**) |
| `iam/lambda-*-permissions.json` | Least-privilege policies, scoped to the single instance ARN |
| `caddy/Caddyfile` | HTTPS reverse proxy on the box; SSE-safe, long timeouts for model loading |
| `scripts/deploy_lambdas.sh` | Idempotent one-shot deploy of both Lambdas, the Function URL, and the schedule |

## Deploy

Run from **AWS CloudShell** (already authenticated — no local AWS CLI needed):

```bash
git clone https://github.com/vjkarthik98/multimodal-rag-assistant.git
cd multimodal-rag-assistant/deploy/aws/scripts

# Pre-domain: point at the Elastic IP. Post-domain: the HTTPS subdomain.
APP_URL="http://3.208.159.124:8000" bash deploy_lambdas.sh
```

Safe to re-run; it updates in place. The script prints the Function URL and a
verification checklist when it finishes.

## Verify

```bash
# 1. Cold start
aws ec2 stop-instances --instance-ids i-02efa81c8876a014e
#    then open the Function URL — expect the interstitial, then a redirect

# 2. Idle stop (watch for ~25 min after the box has been up 15+ min)
aws logs tail /aws/lambda/magik-idle-stop --follow
aws ec2 describe-instances --instance-ids i-02efa81c8876a014e \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```

`DRY_RUN=true` on the idle-stop Lambda logs its decision without stopping
anything — useful for confirming the thresholds before trusting it.

## Attaching a custom domain

1. `magik.<domain>` → CNAME/ALIAS → the Lambda Function URL host (or a
   CloudFront distribution in front of it, if you want a cert on the apex).
2. `magik-app.<domain>` → A → the Elastic IP.
3. On the box: install Caddy, drop in `caddy/Caddyfile`, replace
   `APP_DOMAIN_PLACEHOLDER` with `magik-app.<domain>`, `systemctl enable --now caddy`.
   Ports **80 and 443** must be open — 80 is required for the ACME challenge.
4. Re-run `deploy_lambdas.sh` with `APP_URL=https://magik-app.<domain>`.
5. **Then remove port 8000 from public ingress** — Caddy reaches the container
   over localhost, so the app port no longer needs to face the internet.

## Operational notes

- **Never** put the model cache on instance-store (`/opt/dlami/nvme`). It is
  wiped on every stop/start, and this design stops the box constantly — the
  ~20GB of weights would re-download on every single wake. They live on the EBS
  root volume via `/opt/magik/.hf_cache`.
- The wake gateway health-checks `APP_URL` **from outside AWS**, so that
  endpoint must be publicly reachable. That is why port 8000 stays open until
  Caddy takes over on 443.
- Idle-stop fails **safe**: if CloudWatch or SSM cannot be read, it treats the
  instance as busy and does nothing. A missed stop costs ~$0.15; a wrong stop
  kills a live session or a running deploy.
- `cd.yml`'s deploy job calls `StartInstances` itself, so a tagged release
  deploys correctly whether the box is running or stopped.
