# Uptime Kuma — Passive, Push-Based Uptime Monitoring

**Not deployed yet.** This directory is the ready-to-run config; provisioning
the actual host is a deliberate, separate action — see the approved plan's
checkpoint on new AWS infrastructure. Nothing here runs automatically.

## Why a separate host

The GPU box (`i-02efa81c8876a014e`) sleeps by design. A status page hosted
*on* it would go dark exactly when it's most useful to check — and any
monitor that *polls* the app to check it's up would itself trigger a wake via
`deploy/aws/lambda/wake_gateway/handler.py`, which defeats scale-to-zero
entirely. So:

- Uptime Kuma runs on a small, **separate, always-on** host (a
  `t4g.micro`/`t3.micro` or whatever's free-tier eligible — roughly $3-8/mo,
  a fixed cost independent of the GPU box's $12-vs-$1,340/mo tradeoff).
- It never polls the app. It only **receives pushes** — from
  `wake_gateway/handler.py` (when a real visitor's request confirms the app
  is genuinely healthy) and `idle_stop/handler.py` (which already runs every
  5 minutes regardless of monitoring, and now also reports "up, latency Xms"
  while the instance is running and "down" the moment it actually stops it).
  See both handlers' docstrings for the full reasoning.

## One-time provisioning

1. **Launch the small host** (outside the scope of this repo's automation —
   a manual AWS Console / CLI action, deliberately not scripted here so it's
   never accidentally re-run). Open port 443 (and 80 for the ACME challenge).
2. **DNS**: point a new subdomain (e.g. `status.vk-ai.online`) at the host's
   IP. This must be a *different* subdomain from `magik.vk-ai.online` — that
   one lives on the GPU box.
3. **Edit `Caddyfile`**: replace `STATUS_DOMAIN_PLACEHOLDER` with the real
   subdomain, and `KUMA_BASICAUTH_HASH_PLACEHOLDER` with
   `docker run --rm caddy:2 caddy hash-password --plaintext '<your password>'`.
4. **Bring it up**:
   ```bash
   docker compose up -d
   ```
5. **First-run Kuma setup** (via the basic-auth-gated dashboard at
   `https://status.vk-ai.online/`): create the admin account, then add a
   **Push** monitor (Monitor Type: Push). Kuma generates a unique heartbeat
   URL — copy it.
6. **Wire it into the Lambdas** — re-run the deploy script with the push URL:
   ```bash
   cd ../../deploy/aws/scripts
   KUMA_PUSH_URL="https://status.vk-ai.online/api/push/<token>" bash deploy_lambdas.sh
   ```
   Both `KUMA_PUSH_URL` env vars (wake_gateway and idle_stop) get set in the
   same pass — see `deploy_lambdas.sh`'s `KUMA_PUSH_URL` variable.
7. **Enable the public status page** in Kuma (Settings → Status Pages →
   create one, add the push monitor to it). Label it honestly — something
   like *"MAGIK sleeps when idle to save cost — gaps below are the intended
   wake-on-demand design, not incidents."* This page (`/status/<slug>`, left
   open in the Caddyfile — no basic_auth) is the link for the README badge
   and the portfolio site.

## Verifying it stayed passive

After step 6, watch the GPU box's real start/stop pattern for 24-48h
(CloudWatch, or `idle_stop`'s own CloudWatch Logs) to confirm the new push
hooks added **zero** extra wake events — the box should still only wake on
real visitor traffic and still stop after the same idle window as before.
This is the plan's own verification requirement for this phase; don't skip it.
