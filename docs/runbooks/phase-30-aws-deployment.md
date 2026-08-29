# Phase 30 — AWS Deployment: End-to-End Implementation Plan

> Owner: you (solo). Reviewer persona: senior GenAI/DevOps engineer.
> Target: a public `https://` demo of MAGIK that HR / hiring managers can click, that
> costs near-nothing while idle, and that redeploys with a single `git push --tags`.

**Target hardware:** AWS EC2 **`g6e.xlarge`** — 1× NVIDIA **L40S**, 48 GB VRAM, 32 GB RAM, 4 vCPU, `us-east-1`.
**Budget:** $200 credit, open-ended runway. The entire architecture optimizes for *GPU-off-by-default,
wake-on-visit*, not raw throughput.

This document is the single source of truth for Phase 30. It supersedes:
- the prior single-GPU sizing assumptions in `Dockerfile`, `config.py`, `device_manager.py`, `install_cuda.sh`, `.claude/skills/devops/SKILL.md` (corrected in Stage 1 — now all g6e.xlarge / L40S);
- the "Phase 30 handoff" 7-item table in `docs/runbooks/ci-cd.md` (completed here).

Everything upstream of AWS (CI, Tier-1 eval gate, Dockerfile multi-target build, `cd.yml`) is
**already built and working** — Phase 30 is purely the AWS substrate plus the repo corrections needed
to land on this specific GPU.

---

## 0. How to read this plan

- **Two lanes per stage**: 💻 *VSCode / repo / GitHub* work vs ☁️ *AWS* work, exactly as requested.
  They are **not** parallel tracks — the stage order **is** the dependency order.
- **Every automation is preceded by the manual version of the same action.** You will `docker run` by
  hand (Stage 3) before `cd.yml` does it (Stage 4); you will stop/start the box by hand before the
  Lambda does it (Stage 6). A later failure then means "the automation is wrong," never "was this ever
  possible?"
- **Appendices A–G hold the actual artifacts** (Lambda code, Caddyfile, IAM JSON, user-data, Terraform
  skeleton, env inventory, validation matrix). The stages reference them; you copy-paste from there.
- Estimated effort: **2–4 focused days** for a first-timer, most of it in Stages 2, 4, and 6.

---

## 1. Target architecture (what "done" looks like)

```
                        ┌───────────────────────────────────────────────┐
   Recruiter / HR ─────▶│  https://yourdomain.com   (portfolio link)     │
                        └───────────────────────┬───────────────────────┘
                                                │  DNS (Route53 / registrar)
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │  CloudFront + ACM cert   (always on, ~$0)      │
                        └───────────────────────┬───────────────────────┘
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │  Lambda: magik-wake-gateway   (always on, ~$0) │
                        │   • DescribeInstances(magik-prod)              │
                        │   • stopped → StartInstances + "waking…" page  │
                        │   • running+healthy → 302 → app.yourdomain.com │
                        └───────────────────────┬───────────────────────┘
                                                │ (only once the box is healthy)
                                                ▼
   app.yourdomain.com ─── DNS A/ALIAS ──▶ Elastic IP ──▶ ┌──────────────────────────────┐
                                                          │  EC2 g6e.xlarge (L40S 48GB)  │
                                                          │  ┌────────────────────────┐  │
                                                          │  │ Caddy :443 (auto-HTTPS)│  │
                                                          │  │   └▶ reverse_proxy     │  │
                                                          │  │        localhost:8000  │  │
                                                          │  └──────────┬─────────────┘  │
                                                          │  ┌──────────▼─────────────┐  │
                                                          │  │ Docker: magik-current  │  │
                                                          │  │  FastAPI + llama-server│  │
                                                          │  │  + React UI (built)    │  │
                                                          │  └────────────────────────┘  │
                                                          │  EBS root 100GB gp3:         │
                                                          │   /opt/magik/{.hf_cache,     │
                                                          │     data,logs,.env}          │
                                                          └──────────────┬───────────────┘
                                                                         │ TLS out
                                          ┌──────────────────────────────┼───────────────────────┐
                                          ▼                              ▼                       ▼
                                    Qdrant Cloud                   Upstash Redis           MongoDB Atlas
                                    (vectors)                      (short-term mem)        (long-term mem)

   Idle path:  CloudWatch alarm (NetworkIn low ~15–20 min) → SNS → Lambda: magik-idle-stop → StopInstances
   Deploy path: git push --tags → GitHub Actions cd.yml → build→GHCR→OIDC→SSM→docker pull/swap→health
```

**Design decisions worth stating (ADRs):**
- **ADR-30-1: Scale-to-zero over always-on.** GPU compute is the entire cost. Idle-stop + wake-on-visit
  turns a ~$1,340/mo box into a ~$12/mo box plus a few dollars per active hour. This is *the* reason the
  plan exists.
- **ADR-30-2: Lambda wake gateway, not an ALB.** An ALB is ~$16–20/mo fixed, 24/7, for zero-traffic
  periods that dominate a job search. A Lambda Function URL / CloudFront-fronted Lambda is free-tier for
  this volume. Cost dominates; the slight extra complexity is worth it.
- **ADR-30-3: Two subdomains (`yourdomain.com` gateway + `app.yourdomain.com` app), not one seamless
  edge.** A single-domain seamless flow needs Lambda@Edge (region-locked, no env vars, 5s cap). The
  two-hop version uses an ordinary Lambda + Caddy — simpler to write, test, debug. Consolidate later as
  polish, not a blocker.
- **ADR-30-4: SSM, not SSH, for deploys.** `cd.yml` deploys via `ssm:SendCommand`. No inbound SSH from
  GitHub, no SSH key in CI. SSH stays locked to your IP for break-glass admin only.
- **ADR-30-5: OIDC, not static AWS keys.** GitHub Actions assumes a scoped IAM role via OIDC. No
  long-lived `AWS_ACCESS_KEY_ID` in repo secrets, ever.

---

## 2. Cost model (real numbers, verify current rates before committing)

On-demand `g6e.xlarge` in `us-east-1` is roughly **$1.86/hr** at time of writing — **confirm in the AWS
Pricing Calculator**, rates move.

**Fixed monthly (accrues whether the box is on or off):**

| Item | Qty | ~Monthly |
|---|---|---|
| EBS gp3 root volume | 100 GB | ~$8.00 |
| Elastic IP (public IPv4, charged even when attached) | 1 | ~$3.60 |
| Route53 hosted zone (only if using a custom domain in Route53) | 1 | ~$0.50 |
| CloudFront + Lambda + EventBridge + CloudWatch + SNS | — | ~$0 (free tier at this volume) |
| **Fixed baseline** | | **~$12/mo** |

**Variable (GPU compute, only while awake):**

| Usage scenario | Awake hrs/mo | GPU cost/mo | Total/mo | $200 runway |
|---|---|---|---|---|
| Light recruiter traffic (a few visits) | ~3 | ~$5.60 | ~$18 | **~11 months** |
| Moderate (weekly demos + your own testing) | ~10 | ~$18.60 | ~$31 | **~6.5 months** |
| Heavy / you forget idle-stop | ~40 | ~$74 | ~$86 | **~2.3 months** |
| **Always-on (no scale-to-zero)** | 730 | ~$1,358 | ~$1,358 | **< 4 days** |

The bottom row is why Stage 6 is non-negotiable. A **Budget alarm (Stage 0)** is the backstop for the
"you forget" row.

---

## 3. Pre-flight — before Stage 0

Confirm each of these is true; each has bitten someone:

- [ ] You can log into the AWS console with an **IAM user or Identity Center user**, not the root account.
- [ ] Your G/VT vCPU quota shows **Applied: 4** (Service Quotas → EC2 → "Running On-Demand G and VT instances"), not merely "requested."
- [ ] You have a GitHub repo with `cd.yml`, `ci.yml`, `eval-gate.yml` present (you do).
- [ ] Your managed data services (Qdrant Cloud, Upstash Redis, MongoDB Atlas) are provisioned and you have their connection strings — the box connects **out** to these; they are not part of AWS provisioning.
- [ ] You have a GitHub PAT with `read:packages` scope (for the box to pull from GHCR) and one with `repo` scope is NOT needed here.
- [ ] `docker build --target runtime .` works on your laptop today (build-time CUDA compile needs no GPU).

---

## Stage 0 — Cost & account guardrails

**Objective:** make a surprise bill structurally impossible before any billable resource exists.

**☁️ AWS**
1. **Budgets** (Billing console): create a cost budget on the credit; alert emails at **25 / 50 / 75 / 90%**. Confirm the email.
2. **Cost Anomaly Detection**: enable (free) — catches a runaway resource in hours, not at month-end.
3. **IAM/Identity Center**: daily-driver principal with only what's needed (EC2, IAM, SSM, Route53, CloudFront, ACM, Lambda, EventBridge, CloudWatch, SNS, Budgets-read). Root stays for account-level only.
4. Re-verify the G/VT quota is **Applied: 4**.

**💻 VSCode**
- Install AWS CLI v2 locally; `aws configure` with the IAM user's keys (region `us-east-1`, output `json`). Verify: `aws sts get-caller-identity`. (Console suffices for everything, but you'll want the CLI for Stage 4/6 checks.)

**Gate:** budget-alarm confirmation email received and clicked.

---

## Stage 1 — Repo corrections for g6e.xlarge (💻 pure local, no AWS)

**Code-side corrections: ✅ DONE.** The image and comments are already on L40S / sm_89 —
`Dockerfile` (`-DCMAKE_CUDA_ARCHITECTURES=89`), `requirements.txt`, `install_cuda.sh`, and all
`config.py` / `device_manager.py` / `model_loader.py` / `ingestion_pipeline.py` / CLAUDE.md
references now describe g6e.xlarge / L40S 48GB only. Nothing pending in the repo.

**Remaining Stage-1 work is the box's `.env`** (per-hardware / per-deployment values that live on the
instance, never in `config.py` — you apply these in Stage 3):

| # | box `.env` key | Set to | Why |
|---|---|---|---|
| 1 | `VRAM_BUDGET_GB` | `44` | the nominal 48GB L40S shows as ~44.4GB actually visible to CUDA once the OS/driver take their share (confirmed live via device_manager's startup log — `46` silently never binds, since device_manager.py does `min(actual_free_vram, VRAM_BUDGET_GB)` and 46 exceeds the real total); the code default is conservative and merely *caps* usage, so the box works but uses less than half the card until this is raised |
| 2 | `MAX_CONCURRENT_GPU_JOBS` | leave `3` for first deploy | conservative default; raise **only** after watching real VRAM headroom with `nvidia-smi` — don't tune blind |
| 3 | `CORS_ORIGINS` | `https://yourdomain.com,https://app.yourdomain.com` | wildcard CORS default on a public, auth-gated app is a hardening gap; lock to your domains before go-live |
| 4 | `QWEN2_VL_LOAD_IN_8BIT` | decide on real hardware: leave `False` (fp16) **or** set `=true` (INT8) | fp16 is higher quality (no quantisation loss) and fits comfortably in 48 GB; INT8 (~9.5GB) frees VRAM if you ever hit pressure under peak concurrent ingestion. Confirm headroom with `nvidia-smi` during a real image-captioning request before finalizing. |

**Provisioning note (not an `.env` value):** the model cache dir `/opt/magik/.hf_cache` (mounted by
`cd.yml`'s `docker run`) must live on the **EBS root volume**, never on instance-store scratch — or
every stop/start (your whole cost strategy) re-downloads 17.7 GB+ from HF. Enforced in Stage 3 step 2.

**Gate:** `docker build --target runtime -t magik:latest .` succeeds locally (the sm_89 CUDA compile
runs on a CPU-only laptop — no physical GPU needed at build time).

---

## Stage 2 — Provision the EC2 instance (☁️ AWS)

> **SUPERSEDED 2026-08-21:** Stage 2 (and the "Appendix E — Terraform skeleton"
> it used to point to as optional/future work) is now **implemented and
> current**, not a skeleton — see `deploy/aws/terraform/`. This happened
> because the entire prior EC2 fleet + EBS volumes were manually deleted,
> which is exactly the failure mode "codify after you've launched by hand
> once" was meant to protect against. Run `terraform init && terraform plan`
> in that directory instead of hand-clicking through the steps below; this
> section is kept as narrative/historical context for what Terraform is
> actually doing under the hood, not as a runbook to follow by hand anymore.
> Two things NOT in Terraform's scope, still exactly as described below: the
> Lambda/API-Gateway wake-gateway + idle-stop stack (`deploy_lambdas.sh`,
> unaffected by an AWS-account change except for the two hardcoded instance
> ARNs in `deploy/aws/iam/lambda-*-permissions.json`), and DNS (still a manual
> GoDaddy A-record edit — see Stage 6a).

**Objective:** a running, reachable, correctly-sized box with **persistent** storage and SSM reachability.

**☁️ AWS — EC2 → Launch instance**
1. **AMI:** *AWS Deep Learning AMI (GPU, PyTorch)*, Ubuntu — ships NVIDIA driver + CUDA + Docker + nvidia-container-toolkit, saving you a manual `install_cuda.sh` pass. (Confirm the toolkit in Stage 3.)
2. **Type:** `g6e.xlarge`.
3. **Key pair:** create new, download the `.pem`, store it where you'll find it (break-glass SSH only).
4. **Storage:** root **EBS 100 GB, gp3** (bump from the AMI default; 17.7 GB models + Docker layers + OS + headroom). Confirm **gp3**, not gp2.
5. **IAM instance profile:** create/attach one with **`AmazonSSMManagedInstanceCore`** now — this is what lets `cd.yml`'s SSM deploy reach the box and lets you avoid opening SSH to the world.
6. **Tag:** `Name = magik-prod` (this exact string — `cd.yml` looks it up by tag; see `docs/runbooks/ci-cd.md`).
7. **Security group:** least privilege (see below). `0.0.0.0/0` on 8000 is acceptable **only** for Stage 3 manual testing; lock it down before go-live.
8. **User-data (optional but recommended):** paste **Appendix D** to auto-create `/opt/magik/{.hf_cache,data,logs}` on first boot and install Caddy. Skippable if you prefer to do it by hand in Stage 3.

**Security group (final state — reach this by Stage 8):**

| Port | Source | Purpose |
|---|---|---|
| 22 | `YOUR.IP.ADDR.ESS/32` | break-glass SSH only |
| 80 | `0.0.0.0/0` | Caddy ACME HTTP-01 challenge + redirect to 443 |
| 443 | `0.0.0.0/0` | Caddy HTTPS for `app.yourdomain.com` |
| 8000 | `127.0.0.1` (via Caddy only) or CloudFront prefix list | app; not directly public in final state |

**☁️ AWS — networking**
9. Allocate an **Elastic IP**, associate to the instance. Without it the public IP changes on every stop/start and your DNS silently rots.

**💻 VSCode (optional, after you've launched it by hand once)**
- Codify the box as Terraform under `deploy/aws/terraform/` (skeleton in **Appendix E**), so a lost box is a `terraform apply`, not a from-memory rebuild. Not required for v1; a hand-launched single box is legitimate.

**Gate:** `ssh -i key.pem ubuntu@<EIP>` connects; `nvidia-smi` shows **L40S, ~48 GB**; the instance
appears **Online** in Systems Manager → Fleet Manager.

---

## Stage 3 — Manual bootstrap (☁️ on the box — prove it before automating)

> **UPDATED 2026-08-21:** the model cache now lives on its own dedicated
> 100GiB EBS volume (`/opt/magik/.hf_cache`), separate from the 100GiB root
> volume (OS, Docker, `/opt/magik/{data,logs,.env}`) — not a single combined
> `/opt/magik` directory tree on one volume as items 2–3 below originally
> described. Terraform (`deploy/aws/terraform/ec2.tf`) creates and attaches
> both volumes; run `deploy/aws/scripts/bootstrap_instance.sh <production|staging>`
> first (formats/mounts the model volume by UUID, creates the rest of the
> directory tree) — it replaces item 2 below and half of item 3 (the mount,
> not the actual model download, which still needs the steps that follow).

**Objective:** the app runs correctly here via hand-typed commands, before CI/CD touches it.

**On the box (SSH or SSM Session Manager):**
1. **Verify GPU Docker:** `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` → shows the L40S. If it fails, install `nvidia-container-toolkit` (rare on the DLAMI).
2. **Persistent dirs on EBS:** run `sudo bash deploy/aws/scripts/bootstrap_instance.sh production` (or `staging`) — mounts the dedicated model volume by UUID at `/opt/magik/.hf_cache`, creates `/opt/magik/{data,logs}` on the root volume, and a placeholder `/opt/magik/.env`. Confirm `df -h /opt/magik/.hf_cache` resolves to the **dedicated model EBS volume**, `df -h /opt/magik` (everything else) to the **root volume** — two separate block devices, not one.
3. **Download models once:** `git clone <repo> /tmp/magik-src` (only to get the downloader), then
   `HF_HOME=/opt/magik/.hf_cache python3 /tmp/magik-src/app/bin/models/download_all_models.py`. ~20 GB from HF Hub, lands on the dedicated model volume.
4. **Real `.env` at `/opt/magik/.env`:** copy from **Appendix F**, fill real secrets, include the Stage 1 items 4–6 values (`VRAM_BUDGET_GB=44`, `MAX_CONCURRENT_GPU_JOBS=3`, `CORS_ORIGINS=...`). This file never enters git.
5. **Pull & run manually** (first time, build locally if nothing is in GHCR yet):
   ```bash
   echo <GHCR_PAT> | docker login ghcr.io -u <owner> --password-stdin
   docker pull ghcr.io/<owner>/<repo>:latest
   docker run -d --name magik-current --gpus all --restart unless-stopped \
     -p 8000:8000 --env-file /opt/magik/.env \
     -v /opt/magik/.hf_cache:/app/.hf_cache \
     -v /opt/magik/data:/app/data \
     -v /opt/magik/logs:/app/logs \
     ghcr.io/<owner>/<repo>:latest
   ```
6. **Smoke test — not just /health:**
   ```bash
   curl -sf http://localhost:8000/health          # 200 {"status":"ok",...}
   # then a REAL end-to-end query via the API (register/login a user, upload a small doc, ask a question)
   ```

**Gate:** `/health` = 200 **and** one real grounded answer returns from a real query. (A green health
check with a broken model stack is the classic false-positive — send real traffic.)

---

## Stage 4 — Wire up CI/CD automation (💻 GitHub + ☁️ AWS)

**Objective:** replace Stage 3's manual `docker run` with `git tag vX.Y.Z && git push --tags`.

**☁️ AWS**
1. **OIDC provider:** IAM → Identity providers → add `token.actions.githubusercontent.com` (audience `sts.amazonaws.com`). One-time per account.
2. **Deploy role:** create the role `cd.yml` assumes — **trust policy** + **least-privilege permissions** in **Appendix C**. Scope to your `repo:OWNER/REPO:ref:refs/tags/*` and to the **one instance ARN**, not `*`.
3. **GHCR PAT in SSM:** store the `read:packages` PAT at Parameter Store `/magik/ghcr_pat` as **SecureString** (so it never appears in an Actions log — the box fetches it via `ssm get-parameter`).
4. Confirm SSM Agent **Online** (Fleet Manager).

**💻 GitHub — repo → Settings → Secrets and variables → Actions**
5. **Variables:** `AWS_DEPLOY_ROLE_ARN=<role arn>`, `EC2_INSTANCE_TAG=magik-prod`, `AWS_REGION=us-east-1`.
6. **Secrets (optional):** `SLACK_WEBHOOK_URL` for deploy-failure alerts (no-op until set, per `cd.yml`).
7. **Ship it:**
   ```bash
   git add -A && git commit -m "Stage 1 g6e.xlarge corrections"   # if not already committed
   git tag v1.0.0
   git push origin main --tags
   ```
8. Watch the **Actions** tab. `cd.yml` should: `build-push` (compile sm_89 CUDA image → GHCR) → `deploy` (OIDC → resolve instance by tag → wake → SSM pull/swap/health-check, auto-rollback on failure) → `post-deploy-eval` (dispatch Tier-2, expected to queue — see Stage 5).

**Gate:** a **green** `cd.yml` run; the deploy step's `$GITHUB_STEP_SUMMARY` shows the new tag; `/health`
on the box reflects it.

---

## Stage 5 — Self-hosted GPU runner for Tier-2 eval (☁️ on the box)

**Objective:** give `eval-gate.yml`'s nightly `tier2-full-suite` (`[self-hosted, gpu]`) somewhere to run.

**☁️ AWS (on the box)**
1. GitHub → Settings → Actions → Runners → *New self-hosted runner* → run the generated `config.sh` on the box, labels `self-hosted,gpu`.
2. Install as a service so it survives reboots/wakes: `sudo ./svc.sh install && sudo ./svc.sh start`.

**💻 VSCode:** verify `eval-gate.yml`'s job is `runs-on: [self-hosted, gpu]` — no change expected.

**Known limitation (flag, don't paper over):** the box is normally **stopped** (Stage 6), so at 03:00 UTC
the runner is usually offline and the nightly job **queues indefinitely** — it doesn't fail loudly, it
waits. Per `ci-cd.md`, Tier-2 is "informational until baseline v4" (an eval-engineer task, not Phase 30),
so this is acceptable now. **Future fix:** an EventBridge Scheduler rule that `StartInstances` ~5 min
before 03:00 UTC; normal idle-stop reclaims it after the run.

**Gate:** runner shows **Idle** (not offline) in the repo runner list while the box is up.

---

## Stage 6 — Domain, HTTPS, wake-on-visit, idle-stop (💻 + ☁️ — the heart of the plan)

**Objective:** one stable `https://` URL, always answering, that starts the GPU box on first visit and
stops it after ~15–20 min of no traffic.

### 6a — Domain + certs (☁️)
1. **Register a domain** — Route53 or an external registrar (Namecheap/Porkbun, ~$8–12/yr). *Note:* many
   promotional AWS credits **exclude** domain registration; an external registrar sidesteps the question.
2. **ACM cert** for `yourdomain.com` (+ `*.yourdomain.com`), issued **in `us-east-1`** (CloudFront
   requirement), validated via the DNS record ACM provides.

### 6b — The app subdomain, direct to the box (☁️ + on the box)
3. **DNS:** `app.yourdomain.com` → A/ALIAS → the box's **Elastic IP**.
4. **Caddy on the box** reverse-proxies `app.yourdomain.com` → `localhost:8000`, auto-issuing its own
   Let's Encrypt cert on first start (needs DNS already pointing here + port 80 open for ACME).
   Caddyfile in **Appendix B**; install via user-data (**Appendix D**) or by hand.

### 6c — The wake gateway (💻 write, ☁️ deploy)
5. **`magik-wake-gateway` Lambda** — code in **Appendix A**. Logic:
   - `DescribeInstances(Name=magik-prod)`.
   - `stopped`/`stopping` → `StartInstances` (idempotent) → return the self-refreshing "waking up ~60–90s" HTML.
   - `running` → `GET https://app.yourdomain.com/health` (short timeout) → healthy ⇒ 302 to `https://app.yourdomain.com`; not-yet ⇒ same waking page.
   - IAM role: `ec2:DescribeInstances` + `ec2:StartInstances` scoped to the one instance (**Appendix C**, gateway role).
6. **CloudFront** distribution: origin = the Lambda (Function URL or API Gateway HTTP API), ACM cert
   attached, so `https://yourdomain.com` is real HTTPS with no warnings and is the always-on front door.
7. **DNS:** `yourdomain.com` (apex) → ALIAS → the CloudFront distribution.

### 6d — Idle-stop (☁️)
8. **CloudWatch alarm** on `NetworkIn` (or `CPUUtilization`) below threshold for **15–20 min** → **SNS** →
   **`magik-idle-stop` Lambda** (**Appendix A**) → `StopInstances`. Role: `ec2:StopInstances` +
   `ec2:DescribeInstances` scoped to the instance.

**💻 VSCode / repo**
- Commit both Lambdas under `deploy/aws/lambda/wake_gateway/` and `deploy/aws/lambda/idle_stop/`, the
  Caddyfile under `deploy/aws/caddy/Caddyfile`, and (optionally) the whole wake stack as Terraform/SAM.
- **Unit-test the gateway's decision logic** (mock the boto3 EC2 client): stopped→wake, booting→wait,
  healthy→redirect. It's a small pure-ish function; cover it.

**Gate (the money test):** stop the instance by hand → open `https://yourdomain.com` in a **private
window** → confirm the waking page → confirm redirect to `https://app.yourdomain.com` once healthy →
confirm **zero cert warnings** anywhere → leave it idle → confirm it **auto-stops** in 15–20 min (check
the EC2 console, not just the alarm state).

---

## Stage 7 — Demo access & abuse control (application-level)

**Objective:** an HR visitor is in with zero friction; a bot that finds the link can't run up the bill.

1. **Dedicated demo account** via your own `app/auth/` flow — *not* your dev/test login — seeded with a
   small, non-sensitive corpus (e.g. a public 10-K). A "Try the live demo" button that auto-logs-in as
   this account beats "type these credentials."
2. **Tighter per-user rate limit** for that account (`app/auth/rate_limit.py` supports per-user limits) —
   meaningfully below a real user's ceiling; the public link is the one realistic abuse vector for GPU burn.
3. **Confirm `CORS_ORIGINS`** is locked (Stage 1 item 6) and the security group no longer exposes 8000 to `0.0.0.0/0`.

**Gate:** demo-button login works from a fresh browser; exceeding the rate limit throttles rather than silently absorbing.

---

## Stage 8 — Validate end-to-end & go live (💻 + ☁️)

**💻 VSCode / repo**
1. Update `docs/runbooks/ci-cd.md`'s "Phase 30 handoff" table — mark all 7 items **done**, with dates.
2. Replace the `cd.yml:94` ADR-5 TODO comment with "implemented — see `docs/runbooks/phase-30-aws-deployment.md`".
3. Run **Appendix G validation matrix** top to bottom; commit results (normal PR through `ci.yml` + Tier-1 gate).

**☁️ AWS**
4. Run the **full cold cycle twice**, ~a day apart: stopped → visit → wake → interact → idle → auto-stop.
   First run proves it can; second proves it's not a fluke.
5. Watch **Cost Explorer daily for 48 h** — confirm real spend tracks the §2 model.
6. Put **one URL** (`https://yourdomain.com`) on portfolio / résumé / LinkedIn.

**Definition of Done (Phase 30):**
- [ ] Public HTTPS endpoint works from a from-zero cold stop, no human touching AWS.
- [ ] No secret in image, repo, or Actions log — all via SSM / box `.env`.
- [ ] Wake-on-visit + idle-stop both proven twice.
- [ ] `git push --tags` deploys and auto-rolls-back on health failure (tested once by shipping a deliberately broken tag to a staging tag, then reverting).
- [ ] Budget alarm configured and confirmation-clicked.
- [ ] The canonical demo URL is documented in `README.md`.

---

## Appendix A — Wake gateway & idle-stop Lambdas (Python 3.12, boto3)

`deploy/aws/lambda/wake_gateway/handler.py`
```python
import os
import urllib.request

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
INSTANCE_TAG = os.environ.get("EC2_INSTANCE_TAG", "magik-prod")
APP_URL = os.environ["APP_URL"]              # https://app.yourdomain.com
HEALTH_URL = f"{APP_URL}/health"
HEALTH_TIMEOUT_S = float(os.environ.get("HEALTH_TIMEOUT_S", "3"))

ec2 = boto3.client("ec2", region_name=REGION)

WAKING_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>MAGIK — starting…</title>
<style>body{{font-family:system-ui;background:#0a0e14;color:#e6edf3;display:grid;
place-items:center;height:100vh;margin:0}}.c{{text-align:center;max-width:32rem;padding:2rem}}
.s{{width:2.5rem;height:2.5rem;border:3px solid #1f6feb;border-top-color:transparent;
border-radius:50%;animation:r 1s linear infinite;margin:0 auto 1.5rem}}
@keyframes r{{to{{transform:rotate(360deg)}}}}</style></head>
<body><div class="c"><div class="s"></div>
<h1>Waking up MAGIK…</h1>
<p>Loading the AI model stack on a GPU server. This takes ~60–90 seconds on first visit.
This page refreshes automatically.</p></div></body></html>"""


def _resolve_instance():
    r = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_TAG]},
            {"Name": "instance-state-name",
             "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )
    for res in r["Reservations"]:
        for inst in res["Instances"]:
            return inst["InstanceId"], inst["State"]["Name"]
    return None, None


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=HEALTH_TIMEOUT_S) as resp:
            return resp.status == 200
    except Exception:
        return False


def _page(status, body, headers=None):
    h = {"content-type": "text/html; charset=utf-8", "cache-control": "no-store"}
    if headers:
        h.update(headers)
    return {"statusCode": status, "headers": h, "body": body}


def handler(event, _context):
    instance_id, state = _resolve_instance()
    if instance_id is None:
        return _page(503, "<h1>Demo instance not found.</h1>")

    if state in ("stopped", "stopping"):
        # start-instances on stopping just queues; on stopped it starts. Idempotent enough.
        ec2.start_instances(InstanceIds=[instance_id])
        return _page(200, WAKING_HTML)

    if state == "running" and _healthy():
        return _page(302, "", {"location": APP_URL})

    # pending, or running-but-not-yet-healthy
    return _page(200, WAKING_HTML)
```

`deploy/aws/lambda/idle_stop/handler.py`
```python
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
INSTANCE_TAG = os.environ.get("EC2_INSTANCE_TAG", "magik-prod")
ec2 = boto3.client("ec2", region_name=REGION)


def handler(event, _context):
    # Triggered by SNS from a CloudWatch low-NetworkIn alarm.
    r = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_TAG]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    ids = [i["InstanceId"] for res in r["Reservations"] for i in res["Instances"]]
    if ids:
        ec2.stop_instances(InstanceIds=ids)
    return {"stopped": ids}
```

Gateway env vars: `EC2_INSTANCE_TAG=magik-prod`, `APP_URL=https://app.yourdomain.com`. Timeout ≥ 5 s,
memory 128 MB.

---

## Appendix B — `deploy/aws/caddy/Caddyfile`

```
app.yourdomain.com {
    encode zstd gzip
    reverse_proxy localhost:8000 {
        health_uri /health
        health_interval 10s
    }
}
```
Caddy issues + renews the Let's Encrypt cert for `app.yourdomain.com` automatically once DNS points at
the box and ports 80/443 are open. Install: `sudo apt install -y caddy` (or the official repo), place
this at `/etc/caddy/Caddyfile`, `sudo systemctl enable --now caddy`.

---

## Appendix C — IAM policies (least privilege)

**Deploy role trust (GitHub OIDC → assume):**
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike": {"token.actions.githubusercontent.com:sub": "repo:<OWNER>/<REPO>:ref:refs/tags/*"}
    }
  }]
}
```

**Deploy role permissions (what `cd.yml` may do — one instance, not `*`):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["ec2:DescribeInstances"], "Resource": "*"},
    {"Effect": "Allow", "Action": ["ec2:StartInstances"],
     "Resource": "arn:aws:ec2:us-east-1:<ACCOUNT_ID>:instance/<INSTANCE_ID>"},
    {"Effect": "Allow", "Action": ["ssm:SendCommand"],
     "Resource": [
       "arn:aws:ec2:us-east-1:<ACCOUNT_ID>:instance/<INSTANCE_ID>",
       "arn:aws:ssm:us-east-1::document/AWS-RunShellScript"
     ]},
    {"Effect": "Allow",
     "Action": ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations", "ssm:DescribeInstanceInformation"],
     "Resource": "*"}
  ]
}
```

**Wake-gateway Lambda role** (in addition to `AWSLambdaBasicExecutionRole`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["ec2:DescribeInstances"], "Resource": "*"},
    {"Effect": "Allow", "Action": ["ec2:StartInstances"],
     "Resource": "arn:aws:ec2:us-east-1:<ACCOUNT_ID>:instance/<INSTANCE_ID>"}
  ]
}
```

**Idle-stop Lambda role**: same shape, `ec2:StopInstances` instead of `StartInstances`.

**EC2 instance profile**: attach the AWS-managed `AmazonSSMManagedInstanceCore`. Add
`ssm:GetParameter` on `/magik/*` if the box reads the GHCR PAT from SSM at deploy time (it does, via
`cd.yml`'s SSM script).

---

## Appendix D — EC2 user-data (first-boot bootstrap, optional)

```bash
#!/usr/bin/env bash
set -euxo pipefail

# Persistent app dirs on the EBS root volume (NOT instance-store).
mkdir -p /opt/magik/{.hf_cache,data,logs}
chown -R ubuntu:ubuntu /opt/magik

# Caddy (auto-HTTPS reverse proxy for app.yourdomain.com).
apt-get update
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  > /etc/apt/sources.list.d/caddy-stable.list
apt-get update
apt-get install -y caddy

# Drop the Caddyfile (edit domain first, or template it in via SSM later).
cat >/etc/caddy/Caddyfile <<'EOF'
app.yourdomain.com {
    encode zstd gzip
    reverse_proxy localhost:8000 {
        health_uri /health
        health_interval 10s
    }
}
EOF
systemctl enable --now caddy

# NOTE: models and .env are provisioned in Stage 3 (too large / too secret for user-data).
```

---

## Appendix E — Terraform (`deploy/aws/terraform/`) — IMPLEMENTED

> **UPDATED 2026-08-21:** no longer a skeleton/optional-future-step — this was
> built and applied for real after the entire prior EC2 fleet + EBS volumes
> were deleted (proof that "codify after you've launched by hand once" needed
> to actually happen, not stay deferred indefinitely). Actual file layout,
> which diverged from the sketch below in a few ways worth noting explicitly:

```
deploy/aws/terraform/
  versions.tf           # terraform + provider version pins (aws ~>5.0, tls, local); local backend, not S3+DynamoDB
  providers.tf          # aws provider (region + configurable local operator profile), data.aws_caller_identity
  variables.tf          # instance_type=g6e.xlarge, root/model volume sizes=100 each, admin_ssh_cidr, github_owner/repo (+ numeric IDs), create_staging, staging_model_snapshot_id
  network.tf            # aws_vpc + subnet + IGW + route table (fresh account had NO default VPC — not assumed, built)
  security_groups.tf    # production (22/80/443) and staging (zero ingress) — the Stage 2 table, as code
  key_pair.tf            # tls_private_key + aws_key_pair + local_sensitive_file (break-glass SSH; CI itself never uses this, SSM only)
  ec2.tf                 # data.aws_ami.deep_learning (most_recent, name-pattern not pinned ID) + aws_instance.production/staging + dedicated model aws_ebs_volume (staging's clonable via snapshot_id) + aws_eip (production only)
  iam.tf                 # GitHub OIDC provider + magik-deploy-role (trust policy built from deploy/aws/iam/github-oidc-trust-policy.json's sub patterns, dynamic account ID) + magik-ec2-role/instance profile (permissions built from ec2-instance-profile-permissions.json, dynamic account ID)
  outputs.tf              # production_public_ip (for the GoDaddy A record), production_instance_arn (feeds the Lambda permission JSONs), magik_deploy_role_arn (feeds the AWS_DEPLOY_ROLE_ARN repo variable), ssh key path
  terraform.tfvars.example
```

Deliberately **not** in this Terraform's scope (see `deploy/aws/README.md` and
the plan that drove this rebuild): the wake-gateway/idle-stop Lambdas, API
Gateway, and EventBridge schedule rule — those stay owned by
`deploy/aws/scripts/deploy_lambdas.sh` (imperative, idempotent, already
correct), because re-expressing them in Terraform risked generating a new API
Gateway invoke URL and silently breaking an already-embedded portfolio "launch
demo" link for no benefit. CloudFront/ACM/apex-ALIAS was in the original
sketch but was never actually part of the as-built system either time — DNS
stays a manual GoDaddy A-record edit (Stage 6a), Caddy on the box terminates
TLS via Let's Encrypt directly.

Secrets stay **out** of `.tf`/state, same principle as the original sketch:
SSM parameters (`/magik/*`) are referenced by name/ARN pattern in `iam.tf`'s
policy documents, never declared with real values in Terraform.

---

## Appendix F — Box `.env` inventory (`/opt/magik/.env`, never committed)

**Superseded in part by Phase 31** (project memory "secrets-management-gap",
closed in `docs/runbooks/phase-31-monitoring.md`'s "Secrets migration"
section): `MONGO_URI`, `JWT_SECRET_KEY`, `SECRET_KEY`, `GOOGLE_CLIENT_SECRET`,
and `SMTP_PASSWORD` below no longer need to live in this file — `cd.yml`
now fetches them from SSM Parameter Store on every deploy instead. This
Appendix is kept as-written for historical accuracy (it's what Phase 30
actually shipped); don't treat it as the current state for those five keys.

Secrets + genuinely per-hardware values only (everything else has a tuned default in `config.py`):
```bash
# --- environment / hardware (per-box, not secret) ---
ENV=production
DEBUG=false
VRAM_BUDGET_GB=44                 # Stage 1 item 4 (48GB L40S shows as ~44.4GB to CUDA; 46 is a silent no-op)
MAX_CONCURRENT_GPU_JOBS=3         # Stage 1 item 5 (raise only after watching real headroom)
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com   # Stage 1 item 6

# --- managed data services (secret) ---
QDRANT_URL=
QDRANT_API_KEY=
REDIS_URL=
REDIS_TOKEN=
MONGO_URI=

# --- tools / models (secret) ---
TAVILY_API_KEY=
HF_TOKEN=

# --- auth (secret) ---
JWT_SECRET_KEY=
SECRET_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=https://app.yourdomain.com/auth/oauth/callback

# --- email (if used) ---
SMTP_USER=
SMTP_PASSWORD=
```
Cross-check names against the committed `.env.example` before first boot; a missing key surfaces as a
`config.py` validation error at startup, which `startup_validator.py` turns into a loud refuse-to-serve.

---

## Appendix G — Validation matrix (run in Stage 8)

| # | Check | Command / action | Pass criterion |
|---|---|---|---|
| 1 | Image builds for L40S | `docker build --target runtime .` | build ok incl. `llama_supports_gpu_offload()` assert |
| 2 | GPU visible in Docker | `docker run --rm --gpus all …base nvidia-smi` | L40S, ~48 GB |
| 3 | Models on EBS | `df -h /opt/magik/.hf_cache` | on EBS root, ~20 GB present |
| 4 | Health | `curl -sf https://app.yourdomain.com/health` | 200, correct version |
| 5 | Real query | login → upload → ask | grounded answer with sources |
| 6 | Deploy pipeline | `git tag v1.0.1 && git push --tags` | `cd.yml` green; box on new tag |
| 7 | Auto-rollback | deploy a deliberately broken tag | health fails → `magik-previous` restored → job fails loudly |
| 8 | Wake-on-visit | stop box → open `https://yourdomain.com` | waking page → redirect when healthy, no cert warning |
| 9 | Idle-stop | leave idle 20 min | instance state → `stopped` in EC2 console |
| 10 | No public 8000 | `curl http://<EIP>:8000/health` from outside | refused/timeout (only Caddy/CloudFront reach it) |
| 11 | CORS locked | cross-origin `fetch` from a random origin | blocked by CORS |
| 12 | Secrets absent | `git grep -iE 'api_key|secret|mongodb\+srv' -- ':!*.example'` | nothing real; Actions logs clean |
| 13 | Cost | Cost Explorer, 48 h | tracks §2 model (~$12 fixed + few $/active hr) |
| 14 | Budget alarm | Billing → Budgets | exists, email confirmed |

---

## Appendix H — Rollback & incident runbook

**A deploy went bad (health check failed):** `cd.yml`'s SSM script already auto-rolled back to
`magik-previous` and the job failed loudly. Confirm: `curl .../health` returns the *old* version. No
action needed beyond fixing forward.

**A deploy passed health but is wrong (bad answers, etc.):** manual rollback on the box:
```bash
docker rm -f magik-current
docker rename magik-previous magik-current
docker start magik-current
```

**Wake gateway shows "waking" forever:** box is up but unhealthy. SSH in →
`docker logs magik-current --tail 100`. Common causes: missing `.hf_cache` mount (Stage 1 item 7),
bad `.env` (Appendix F), or GPU not visible (`nvidia-smi` in-container).

**Bill climbing unexpectedly:** check EC2 — is the box stuck `running`? Verify the CloudWatch idle
alarm state and the `magik-idle-stop` Lambda's recent invocations. Stop it by hand:
`aws ec2 stop-instances --instance-ids <id>`. Then debug the alarm threshold.

**Models re-download on every wake:** `/opt/magik/.hf_cache` is on instance-store, not EBS
(Stage 1 item 7 / Stage 3 step 2). Move it to the EBS root volume.

**Nightly Tier-2 eval never runs:** expected while the box is stopped (Stage 5). Not an incident.

**Total loss of the box:** relaunch from the DLAMI (or `terraform apply` if you did Appendix E),
re-associate the Elastic IP, re-run Stage 3 (models + `.env`), re-register the runner (Stage 5).
Managed data (Qdrant/Redis/Mongo) is untouched — it lives outside AWS.

---

## Appendix I — Security hardening checklist (before go-live)

- [ ] SG port 8000 no longer `0.0.0.0/0` — reachable only via Caddy(localhost)/CloudFront.
- [ ] SSH (22) restricted to your `/32`, key-only (no password auth).
- [ ] `CORS_ORIGINS` locked to your two domains (no `*`).
- [ ] All secrets in box `.env` (0600, ubuntu-owned) or SSM SecureString — none in git, image, or Actions logs.
- [ ] OIDC role + both Lambda roles scoped to the single instance ARN, not `*`.
- [ ] Public demo account is separate from your dev account, rate-limited tighter, seeded with non-sensitive data.
- [ ] Auth still enforced end-to-end (the demo auto-login uses a real account, doesn't bypass `get_current_user`).
- [ ] ACM + Caddy certs valid; no mixed-content or cert warnings in the full chain.
- [ ] Budget alarm + Cost Anomaly Detection active.
```
