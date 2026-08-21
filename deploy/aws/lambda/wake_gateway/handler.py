"""MAGIK wake gateway — public entry point for the live demo.

The GPU instance (g6e.xlarge / L40S, ~$1.86/hr) is stopped by default. This
Lambda is the always-on front door that makes that invisible to a visitor:

    stopped        -> StartInstances, show a live-updating "waking up" page
    booting        -> same page, step 1 ("waking GPU server") active
    running        -> same page, step 2 ("loading AI models") active
    running+stuck  -> distinct "taking longer than usual" state once the
                       instance has been running past STUCK_MINUTES without
                       /health ever going green — see below
    healthy        -> step 3 lights up, then a client-side redirect to the app

Deployed behind an API Gateway HTTP API. Lambda Function URLs were tried
first but returned a persistent 403 Forbidden in this account despite
correct auth-type and resource-policy configuration — API Gateway's older
permission model worked immediately, so that's the standard front door now.
Runs outside a VPC so it can reach the instance's public endpoint directly.

Cost note: this is the piece that makes scale-to-zero viable. Always-on would
be ~$1,340/mo; stopped-by-default plus this gateway is ~$12/mo fixed plus a
few dollars per active hour.

── Live status page (2026-08-21 redesign) ──────────────────────────────────
Originally this served a full HTML page on every hit, refreshed via a bare
`<meta http-equiv="refresh">` — every ~7s the whole page flashed and
re-rendered from scratch, and every reload was indistinguishable from every
other one (no visible progress, just the same paragraph repeating). Visitors
watching this live (hiring managers clicking a portfolio link) had no signal
that anything was actually happening.

Now the FIRST hit still renders a full HTML page (with the CSS/JS shell
inlined, self-contained, no external assets), but every subsequent update is
driven by that page's own JS polling this SAME Lambda URL with `?check=1`
and getting back a small JSON status object — the DOM updates in place
(progress steps light up, message text changes) with no page flash, and a
`window.location.replace()` fires client-side the moment status is "ready".
This is why the handler below branches on `queryStringParameters.check`:
that's the one thing separating an initial page load from a background poll
tick from that page's own JS — everything else (instance lookup, health
check, Kuma push) is identical for both.

Stuck-state detection (STUCK_MINUTES, default 6): every request while
state=="running" recomputes minutes-since-LaunchTime (AWS resets LaunchTime on
every StartInstances call, so this is genuinely "minutes since this boot").
Before this existed, a visitor reloading a wedged instance (crashed app,
unreachable Qdrant/Redis/Mongo, bad deploy) saw the exact same "about a
minute" copy indefinitely, with nothing distinguishing a normal 45-second boot
from an instance stuck for 20 minutes — that indistinguishability was the bug.

Uptime monitoring hook (optional, KUMA_PUSH_URL): when set, this function
pushes an "up" heartbeat the moment it confirms the app is genuinely healthy
and about to redirect a real visitor, and a "down" heartbeat the moment it
detects the stuck state above — see _push_kuma() below. This is deliberately
a PUSH from here, not a POLL from Kuma: Kuma never calls this gateway itself,
because doing so would BE a visitor request and would wake the instance on
its own, defeating the entire point of scale-to-zero. Every push this
function makes is therefore a side effect of a real visitor already being
here — it can never be the cause of a wake. See
deploy/aws/lambda/idle_stop/handler.py for the independent, schedule-driven
"up"/"down" pushes that don't depend on a visitor showing up at all.
"""

from __future__ import annotations

import html
import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

INSTANCE_TAG = os.environ.get("EC2_INSTANCE_TAG", "magik-prod")
APP_URL = os.environ.get("APP_URL", "").rstrip("/")
HEALTH_URL = os.environ.get("HEALTH_URL") or (f"{APP_URL}/health" if APP_URL else "")
HEALTH_TIMEOUT_S = float(os.environ.get("HEALTH_TIMEOUT_S", "3"))
REFRESH_SECONDS = int(os.environ.get("REFRESH_SECONDS", "7"))

# EC2 has been "running" this long without /health ever going green -> stop
# reporting plain "loading" and switch to an explicit "taking longer than
# usual" state instead. Model loading normally finishes well under this;
# past it, something is genuinely wrong (crashed app, unreachable
# Qdrant/Redis/Mongo dependency, bad deploy) and a visitor endlessly seeing
# the *identical* "loading" message with no escalation was the actual bug
# being fixed here — not just the copy.
STUCK_MINUTES = float(os.environ.get("STUCK_MINUTES", "6"))

# Optional — unset by default, so this is a strict no-op until Phase F's Kuma
# host is actually provisioned and this env var is deliberately set (see
# deploy/aws/scripts/deploy_lambdas.sh and monitoring/uptime-kuma/).
KUMA_PUSH_URL = os.environ.get("KUMA_PUSH_URL", "").rstrip("/")
KUMA_PUSH_TIMEOUT_S = float(os.environ.get("KUMA_PUSH_TIMEOUT_S", "2"))

# Short timeouts + no retries: this Lambda sits in front of a human staring at
# a browser tab. A slow AWS call must not turn into a 30s blank page — better
# to serve the interstitial and let the page's own poll loop drive the next
# attempt.
_boto_cfg = Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2})
ec2 = boto3.client("ec2", config=_boto_cfg)

_WAKING_STATES = {"pending", "stopping", "shutting-down"}

# Every state the status JSON / page can be in. "waking" and "loading" are
# the two ordinary steps of the 3-step progress UI; "stuck"/"capacity"/
# "error" are the distinct off-the-happy-path states the redesign exists to
# make visible instead of indistinguishable.
_STEP_LABELS = ["Waking GPU server", "Loading AI models", "Redirecting to sign-in"]


def _resp(status: int, body: str, content_type: str = "text/html; charset=utf-8") -> dict:
    return {
        "statusCode": status,
        "headers": {
            "content-type": content_type,
            "cache-control": "no-store, no-cache, must-revalidate",
        },
        "body": body,
    }


def _json_resp(payload: dict) -> dict:
    return _resp(200, json.dumps(payload), content_type="application/json")


def _push_kuma(status: str, msg: str) -> None:
    """Fire-and-forget heartbeat to Uptime Kuma's push monitor. No-op if
    KUMA_PUSH_URL is unset. Never allowed to affect the visitor-facing
    response — any failure here is swallowed, same fail-open posture as
    everything else in this Lambda that isn't the core wake/redirect path.
    """
    if not KUMA_PUSH_URL:
        return
    try:
        url = f"{KUMA_PUSH_URL}?status={status}&msg={msg}"
        req = urllib.request.Request(url, headers={"User-Agent": "magik-wake-gateway"})
        urllib.request.urlopen(req, timeout=KUMA_PUSH_TIMEOUT_S).close()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        log.info("kuma push failed (non-fatal): %s", exc)


def _find_instance() -> tuple[str | None, str | None, datetime | None]:
    """Return (instance_id, state, launch_time) for the tagged instance, or
    (None, None, None). AWS resets LaunchTime to the moment of the most
    recent StartInstances call — not the instance's original creation time —
    so "minutes since LaunchTime" while state=="running" is exactly "minutes
    since this boot," which is what stuck-state detection needs below.
    """
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_TAG]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped", "shutting-down"],
            },
        ]
    )
    for reservation in resp.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            return inst["InstanceId"], inst["State"]["Name"], inst.get("LaunchTime")
    return None, None, None


def _is_healthy() -> bool:
    if not HEALTH_URL:
        return False
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "magik-wake-gateway"})
        with urllib.request.urlopen(req, timeout=HEALTH_TIMEOUT_S) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        log.info("health check not ready: %s", exc)
        return False


def _compute_status() -> dict:
    """The one place that resolves 'what's actually going on right now' into
    a small JSON-serializable status object. Called for BOTH the initial page
    load and every background poll tick — the page's own JS only ever needs
    to understand this shape, never the instance/EC2 details behind it.

    Shape: {"state": ..., "step": 0|1|2, "message": str, "redirect": str|None,
            "elapsed_minutes": float|None}

    states: "waking" (step 0), "loading" (step 1), "ready" (step 2, redirect
    set), "stuck", "capacity", "error" (message explains what's wrong).
    """
    if not APP_URL:
        log.error("APP_URL is not configured")
        return {"state": "error", "step": None, "message": "Gateway is not fully configured.",
                "redirect": None, "elapsed_minutes": None, "retry": False}

    try:
        instance_id, state, launch_time = _find_instance()
    except ClientError as exc:
        log.exception("describe_instances failed")
        code = exc.response["Error"]["Code"]
        return {"state": "error", "step": None,
                "message": f"Could not query the demo instance ({code}).",
                "redirect": None, "elapsed_minutes": None, "retry": True}

    if instance_id is None:
        log.error("no instance tagged Name=%s", INSTANCE_TAG)
        return {"state": "error", "step": None, "message": "The demo instance could not be found.",
                "redirect": None, "elapsed_minutes": None, "retry": False}

    log.info("instance %s state=%s", instance_id, state)

    if state == "stopped":
        try:
            ec2.start_instances(InstanceIds=[instance_id])
            log.info("start_instances issued for %s", instance_id)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            # IncorrectInstanceState = someone/something else already started it
            # between our describe and start. Harmless; fall through.
            if code not in ("IncorrectInstanceState",):
                log.exception("start_instances failed")
                if code == "InsufficientInstanceCapacity":
                    return {
                        "state": "capacity", "step": None,
                        "message": "AWS currently has no spare capacity for this GPU "
                                   "instance type. This is on AWS's side, not the app — "
                                   "it usually clears within a few minutes.",
                        "redirect": None, "elapsed_minutes": None, "retry": True,
                    }
                return {"state": "error", "step": None,
                        "message": f"Could not start the demo instance ({code}).",
                        "redirect": None, "elapsed_minutes": None, "retry": True}
        return {"state": "waking", "step": 0,
                "message": "Starting the GPU server.",
                "redirect": None, "elapsed_minutes": None, "retry": True}

    if state in _WAKING_STATES:
        return {"state": "waking", "step": 0,
                "message": "Starting the GPU server.",
                "redirect": None, "elapsed_minutes": None, "retry": True}

    # state == "running" from here on
    if _is_healthy():
        log.info("healthy -> redirecting to %s", APP_URL)
        _push_kuma("up", "woken_by_visitor")  # real visitor confirmed the app is genuinely serving
        return {"state": "ready", "step": 2, "message": "Ready — redirecting you now.",
                "redirect": APP_URL, "elapsed_minutes": None, "retry": False}

    elapsed_minutes = 0.0
    if launch_time is not None:
        elapsed_minutes = (datetime.now(timezone.utc) - launch_time).total_seconds() / 60

    if elapsed_minutes >= STUCK_MINUTES:
        log.warning(
            "instance %s running %.1fm without going healthy (threshold %.1fm)",
            instance_id, elapsed_minutes, STUCK_MINUTES,
        )
        _push_kuma("down", f"unhealthy_after_{elapsed_minutes:.0f}m")
        return {
            "state": "stuck", "step": 1,
            "message": f"The server has been running for about {elapsed_minutes:.0f} "
                       "minutes but the app still isn't answering — normally this "
                       "takes under a minute.",
            "redirect": None, "elapsed_minutes": round(elapsed_minutes, 1), "retry": True,
        }

    return {"state": "loading", "step": 1,
            "message": "The GPU instance is up and the AI model stack is loading.",
            "redirect": None, "elapsed_minutes": round(elapsed_minutes, 1), "retry": True}


# ── Page shell (rendered once, on the very first hit) ───────────────────────
# Everything after this point is presentation only — _compute_status() above
# is the single source of truth both this shell's first paint AND every
# later poll tick read from, so the two can never show contradictory info.

_PAGE_CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;
background:#0a0e14;color:#e6edf3;
font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.c{width:100%;max-width:30rem;padding:2.5rem 1.75rem;text-align:center}
h1{margin:0 0 .4rem;font-size:1.4rem;font-weight:600;letter-spacing:-.01em}
.sub{margin:0 0 2rem;color:#9198a1;font-size:.9375rem}
.steps{display:flex;align-items:flex-start;justify-content:center;gap:0;margin:0 0 1.75rem}
.step{flex:1;display:flex;flex-direction:column;align-items:center;position:relative}
.step:not(:last-child)::after{content:"";position:absolute;top:.9rem;left:56%;width:88%;
height:2px;background:#21262d;z-index:0}
.step.done:not(:last-child)::after{background:#2ea043}
.dot{width:1.9rem;height:1.9rem;border-radius:50%;display:grid;place-items:center;
background:#161b22;border:2px solid #21262d;font-size:.8rem;font-weight:700;
color:#6e7681;position:relative;z-index:1;flex-shrink:0}
.step.done .dot{background:#2ea04322;border-color:#2ea043;color:#2ea043}
.step.active .dot{border-color:#1f6feb;color:#1f6feb}
.dot .spin{display:none}
.step.active .dot .num{display:none}
.step.active .dot .spin{display:block;width:.9rem;height:.9rem;border:2px solid #1f6feb44;
border-top-color:#1f6feb;border-radius:50%;animation:r 0.9s linear infinite}
.step.err .dot{border-color:#f85149;color:#f85149}
.step-label{margin-top:.6rem;font-size:.7rem;color:#6e7681;max-width:6.5rem;line-height:1.3}
.step.active .step-label{color:#c9d1d9}
.step.done .step-label{color:#8b949e}
@keyframes r{to{transform:rotate(360deg)}}
.msg{background:#161b22;border:1px solid #21262d;border-radius:.6rem;
padding:1rem 1.1rem;font-size:.875rem;color:#c9d1d9;text-align:left;margin-bottom:1rem}
.msg.warn{border-color:#9e6a03;background:#3b2a0022}
.msg.err{border-color:#f85149;background:#3d151322}
.n{font-size:.8125rem;color:#6e7681;margin:0}
"""

_PAGE_JS = """
function render(s){
  var steps=document.querySelectorAll('.step');
  steps.forEach(function(el,i){
    el.classList.remove('done','active','err');
    if(s.state==='error'||s.state==='capacity'){
      if(i===0) el.classList.add('err');
    } else if(s.state==='stuck'){
      if(i<1) el.classList.add('done'); else if(i===1) el.classList.add('err');
    } else if(s.step===null){
      // no step info (config error) — leave all neutral
    } else if(i<s.step){ el.classList.add('done'); }
    else if(i===s.step){ el.classList.add('active'); }
  });
  var msgEl=document.getElementById('msg');
  msgEl.textContent=s.message;
  msgEl.className='msg'+(s.state==='error'?' err':(s.state==='capacity'||s.state==='stuck'?' warn':''));
  var h1=document.getElementById('h1');
  if(s.state==='ready') h1.textContent='Ready!';
  else if(s.state==='stuck') h1.textContent='Taking longer than usual';
  else if(s.state==='capacity') h1.textContent='Waiting on AWS capacity';
  else if(s.state==='error') h1.textContent='Demo temporarily unavailable';
  else h1.textContent='Thanks for checking out MAGIK';
  if(s.state==='ready' && s.redirect){
    setTimeout(function(){ window.location.replace(s.redirect); }, 600);
    return;
  }
  if(s.retry){
    setTimeout(poll, s.state==='capacity' ? 20000 : REFRESH_MS);
  }
}
function poll(){
  fetch(window.location.pathname + '?check=1', {cache:'no-store'})
    .then(function(r){ return r.json(); })
    .then(render)
    .catch(function(){ setTimeout(poll, REFRESH_MS); });
}
"""


def _render_shell(initial: dict) -> str:
    steps_html = "".join(
        f'<div class="step"><div class="dot"><span class="num">{i + 1}</span>'
        f'<span class="spin"></span></div><div class="step-label">{label}</div></div>'
        for i, label in enumerate(_STEP_LABELS)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAGIK — starting up</title><style>{_PAGE_CSS}</style></head>
<body><div class="c">
<h1 id="h1">Thanks for checking out MAGIK</h1>
<p class="sub">This live demo scales its GPU server to zero when idle to keep hosting costs down.</p>
<div class="steps">{steps_html}</div>

<div id="msg" class="msg">{html.escape(initial["message"])}</div>
<p class="n">No need to click or refresh — this page updates itself and will take you to
sign-in the moment it's ready.</p>
</div>
<script>
var REFRESH_MS = {REFRESH_SECONDS * 1000};
{_PAGE_JS}
render({json.dumps(initial)});
</script>
</body></html>"""


def handler(event, context):  # noqa: ARG001 - Lambda signature
    qs = event.get("queryStringParameters") or {}
    is_poll = qs.get("check") == "1"

    status = _compute_status()

    if is_poll:
        return _json_resp(status)

    # Initial page load: config errors and "instance not found" are fatal
    # and unlikely to change from one visitor reload to the next — the
    # shell's own JS already reads status["retry"]=false and skips
    # scheduling another poll, so this just needs the right HTTP status code.
    status_code = 200
    if status["state"] == "error" and not status.get("retry"):
        status_code = 500 if "configured" in status["message"] else 503

    return _resp(status_code, _render_shell(status))
