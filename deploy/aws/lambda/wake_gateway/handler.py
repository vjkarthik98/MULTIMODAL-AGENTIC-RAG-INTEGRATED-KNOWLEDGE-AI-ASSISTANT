"""MAGIK wake gateway — public entry point for the live demo.

The GPU instance (g6e.xlarge / L40S, ~$1.86/hr) is stopped by default. This
Lambda is the always-on front door that makes that invisible to a visitor:

    stopped        -> StartInstances, show a self-refreshing "warming up" page
    booting        -> same page (instance not running yet)
    running        -> same page, now saying so explicitly (models loading)
    running+stuck  -> distinct "taking longer than usual" page once the
                       instance has been running past STUCK_MINUTES without
                       /health ever going green — see below
    healthy        -> 302 to the app

Deployed behind an API Gateway HTTP API. Lambda Function URLs were tried
first but returned a persistent 403 Forbidden in this account despite
correct auth-type and resource-policy configuration — API Gateway's older
permission model worked immediately, so that's the standard front door now.
Runs outside a VPC so it can reach the instance's public endpoint directly.

Cost note: this is the piece that makes scale-to-zero viable. Always-on would
be ~$1,340/mo; stopped-by-default plus this gateway is ~$12/mo fixed plus a
few dollars per active hour.

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
REFRESH_SECONDS = os.environ.get("REFRESH_SECONDS", "7")

# EC2 has been "running" this long without /health ever going green -> stop
# showing the same indistinguishable "starting up" copy forever and switch to
# an explicit "this is taking longer than usual" page instead. Model loading
# normally finishes well under this; past it, something is genuinely wrong
# (crashed app, unreachable Qdrant/Redis/Mongo dependency, bad deploy) and a
# visitor endlessly reloading the *identical* "about a minute" message with no
# escalation was the actual bug being fixed here — not just the copy.
STUCK_MINUTES = float(os.environ.get("STUCK_MINUTES", "6"))

# Optional — unset by default, so this is a strict no-op until Phase F's Kuma
# host is actually provisioned and this env var is deliberately set (see
# deploy/aws/scripts/deploy_lambdas.sh and monitoring/uptime-kuma/).
KUMA_PUSH_URL = os.environ.get("KUMA_PUSH_URL", "").rstrip("/")
KUMA_PUSH_TIMEOUT_S = float(os.environ.get("KUMA_PUSH_TIMEOUT_S", "2"))

# Short timeouts + no retries: this Lambda sits in front of a human staring at
# a browser tab. A slow AWS call must not turn into a 30s blank page — better
# to serve the interstitial and let the page refresh drive the next attempt.
_boto_cfg = Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2})
ec2 = boto3.client("ec2", config=_boto_cfg)

_WAKING_STATES = {"pending", "stopping", "shutting-down"}


def _html(title: str, heading: str, body: str, refresh: bool) -> str:
    meta = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">' if refresh else ""
    spinner = (
        '<div class="s"></div>'
        if refresh
        else '<div class="s" style="animation:none;border-top-color:#1f6feb"></div>'
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{meta}
<title>{title}</title><style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;
background:#0a0e14;color:#e6edf3;
font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.c{{max-width:34rem;padding:2.5rem 1.5rem;text-align:center}}
.s{{width:2.75rem;height:2.75rem;margin:0 auto 1.75rem;border:3px solid #21262d;
border-top-color:#1f6feb;border-radius:50%;animation:r 1s linear infinite}}
@keyframes r{{to{{transform:rotate(360deg)}}}}
h1{{margin:0 0 .75rem;font-size:1.5rem;font-weight:600;letter-spacing:-.01em}}
p{{margin:0 0 .5rem;color:#9198a1}}
.n{{margin-top:1.75rem;font-size:.8125rem;color:#6e7681}}
</style></head><body><div class="c">{spinner}<h1>{heading}</h1>{body}</div></body></html>"""


def _resp(status: int, body: str, headers: dict[str, str] | None = None) -> dict:
    base = {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store, no-cache, must-revalidate",
    }
    if headers:
        base.update(headers)
    return {"statusCode": status, "headers": base, "body": body}


def _waking_page(*, instance_running: bool) -> dict:
    # Once EC2 itself is confirmed running, say so explicitly — "starting the
    # GPU server" is no longer true at that point, and telling a hiring
    # manager the *actual* current step (models loading, not server boot)
    # is what makes the wait legible instead of a generic spinner.
    detail = (
        "<p>The GPU instance is up and the AI model stack is loading.</p>"
        if instance_running
        else "<p>Starting the GPU server, then loading the AI model stack.</p>"
    )
    return _resp(
        200,
        _html(
            "MAGIK — starting up",
            "Thanks for checking out MAGIK",
            detail
            + "<p>This live demo scales its GPU server to zero when idle to keep hosting "
            "costs down, so the first visit takes about a minute to spin back up.</p>"
            '<p class="n">You\'ll be redirected to the sign-in page automatically the '
            "moment it's ready — no need to click or refresh anything.</p>",
            refresh=True,
        ),
    )


def _stuck_page(elapsed_minutes: float) -> dict:
    return _resp(
        200,
        _html(
            "MAGIK — still starting",
            "This is taking longer than usual",
            f"<p>The GPU instance has been running for about {elapsed_minutes:.0f} minutes "
            "but the app still isn't answering — normally this takes under a minute.</p>"
            "<p>You don't need to do anything: this page keeps checking automatically "
            "and will redirect you to sign-in the moment it recovers.</p>"
            '<p class="n">If this keeps happening after a few more minutes, please try '
            "again a bit later.</p>",
            refresh=True,
        ),
    )


def _error_page(msg: str, status: int = 503) -> dict:
    return _resp(
        status,
        _html(
            "MAGIK — unavailable",
            "Demo temporarily unavailable",
            f"<p>{msg}</p>"
            '<p class="n">If this persists, the instance may need attention.</p>',
            refresh=False,
        ),
    )


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


def handler(event, context):  # noqa: ARG001 - Lambda signature
    if not APP_URL:
        log.error("APP_URL is not configured")
        return _error_page("Gateway is not fully configured.", 500)

    try:
        instance_id, state, launch_time = _find_instance()
    except ClientError as exc:
        log.exception("describe_instances failed")
        return _error_page(f"Could not query the demo instance ({exc.response['Error']['Code']}).")

    if instance_id is None:
        log.error("no instance tagged Name=%s", INSTANCE_TAG)
        return _error_page("The demo instance could not be found.")

    log.info("instance %s state=%s", instance_id, state)

    if state == "stopped":
        try:
            ec2.start_instances(InstanceIds=[instance_id])
            log.info("start_instances issued for %s", instance_id)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            # IncorrectInstanceState = someone/something else already started it
            # between our describe and start. Harmless; fall through to the page.
            if code not in ("IncorrectInstanceState",):
                log.exception("start_instances failed")
                if code == "InsufficientInstanceCapacity":
                    return _error_page(
                        "AWS currently has no capacity for this GPU instance type. "
                        "Please try again in a few minutes."
                    )
                return _error_page(f"Could not start the demo instance ({code}).")
        return _waking_page(instance_running=False)

    if state in _WAKING_STATES:
        return _waking_page(instance_running=False)

    if state == "running" and _is_healthy():
        log.info("healthy -> redirecting to %s", APP_URL)
        _push_kuma("up", "woken_by_visitor")  # real visitor confirmed the app is genuinely serving
        return _resp(302, "", {"location": APP_URL})

    # running but not yet answering /health — models still loading, OR the
    # app is genuinely stuck (crashed, a dependency it needs is unreachable,
    # a bad deploy). Previously both looked identical to the visitor forever;
    # now a real elapsed-time check tells them apart.
    elapsed_minutes = 0.0
    if launch_time is not None:
        elapsed_minutes = (datetime.now(timezone.utc) - launch_time).total_seconds() / 60

    if elapsed_minutes >= STUCK_MINUTES:
        log.warning(
            "instance %s running %.1fm without going healthy (threshold %.1fm)",
            instance_id, elapsed_minutes, STUCK_MINUTES,
        )
        _push_kuma("down", f"unhealthy_after_{elapsed_minutes:.0f}m")
        return _stuck_page(elapsed_minutes)

    return _waking_page(instance_running=True)
