"""MAGIK wake gateway — public entry point for the live demo.

The GPU instance (g6e.xlarge / L40S, ~$1.86/hr) is stopped by default. This
Lambda is the always-on front door that makes that invisible to a visitor:

    stopped   -> StartInstances, show a self-refreshing "warming up" page
    booting   -> same page (the instance is up but /health isn't answering yet)
    healthy   -> 302 to the app

Deployed behind a Lambda Function URL (and later CloudFront + a custom domain;
neither changes this code). Runs outside a VPC so it can reach the instance's
public endpoint directly.

Cost note: this is the piece that makes scale-to-zero viable. Always-on would
be ~$1,340/mo; stopped-by-default plus this gateway is ~$12/mo fixed plus a
few dollars per active hour.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request

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


def _waking_page() -> dict:
    return _resp(
        200,
        _html(
            "MAGIK — starting up",
            "Waking up MAGIK",
            "<p>Starting the GPU server and loading the multimodal model stack.</p>"
            "<p>This takes about a minute on the first visit.</p>"
            '<p class="n">This page refreshes automatically — no need to reload.</p>',
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


def _find_instance() -> tuple[str | None, str | None]:
    """Return (instance_id, state) for the tagged instance, or (None, None)."""
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
            return inst["InstanceId"], inst["State"]["Name"]
    return None, None


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
        instance_id, state = _find_instance()
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
        return _waking_page()

    if state in _WAKING_STATES:
        return _waking_page()

    if state == "running" and _is_healthy():
        log.info("healthy -> redirecting to %s", APP_URL)
        return _resp(302, "", {"location": APP_URL})

    # running but not yet answering /health — models still loading
    return _waking_page()
