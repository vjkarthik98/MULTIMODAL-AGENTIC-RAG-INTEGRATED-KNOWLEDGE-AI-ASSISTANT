"""MAGIK idle-stop — stops the GPU instance when nobody is using it.

Invoked on a schedule (EventBridge, every 5 minutes). Decides from CloudWatch
NetworkIn over a trailing window.

Two guards matter more than the threshold itself, and both come from real
failure modes rather than theory:

  * MIN_UPTIME_MINUTES — a freshly woken box spends several minutes loading
    ~18GB of models off EBS with almost no network traffic. Without this guard
    the very first idle check would stop the instance a visitor is actively
    waiting on, and the wake gateway would start it again: a wake/stop loop
    that burns money and never serves anyone.
  * in-flight SSM commands — cd.yml deploys via SSM and a deploy can run 20+
    minutes (image pull + model load) with low network traffic. Stopping
    mid-deploy would corrupt a release.
  * a busy self-hosted GitHub Actions runner — Tier-2 eval (`eval-gate.yml`)
    runs `[self-hosted, gpu]`, i.e. on this same box. A GPU-bound eval job is
    CPU/GPU-heavy but produces almost no *external* NetworkIn, so without this
    guard the CloudWatch signal alone would look idle mid-eval and this Lambda
    would stop the instance out from under a running Tier-2 suite — corrupting
    the run and knocking the runner itself offline until the box wakes again.

Deliberately conservative: the cost of one extra idle interval (~$0.15) is far
below the cost of stopping a live session, a running deploy, or a running eval.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

INSTANCE_TAG = os.environ.get("EC2_INSTANCE_TAG", "magik-prod")
IDLE_MINUTES = int(os.environ.get("IDLE_MINUTES", "20"))
MIN_UPTIME_MINUTES = int(os.environ.get("MIN_UPTIME_MINUTES", "15"))
# Bytes over a 5-minute period. Background chatter (SSM agent heartbeat, OS
# housekeeping) typically sits well under 1MB per period; a single active user
# session is orders of magnitude above it.
NETWORK_IN_THRESHOLD_BYTES = int(os.environ.get("NETWORK_IN_THRESHOLD_BYTES", "1000000"))
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

# GitHub Actions self-hosted runner "busy" check. The token needs repository
# Administration:read (fine-grained) or `repo` scope (classic) — the same
# minimum GitHub requires for the "list self-hosted runners" endpoint. Stored
# in SSM the same way /magik/ghcr_pat is: never touches a GH Actions log.
GITHUB_REPO = os.environ.get("GITHUB_REPO", "vjkarthik98/multimodal-rag-assistant")
GITHUB_RUNNER_LABEL = os.environ.get("GITHUB_RUNNER_LABEL", "gpu")
GITHUB_TOKEN_PARAM = os.environ.get("GITHUB_TOKEN_PARAM", "/magik/github_actions_pat")

_cfg = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 3})
ec2 = boto3.client("ec2", config=_cfg)
cw = boto3.client("cloudwatch", config=_cfg)
ssm = boto3.client("ssm", config=_cfg)


def _find_running_instance() -> tuple[str | None, datetime | None]:
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_TAG]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    for reservation in resp.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            return inst["InstanceId"], inst.get("LaunchTime")
    return None, None


def _uptime_minutes(instance_id: str, launch_time: datetime | None) -> float:
    """Minutes since the instance last entered the running state.

    LaunchTime is the ORIGINAL launch, not the most recent start, so it is
    useless for a stop/start box. The last state transition is the real signal;
    LaunchTime is only a fallback if that is unavailable.
    """
    try:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        # e.g. "2026-07-29 09:12:33 GMT"
        raw = (inst.get("StateTransitionReason") or "").strip()
        if "(" in raw and ")" in raw:
            stamp = raw[raw.index("(") + 1 : raw.rindex(")")].replace(" GMT", "")
            started = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - started).total_seconds() / 60
    except (ClientError, ValueError, KeyError, IndexError) as exc:
        log.warning("could not derive state-transition time: %s", exc)

    if launch_time:
        return (datetime.now(timezone.utc) - launch_time).total_seconds() / 60
    return 0.0


def _has_inflight_ssm(instance_id: str) -> bool:
    try:
        resp = ssm.list_commands(InstanceId=instance_id, MaxResults=10)
        for cmd in resp.get("Commands", []):
            if cmd.get("Status") in ("Pending", "InProgress", "Delayed"):
                log.info("in-flight SSM command %s (%s)", cmd.get("CommandId"), cmd.get("Status"))
                return True
    except ClientError as exc:
        # Fail SAFE: if we cannot tell, assume something is running and do not stop.
        log.warning("list_commands failed, assuming busy: %s", exc)
        return True
    return False


def _runner_busy() -> bool:
    """True if the self-hosted GPU runner is currently executing a job.

    Absence of a usable token or an unreachable API is treated as busy — same
    fail-safe posture as the other guards: a missed stop costs ~$0.15, a wrong
    one can corrupt a running eval.
    """
    try:
        token = ssm.get_parameter(Name=GITHUB_TOKEN_PARAM, WithDecryption=True)["Parameter"]["Value"]
    except ClientError as exc:
        log.warning("could not read %s, assuming runner busy: %s", GITHUB_TOKEN_PARAM, exc)
        return True

    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/runners",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "magik-idle-stop",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.load(r)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        log.warning("GitHub runners API call failed, assuming busy: %s", exc)
        return True

    for runner in data.get("runners", []):
        labels = [lbl.get("name") for lbl in runner.get("labels", [])]
        if GITHUB_RUNNER_LABEL in labels and runner.get("busy"):
            log.info("runner %s (labels=%s) is busy", runner.get("name"), labels)
            return True
    return False


def _is_idle(instance_id: str) -> bool:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=IDLE_MINUTES)
    try:
        resp = cw.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="NetworkIn",
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start,
            EndTime=end,
            Period=300,
            Statistics=["Sum"],
        )
    except ClientError as exc:
        log.warning("CloudWatch query failed, assuming active: %s", exc)
        return False

    points = resp.get("Datapoints", [])
    if not points:
        # No metrics yet (just started, or metrics lagging) — do not stop.
        log.info("no NetworkIn datapoints; treating as active")
        return False

    peak = max(p["Sum"] for p in points)
    log.info(
        "NetworkIn over %dm: %d datapoints, peak=%.0f bytes (threshold=%d)",
        IDLE_MINUTES,
        len(points),
        peak,
        NETWORK_IN_THRESHOLD_BYTES,
    )
    return peak < NETWORK_IN_THRESHOLD_BYTES


def handler(event, context):  # noqa: ARG001 - Lambda signature
    instance_id, launch_time = _find_running_instance()
    if not instance_id:
        return {"action": "none", "reason": "no running instance"}

    uptime = _uptime_minutes(instance_id, launch_time)
    if uptime < MIN_UPTIME_MINUTES:
        msg = f"uptime {uptime:.1f}m < {MIN_UPTIME_MINUTES}m guard"
        log.info("skip: %s", msg)
        return {"action": "none", "instance": instance_id, "reason": msg}

    if _has_inflight_ssm(instance_id):
        return {"action": "none", "instance": instance_id, "reason": "SSM command in flight"}

    if _runner_busy():
        return {"action": "none", "instance": instance_id, "reason": "self-hosted runner busy"}

    if not _is_idle(instance_id):
        return {"action": "none", "instance": instance_id, "reason": "recent network activity"}

    if DRY_RUN:
        log.info("DRY_RUN: would stop %s", instance_id)
        return {"action": "dry-run", "instance": instance_id}

    ec2.stop_instances(InstanceIds=[instance_id])
    log.info("STOPPED %s after %.1fm uptime, idle %dm", instance_id, uptime, IDLE_MINUTES)
    return {
        "action": "stopped",
        "instance": instance_id,
        "uptime_minutes": round(uptime, 1),
        "idle_window_minutes": IDLE_MINUTES,
    }
