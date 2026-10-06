"""Idle-stop must not stop the box mid quality-report — and must not be pinned
on forever by a report that died.

deploy/aws/scripts/run_quality_report.sh runs from an SSM *session*, which
neither _has_inflight_ssm (send-command only) nor _runner_busy (GitHub runner
only) can see, and its judge phase produces almost no NetworkIn. The
`magik:eval-busy-until` lease is the guard. boto3/botocore are stubbed the same
way test_wake_gateway_handler.py does, for the same reason (the handler ships
standalone to Lambda's runtime; this repo does not depend on boto3).
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

HANDLER = Path("deploy/aws/lambda/idle_stop/handler.py")


class _FakeClientError(Exception):
    def __init__(self, code: str = "Boom"):
        self.response = {"Error": {"Code": code}}


class _FakeEC2:
    def __init__(self, tags=None, fail=False):
        self.tags = tags or []
        self.fail = fail
        self.stopped = []

    def describe_instances(self, **kwargs):
        if self.fail:
            raise _FakeClientError()
        return {"Reservations": [{"Instances": [{"InstanceId": "i-1", "Tags": self.tags}]}]}

    def stop_instances(self, InstanceIds):
        self.stopped.extend(InstanceIds)


@pytest.fixture
def idle_stop(monkeypatch):
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: object()
    fake_botocore = types.ModuleType("botocore")
    fake_config = types.ModuleType("botocore.config")
    fake_config.Config = lambda **kw: None
    fake_exc = types.ModuleType("botocore.exceptions")
    fake_exc.ClientError = _FakeClientError
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exc)

    spec = importlib.util.spec_from_file_location("idle_stop_handler_under_test", HANDLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tag(value):
    return [{"Key": "Name", "Value": "magik-prod"}, {"Key": "magik:eval-busy-until", "Value": value}]


def test_live_lease_is_busy(idle_stop):
    idle_stop.ec2 = _FakeEC2(_tag(str(int(time.time()) + 600)))
    assert idle_stop._eval_busy("i-1") is True


def test_expired_lease_is_not_busy(idle_stop):
    """A script that died without cleanup stops protecting the box."""
    idle_stop.ec2 = _FakeEC2(_tag(str(int(time.time()) - 1)))
    assert idle_stop._eval_busy("i-1") is False


def test_lease_too_far_ahead_is_ignored(idle_stop):
    idle_stop.ec2 = _FakeEC2(_tag(str(int(time.time()) + 10 * 86400)))
    assert idle_stop._eval_busy("i-1") is False


@pytest.mark.parametrize("tags", [[], _tag("not-a-number")])
def test_missing_or_malformed_tag_is_not_busy(idle_stop, tags):
    idle_stop.ec2 = _FakeEC2(tags)
    assert idle_stop._eval_busy("i-1") is False


def test_api_error_fails_open(idle_stop):
    idle_stop.ec2 = _FakeEC2(fail=True)
    assert idle_stop._eval_busy("i-1") is False


def test_handler_skips_stop_while_lease_held(idle_stop, monkeypatch):
    ec2 = _FakeEC2(_tag(str(int(time.time()) + 600)))
    idle_stop.ec2 = ec2
    monkeypatch.setattr(idle_stop, "_find_running_instance", lambda: ("i-1", None))
    monkeypatch.setattr(idle_stop, "_push_kuma_up_if_healthy", lambda: None)
    monkeypatch.setattr(idle_stop, "_uptime_minutes", lambda *a: 120.0)
    monkeypatch.setattr(idle_stop, "_has_inflight_ssm", lambda _: False)
    monkeypatch.setattr(idle_stop, "_runner_busy", lambda: False)
    monkeypatch.setattr(idle_stop, "_is_idle", lambda _: True)

    result = idle_stop.handler({}, None)
    assert result["reason"] == "quality report running"
    assert ec2.stopped == []
