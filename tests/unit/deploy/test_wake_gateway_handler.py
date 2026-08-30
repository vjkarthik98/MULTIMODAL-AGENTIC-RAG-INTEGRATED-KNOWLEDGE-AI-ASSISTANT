"""Regression coverage for the v1.0.1 wake-gateway fix.

deploy/aws/lambda/wake_gateway/handler.py is deployed standalone into AWS
Lambda's managed Python 3.12 runtime, which provides boto3/botocore itself —
this repo does not declare either as a project dependency, and never should,
since the handler is never imported by app/. boto3 and botocore are therefore
stubbed into sys.modules before the handler is imported, not mocked on a real
import: that keeps this test hermetic (no AWS calls, no dependency this repo
doesn't actually ship) while still exercising the real handler code.

Root cause under test (see the handler's own "Human-initiated wake only"
docstring and CHANGELOG.md [1.0.1]): the initial, non-poll request used to
compute real status — including calling StartInstances — before any of the
page's own JS had run, so a bare HTTP GET (bot, scanner, certificate-
transparency crawler) woke the production instance identically to a real
visitor. Production evidence: StartInstances fired roughly every two hours,
around the clock, for a full day.

The invariant these tests defend is stronger than "a browser was here": only
an explicit human click on the Start button may start the instance. Every
passive path — page load, status poll — is hard-wired read-only. The tests
below assert that from both directions: no automatic path can start it, and
the real human path still can.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import types
from datetime import datetime, timezone

import pytest


class _FakeClientError(Exception):
    def __init__(self, code: str):
        self.response = {"Error": {"Code": code}}


class _FakeEC2:
    """Minimal stand-in for the boto3 EC2 client surface this handler uses."""

    def __init__(self, initial_state: str = "stopped", fail_with: str | None = None):
        self.state = initial_state
        self.launch_time: datetime | None = None
        self.start_calls = 0
        self.fail_with = fail_with  # e.g. "InsufficientInstanceCapacity"

    def describe_instances(self, Filters=None):  # noqa: N803 - matches boto3's kwarg casing
        inst = {"InstanceId": "i-fake0000000000000", "State": {"Name": self.state}}
        if self.launch_time is not None:
            inst["LaunchTime"] = self.launch_time
        return {"Reservations": [{"Instances": [inst]}]}

    def start_instances(self, InstanceIds=None):  # noqa: N803
        self.start_calls += 1
        if self.fail_with:
            raise _FakeClientError(self.fail_with)
        self.state = "pending"
        self.launch_time = datetime.now(timezone.utc)


@pytest.fixture
def wake_handler(monkeypatch):
    """Import the Lambda handler with boto3/botocore stubbed and a fresh
    _FakeEC2 wired in as `handler.ec2`."""
    state: dict = {"ec2": _FakeEC2()}

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: state["ec2"]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    fake_botocore = types.ModuleType("botocore")
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = lambda **kw: None
    fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
    fake_botocore_exceptions.ClientError = _FakeClientError
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore_exceptions)

    monkeypatch.setenv("APP_URL", "https://magik.vk-ai.online")
    monkeypatch.setenv("WAKE_TOKEN_SECRET", "test-secret")
    monkeypatch.setenv("WAKE_TOKEN_MAX_AGE_S", "90")
    monkeypatch.syspath_prepend("deploy/aws/lambda/wake_gateway")

    sys.modules.pop("handler", None)
    module = importlib.import_module("handler")
    module.ec2 = state["ec2"]
    yield module
    sys.modules.pop("handler", None)


def _event(qs: dict | None = None) -> dict:
    return {"queryStringParameters": qs}


def _page_token(html_body: str) -> str:
    marker = 'var TOKEN = "'
    start = html_body.index(marker) + len(marker)
    return html_body[start : html_body.index('"', start)]


def _click_start(handler_mod) -> dict:
    """Do exactly what a human clicking the Start button does: load the page,
    then issue ?wake=1 with the token that page carried."""
    page = handler_mod.handler(_event(None), None)
    token = _page_token(page["body"])
    resp = handler_mod.handler(_event({"wake": "1", "t": token}), None)
    return json.loads(resp["body"])


# ── Nothing automatic may ever start the instance ──────────────────────────


def test_bare_http_get_never_wakes_the_instance(wake_handler):
    """The exact production bug: a raw GET with no query params — what any
    bot, scanner, or curl sends — must never call StartInstances."""
    wake_handler.ec2.state = "stopped"
    resp = wake_handler.handler(_event(None), None)
    assert wake_handler.ec2.start_calls == 0
    assert resp["statusCode"] == 200


def test_asleep_page_offers_a_button_and_does_not_self_poll(wake_handler):
    """A stopped instance is reported honestly as "asleep" with retry=False,
    so the page waits for a person instead of spinning on a "starting up"
    message that isn't true."""
    wake_handler.ec2.state = "stopped"
    body = wake_handler.handler(_event(None), None)["body"]
    assert 'id="wake"' in body  # the Start button exists in the shell
    assert '"state": "asleep"' in body or '"state":"asleep"' in body
    assert wake_handler.ec2.start_calls == 0


def test_status_poll_never_wakes_the_instance(wake_handler):
    """?check=1 is the page's own background poll. Even carrying a perfectly
    valid page token it is read-only — polling is not consent."""
    wake_handler.ec2.state = "stopped"
    token = _page_token(wake_handler.handler(_event(None), None)["body"])
    resp = wake_handler.handler(_event({"check": "1", "t": token}), None)
    assert wake_handler.ec2.start_calls == 0
    assert json.loads(resp["body"])["state"] == "asleep"


def test_headless_browser_that_renders_and_polls_still_cannot_wake_it(wake_handler):
    """The gap in the first attempt at this fix: something that executes the
    page's JS and polls like a real browser, but never clicks, is still not a
    human and still must not start anything."""
    wake_handler.ec2.state = "stopped"
    token = _page_token(wake_handler.handler(_event(None), None)["body"])
    for _ in range(10):
        resp = wake_handler.handler(_event({"check": "1", "t": token}), None)
        body = json.loads(resp["body"])
        token = body["token"]
    assert wake_handler.ec2.start_calls == 0


def test_wake_without_a_page_token_is_refused(wake_handler):
    """?wake=1 is public and this repo is public, so the endpoint itself is
    discoverable. It still requires a token from a page we actually served."""
    wake_handler.ec2.state = "stopped"
    resp = wake_handler.handler(_event({"wake": "1"}), None)
    assert wake_handler.ec2.start_calls == 0
    assert json.loads(resp["body"])["state"] == "asleep"


def test_expired_page_token_cannot_wake_it(wake_handler, monkeypatch):
    """A scraped/bookmarked token replayed later — the shape of the observed
    ~2-hourly false wakes — is rejected once stale."""
    monkeypatch.setenv("WAKE_TOKEN_MAX_AGE_S", "1")
    importlib.reload(wake_handler)
    wake_handler.ec2.state = "stopped"
    stale = wake_handler._issue_token(wake_handler.TOKEN_PAGE)
    time.sleep(1.2)
    wake_handler.handler(_event({"wake": "1", "t": stale}), None)
    assert wake_handler.ec2.start_calls == 0


def test_page_token_cannot_be_replayed_as_a_wake_grant(wake_handler):
    """The token kind is inside the signed payload, so relabeling a page
    token as a wake grant fails the signature rather than escalating."""
    wake_handler.ec2.state = "stopped"
    page = wake_handler._issue_token(wake_handler.TOKEN_PAGE)
    forged = page.replace("page.", "wake.", 1)
    assert wake_handler._valid_token(forged, wake_handler.TOKEN_WAKE) is False
    wake_handler.handler(_event({"check": "1", "w": forged}), None)
    assert wake_handler.ec2.start_calls == 0


def test_missing_secret_fails_closed_not_open(wake_handler, monkeypatch):
    """WAKE_TOKEN_SECRET unset (an old or misconfigured deploy) must refuse
    every start, never silently revert to the pre-fix always-allow behavior."""
    monkeypatch.setenv("WAKE_TOKEN_SECRET", "")
    importlib.reload(wake_handler)
    wake_handler.ec2.state = "stopped"
    token = wake_handler._issue_token(wake_handler.TOKEN_PAGE)
    wake_handler.handler(_event({"wake": "1", "t": token}), None)
    assert wake_handler.ec2.start_calls == 0


# ── A real human click still works ─────────────────────────────────────────


def test_human_click_starts_the_instance_exactly_once(wake_handler):
    wake_handler.ec2.state = "stopped"
    body = _click_start(wake_handler)
    assert wake_handler.ec2.start_calls == 1
    assert body["state"] == "waking"
    assert body["wake"], "a wake grant must be issued so retries can proceed"


def test_capacity_retry_after_a_human_click_keeps_retrying(wake_handler):
    """Reproduces the live 2026-08-30 incident: AWS returned
    InsufficientInstanceCapacity for ~36 minutes while the instance stayed
    "stopped". The human already consented, so those polls must keep
    retrying — without the wake grant the click would silently do nothing."""
    wake_handler.ec2 = _FakeEC2(initial_state="stopped", fail_with="InsufficientInstanceCapacity")
    body = _click_start(wake_handler)
    assert body["state"] == "capacity"
    grant = body["wake"]

    for _ in range(10):
        resp = wake_handler.handler(_event({"check": "1", "w": grant}), None)
        body = json.loads(resp["body"])
        assert body["state"] == "capacity"
        grant = body["wake"]

    assert wake_handler.ec2.start_calls == 11  # 1 click + 10 authorized retries


def test_running_instance_needs_no_click_and_is_never_started_again(wake_handler):
    """If the box is already up (someone else woke it, or a deploy did), the
    visitor gets the normal loading/ready flow with no button and no start."""
    wake_handler.ec2.state = "running"
    resp = wake_handler.handler(_event({"check": "1"}), None)
    assert wake_handler.ec2.start_calls == 0
    assert json.loads(resp["body"])["state"] in {"loading", "ready", "stuck"}
