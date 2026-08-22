"""
Demo-account login tests.

The one fixed, publicly-shared demo login (settings.DEMO_ACCOUNT_EMAIL) must
never be handed an email OTP challenge — nobody holding those credentials can
read that mailbox. The bypass therefore has to hold even when the Mongo
`is_demo` flag was never written (fresh environment, or an account that
predates the flag), which is exactly what broke it before.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.auth.models import UserInDB, UserPublic, UserRole
from app.auth.service import AuthService, _hash_password, is_demo_account
from app.core.config import settings

DEMO_PASSWORD = "Demo@2026"  # pragma: allowlist secret — public demo credential


@pytest.fixture
def demo_email():
    return (settings.DEMO_ACCOUNT_EMAIL or "magikaiassistant@gmail.com").strip().lower()


@pytest.fixture
def unflagged_demo_user(mock_mongo_col, demo_email):
    """The demo account as it exists on an environment where the seed script
    was never run: right email + password, no is_demo field at all."""
    doc = UserInDB(
        email=demo_email,
        hashed_password=_hash_password(DEMO_PASSWORD),
        auth_providers=["email"],
        is_active=True,
    ).model_dump()
    doc.pop("is_demo")  # the pre-flag document shape
    mock_mongo_col.insert_one(doc)
    return doc


@pytest.fixture
def client(mock_mongo_col):
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _public(email: str, *, is_demo: bool) -> UserPublic:
    from datetime import datetime, timezone

    return UserPublic(
        user_id="u1",
        email=email,
        role=UserRole.USER,
        is_active=True,
        is_demo=is_demo,
        created_at=datetime.now(timezone.utc),
    )


# ── is_demo_account() ─────────────────────────────────────────────────────────

def test_configured_email_is_demo_without_the_flag(demo_email):
    assert is_demo_account(_public(demo_email, is_demo=False)) is True


def test_configured_email_matched_case_insensitively(demo_email):
    assert is_demo_account(_public(demo_email.upper(), is_demo=False)) is True


def test_stored_flag_still_wins_for_any_other_email():
    assert is_demo_account(_public("someone@example.com", is_demo=True)) is True


def test_ordinary_account_is_not_demo():
    assert is_demo_account(_public("alice@example.com", is_demo=False)) is False


def test_bypass_disabled_when_setting_is_blank(monkeypatch, demo_email):
    monkeypatch.setattr(settings, "DEMO_ACCOUNT_EMAIL", "")
    assert is_demo_account(_public(demo_email, is_demo=False)) is False


# ── POST /auth/login ──────────────────────────────────────────────────────────

def test_demo_login_skips_otp_without_the_flag(client, unflagged_demo_user, demo_email):
    with patch("app.auth.email_service.send_otp_email") as send:
        r = client.post("/auth/login", json={"email": demo_email, "password": DEMO_PASSWORD})

    assert r.status_code == 200
    body = r.json()
    assert body.get("otp_required") is not True
    assert body["access_token"] and body["refresh_token"]
    send.assert_not_called()


def test_demo_login_backfills_the_stored_flag(
    client, unflagged_demo_user, demo_email, mock_mongo_col
):
    with patch("app.auth.email_service.send_otp_email"):
        client.post("/auth/login", json={"email": demo_email, "password": DEMO_PASSWORD})

    assert mock_mongo_col.find_one({"email": demo_email})["is_demo"] is True


def test_demo_login_survives_a_failed_backfill(client, unflagged_demo_user, demo_email):
    """A Mongo hiccup on the self-healing write must not cost the recruiter
    their login — the bypass never depended on that write succeeding."""
    with patch.object(AuthService, "mark_demo", side_effect=RuntimeError("mongo down")):
        r = client.post("/auth/login", json={"email": demo_email, "password": DEMO_PASSWORD})

    assert r.status_code == 200
    assert r.json()["access_token"]


def test_demo_login_rejects_the_wrong_password(client, unflagged_demo_user, demo_email):
    r = client.post("/auth/login", json={"email": demo_email, "password": "NotThePassword1!"})
    assert r.status_code == 401


def test_ordinary_login_still_gets_an_otp_challenge(client, registered_user_a, user_a_data):
    with patch("app.auth.email_service.send_otp_email"), patch("app.auth.otp_store.store_otp"):
        r = client.post(
            "/auth/login",
            json={"email": user_a_data["email"], "password": user_a_data["password"]},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["otp_required"] is True
    assert "access_token" not in body
