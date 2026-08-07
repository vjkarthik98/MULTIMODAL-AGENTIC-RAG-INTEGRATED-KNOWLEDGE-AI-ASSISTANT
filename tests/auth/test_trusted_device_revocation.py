"""
Trusted-device revocation on trust-reset events.

A trusted-device token is a standing OTP exemption for one browser, stored
in Redis for 30 days, entirely separate from the JWT blacklist. Revoking
tokens without revoking it produces the exact behaviour that prompted these
tests: the user is signed out on password change, then signs straight back in
with no email code — for the remaining life of the device token, not just
once. Password change, password reset, and logout-all must all clear both.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_handler import issue_tokens


@pytest.fixture
def client(mock_mongo_col):
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def token_for_a(registered_user_a):
    return issue_tokens(
        registered_user_a.user_id,
        registered_user_a.email,
        registered_user_a.role.value,
    )["access_token"]


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_password_change_revokes_trusted_devices(
    client, token_for_a, registered_user_a, user_a_data
):
    with patch("app.auth.otp_store.revoke_device_tokens") as revoke, patch(
        "app.auth.router.revoke_all_user_tokens"
    ):
        r = client.post(
            "/auth/password",
            json={
                "current_password": user_a_data["password"],
                "new_password": "BrandNewPass77!",
            },
            headers=_bearer(token_for_a),
        )

    assert r.status_code == 200
    revoke.assert_called_once_with(registered_user_a.user_id)


def test_logout_all_revokes_trusted_devices(client, token_for_a, registered_user_a):
    with patch("app.auth.otp_store.revoke_device_tokens") as revoke, patch(
        "app.auth.router.revoke_all_user_tokens"
    ):
        r = client.post("/auth/logout-all", headers=_bearer(token_for_a))

    assert r.status_code == 200
    revoke.assert_called_once_with(registered_user_a.user_id)


def test_password_change_survives_redis_being_down(
    client, token_for_a, user_a_data, mock_mongo_col
):
    """Redis unavailable must not fail the password change itself — the
    password is already written by then, so a 500 here would tell the user
    their change failed when it didn't."""
    with patch(
        "app.auth.otp_store.revoke_device_tokens", side_effect=RuntimeError("Redis unavailable")
    ), patch("app.auth.router.revoke_all_user_tokens"):
        r = client.post(
            "/auth/password",
            json={
                "current_password": user_a_data["password"],
                "new_password": "BrandNewPass77!",
            },
            headers=_bearer(token_for_a),
        )

    assert r.status_code == 200


def test_wrong_current_password_revokes_nothing(client, token_for_a):
    with patch("app.auth.otp_store.revoke_device_tokens") as revoke, patch(
        "app.auth.router.revoke_all_user_tokens"
    ) as revoke_jwt:
        r = client.post(
            "/auth/password",
            json={"current_password": "NotMyPassword1!", "new_password": "BrandNewPass77!"},
            headers=_bearer(token_for_a),
        )

    assert r.status_code == 400
    revoke.assert_not_called()
    revoke_jwt.assert_not_called()
