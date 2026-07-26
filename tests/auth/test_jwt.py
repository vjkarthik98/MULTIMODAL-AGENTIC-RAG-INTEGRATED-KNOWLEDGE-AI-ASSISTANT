"""
Phase 27 — JWT token tests.
Covers: issue, verify, expiry, refresh, wrong type, tampered signature.
"""
from __future__ import annotations

import time
import uuid

import pytest
from jose import jwt

from app.auth.jwt_handler import issue_tokens, refresh_access_token, verify_token
from app.core.config import settings


USER_ID = str(uuid.uuid4())
EMAIL = "jwt_test@example.com"
ROLE = "user"


# ── Issue ─────────────────────────────────────────────────────────────────────

def test_issue_tokens_returns_pair():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


def test_access_token_payload():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    payload = verify_token(tokens["access_token"], expected_type="access")
    assert payload["sub"] == USER_ID
    assert payload["email"] == EMAIL
    assert payload["role"] == ROLE
    assert payload["type"] == "access"
    assert "jti" in payload
    assert "exp" in payload
    assert "iat" in payload


def test_refresh_token_payload():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    payload = verify_token(tokens["refresh_token"], expected_type="refresh")
    assert payload["sub"] == USER_ID
    assert payload["type"] == "refresh"


def test_access_and_refresh_have_different_jti():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    access_p = verify_token(tokens["access_token"], expected_type="access")
    refresh_p = verify_token(tokens["refresh_token"], expected_type="refresh")
    assert access_p["jti"] != refresh_p["jti"]


# ── Verify ────────────────────────────────────────────────────────────────────

def test_verify_wrong_type_raises():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    with pytest.raises(ValueError, match="Expected access token"):
        verify_token(tokens["refresh_token"], expected_type="access")


def test_verify_tampered_token_raises():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    tampered = tokens["access_token"][:-5] + "XXXXX"
    with pytest.raises(ValueError):
        verify_token(tampered)


def test_verify_garbage_raises():
    with pytest.raises(ValueError):
        verify_token("this.is.not.a.jwt")


def test_verify_wrong_secret_raises():
    bad_token = jwt.encode(
        {"sub": USER_ID, "email": EMAIL, "role": ROLE, "type": "access",
         "jti": str(uuid.uuid4()), "exp": int(time.time()) + 3600, "iat": int(time.time())},
        "wrong_secret_key_that_is_long_enough_to_pass",
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(ValueError):
        verify_token(bad_token)


def test_verify_expired_token_raises():
    expired_token = jwt.encode(
        {"sub": USER_ID, "email": EMAIL, "role": ROLE, "type": "access",
         "jti": str(uuid.uuid4()), "exp": int(time.time()) - 1, "iat": int(time.time()) - 100},
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(ValueError, match="Invalid or expired"):
        verify_token(expired_token)


# ── Refresh ───────────────────────────────────────────────────────────────────

def test_refresh_issues_new_access_token():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    new_tokens = refresh_access_token(tokens["refresh_token"])
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens

    payload = verify_token(new_tokens["access_token"], expected_type="access")
    assert payload["sub"] == USER_ID


def test_refresh_with_access_token_raises():
    tokens = issue_tokens(USER_ID, EMAIL, ROLE)
    with pytest.raises(ValueError):
        refresh_access_token(tokens["access_token"])


def test_refresh_with_garbage_raises():
    with pytest.raises(ValueError):
        refresh_access_token("not.a.valid.token.at.all")
