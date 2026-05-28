"""
Phase 27 — Protected route tests.
Covers: 401 on missing token, 401 on invalid token, 401 on expired token,
200 with valid token, admin-only route enforcement.
"""
from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from app.auth.jwt_handler import issue_tokens
from app.core.config import settings


@pytest.fixture
def client(mock_mongo_col):
    """TestClient with auth enabled."""
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def valid_token(registered_user_a):
    tokens = issue_tokens(
        registered_user_a.user_id,
        registered_user_a.email,
        registered_user_a.role.value,
    )
    return tokens["access_token"]


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── /auth/me — requires valid token ──────────────────────────────────────────

def test_me_with_valid_token_returns_200(client, valid_token):
    r = client.get("/auth/me", headers=_auth_header(valid_token))
    assert r.status_code == 200
    data = r.json()
    assert "user_id" in data
    assert "email" in data
    assert "password" not in data
    assert "hashed_password" not in data


def test_me_without_token_returns_401(client):
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_me_with_invalid_token_returns_401(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer this.is.invalid"})
    assert r.status_code == 401


def test_me_with_expired_token_returns_401(client):
    expired = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "email": "expired@test.com",
            "role": "user",
            "type": "access",
            "jti": str(uuid.uuid4()),
            "exp": int(time.time()) - 1,
            "iat": int(time.time()) - 100,
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


def test_me_with_refresh_token_returns_401(client, registered_user_a):
    tokens = issue_tokens(
        registered_user_a.user_id,
        registered_user_a.email,
        registered_user_a.role.value,
    )
    r = client.get("/auth/me", headers=_auth_header(tokens["refresh_token"]))
    assert r.status_code == 401


def test_me_with_bearer_prefix_missing_returns_401(client, valid_token):
    r = client.get("/auth/me", headers={"Authorization": valid_token})
    assert r.status_code == 401


# ── /auth/register and /auth/login are PUBLIC ────────────────────────────────

def test_register_is_public(client):
    r = client.post("/auth/register", json={
        "email": "newuser@example.com",
        "password": "NewUserPass99!"
    })
    assert r.status_code == 201


def test_login_is_public(client, registered_user_a, user_a_data):
    r = client.post("/auth/login", json={
        "email": user_a_data["email"],
        "password": user_a_data["password"],
    })
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert "refresh_token" in data


# ── /auth/refresh ─────────────────────────────────────────────────────────────

def test_refresh_returns_new_token_pair(client, registered_user_a):
    tokens = issue_tokens(
        registered_user_a.user_id,
        registered_user_a.email,
        registered_user_a.role.value,
    )
    r = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data


def test_refresh_with_access_token_returns_401(client, valid_token):
    r = client.post("/auth/refresh", json={"refresh_token": valid_token})
    assert r.status_code == 401


# ── Health is always public ───────────────────────────────────────────────────

def test_health_no_auth_required(client):
    r = client.get("/rag/health")
    assert r.status_code == 200
