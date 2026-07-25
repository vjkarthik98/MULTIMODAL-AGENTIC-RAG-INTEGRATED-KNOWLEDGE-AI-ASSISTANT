"""
Phase 27 — GDPR purge tests.
Covers: /auth/me DELETE purges data from all stores, unauthenticated purge blocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_handler import issue_tokens


@pytest.fixture
def client(mock_mongo_col):
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


def test_gdpr_delete_me_requires_auth(client):
    r = client.delete("/auth/me")
    assert r.status_code == 401


def test_gdpr_delete_me_with_valid_token_returns_200(client, valid_token):
    with patch("app.memory.memory_manager.MemoryManager.gdpr_purge", return_value=None), \
         patch("app.core.infra_registry.InfraRegistry.get_bm25", return_value=None):
        r = client.delete("/auth/me", headers=_auth_header(valid_token))
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "deleted" in data["message"].lower()


def test_gdpr_delete_me_calls_memory_purge(client, registered_user_a, valid_token):
    """Redis purge_user must be called with the authenticated user's user_id."""
    purge_calls = []

    mock_redis_mem = MagicMock()
    mock_redis_mem.purge_user = lambda uid: purge_calls.append(uid)

    # Patch token_blacklist's _redis() so JWT verification uses a clean stub
    # (prevents MagicMock.__int__=1 from being mistaken as current_gen=1)
    mock_bl_redis = MagicMock()
    mock_bl_redis.get.return_value = None   # TOKEN_GEN key absent → gen=0
    mock_bl_redis.exists.return_value = 0   # no revoked JTIs

    with patch("app.auth.token_blacklist._redis", return_value=mock_bl_redis), \
         patch("app.core.infra_registry.infra.get_memory", return_value=mock_redis_mem), \
         patch("app.core.infra_registry.InfraRegistry.get_bm25", return_value=None):
        r = client.delete("/auth/me", headers=_auth_header(valid_token))
        assert r.status_code == 200

    assert len(purge_calls) == 1
    assert purge_calls[0] == registered_user_a.user_id


def test_gdpr_delete_me_deactivates_account(client, registered_user_a, valid_token, mock_mongo_col):
    with patch("app.memory.memory_manager.MemoryManager.gdpr_purge", return_value=None), \
         patch("app.core.infra_registry.InfraRegistry.get_bm25", return_value=None):
        r = client.delete("/auth/me", headers=_auth_header(valid_token))
        assert r.status_code == 200

    doc = mock_mongo_col.find_one({"user_id": registered_user_a.user_id})
    assert doc is None  # GDPR hard-deletes the user document
