"""
Phase 27 — Admin API tests.
Covers: list users, get user, promote/demote role, activate/deactivate,
GDPR purge, 403 for non-admins, self-protection guards.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.jwt_handler import issue_tokens


@pytest.fixture
def client(mock_mongo_col):
    from app.main import app
    # Patch the admin router's _get_users_col to use our in-memory collection
    with patch("app.auth.admin_router._get_users_col", return_value=mock_mongo_col):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Access control ─────────────────────────────────────────────────────────────

def test_admin_list_users_requires_auth(client):
    r = client.get("/admin/users")
    assert r.status_code == 401


def test_admin_list_users_requires_admin_role(client, token_a):
    r = client.get("/admin/users", headers=_bearer(token_a))
    assert r.status_code == 403


def test_admin_list_users_succeeds_for_admin(client, admin_token):
    r = client.get("/admin/users", headers=_bearer(admin_token))
    assert r.status_code == 200
    data = r.json()
    assert "users" in data
    assert "total" in data


def test_admin_response_never_contains_password(client, admin_token):
    r = client.get("/admin/users", headers=_bearer(admin_token))
    assert r.status_code == 200
    for user in r.json()["users"]:
        assert "password" not in user
        assert "hashed_password" not in user


# ── List users ────────────────────────────────────────────────────────────────

def test_admin_list_users_shows_all_users(
    client, admin_token, registered_user_a, registered_user_b
):
    r = client.get("/admin/users", headers=_bearer(admin_token))
    assert r.status_code == 200
    emails = [u["email"] for u in r.json()["users"]]
    assert registered_user_a.email in emails
    assert registered_user_b.email in emails


def test_admin_list_users_pagination(client, admin_token, registered_user_a):
    r = client.get("/admin/users?skip=0&limit=1", headers=_bearer(admin_token))
    assert r.status_code == 200
    assert len(r.json()["users"]) <= 1


# ── Get single user ───────────────────────────────────────────────────────────

def test_admin_get_user_success(client, admin_token, registered_user_a):
    r = client.get(f"/admin/users/{registered_user_a.user_id}", headers=_bearer(admin_token))
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == registered_user_a.user_id
    assert data["email"] == registered_user_a.email
    assert "stats" in data


def test_admin_get_user_not_found(client, admin_token):
    r = client.get("/admin/users/nonexistent-id-xyz", headers=_bearer(admin_token))
    assert r.status_code == 404


def test_admin_get_user_forbidden_for_regular_user(client, token_a, registered_user_b):
    r = client.get(f"/admin/users/{registered_user_b.user_id}", headers=_bearer(token_a))
    assert r.status_code == 403


# ── Role update ───────────────────────────────────────────────────────────────

def test_admin_promote_user_to_admin(client, admin_token, registered_user_a, mock_mongo_col):
    r = client.patch(
        f"/admin/users/{registered_user_a.user_id}/role",
        json={"role": "admin"},
        headers=_bearer(admin_token),
    )
    assert r.status_code == 200
    assert r.json()["role"] == "admin"

    doc = mock_mongo_col.find_one({"user_id": registered_user_a.user_id})
    assert doc["role"] == "admin"


def test_admin_demote_user(client, admin_token, registered_user_a, mock_mongo_col):
    # First promote
    mock_mongo_col.update_one(
        {"user_id": registered_user_a.user_id}, {"$set": {"role": "admin"}}
    )
    # Then demote
    r = client.patch(
        f"/admin/users/{registered_user_a.user_id}/role",
        json={"role": "user"},
        headers=_bearer(admin_token),
    )
    assert r.status_code == 200
    assert r.json()["role"] == "user"


def test_admin_cannot_demote_own_role(client, admin_user, admin_token):
    r = client.patch(
        f"/admin/users/{admin_user.user_id}/role",
        json={"role": "user"},
        headers=_bearer(admin_token),
    )
    assert r.status_code == 400
    assert "own admin role" in r.json()["detail"]


# ── Status update ─────────────────────────────────────────────────────────────

def test_admin_deactivate_user(client, admin_token, registered_user_a, mock_mongo_col):
    r = client.patch(
        f"/admin/users/{registered_user_a.user_id}/status",
        json={"is_active": False},
        headers=_bearer(admin_token),
    )
    assert r.status_code == 200
    assert r.json()["is_active"] is False

    doc = mock_mongo_col.find_one({"user_id": registered_user_a.user_id})
    assert doc["is_active"] is False


def test_admin_reactivate_user(client, admin_token, registered_user_a, mock_mongo_col):
    mock_mongo_col.update_one(
        {"user_id": registered_user_a.user_id}, {"$set": {"is_active": False}}
    )
    r = client.patch(
        f"/admin/users/{registered_user_a.user_id}/status",
        json={"is_active": True},
        headers=_bearer(admin_token),
    )
    assert r.status_code == 200
    assert r.json()["is_active"] is True


def test_admin_cannot_deactivate_own_account(client, admin_user, admin_token):
    r = client.patch(
        f"/admin/users/{admin_user.user_id}/status",
        json={"is_active": False},
        headers=_bearer(admin_token),
    )
    assert r.status_code == 400
    assert "own account" in r.json()["detail"]


# ── GDPR purge ────────────────────────────────────────────────────────────────

def test_admin_purge_user(client, admin_token, registered_user_a, mock_mongo_col):
    with patch("app.memory.memory_manager.MemoryManager.gdpr_purge", return_value=None), \
         patch("app.core.infra_registry.InfraRegistry.get_bm25", return_value=None):
        r = client.delete(
            f"/admin/users/{registered_user_a.user_id}",
            headers=_bearer(admin_token),
        )
    assert r.status_code == 200
    assert "deleted" in r.json()["message"].lower()

    doc = mock_mongo_col.find_one({"user_id": registered_user_a.user_id})
    assert doc is None


def test_admin_cannot_purge_own_account(client, admin_user, admin_token):
    r = client.delete(
        f"/admin/users/{admin_user.user_id}",
        headers=_bearer(admin_token),
    )
    assert r.status_code == 400


def test_admin_purge_not_found(client, admin_token):
    with patch("app.memory.memory_manager.MemoryManager.gdpr_purge", return_value=None), \
         patch("app.core.infra_registry.InfraRegistry.get_bm25", return_value=None):
        r = client.delete("/admin/users/nonexistent-xyz", headers=_bearer(admin_token))
    assert r.status_code == 404


# ── Platform stats ────────────────────────────────────────────────────────────

def test_admin_stats_returns_counts(client, admin_token, registered_user_a, registered_user_b):
    r = client.get("/admin/stats", headers=_bearer(admin_token))
    assert r.status_code == 200
    data = r.json()
    assert "total_users" in data
    assert "active_users" in data
    assert "admin_users" in data
    # admin fixture + user_a + user_b = 3
    assert data["total_users"] >= 3


def test_admin_stats_forbidden_for_regular_user(client, token_a):
    r = client.get("/admin/stats", headers=_bearer(token_a))
    assert r.status_code == 403


# ── System health ─────────────────────────────────────────────────────────────

def test_admin_system_health_returns_data(client, admin_token):
    r = client.get("/admin/system/health", headers=_bearer(admin_token))
    assert r.status_code == 200
    data = r.json()
    assert "infra" in data
    assert "models" in data
    assert "timestamp" in data


def test_admin_system_health_forbidden_for_user(client, token_a):
    r = client.get("/admin/system/health", headers=_bearer(token_a))
    assert r.status_code == 403


# ── Audit log ─────────────────────────────────────────────────────────────────

def test_admin_audit_log_returns_entries(client, admin_token):
    r = client.get("/admin/system/audit", headers=_bearer(admin_token))
    assert r.status_code == 200
    data = r.json()
    assert "entries" in data
    assert "count" in data


def test_admin_audit_log_forbidden_for_user(client, token_a):
    r = client.get("/admin/system/audit", headers=_bearer(token_a))
    assert r.status_code == 403
