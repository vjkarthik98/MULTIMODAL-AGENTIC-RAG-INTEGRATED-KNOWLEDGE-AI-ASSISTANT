"""
Phase 27 — Registration and login tests.
Covers: happy path, duplicate email, weak password, wrong password, inactive account.
"""
from __future__ import annotations

import pytest

from app.auth.models import RegisterRequest
from app.auth.service import AuthService


@pytest.fixture
def svc(mock_mongo_col):
    return AuthService()


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_success(svc):
    req = RegisterRequest(email="user@example.com", password="StrongPass99!")
    user = svc.register(req)

    assert user.email == "user@example.com"
    assert user.user_id
    assert user.is_active is True
    assert user.role.value == "user"


def test_register_email_normalised_to_lowercase(svc):
    req = RegisterRequest(email="UPPER@EXAMPLE.COM", password="StrongPass99!")
    user = svc.register(req)
    assert user.email == "upper@example.com"


def test_register_duplicate_email_raises(svc):
    req = RegisterRequest(email="dup@example.com", password="StrongPass99!")
    svc.register(req)
    with pytest.raises(ValueError, match="already exists"):
        svc.register(req)


def test_register_invalid_email_raises():
    with pytest.raises(ValueError):
        RegisterRequest(email="not-an-email", password="StrongPass99!")


def test_register_password_too_short_raises():
    with pytest.raises(ValueError):
        RegisterRequest(email="user@example.com", password="short")


def test_register_password_never_stored_plaintext(svc, mock_mongo_col):
    req = RegisterRequest(email="secure@example.com", password="StrongPass99!")
    user = svc.register(req)

    doc = mock_mongo_col.find_one({"user_id": user.user_id})
    assert doc is not None
    assert doc["hashed_password"] != req.password
    assert doc["hashed_password"].startswith("$argon2") or doc["hashed_password"].startswith("$2b$")


def test_register_password_not_in_response(svc):
    req = RegisterRequest(email="safe@example.com", password="StrongPass99!")
    user = svc.register(req)
    user_dict = user.model_dump()
    assert "password" not in user_dict
    assert "hashed_password" not in user_dict


# ── Login ─────────────────────────────────────────────────────────────────────

def test_login_success(svc, registered_user_a, user_a_data):
    user = svc.authenticate(user_a_data["email"], user_a_data["password"])
    assert user.user_id == registered_user_a.user_id
    assert user.email == registered_user_a.email


def test_login_wrong_password_raises(svc, registered_user_a, user_a_data):
    with pytest.raises(ValueError, match="Incorrect"):
        svc.authenticate(user_a_data["email"], "wrongpassword")


def test_login_unknown_email_raises(svc):
    with pytest.raises(ValueError, match="Incorrect"):
        svc.authenticate("nobody@example.com", "somepassword")


def test_login_updates_last_login(svc, registered_user_a, user_a_data, mock_mongo_col):
    svc.authenticate(user_a_data["email"], user_a_data["password"])
    doc = mock_mongo_col.find_one({"user_id": registered_user_a.user_id})
    assert doc["last_login"] is not None


def test_login_inactive_account_raises(svc, registered_user_a, user_a_data, mock_mongo_col):
    svc.deactivate(registered_user_a.user_id)
    with pytest.raises(ValueError, match="disabled"):
        svc.authenticate(user_a_data["email"], user_a_data["password"])
