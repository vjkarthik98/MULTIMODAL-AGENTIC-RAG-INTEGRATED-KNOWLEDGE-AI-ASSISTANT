"""
Shared fixtures for Phase 27 auth tests.
All tests use an in-memory MongoDB mock so no live Atlas connection is needed.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.auth.jwt_handler import issue_tokens
from app.auth.models import RegisterRequest, UserPublic, UserRole
from app.core.config import settings


# ── In-memory MongoDB stub ────────────────────────────────────────────────────

class _InMemoryCollection:
    """Minimal MongoDB Collection mock backed by a plain dict."""

    def __init__(self) -> None:
        self._docs: Dict[str, Dict] = {}

    def _matches(self, doc: dict, query: dict) -> bool:
        for k, v in query.items():
            if doc.get(k) != v:
                return False
        return True

    def insert_one(self, doc: dict) -> None:
        self._docs[doc["user_id"]] = dict(doc)

    def find_one(self, query: dict, projection: Optional[dict] = None) -> Optional[dict]:
        for doc in self._docs.values():
            if self._matches(doc, query):
                return self._project(dict(doc), projection)
        return None

    def find(self, query: Optional[dict] = None, projection: Optional[dict] = None):
        q = query or {}
        results = [
            self._project(dict(doc), projection)
            for doc in self._docs.values()
            if self._matches(doc, q)
        ]
        return _InMemoryCursor(results)

    def _project(self, doc: dict, projection: Optional[dict]) -> dict:
        if not projection:
            return doc
        # Remove excluded fields (value=0) and _id
        result = dict(doc)
        for k, v in projection.items():
            if v == 0 and k in result:
                del result[k]
        result.pop("_id", None)
        return result

    def count_documents(self, query: dict) -> int:
        return sum(1 for doc in self._docs.values() if self._matches(doc, query))

    def update_one(self, query: dict, update: dict) -> "_UpdateResult":
        for doc in self._docs.values():
            if self._matches(doc, query):
                set_fields = update.get("$set", {})
                doc.update(set_fields)
                return _UpdateResult(matched=1, modified=1)
        return _UpdateResult(matched=0, modified=0)

    def delete_one(self, query: dict) -> "_DeleteResult":
        for uid, doc in list(self._docs.items()):
            if self._matches(doc, query):
                del self._docs[uid]
                return _DeleteResult(deleted=1)
        return _DeleteResult(deleted=0)

    def delete_many(self, query: dict) -> None:
        to_delete = [
            uid for uid, doc in self._docs.items()
            if self._matches(doc, query)
        ]
        for uid in to_delete:
            del self._docs[uid]

    def clear(self) -> None:
        self._docs.clear()


class _InMemoryCursor:
    def __init__(self, docs: list) -> None:
        self._docs = docs

    def skip(self, n: int) -> "_InMemoryCursor":
        return _InMemoryCursor(self._docs[n:])

    def limit(self, n: int) -> "_InMemoryCursor":
        return _InMemoryCursor(self._docs[:n] if n > 0 else self._docs)

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)


class _UpdateResult:
    def __init__(self, matched: int, modified: int):
        self.matched_count = matched
        self.modified_count = modified


class _DeleteResult:
    def __init__(self, deleted: int):
        self.deleted_count = deleted


# Global collection instances shared across fixtures in a test session
_users_col = _InMemoryCollection()


@pytest.fixture(autouse=True)
def _reset_users_collection():
    """Wipe the in-memory user store before each test."""
    _users_col.clear()
    yield
    _users_col.clear()


@pytest.fixture
def mock_mongo_col(monkeypatch):
    """Patch _get_mongo_collection in auth.service to use in-memory store."""
    monkeypatch.setattr(
        "app.auth.service._get_mongo_collection",
        lambda: _users_col,
    )
    return _users_col


# ── Pre-built user fixtures ───────────────────────────────────────────────────

@pytest.fixture
def user_a_data():
    return {"email": "alice@example.com", "password": "AlicePass123!"}


@pytest.fixture
def user_b_data():
    return {"email": "bob@example.com", "password": "BobPass456!"}


@pytest.fixture
def registered_user_a(mock_mongo_col, user_a_data):
    from app.auth.service import AuthService
    svc = AuthService()
    user = svc.register(RegisterRequest(**user_a_data))
    return user


@pytest.fixture
def registered_user_b(mock_mongo_col, user_b_data):
    from app.auth.service import AuthService
    svc = AuthService()
    user = svc.register(RegisterRequest(**user_b_data))
    return user


@pytest.fixture
def token_a(registered_user_a):
    tokens = issue_tokens(
        registered_user_a.user_id,
        registered_user_a.email,
        registered_user_a.role.value,
    )
    return tokens["access_token"]


@pytest.fixture
def token_b(registered_user_b):
    tokens = issue_tokens(
        registered_user_b.user_id,
        registered_user_b.email,
        registered_user_b.role.value,
    )
    return tokens["access_token"]


@pytest.fixture
def admin_user(mock_mongo_col):
    from app.auth.models import UserInDB
    from app.auth.service import _hash_password
    user = UserInDB(
        email="admin@example.com",
        hashed_password=_hash_password("AdminPass999!"),
        role=UserRole.ADMIN,
    )
    mock_mongo_col.insert_one(user.model_dump())
    return UserPublic(
        user_id=user.user_id,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@pytest.fixture
def admin_token(admin_user):
    tokens = issue_tokens(admin_user.user_id, admin_user.email, admin_user.role.value)
    return tokens["access_token"]
