"""Tests for app/auth/metrics.py — the Phase 1 monitoring gap this closes:
app/auth/ had zero Prometheus metrics before this (failed-login rate,
MFA-failure rate, rate-limit rejections were invisible except by grepping
logs). Verifies both the metrics module itself (importable, safe no-op
behavior) and that the three real call sites (auth/service.py::authenticate,
auth/mfa.py::MFAService.verify_login, auth/rate_limit.py::
check_user_rate_limit) actually increment their counter on failure — not
just that the counters exist.

Reuses this package's existing fixtures/patterns: `svc`/`mock_mongo_col`/
`registered_user_a`/`user_a_data` from conftest.py, and the same
`_col`-patching / fake-Redis approach test_mfa.py already uses for
MFAService and the rate limiter.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pyotp
import pytest

from app.auth.service import AuthService


@pytest.fixture
def svc(mock_mongo_col):
    return AuthService()


class TestAuthMetricsImport:
    def test_login_failures_importable(self):
        from app.auth.metrics import auth_login_failures_total

        assert auth_login_failures_total is not None

    def test_mfa_failures_importable(self):
        from app.auth.metrics import auth_mfa_failures_total

        assert auth_mfa_failures_total is not None

    def test_rate_limit_rejections_importable(self):
        from app.auth.metrics import auth_rate_limit_rejections_total

        assert auth_rate_limit_rejections_total is not None

    def test_record_helpers_do_not_raise(self):
        from app.auth.metrics import (
            record_login_failure,
            record_mfa_failure,
            record_rate_limit_rejection,
        )

        record_login_failure("wrong_password")
        record_mfa_failure()
        record_rate_limit_rejection()


class TestLoginFailureMetric:
    def _count(self, reason: str) -> float:
        from app.auth.metrics import auth_login_failures_total

        metric = auth_login_failures_total.labels(reason=reason)
        return metric._value.get()  # prometheus_client Counter internal value

    def test_wrong_password_increments_reason_label(self, svc, registered_user_a, user_a_data):
        before = self._count("wrong_password")
        with pytest.raises(ValueError, match="Incorrect"):
            svc.authenticate(user_a_data["email"], "wrongpassword")
        after = self._count("wrong_password")
        assert after == before + 1

    def test_unknown_email_increments_no_account_label(self, svc):
        before = self._count("no_account")
        with pytest.raises(ValueError, match="No account"):
            svc.authenticate("nobody@example.com", "somepassword")
        after = self._count("no_account")
        assert after == before + 1

    def test_successful_login_does_not_increment(self, svc, registered_user_a, user_a_data):
        before = self._count("wrong_password")
        svc.authenticate(user_a_data["email"], user_a_data["password"])
        after = self._count("wrong_password")
        assert after == before


class TestMfaFailureMetric:
    def _make_mongo_doc(self, user_id: str, extra: dict = None) -> dict:
        doc = {"user_id": user_id, "email": f"{user_id}@test.com", "mfa_enabled": False}
        if extra:
            doc.update(extra)
        return doc

    def _count(self) -> float:
        from app.auth.metrics import auth_mfa_failures_total

        return auth_mfa_failures_total._value.get()

    def test_wrong_totp_code_increments_counter(self):
        from app.auth.mfa import MFAService, _issue_mfa_token

        svc = MFAService()
        uid = str(uuid.uuid4())
        secret = pyotp.random_base32()

        doc = self._make_mongo_doc(
            uid, {"mfa_enabled": True, "mfa_secret": secret, "mfa_backup_hashes": []}
        )
        mock_col = MagicMock()
        mock_col.find_one.return_value = doc
        mfa_token = _issue_mfa_token(uid)

        before = self._count()
        with patch("app.auth.mfa._col", return_value=mock_col):
            with pytest.raises(ValueError, match="Invalid MFA"):
                svc.verify_login(mfa_token, "000000")
        after = self._count()
        assert after == before + 1

    def test_correct_totp_code_does_not_increment(self):
        from app.auth.mfa import MFAService, _issue_mfa_token

        svc = MFAService()
        uid = str(uuid.uuid4())
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)

        doc = self._make_mongo_doc(
            uid, {"mfa_enabled": True, "mfa_secret": secret, "mfa_backup_hashes": []}
        )
        mock_col = MagicMock()
        mock_col.find_one.return_value = doc
        mfa_token = _issue_mfa_token(uid)

        before = self._count()
        with patch("app.auth.mfa._col", return_value=mock_col):
            svc.verify_login(mfa_token, totp.now())
        after = self._count()
        assert after == before


class TestRateLimitRejectionMetric:
    """Fake Redis stub — same shape as test_mfa.py's _make_fake_redis(), only
    the two methods app/auth/rate_limit.py actually calls (incr/expire)."""

    def _make_fake_cache(self, start_count: int):
        store = {"count": start_count}

        def incr(key):
            store["count"] += 1
            return store["count"]

        def expire(key, ttl):
            pass

        cache = MagicMock()
        cache.incr.side_effect = incr
        cache.expire.side_effect = expire
        return cache

    def _count(self) -> float:
        from app.auth.metrics import auth_rate_limit_rejections_total

        return auth_rate_limit_rejections_total._value.get()

    def test_over_limit_increments_counter(self, monkeypatch):
        from app.auth.rate_limit import check_user_rate_limit
        from app.core.config import settings

        monkeypatch.setattr(settings, "AUTH_ENABLED", True)
        fake_infra = MagicMock()
        fake_infra.get_cache.return_value = self._make_fake_cache(start_count=999)
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        before = self._count()
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            check_user_rate_limit("user-123", limit=5)
        after = self._count()
        assert after == before + 1

    def test_under_limit_does_not_increment(self, monkeypatch):
        from app.auth.rate_limit import check_user_rate_limit
        from app.core.config import settings

        monkeypatch.setattr(settings, "AUTH_ENABLED", True)
        fake_infra = MagicMock()
        fake_infra.get_cache.return_value = self._make_fake_cache(start_count=0)
        monkeypatch.setattr("app.core.infra_registry.infra", fake_infra)

        before = self._count()
        check_user_rate_limit("user-123", limit=5)
        after = self._count()
        assert after == before
