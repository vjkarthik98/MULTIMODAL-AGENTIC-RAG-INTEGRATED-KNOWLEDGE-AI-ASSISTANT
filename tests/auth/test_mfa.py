"""Tests for TOTP MFA and JWT token revocation blacklist."""
from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
import pyotp


# ── Token blacklist tests ─────────────────────────────────────────────────────

def _make_fake_redis():
    """In-memory Redis stub for tests — no real Redis needed."""
    store = {}
    r = MagicMock()

    def setex(key, ttl, val):
        store[key] = val

    def exists(key):
        return 1 if key in store else 0

    def incr(key):
        store[key] = store.get(key, 0) + 1
        return store[key]

    def expire(key, ttl):
        pass

    def get(key):
        v = store.get(key)
        return str(v).encode() if v is not None else None

    r.setex.side_effect   = setex
    r.exists.side_effect  = exists
    r.incr.side_effect    = incr
    r.expire.side_effect  = expire
    r.get.side_effect     = get
    return r


class TestTokenBlacklist:

    def test_revoke_and_check(self):
        """A revoked JTI must be detected as revoked."""
        import app.auth.token_blacklist as bl
        fake = _make_fake_redis()
        with patch.object(bl, "_redis", return_value=fake):
            jti = str(uuid.uuid4())
            exp = int(time.time()) + 3600
            bl.revoke_token(jti, exp)
            assert bl.is_revoked(jti) is True

    def test_unknown_jti_not_revoked(self):
        """An unknown JTI must not be flagged as revoked."""
        import app.auth.token_blacklist as bl
        fake = _make_fake_redis()
        with patch.object(bl, "_redis", return_value=fake):
            assert bl.is_revoked(str(uuid.uuid4())) is False

    def test_generation_bump(self):
        """After bumping generation, counter increments."""
        import app.auth.token_blacklist as bl
        fake = _make_fake_redis()
        with patch.object(bl, "_redis", return_value=fake):
            user_id = f"test-gen-{uuid.uuid4()}"
            assert bl.get_user_token_generation(user_id) == 0
            bl.revoke_all_user_tokens(user_id)
            assert bl.get_user_token_generation(user_id) == 1

    def test_expired_token_revoke_no_error(self):
        """Revoking an already-expired token (exp in past) must not raise."""
        import app.auth.token_blacklist as bl
        fake = _make_fake_redis()
        with patch.object(bl, "_redis", return_value=fake):
            jti = str(uuid.uuid4())
            exp = int(time.time()) - 10
            bl.revoke_token(jti, exp)   # should not raise


# ── JWT handler revocation integration ───────────────────────────────────────

class TestJWTRevocation:

    def test_revoked_token_rejected_by_verify(self):
        """verify_token must raise ValueError for a revoked token."""
        import app.auth.token_blacklist as bl
        from app.auth.jwt_handler import issue_tokens, verify_token

        fake = _make_fake_redis()
        with patch.object(bl, "_redis", return_value=fake):
            tokens = issue_tokens("user-rev-test", "rev@test.com", "user")
            access = tokens["access_token"]

            payload = verify_token(access, expected_type="access")
            jti = payload["jti"]
            exp = payload["exp"]

            bl.revoke_token(jti, exp)

            with pytest.raises(ValueError, match="revoked"):
                verify_token(access, expected_type="access")

    def test_logout_all_invalidates_old_tokens(self):
        """Tokens issued before logout-all are rejected via generation check."""
        import app.auth.token_blacklist as bl
        from app.auth.jwt_handler import issue_tokens, verify_token

        uid = f"gen-test-{uuid.uuid4()}"
        fake = _make_fake_redis()

        with patch.object(bl, "_redis", return_value=fake):
            tokens_before = issue_tokens(uid, "gen@test.com", "user")
            access_before = tokens_before["access_token"]

            # Token valid before generation bump
            verify_token(access_before, expected_type="access")

            # Bump generation (simulate logout-all / password change)
            bl.revoke_all_user_tokens(uid)

            # Token issued before bump is now stale
            with pytest.raises(ValueError, match="invalidated"):
                verify_token(access_before, expected_type="access")

    def test_new_token_after_logout_all_is_valid(self):
        """Tokens issued AFTER logout-all carry the new generation and are valid."""
        import app.auth.token_blacklist as bl
        from app.auth.jwt_handler import issue_tokens, verify_token

        uid = f"new-gen-{uuid.uuid4()}"
        fake = _make_fake_redis()

        with patch.object(bl, "_redis", return_value=fake):
            bl.revoke_all_user_tokens(uid)

            tokens_after = issue_tokens(uid, "newgen@test.com", "user")
            payload = verify_token(tokens_after["access_token"], expected_type="access")
            assert payload["sub"] == uid


# ── MFA TOTP tests (mocked MongoDB) ──────────────────────────────────────────

class TestMFAService:

    def _make_mongo_doc(self, user_id: str, extra: dict = None) -> dict:
        doc = {"user_id": user_id, "email": f"{user_id}@test.com",
               "mfa_enabled": False}
        if extra:
            doc.update(extra)
        return doc

    def test_enroll_start_generates_secret(self):
        """enroll_start returns a valid base32 TOTP secret."""
        from app.auth.mfa import MFAService
        svc = MFAService()
        uid = str(uuid.uuid4())
        doc = self._make_mongo_doc(uid)

        mock_col = MagicMock()
        mock_col.find_one.return_value = doc

        with patch("app.auth.mfa._col", return_value=mock_col):
            result = svc.enroll_start(uid, f"{uid}@test.com")

        assert "secret" in result
        assert len(result["secret"]) >= 16    # valid base32
        assert "uri" in result
        assert "otpauth://totp/" in result["uri"]

    def test_verify_enroll_valid_code_enables_mfa(self):
        """verify_enroll with a valid TOTP code marks mfa_enabled=True."""
        from app.auth.mfa import MFAService
        svc = MFAService()
        uid = str(uuid.uuid4())
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)

        doc = self._make_mongo_doc(uid, {"mfa_pending_secret": secret})

        mock_col = MagicMock()
        mock_col.find_one.return_value = doc

        with patch("app.auth.mfa._col", return_value=mock_col):
            backup_codes = svc.verify_enroll(uid, totp.now())

        assert len(backup_codes) == 8
        for code in backup_codes:
            assert len(code) == 10   # 5 bytes hex = 10 chars

    def test_verify_enroll_wrong_code_raises(self):
        """verify_enroll with a wrong code raises ValueError."""
        from app.auth.mfa import MFAService
        svc = MFAService()
        uid = str(uuid.uuid4())
        secret = pyotp.random_base32()
        doc = self._make_mongo_doc(uid, {"mfa_pending_secret": secret})

        mock_col = MagicMock()
        mock_col.find_one.return_value = doc

        with patch("app.auth.mfa._col", return_value=mock_col):
            with pytest.raises(ValueError, match="Invalid TOTP"):
                svc.verify_enroll(uid, "000000")

    def test_verify_login_valid_totp(self):
        """verify_login accepts a valid TOTP code and returns user_id."""
        from app.auth.mfa import MFAService, _issue_mfa_token
        svc = MFAService()
        uid = str(uuid.uuid4())
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)

        doc = self._make_mongo_doc(uid, {
            "mfa_enabled": True,
            "mfa_secret": secret,
            "mfa_backup_hashes": [],
        })

        mock_col = MagicMock()
        mock_col.find_one.return_value = doc
        mfa_token = _issue_mfa_token(uid)

        with patch("app.auth.mfa._col", return_value=mock_col):
            result_uid = svc.verify_login(mfa_token, totp.now())

        assert result_uid == uid

    def test_verify_login_wrong_code_raises(self):
        """verify_login with wrong code raises ValueError."""
        from app.auth.mfa import MFAService, _issue_mfa_token
        svc = MFAService()
        uid = str(uuid.uuid4())
        secret = pyotp.random_base32()

        doc = self._make_mongo_doc(uid, {
            "mfa_enabled": True,
            "mfa_secret": secret,
            "mfa_backup_hashes": [],
        })

        mock_col = MagicMock()
        mock_col.find_one.return_value = doc
        mfa_token = _issue_mfa_token(uid)

        with patch("app.auth.mfa._col", return_value=mock_col):
            with pytest.raises(ValueError, match="Invalid MFA"):
                svc.verify_login(mfa_token, "000000")

    def test_verify_login_backup_code(self):
        """verify_login accepts a valid backup code and burns it."""
        from passlib.context import CryptContext
        from app.auth.mfa import MFAService, _issue_mfa_token
        svc = MFAService()
        uid = str(uuid.uuid4())
        secret = pyotp.random_base32()
        _ctx = CryptContext(schemes=["bcrypt"])

        plain_backup = "AABBCCDDEE"
        hashed_backup = _ctx.hash(plain_backup)

        doc = self._make_mongo_doc(uid, {
            "mfa_enabled": True,
            "mfa_secret": secret,
            "mfa_backup_hashes": [hashed_backup],
        })

        mock_col = MagicMock()
        mock_col.find_one.return_value = doc
        mfa_token = _issue_mfa_token(uid)

        with patch("app.auth.mfa._col", return_value=mock_col):
            result_uid = svc.verify_login(mfa_token, plain_backup)

        assert result_uid == uid
        # Verify the backup code was burned (replaced with "")
        call_args = mock_col.update_one.call_args
        update_doc = call_args[0][1]
        assert update_doc["$set"]["mfa_backup_hashes"][0] == ""

    def test_mfa_token_expired_raises(self):
        """An expired MFA challenge token raises ValueError."""
        from app.auth.mfa import MFAService
        svc = MFAService()
        with pytest.raises(ValueError, match="Invalid or expired"):
            svc.verify_login("this.is.not.a.valid.token", "123456")
