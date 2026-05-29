"""TOTP-based Multi-Factor Authentication (RFC 6238).

Flow:
  1. User calls POST /auth/mfa/enroll  → gets secret + QR code URI + backup codes
  2. User scans QR with authenticator app, then calls POST /auth/mfa/verify-enroll
     with the 6-digit code to confirm enrolment.
  3. On subsequent logins, if MFA is enabled, POST /auth/login returns
     {"mfa_required": true, "mfa_token": "<short-lived token>"}
  4. User calls POST /auth/mfa/verify with the 6-digit TOTP code + mfa_token
     to receive the full JWT access+refresh pair.
  5. Backup codes: 8 single-use codes stored as bcrypt hashes in MongoDB.
"""
from __future__ import annotations

import base64
import io
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pyotp

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ISSUER = getattr(settings, "APP_NAME", "Multimodal RAG Assistant")
_MFA_TOKEN_TTL_MINUTES = 10          # mfa_token expires after 10 min
_BACKUP_CODE_COUNT     = 8


# ── MongoDB helpers ───────────────────────────────────────────────────────────

def _col():
    from app.core.infra_registry import infra
    mongo = infra.get_mongo()
    if mongo is None:
        raise RuntimeError("MongoDB unavailable")
    db = mongo.client[settings.MONGO_DB_NAME]
    return db[settings.AUTH_COLLECTION]


# ── Backup code helpers ───────────────────────────────────────────────────────

def _generate_backup_codes() -> tuple[List[str], List[str]]:
    """Return (plaintext_codes, bcrypt_hashes). Store hashes; show plaintext once."""
    from passlib.context import CryptContext
    _ctx = CryptContext(schemes=["bcrypt"])
    codes, hashes = [], []
    for _ in range(_BACKUP_CODE_COUNT):
        code = secrets.token_hex(5).upper()   # e.g. "A3F9B2C1D4"
        codes.append(code)
        hashes.append(_ctx.hash(code))
    return codes, hashes


def _verify_backup_code(plain: str, stored_hashes: List[str]) -> Optional[int]:
    """Return the index of the matching backup code, or None."""
    from passlib.context import CryptContext
    _ctx = CryptContext(schemes=["bcrypt"])
    for i, h in enumerate(stored_hashes):
        if h and _ctx.verify(plain.strip().upper(), h):
            return i
    return None


# ── MFA token (short-lived JWT for the mfa_required step) ────────────────────

def _issue_mfa_token(user_id: str) -> str:
    """Issue a short-lived token that gates the MFA verify step."""
    from jose import jwt as jose_jwt
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": "mfa_challenge",
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=_MFA_TOKEN_TTL_MINUTES)).timestamp()),
    }
    return jose_jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _verify_mfa_token(token: str) -> str:
    """Verify and return user_id from MFA challenge token. Raises ValueError on failure."""
    from jose import JWTError, jwt as jose_jwt
    try:
        payload = jose_jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as exc:
        raise ValueError("Invalid or expired MFA token") from exc
    if payload.get("type") != "mfa_challenge":
        raise ValueError("Invalid MFA token type")
    return payload["sub"]


# ── Core MFA service ──────────────────────────────────────────────────────────

class MFAService:

    def enroll_start(self, user_id: str, email: str) -> dict:
        """
        Generate a TOTP secret and return enrolment data.
        Does NOT activate MFA — activation requires verify_enroll().
        """
        col = _col()
        doc = col.find_one({"user_id": user_id})
        if not doc:
            raise ValueError("User not found")
        if doc.get("mfa_enabled"):
            raise ValueError("MFA is already enabled for this account")

        secret = pyotp.random_base32()
        totp   = pyotp.TOTP(secret)
        uri    = totp.provisioning_uri(name=email, issuer_name=_ISSUER)

        # Generate QR code as base64 data URI
        try:
            import qrcode
            qr = qrcode.make(uri)
            buf = io.BytesIO()
            qr.save(buf, format="PNG")
            qr_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            qr_b64 = None

        # Store pending secret (not yet active — user must confirm with a valid code)
        col.update_one(
            {"user_id": user_id},
            {"$set": {"mfa_pending_secret": secret, "mfa_enabled": False}},
        )
        logger.info(event="mfa_enroll_started", user_id=user_id)

        return {
            "secret":     secret,          # show once — user stores in authenticator
            "uri":        uri,
            "qr_code":    qr_b64,          # base64 PNG for display in UI
        }

    def verify_enroll(self, user_id: str, code: str) -> List[str]:
        """
        Confirm enrolment by verifying the first TOTP code.
        Returns plaintext backup codes (show once).
        """
        col   = _col()
        doc   = col.find_one({"user_id": user_id})
        if not doc:
            raise ValueError("User not found")

        secret = doc.get("mfa_pending_secret")
        if not secret:
            raise ValueError("No pending MFA enrolment found — call /auth/mfa/enroll first")

        totp = pyotp.TOTP(secret)
        if not totp.verify(code, valid_window=1):    # ±30s tolerance
            raise ValueError("Invalid TOTP code")

        plaintext_codes, hashed_codes = _generate_backup_codes()

        col.update_one(
            {"user_id": user_id},
            {"$set": {
                "mfa_enabled":        True,
                "mfa_secret":         secret,
                "mfa_backup_hashes":  hashed_codes,
                "mfa_enrolled_at":    datetime.now(timezone.utc),
            }, "$unset": {"mfa_pending_secret": ""}},
        )
        logger.info(event="mfa_enroll_completed", user_id=user_id)
        return plaintext_codes     # caller shows these once; never stored in plaintext

    def begin_login(self, user_id: str) -> str:
        """
        Called after password check passes but MFA is required.
        Returns a short-lived mfa_token the client must include in verify_login().
        """
        return _issue_mfa_token(user_id)

    def verify_login(self, mfa_token: str, code: str) -> str:
        """
        Verify TOTP code (or backup code) against the challenge token.
        Returns user_id on success; raises ValueError on failure.
        """
        user_id = _verify_mfa_token(mfa_token)

        col = _col()
        doc = col.find_one({"user_id": user_id})
        if not doc:
            raise ValueError("User not found")
        if not doc.get("mfa_enabled"):
            raise ValueError("MFA not enabled for this user")

        secret = doc.get("mfa_secret", "")
        totp   = pyotp.TOTP(secret)

        if totp.verify(code, valid_window=1):
            logger.info(event="mfa_login_success", user_id=user_id)
            return user_id

        # Try backup codes
        stored_hashes = doc.get("mfa_backup_hashes", [])
        idx = _verify_backup_code(code, stored_hashes)
        if idx is not None:
            # Burn the used backup code — replace with empty string
            stored_hashes[idx] = ""
            col.update_one(
                {"user_id": user_id},
                {"$set": {"mfa_backup_hashes": stored_hashes}},
            )
            logger.warning(event="mfa_backup_code_used", user_id=user_id, index=idx)
            return user_id

        logger.warning(event="mfa_login_failed", user_id=user_id)
        raise ValueError("Invalid MFA code")

    def disable(self, user_id: str, code: str) -> None:
        """Disable MFA after verifying a valid TOTP code."""
        col = _col()
        doc = col.find_one({"user_id": user_id})
        if not doc or not doc.get("mfa_enabled"):
            raise ValueError("MFA is not enabled")

        totp = pyotp.TOTP(doc.get("mfa_secret", ""))
        if not totp.verify(code, valid_window=1):
            raise ValueError("Invalid TOTP code")

        col.update_one(
            {"user_id": user_id},
            {"$unset": {
                "mfa_enabled": "",
                "mfa_secret": "",
                "mfa_backup_hashes": "",
                "mfa_enrolled_at": "",
            }},
        )
        logger.info(event="mfa_disabled", user_id=user_id)

    def is_enabled(self, user_id: str) -> bool:
        col = _col()
        doc = col.find_one({"user_id": user_id}, {"mfa_enabled": 1})
        return bool(doc and doc.get("mfa_enabled"))
