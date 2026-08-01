from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


# ── Stored in MongoDB (never returned to clients) ────────────────────────────


class UserInDB(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    hashed_password: str | None = None  # None for Google-only accounts
    auth_providers: list[str] = Field(
        default_factory=list
    )  # ["email"], ["google"], ["email","google"]
    role: UserRole = UserRole.USER
    is_active: bool = True
    # Set only by app/bin/seed_demo_account.py, never by /auth/register — a
    # fixed, publicly-shared login (recruiters/hiring managers) that skips OTP
    # on every login. See /auth/login in router.py for the bypass.
    is_demo: bool = False
    # Set only by app/bin/seed_test_tenants.py, never by /auth/register — a
    # dedicated, non-public account used exclusively by automated quality
    # tooling (k6 load/stress/multi-user simulation, live-mode Schemathesis,
    # ZAP). Skips OTP for the same non-interactive-automation reason is_demo
    # does (see /auth/login), but is otherwise a completely ordinary account —
    # no tenant-isolation exemption, no elevated quota, nothing. Distinct from
    # is_demo so the one shared public recruiter login is never reused (and
    # never rate-limit-contended) by test tooling.
    is_load_test: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: datetime | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email address")
        if len(v) > 254:
            raise ValueError("Email address too long")
        return v


# ── Returned to clients (no password fields) ─────────────────────────────────


class UserPublic(BaseModel):
    user_id: str
    email: str
    role: UserRole
    is_active: bool
    is_demo: bool = False
    is_load_test: bool = False
    created_at: datetime


# ── Registration / login request bodies ──────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("Password must be at most 128 characters")
        return v


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)
    device_token: str | None = Field(None, max_length=128)  # trusted-device bypass

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.strip().lower()


# ── Token payloads ────────────────────────────────────────────────────────────


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)


class LogoutRequest(BaseModel):
    """Optional body for /auth/logout — lets the client surrender its refresh
    token too, so logout fully ends the session rather than just the access token."""

    refresh_token: str | None = Field(None, min_length=10)


class TokenPayload(BaseModel):
    sub: str  # user_id
    email: str
    role: str
    type: str  # "access" | "refresh"
    jti: str  # unique token id (for future revocation)
    exp: int
    iat: int


# ── Email OTP ─────────────────────────────────────────────────────────────────


class OTPVerifyRequest(BaseModel):
    otp_token: str = Field(..., min_length=10)
    code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


class OTPResendRequest(BaseModel):
    otp_token: str = Field(..., min_length=10)


# ── Forgot / reset password ───────────────────────────────────────────────────


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.strip().lower()


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=10)
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v
