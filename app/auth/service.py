from __future__ import annotations

from datetime import datetime, timezone

from passlib.context import CryptContext

from app.auth.models import RegisterRequest, UserInDB, UserPublic
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Argon2 first, bcrypt as fallback — OWASP recommended
# Explicit parameters per OWASP: time_cost=3, memory_cost=64MB, parallelism=2
_pwd_ctx = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__time_cost=3,
    argon2__memory_cost=65536,
    argon2__parallelism=2,
)


def _hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


def _check_password_strength(password: str, email: str) -> None:
    """Raise ValueError if password is too weak."""
    try:
        from zxcvbn import zxcvbn

        result = zxcvbn(password, user_inputs=[email])
        score = result.get("score", 0)
        if score < settings.PASSWORD_MIN_ZXCVBN_SCORE:
            suggestions = result.get("feedback", {}).get("suggestions", [])
            hint = suggestions[0] if suggestions else "Choose a stronger password."
            raise ValueError(f"Password too weak (score {score}/4). {hint}")
    except ImportError:
        pass


def _get_mongo_collection():
    from app.core.infra_registry import infra

    mongo = infra.get_mongo()
    if mongo is None:
        raise RuntimeError("MongoDB is not available")
    db = mongo.client[settings.MONGO_DB_NAME]
    return db[settings.AUTH_COLLECTION]


def _doc_to_public(doc: dict) -> UserPublic:
    return UserPublic(
        user_id=doc["user_id"],
        email=doc["email"],
        role=doc.get("role", "user"),
        is_active=doc.get("is_active", True),
        created_at=doc.get("created_at", datetime.now(timezone.utc)),
    )


class AuthService:

    def register(self, req: RegisterRequest) -> UserPublic:
        """
        Create or link an email/password account as inactive (pending OTP verification).

        - New email → create inactive account, caller must send OTP and call activate_user()
        - Email exists, unverified (is_active=False) → replace with fresh inactive account
        - Email exists, Google-only → link password, keep active
        - Email exists, already active email account → reject
        """
        col = _get_mongo_collection()
        existing = col.find_one({"email": req.email})

        if existing:
            providers = existing.get("auth_providers") or (
                ["google"] if existing.get("oauth_only") else ["email"]
            )

            # Unverified account (created but OTP never completed) — let user retry
            if "email" in providers and not existing.get("is_active", True):
                _check_password_strength(req.password, req.email)
                col.delete_one({"email": req.email})
                # Fall through to create a fresh account below

            elif "email" in providers:
                raise ValueError("An account with this email already exists.")

            else:
                # Google-only account — link email/password, activate immediately
                _check_password_strength(req.password, req.email)
                col.update_one(
                    {"email": req.email},
                    {
                        "$set": {"hashed_password": _hash_password(req.password)},
                        "$addToSet": {"auth_providers": "email"},
                        "$unset": {"oauth_only": ""},
                    },
                )
                logger.info(event="auth_email_linked_to_google_account", email=req.email)
                doc = col.find_one({"email": req.email})
                return _doc_to_public(doc)

        # Brand-new account — inactive until OTP verified
        _check_password_strength(req.password, req.email)
        user = UserInDB(
            email=req.email,
            hashed_password=_hash_password(req.password),
            auth_providers=["email"],
            is_active=False,  # activated in verify-otp after email confirmation
        )
        col.insert_one(user.model_dump())
        logger.info(event="auth_user_registered_pending", email=req.email, user_id=user.user_id)
        return _doc_to_public(user.model_dump())

    def register_and_activate(self, req: RegisterRequest) -> UserPublic:
        """Register and immediately activate an account (no OTP step).

        Used exclusively by the guest → real user conversion flow, where the
        guest has already demonstrated real intent through product use.
        Normal registration still requires OTP email verification.
        """
        user = self.register(req)
        # If register() returned an already-active user (Google-linked account),
        # activate_user is a no-op; otherwise it sets is_active=True.
        self.activate_user(user.user_id)
        # Re-fetch to return current is_active state
        col = _get_mongo_collection()
        doc = col.find_one({"user_id": user.user_id})
        if doc is None:
            return user
        return _doc_to_public(doc)

    def activate_user(self, user_id: str) -> None:
        """Mark account as active after OTP verification."""
        col = _get_mongo_collection()
        col.update_one(
            {"user_id": user_id},
            {"$set": {"is_active": True, "last_login": datetime.now(timezone.utc)}},
        )
        logger.info(event="auth_user_activated", user_id=user_id)

    def delete_user(self, user_id: str) -> None:
        """Remove an account — used to roll back on OTP send failure."""
        col = _get_mongo_collection()
        col.delete_one({"user_id": user_id})
        logger.info(event="auth_user_deleted", user_id=user_id)

    def authenticate(self, email: str, password: str) -> UserPublic:
        """
        Verify email/password credentials.

        - Google-only account (no "email" provider) → clear error directing to Google,
          or inviting them to register with their email to add a password.
        - Wrong password → generic error (constant-time).
        """
        col = _get_mongo_collection()
        doc = col.find_one({"email": email})

        if doc:
            providers = doc.get("auth_providers") or (
                ["google"] if doc.get("oauth_only") else ["email"]
            )
            has_email_provider = "email" in providers
            has_password = bool(doc.get("hashed_password"))

            if not has_email_provider or not has_password:
                logger.warning(event="auth_login_failed_no_password", email=email)
                raise ValueError(
                    "This account was created with Google sign-in and has no password. "
                    "Sign in with Google, or use the registration form with this email "
                    "to add a password to your account."
                )

        # No account found for this email
        if doc is None:
            # Still run dummy verify to prevent timing-based user enumeration
            dummy = "$argon2id$v=19$m=65536,t=3,p=4$AAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            _verify_password(password, dummy)
            logger.warning(event="auth_login_failed_no_account", email=email)
            raise ValueError("No account found with this email. Please create an account first.")

        if not _verify_password(password, doc["hashed_password"]):
            logger.warning(event="auth_login_failed_wrong_password", email=email)
            raise ValueError("Incorrect password. Please try again.")

        if not doc.get("is_active", True):
            raise ValueError(
                "Email not verified. Please complete registration to activate your account."
            )

        col.update_one(
            {"email": email},
            {"$set": {"last_login": datetime.now(timezone.utc)}},
        )
        logger.info(event="auth_login_success", email=email, user_id=doc["user_id"])
        return _doc_to_public(doc)

    # ── Kept for backwards compatibility (called from older code paths) ────────

    def mark_oauth_only(self, email: str) -> None:
        """Legacy shim — new code uses auth_providers. Sets google provider if not set."""
        col = _get_mongo_collection()
        col.update_one(
            {"email": email},
            {
                "$addToSet": {"auth_providers": "google"},
                "$set": {"oauth_only": True},  # keep for old readers
            },
        )

    # ── Lookups ────────────────────────────────────────────────────────────────

    def get_by_id(self, user_id: str) -> UserPublic | None:
        col = _get_mongo_collection()
        doc = col.find_one({"user_id": user_id})
        return _doc_to_public(doc) if doc else None

    def get_by_email(self, email: str) -> UserPublic | None:
        col = _get_mongo_collection()
        doc = col.find_one({"email": email})
        return _doc_to_public(doc) if doc else None

    def change_password(self, user_id: str, current_password: str, new_password: str) -> None:
        """Verify current password then update to new one."""
        col = _get_mongo_collection()
        doc = col.find_one({"user_id": user_id})
        if not doc:
            raise ValueError("User not found.")

        if not doc.get("hashed_password"):
            raise ValueError(
                "This account has no password yet. "
                "Use the registration form with your email to add one."
            )

        if not _verify_password(current_password, doc["hashed_password"]):
            raise ValueError("Current password is incorrect.")

        email = doc.get("email", "")
        _check_password_strength(new_password, email)

        col.update_one(
            {"user_id": user_id},
            {
                "$set": {"hashed_password": _hash_password(new_password)},
                "$addToSet": {"auth_providers": "email"},
            },
        )
        logger.info(event="auth_password_changed", user_id=user_id)

    def reset_password(self, user_id: str, new_password: str) -> None:
        """Set a new password without requiring the current one (password-reset flow)."""
        col = _get_mongo_collection()
        doc = col.find_one({"user_id": user_id}, {"email": 1})
        if not doc:
            raise ValueError("User not found.")
        email = doc.get("email", "")
        _check_password_strength(new_password, email)
        col.update_one(
            {"user_id": user_id},
            {
                "$set": {"hashed_password": _hash_password(new_password)},
                "$addToSet": {"auth_providers": "email"},
            },
        )
        logger.info(event="auth_password_reset", user_id=user_id)

    def deactivate(self, user_id: str) -> None:
        col = _get_mongo_collection()
        col.update_one({"user_id": user_id}, {"$set": {"is_active": False}})
        logger.info(event="auth_user_deactivated", user_id=user_id)
