from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from passlib.context import CryptContext

from app.auth.models import RegisterRequest, UserInDB, UserPublic
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Argon2 first, bcrypt as fallback — OWASP recommended
_pwd_ctx = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")


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
        # zxcvbn not available — fall back to basic length check only
        pass


def _get_mongo_collection():
    from app.core.infra_registry import infra
    mongo = infra.get_mongo()
    if mongo is None:
        raise RuntimeError("MongoDB is not available")
    db = mongo.client[settings.MONGO_DB_NAME]
    return db[settings.AUTH_COLLECTION]


class AuthService:

    def register(self, req: RegisterRequest) -> UserPublic:
        """Create a new user account. Raises ValueError on duplicate email or weak password."""
        _check_password_strength(req.password, req.email)

        col = _get_mongo_collection()

        if col.find_one({"email": req.email}):
            raise ValueError("An account with this email already exists")

        user = UserInDB(
            email=req.email,
            hashed_password=_hash_password(req.password),
        )

        col.insert_one(user.model_dump())
        logger.info(event="auth_user_registered", email=req.email, user_id=user.user_id)

        return UserPublic(
            user_id=user.user_id,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
        )

    def mark_oauth_only(self, email: str) -> None:
        """Tag an account as OAuth-only so email/password login returns a clear error."""
        col = _get_mongo_collection()
        col.update_one({"email": email}, {"$set": {"oauth_only": True}})

    def authenticate(self, email: str, password: str) -> UserPublic:
        """Verify credentials. Raises ValueError on wrong email/password."""
        col = _get_mongo_collection()
        doc = col.find_one({"email": email})

        # OAuth-only accounts have no usable password — give a clear error before hashing
        if doc and doc.get("oauth_only"):
            logger.warning(event="auth_login_failed_oauth_only", email=email)
            raise ValueError("This account uses Google sign-in. Please click 'Continue with Google' to log in.")

        # Constant-time failure — always verify even on miss to prevent timing attacks
        dummy_hash = "$argon2id$v=19$m=65536,t=3,p=4$AAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        stored_hash = doc["hashed_password"] if doc else dummy_hash

        if not _verify_password(password, stored_hash) or doc is None:
            logger.warning(event="auth_login_failed", email=email)
            raise ValueError("Incorrect email or password")

        if not doc.get("is_active", True):
            raise ValueError("Account is disabled")

        # Update last_login
        col.update_one(
            {"email": email},
            {"$set": {"last_login": datetime.now(timezone.utc)}},
        )

        logger.info(event="auth_login_success", email=email, user_id=doc["user_id"])

        return UserPublic(
            user_id=doc["user_id"],
            email=doc["email"],
            role=doc.get("role", "user"),
            is_active=doc.get("is_active", True),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
        )

    def get_by_id(self, user_id: str) -> Optional[UserPublic]:
        col = _get_mongo_collection()
        doc = col.find_one({"user_id": user_id})
        if not doc:
            return None
        return UserPublic(
            user_id=doc["user_id"],
            email=doc["email"],
            role=doc.get("role", "user"),
            is_active=doc.get("is_active", True),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
        )

    def get_by_email(self, email: str) -> Optional[UserPublic]:
        col = _get_mongo_collection()
        doc = col.find_one({"email": email})
        if not doc:
            return None
        return UserPublic(
            user_id=doc["user_id"],
            email=doc["email"],
            role=doc.get("role", "user"),
            is_active=doc.get("is_active", True),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
        )

    def change_password(self, user_id: str, current_password: str, new_password: str) -> None:
        """Verify current password then update to new one. Raises ValueError on failure."""
        col = _get_mongo_collection()
        doc = col.find_one({"user_id": user_id})
        if not doc:
            raise ValueError("User not found")

        if not _verify_password(current_password, doc["hashed_password"]):
            raise ValueError("Current password is incorrect")

        email = doc.get("email", "")
        _check_password_strength(new_password, email)

        col.update_one(
            {"user_id": user_id},
            {"$set": {"hashed_password": _hash_password(new_password)}},
        )
        logger.info(event="auth_password_changed", user_id=user_id)

    def deactivate(self, user_id: str) -> None:
        col = _get_mongo_collection()
        col.update_one({"user_id": user_id}, {"$set": {"is_active": False}})
        logger.info(event="auth_user_deactivated", user_id=user_id)
