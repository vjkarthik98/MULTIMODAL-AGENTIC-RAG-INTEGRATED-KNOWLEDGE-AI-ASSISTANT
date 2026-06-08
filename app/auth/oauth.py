"""
Google OAuth2 / OIDC social login.

Flow:
  1. GET  /auth/google           → redirect user to Google consent screen
  2. GET  /auth/callback/google  → Google redirects back with ?code=...
                                   → exchange code for tokens
                                   → verify ID token, extract email
                                   → get_or_create user in MongoDB
                                   → issue our own JWT pair
                                   → return TokenPair to client
"""
from __future__ import annotations

import os
import secrets
from typing import Optional
from urllib.parse import urlencode

import httpx
from datetime import datetime, timezone

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GOOGLE_CLIENT_ID: Optional[str]     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET")
OAUTH_REDIRECT_URI: str             = os.getenv(
    "OAUTH_REDIRECT_URI",
    "http://localhost:8000/auth/callback/google",
)

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# In-memory state store (nonce → True). Prevents CSRF on the callback.
# In production use Redis with a short TTL.
_STATE_STORE: dict[str, bool] = {}
_STATE_MAX = 500  # bounded to prevent memory growth


def google_oauth_enabled() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


# ── Step 1 — Build the Google login URL ───────────────────────────────────────

def build_google_auth_url() -> tuple[str, str]:
    """
    Build the URL to redirect the user to Google's consent screen.
    Returns (url, state) — state must be stored and verified on callback.
    """
    if not google_oauth_enabled():
        raise RuntimeError("Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")

    state = secrets.token_urlsafe(32)

    # Bounded state store
    if len(_STATE_STORE) >= _STATE_MAX:
        _STATE_STORE.clear()
    _STATE_STORE[state] = True

    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "online",
        "prompt":        "select_account",  # always show account picker
    }

    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    logger.info(event="google_oauth_redirect_built")
    return url, state


# ── Step 2 — Exchange code for tokens ────────────────────────────────────────

async def exchange_google_code(code: str, state: str) -> dict:
    """
    Exchange the authorization code for Google tokens.
    Verifies the state parameter to prevent CSRF.
    Returns the Google user info dict: {email, name, picture, sub, ...}
    """
    if not _STATE_STORE.pop(state, None):
        raise ValueError("Invalid or expired OAuth state parameter")

    # Exchange code → tokens
    async with httpx.AsyncClient(timeout=10) as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  OAUTH_REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
        )

    if token_resp.status_code != 200:
        logger.warning(
            event="google_token_exchange_failed",
            status=token_resp.status_code,
            body=token_resp.text[:200],
        )
        raise ValueError("Failed to exchange Google authorization code")

    token_data = token_resp.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise ValueError("No access token in Google response")

    # Fetch user info using the access token
    async with httpx.AsyncClient(timeout=10) as client:
        userinfo_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if userinfo_resp.status_code != 200:
        raise ValueError("Failed to fetch Google user info")

    userinfo = userinfo_resp.json()

    if not userinfo.get("email"):
        raise ValueError("Google did not return an email address")

    if not userinfo.get("email_verified", False):
        raise ValueError("Google account email is not verified")

    logger.info(
        event="google_oauth_userinfo_fetched",
        email=userinfo["email"],
    )
    return userinfo


# ── Step 3 — Get or create user ───────────────────────────────────────────────

def get_or_create_oauth_user(email: str, provider: str):
    """
    Account-linking login (Option B):

    - Email already exists (any provider) → add this provider to their list,
      then return the existing account. One user, multiple sign-in methods.
    - Email not seen before → create a new account with this provider only.

    Returns UserPublic.
    """
    from app.auth.service import _get_mongo_collection, _doc_to_public
    from app.auth.models import UserInDB

    col = _get_mongo_collection()
    doc = col.find_one({"email": email})

    if doc:
        providers = doc.get("auth_providers") or (
            ["google"] if doc.get("oauth_only") else ["email"]
        )
        if provider not in providers:
            # Link this OAuth provider to the existing account
            col.update_one(
                {"email": email},
                {
                    "$addToSet": {"auth_providers": provider},
                    "$unset":    {"oauth_only": ""},   # remove legacy flag
                    "$set":      {"last_login": datetime.now(timezone.utc)},
                },
            )
            logger.info(
                event="oauth_provider_linked",
                email=email,
                provider=provider,
                existing_providers=providers,
            )
        else:
            col.update_one(
                {"email": email},
                {"$set": {"last_login": datetime.now(timezone.utc)}},
            )
            logger.info(event="oauth_existing_user_login", email=email, provider=provider)

        doc = col.find_one({"email": email})
        return _doc_to_public(doc)

    # New user — create account with this OAuth provider, no password
    user = UserInDB(
        email=email,
        hashed_password=None,
        auth_providers=[provider],
    )
    col.insert_one(user.model_dump())
    logger.info(event="oauth_user_auto_registered", email=email, provider=provider)
    return _doc_to_public(user.model_dump())
