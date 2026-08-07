"""
Browser session cookies — the only place access/refresh/device tokens and the
CSRF token touch an HTTP response header.

Access, refresh, and device tokens are httpOnly: JavaScript in the browser
(and therefore any XSS payload or malicious extension) cannot read them, and
they never appear in localStorage/sessionStorage or a URL. The frontend only
ever sees the CSRF token, which is deliberately NOT httpOnly (the SPA must be
able to read it and echo it back as a header) but is useless on its own —
without the paired httpOnly session cookie it grants no access.

API/CLI/test clients are unaffected: every endpoint that sets these cookies
still returns the same token pair in the JSON body it always has, so
`Authorization: Bearer <token>` keeps working exactly as before (see
app/auth/dependencies.py). Only the shipped browser frontend has been changed
to stop reading tokens out of that body.
"""

from __future__ import annotations

import secrets

from fastapi import Request, Response

from app.core.config import settings

ACCESS_COOKIE = "magik_access"
REFRESH_COOKIE = "magik_refresh"
DEVICE_COOKIE = "magik_device"
CSRF_COOKIE = "magik_csrf"

_DEVICE_TTL_SECONDS = 30 * 24 * 60 * 60  # matches otp_store._DEVICE_TTL

_COOKIE_KW = {
    "httponly": True,
    "secure": settings.COOKIE_SECURE,
    "samesite": settings.COOKIE_SAMESITE,
    "domain": settings.COOKIE_DOMAIN,
}


def set_auth_cookies(
    response: Response,
    tokens: dict,
    device_token: str | None = None,
) -> str:
    """Set the httpOnly access/refresh cookies (+ device cookie, if issued)
    and a fresh, JS-readable CSRF cookie. Returns the new CSRF token so a
    caller that also wants to hand it back in the JSON body (nice-to-have for
    non-browser clients) can do so."""
    response.set_cookie(
        ACCESS_COOKIE,
        tokens["access_token"],
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
        **_COOKIE_KW,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        tokens["refresh_token"],
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/auth",
        **_COOKIE_KW,
    )
    if device_token:
        response.set_cookie(
            DEVICE_COOKIE,
            device_token,
            max_age=_DEVICE_TTL_SECONDS,
            path="/auth",
            **_COOKIE_KW,
        )

    csrf_token = secrets.token_urlsafe(32)
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
        httponly=False,  # the SPA must be able to read this one
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
    )
    return csrf_token


def clear_auth_cookies(response: Response) -> None:
    """Delete every session cookie — used on logout/logout-all/password
    change/account deletion. `path` must match what set_cookie used or the
    browser treats it as a different cookie and leaves the old one in place."""
    response.delete_cookie(ACCESS_COOKIE, path="/", domain=settings.COOKIE_DOMAIN)
    response.delete_cookie(REFRESH_COOKIE, path="/auth", domain=settings.COOKIE_DOMAIN)
    response.delete_cookie(DEVICE_COOKIE, path="/auth", domain=settings.COOKIE_DOMAIN)
    response.delete_cookie(CSRF_COOKIE, path="/", domain=settings.COOKIE_DOMAIN)


def read_access_cookie(request: Request) -> str | None:
    return request.cookies.get(ACCESS_COOKIE) or None


def read_refresh_cookie(request: Request) -> str | None:
    return request.cookies.get(REFRESH_COOKIE) or None


def read_device_cookie(request: Request) -> str | None:
    return request.cookies.get(DEVICE_COOKIE) or None
