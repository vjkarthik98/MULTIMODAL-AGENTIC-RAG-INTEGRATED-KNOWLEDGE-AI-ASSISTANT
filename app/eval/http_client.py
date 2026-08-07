"""Authenticated, retry-aware HTTP client shared by every eval runner.

WHY THIS EXISTS — a measurement bug that silently destroyed most of a Tier-2 run
(CD run 31139999120, v0.30.0, 2026-08-07):

  * tier2-eval.yml mints ONE access token, before the suite starts, and passes it
    to the container as a static `EVAL_ACCESS_TOKEN`.
  * `settings.ACCESS_TOKEN_EXPIRE_MINUTES` defaults to **30**.
  * the full suite ran for **44 minutes**.

The e2e sub-suite runs near the end, so it crossed the 30-minute mark mid-loop.
Result: 73 of 90 e2e queries never reached the model at all —

    [BREACH/ERROR] e2e.query_error_img-0011: 401 Client Error: Unauthorized
    ... 30 more 401s ...
    [BREACH/ERROR] e2e.query_error_route-0009: 429 Client Error: Too Many Requests
    ... 43 more 429s ...

and the 429 cascade is a direct consequence of the 401s: a rejected request costs
no LLM time, so the loop stopped being self-throttling and immediately outran
`settings.RATE_LIMIT_RPM` (60).

The damage was not "some queries failed" — it was that e2e's SCORES were computed
from the ~17 rows that happened to run before expiry, then reported as if they
measured the system. e2e.hit_rate=0.24 / recall@5=0.20 / hallucination_rate=0.79
are largely artifacts of that, not model quality.

WHAT THIS FIXES

  1. Mints the token IN-PROCESS (`app.auth.jwt_handler.issue_tokens`). The eval
     runs inside the app container via `docker exec`, so it already has
     JWT_SECRET_KEY and the whole app importable — there is no reason to depend
     on a token minted minutes earlier by the workflow.
  2. Refreshes PROACTIVELY at `_REFRESH_MARGIN_SEC` before expiry, so a long
     suite never presents a stale token in the first place.
  3. Retries once on a 401 with a force-refreshed token, covering the residual
     race (clock skew, a token revoked mid-run by `gen` bump).
  4. Backs off and retries on 429, honouring `Retry-After` when present. A rate
     limit is a "come back later", never a data point.

`EVAL_ACCESS_TOKEN` is still honoured as a fallback so anything driving the eval
from OUTSIDE the container (a laptop against a remote server, where jwt_handler
has no usable secret) keeps working exactly as before.
"""

from __future__ import annotations

import os
import time
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Refresh this far ahead of the token's real expiry. Comfortably longer than the
# slowest single generation call observed on the box (p99 ~28s) so a token can
# never expire *during* the request it was attached to.
_REFRESH_MARGIN_SEC = 120

# A 429 means the limiter is shedding load, not that the system is wrong. Retry
# with a bounded backoff rather than recording a failure.
_RATE_LIMIT_MAX_RETRIES = 4
_RATE_LIMIT_BASE_SLEEP_SEC = 5.0
_RATE_LIMIT_MAX_SLEEP_SEC = 60.0


class EvalAuth:
    """Supplies a never-expired bearer token for the eval tenant."""

    def __init__(self, user_id: str, email: str = "eval@magik.local", role: str = "user") -> None:
        self._user_id = user_id
        self._email = email
        self._role = role
        self._token: str = ""
        self._expires_at: float = 0.0
        # A token handed in by the workflow. Used only until we successfully mint
        # our own; it carries no expiry we can read without decoding, so it is
        # treated as "valid until something 401s".
        self._static_token: str = os.getenv("EVAL_ACCESS_TOKEN", "").strip()
        self._can_mint: bool = True

    def _mint(self) -> bool:
        """Mint a fresh access token in-process. Returns True on success."""
        try:
            from app.auth.jwt_handler import issue_tokens

            token = (issue_tokens(self._user_id, self._email, self._role) or {}).get("access_token")
            if not token:
                raise ValueError("issue_tokens returned no access_token")
        except Exception as exc:
            # Expected when the eval drives a REMOTE server from outside the
            # container (no JWT_SECRET_KEY here). Fall back to the static token
            # and stop retrying — retrying an import that cannot work just adds
            # noise to every single request.
            self._can_mint = False
            logger.warning(event="eval_token_mint_unavailable", error=str(exc))
            return False

        self._token = token
        self._expires_at = time.time() + (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        logger.info(
            event="eval_token_minted",
            expires_in_sec=int(self._expires_at - time.time()),
        )
        return True

    def token(self, force_refresh: bool = False) -> str:
        """Current bearer token, refreshed if it is expired or close to it."""
        if self._can_mint:
            stale = (
                force_refresh
                or not self._token
                or time.time() >= (self._expires_at - _REFRESH_MARGIN_SEC)
            )
            if stale and self._mint():
                return self._token
            if self._token and not force_refresh:
                return self._token
        return self._static_token

    def headers(self, force_refresh: bool = False) -> dict[str, str]:
        tok = self.token(force_refresh=force_refresh)
        return {"Authorization": f"Bearer {tok}"} if tok else {}


def _sleep_for_rate_limit(resp: Any, attempt: int) -> float:
    """Seconds to wait before retrying a 429, honouring Retry-After."""
    retry_after = ""
    try:
        retry_after = (resp.headers or {}).get("Retry-After", "")
    except Exception:
        retry_after = ""
    if retry_after:
        try:
            return min(float(retry_after), _RATE_LIMIT_MAX_SLEEP_SEC)
        except (TypeError, ValueError):
            pass
    return min(_RATE_LIMIT_BASE_SLEEP_SEC * (2**attempt), _RATE_LIMIT_MAX_SLEEP_SEC)


def post_json(
    url: str,
    payload: dict[str, Any],
    auth: EvalAuth,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST `payload` to `url` with a live token, transparently handling 401/429.

    Raises the underlying requests exception if the call genuinely fails, so a
    real server error is still recorded as one — only expiry and rate limiting
    are absorbed, because neither is a property of the system under test.
    """
    import requests

    refreshed_once = False
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        resp = requests.post(url, json=payload, headers=auth.headers(), timeout=timeout)

        if resp.status_code == 401 and not refreshed_once:
            # Token died earlier than the margin predicted (clock skew, or a
            # `gen` bump revoked it). Re-mint once and replay.
            refreshed_once = True
            logger.warning(event="eval_token_expired_retrying", url=url)
            auth.token(force_refresh=True)
            continue

        if resp.status_code == 429 and attempt < _RATE_LIMIT_MAX_RETRIES:
            wait = _sleep_for_rate_limit(resp, attempt)
            logger.warning(
                event="eval_rate_limited_backing_off",
                url=url,
                attempt=attempt + 1,
                sleep_sec=wait,
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    # Exhausted rate-limit retries — surface it rather than returning a fake row.
    resp.raise_for_status()
    return resp.json()
