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

CORRECTION (found 2026-08-22, quality-live.yml's ragas-report/deepeval-report
jobs): the paragraph above is only half true. It assumed minting OUTSIDE the
container fails with an exception (missing dependency, unreachable import),
which correctly falls back to EVAL_ACCESS_TOKEN. That assumption breaks the
moment the caller has ALSO run `pip install -r requirements.txt` (exactly what
quality-live.yml already did before this fix) — `issue_tokens()` then imports
and signs successfully, using whatever JWT_SECRET_KEY happens to be present in
that environment, which is NOT the real server's secret. The result is a
syntactically valid token with the wrong signature: every request gets a real
401, `post_json()`'s one 401-retry re-mints an equally-wrong token, and the
whole run fails end to end with no report ever produced — which is exactly why
quality-reports/ragas/ and quality-reports/deepeval/ have only ever held a
`.gitkeep`. `_can_mint` only ever measured "did minting throw", never "did the
server actually accept what we minted", so this failure mode never self-healed.

Fixed by giving EvalAuth a genuine third path, not just a better fallback:
EVAL_REPORTER_EMAIL / EVAL_REPORTER_PASSWORD, when both are set, make it call
the real POST /auth/login on EVAL_SERVER_URL instead of minting in-process —
producing a token the server issued itself, so it is correct by construction
regardless of what JWT_SECRET_KEY (if any) is present locally. This is the
supported way to run the eval suite against a real deployed server from
outside its container/secret boundary; EVAL_ACCESS_TOKEN remains for the case
where a token was obtained some other way and handed in directly.
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
        # Real login credentials for a dedicated, non-interactive eval-reporter
        # account (see app/bin/seed_eval_reporter.py). When both are set, _mint()
        # calls the real POST /auth/login instead of minting in-process — the
        # only path that is correct-by-construction when this process does not
        # share the server's own JWT_SECRET_KEY. Takes priority over in-process
        # minting (checked first in _mint()), never over an explicit EVAL_ACCESS_TOKEN
        # someone deliberately handed in — this class does not second-guess that.
        self._login_email: str = os.getenv("EVAL_REPORTER_EMAIL", "").strip()
        self._login_password: str = os.getenv("EVAL_REPORTER_PASSWORD", "")
        self._can_mint: bool = True

    def _login(self) -> bool:
        """Fetch a real token via POST /auth/login against EVAL_SERVER_URL.

        Used instead of in-process minting when EVAL_REPORTER_EMAIL /
        EVAL_REPORTER_PASSWORD are set — the token the server hands back is
        correct by construction (the server signed it itself), unlike an
        in-process mint outside the server's own secret boundary, which signs
        with the wrong key and gets silently, permanently rejected. See this
        module's docstring "CORRECTION" note for the incident that found this.
        """
        try:
            import requests

            from app.eval.runners.generation_runner import _SERVER_URL

            resp = requests.post(
                f"{_SERVER_URL}/auth/login",
                json={"email": self._login_email, "password": self._login_password},
                timeout=30,
            )
            resp.raise_for_status()
            token = resp.json().get("access_token")
            if not token:
                raise ValueError("login response had no access_token")
        except Exception as exc:
            logger.error(event="eval_login_failed", error=str(exc))
            return False

        self._token = token
        self._expires_at = time.time() + (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        logger.info(
            event="eval_token_login",
            expires_in_sec=int(self._expires_at - time.time()),
        )
        return True

    def _mint(self) -> bool:
        """Obtain a fresh access token. Returns True on success."""
        if self._login_email and self._login_password:
            return self._login()

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


def post_sse(
    url: str,
    payload: dict[str, Any],
    auth: EvalAuth,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST to an SSE endpoint (/rag/query/stream) and assemble the streamed
    response into the same {"answer", "sources"} shape post_json() returns,
    so callers don't need two code paths.

    Added for the hallucination-reduction initiative (Phase 2, 2026-08-13):
    every other eval runner posts to /rag/query (query_pipeline.py), never to
    /rag/query/stream (rag_pipeline.stream(), the endpoint the UI actually
    calls — which also runs `_conversational_rewrap` after verification, a
    tone pass no other eval path exercises). See thresholds.yaml's "KNOWN
    COVERAGE GAP" comment for the full rationale.

    Parses the exact __type__ event protocol app/api/api_routes.py's
    event_stream() emits: plain `data: "<token>"` (JSON-encoded string)
    chunks concatenate into the answer; `{"__type__":"replace",...}`
    supersedes them (the canonical guarded answer); `{"__type__":"sources"}`
    carries the citation array; `{"__type__":"refusal"}` marks an explicit
    refusal. `data: [DONE]` ends the stream (a literal sentinel, not JSON —
    matches the API's own framing exactly).

    The VerificationReport is NOT recoverable here — rag_pipeline.stream()
    never puts it on the wire, only logs it (event=rag_stream_verification_
    result). This mode is for grading the ANSWER TEXT real users receive
    (post-rewrap), not for verification-metric collection — Phase 1's
    non-streaming path already covers that.
    """
    import json as _json

    import requests

    refreshed_once = False
    resp = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        resp = requests.post(
            url, json=payload, headers=auth.headers(), timeout=timeout, stream=True
        )

        if resp.status_code == 401 and not refreshed_once:
            refreshed_once = True
            logger.warning(event="eval_token_expired_retrying", url=url)
            resp.close()
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
            resp.close()
            time.sleep(wait)
            continue

        resp.raise_for_status()

        answer_parts: list[str] = []
        final_answer: str | None = None
        sources: list[Any] = []
        refused = False
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[len("data: ") :]
            if data == "[DONE]":
                break
            if data.startswith('{"__type__"'):
                try:
                    evt = _json.loads(data)
                except ValueError:
                    continue
                kind = evt.get("__type__")
                if kind == "sources":
                    sources = evt.get("data") or []
                elif kind == "replace":
                    final_answer = evt.get("data")
                elif kind == "refusal":
                    refused = True
                continue
            try:
                answer_parts.append(_json.loads(data))
            except ValueError:
                continue

        full_answer = (final_answer if final_answer is not None else "".join(answer_parts)).strip()
        return {"answer": "" if refused else full_answer, "sources": sources, "refused": refused}

    # Exhausted rate-limit retries.
    if resp is not None:
        resp.raise_for_status()
    return {"answer": "", "sources": [], "refused": False}
