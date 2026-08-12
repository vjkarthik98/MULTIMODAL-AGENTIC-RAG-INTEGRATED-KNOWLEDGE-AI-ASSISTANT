"""Prometheus metrics for the auth layer.

Registered once at import time — consistent with the pattern used in
app/guardrails/metrics.py (imported at module top level by every caller,
never lazily inside a function; see app/core/metrics.py's module docstring
for the production incident that pattern exists to avoid).

No user-identifying labels (email, user_id) on any metric here: only counts
plus a bounded `reason` label — per the monitoring layer's rule against
high-cardinality or PII-bearing Prometheus labels. The existing
structlog warnings at each call site (auth_login_failed_wrong_password, etc.)
already carry the email for operator debugging; these Counters are for
alerting/dashboards, not investigation, so they deliberately carry less.
"""

from __future__ import annotations

from prometheus_client import Counter

auth_login_failures_total = Counter(
    "auth_login_failures_total",
    "Failed login attempts by reason",
    ["reason"],
)

auth_mfa_failures_total = Counter(
    "auth_mfa_failures_total",
    "Failed TOTP/backup-code MFA verification attempts",
)

auth_rate_limit_rejections_total = Counter(
    "auth_rate_limit_rejections_total",
    "Requests rejected by the per-user rate limiter",
)


def record_login_failure(reason: str) -> None:
    auth_login_failures_total.labels(reason=reason).inc()


def record_mfa_failure() -> None:
    auth_mfa_failures_total.inc()


def record_rate_limit_rejection() -> None:
    auth_rate_limit_rejections_total.inc()
