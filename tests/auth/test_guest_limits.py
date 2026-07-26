"""Guest quota enforcement — per-session AND aggregate per-IP.

A per-guest_id counter alone is trivially reset by opening a new tab/incognito
window (each mints a fresh guest_id via POST /auth/guest with its own quota).
These tests prove the aggregate per-IP cap actually bounds total usage across
however many guest sessions are created from one IP.
"""
from __future__ import annotations

import uuid

import pytest

from app.auth.guest_service import (
    check_and_increment_queries,
    check_and_increment_uploads,
    create_guest_session,
)
from app.core.config import settings


def _fake_ip() -> str:
    # Not a real IP — just a unique Redis-key suffix. Must be unique enough
    # to avoid colliding with leftover keys from previous test runs against
    # the same live Redis: guest_ip:{...} counters live for GUEST_SESSION_TTL_HOURS
    # (24h by default), so a small keyspace (e.g. one octet, 255 values) causes
    # flaky cross-run collisions — use the full uuid4 hex instead.
    return f"test-ip-{uuid.uuid4().hex}"


class TestGuestPerSessionLimit:

    def test_query_limit_enforced_per_session(self):
        gid = create_guest_session()
        ip = _fake_ip()
        results = [check_and_increment_queries(gid, ip) for _ in range(settings.GUEST_QUERY_LIMIT + 2)]
        assert results[:settings.GUEST_QUERY_LIMIT] == [True] * settings.GUEST_QUERY_LIMIT
        assert results[settings.GUEST_QUERY_LIMIT:] == [False, False]

    def test_upload_limit_enforced_per_session(self):
        gid = create_guest_session()
        ip = _fake_ip()
        results = [check_and_increment_uploads(gid, ip) for _ in range(settings.GUEST_FILE_LIMIT + 2)]
        assert results[:settings.GUEST_FILE_LIMIT] == [True] * settings.GUEST_FILE_LIMIT
        assert results[settings.GUEST_FILE_LIMIT:] == [False, False]


class TestGuestAggregatePerIpLimit:

    def test_new_guest_session_does_not_reset_ip_quota(self):
        # Two DIFFERENT guest_ids (simulating two browser tabs) from the SAME
        # IP must share one aggregate query budget, not get 5+5 independently.
        ip = _fake_ip()
        g1 = create_guest_session()
        g2 = create_guest_session()

        allowed_g1 = sum(check_and_increment_queries(g1, ip) for _ in range(settings.GUEST_QUERY_LIMIT))
        assert allowed_g1 == settings.GUEST_QUERY_LIMIT

        # g2 is a brand-new guest_id — its OWN per-session counter is at 0 —
        # but the IP aggregate is already at GUEST_QUERY_LIMIT, so unless
        # GUEST_IP_QUERY_LIMIT is more than double GUEST_QUERY_LIMIT, g2 must
        # still be bounded by the shared IP budget.
        remaining_ip_budget = max(0, settings.GUEST_IP_QUERY_LIMIT - settings.GUEST_QUERY_LIMIT)
        allowed_g2 = sum(check_and_increment_queries(g2, ip) for _ in range(settings.GUEST_QUERY_LIMIT))
        assert allowed_g2 <= min(settings.GUEST_QUERY_LIMIT, remaining_ip_budget)

    def test_many_fresh_sessions_capped_by_ip_aggregate(self):
        # Simulate opening enough tabs to exceed GUEST_IP_QUERY_LIMIT even
        # though each individual session respects its own smaller limit.
        ip = _fake_ip()
        total_allowed = 0
        n_tabs = (settings.GUEST_IP_QUERY_LIMIT // settings.GUEST_QUERY_LIMIT) + 3
        for _ in range(n_tabs):
            gid = create_guest_session()
            for _ in range(settings.GUEST_QUERY_LIMIT):
                if check_and_increment_queries(gid, ip):
                    total_allowed += 1
        assert total_allowed == settings.GUEST_IP_QUERY_LIMIT

    def test_different_ips_do_not_share_budget(self):
        # Two guests from DIFFERENT IPs must not interfere with each other.
        gid_a = create_guest_session()
        gid_b = create_guest_session()
        ip_a, ip_b = _fake_ip(), _fake_ip()

        allowed_a = sum(check_and_increment_queries(gid_a, ip_a) for _ in range(settings.GUEST_QUERY_LIMIT))
        allowed_b = sum(check_and_increment_queries(gid_b, ip_b) for _ in range(settings.GUEST_QUERY_LIMIT))
        assert allowed_a == settings.GUEST_QUERY_LIMIT
        assert allowed_b == settings.GUEST_QUERY_LIMIT

    def test_upload_aggregate_per_ip_enforced(self):
        ip = _fake_ip()
        total_allowed = 0
        n_tabs = (settings.GUEST_IP_FILE_LIMIT // settings.GUEST_FILE_LIMIT) + 3
        for _ in range(n_tabs):
            gid = create_guest_session()
            for _ in range(settings.GUEST_FILE_LIMIT):
                if check_and_increment_uploads(gid, ip):
                    total_allowed += 1
        assert total_allowed == settings.GUEST_IP_FILE_LIMIT
