"""SSRF guard — consolidated IP/scheme/hostname blocking.

Replaces the ad-hoc string-prefix matching in web_search.py and
api_routes.py with proper CIDR-based checking via Python stdlib ipaddress.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import List, Optional
from urllib.parse import urlparse

import structlog

from app.guardrails.exceptions import GuardrailBlocked
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Populated from policies.yaml at module init via _load_policy()
_BLOCKED_CIDRS: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
_BLOCKED_SCHEMES: List[str] = []
_BLOCKED_HOSTNAMES: List[str] = []
_INITIALIZED = False


def _load_policy() -> None:
    global _BLOCKED_CIDRS, _BLOCKED_SCHEMES, _BLOCKED_HOSTNAMES, _INITIALIZED
    if _INITIALIZED:
        return
    try:
        from app.guardrails._policy_loader import get_policy
        p = get_policy()
        ssrf_cfg = p.get("ssrf", {})
        _BLOCKED_CIDRS = [
            ipaddress.ip_network(cidr, strict=False)
            for cidr in ssrf_cfg.get("blocked_cidrs", [])
        ]
        _BLOCKED_SCHEMES = [s.lower() for s in ssrf_cfg.get("blocked_schemes", [])]
        _BLOCKED_HOSTNAMES = [h.lower() for h in ssrf_cfg.get("blocked_hostnames", [])]
    except Exception as e:
        logger.warning("ssrf_policy_load_failed", error=str(e))
        # Safe defaults if policy file unavailable
        _BLOCKED_CIDRS = [
            ipaddress.ip_network(c, strict=False)
            for c in [
                "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12",
                "192.168.0.0/16", "169.254.0.0/16", "0.0.0.0/8",
            ]
        ]
        _BLOCKED_SCHEMES = ["file", "ftp", "gopher", "dict", "ldap"]
        _BLOCKED_HOSTNAMES = ["localhost", "169.254.169.254", "metadata.google.internal"]
    _INITIALIZED = True


def _ip_is_blocked(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in _BLOCKED_CIDRS)
    except ValueError:
        return False


def _resolve_and_check(hostname: str) -> bool:
    """Resolve hostname to IPs and check each against blocked CIDRs.

    Guards against DNS rebinding: a hostname that resolves to a private IP
    is blocked even if the hostname itself looks innocent.
    """
    try:
        results = socket.getaddrinfo(hostname, None)
        for result in results:
            ip_str = result[4][0]
            if _ip_is_blocked(ip_str):
                return True
    except (socket.gaierror, OSError):
        pass
    return False


def is_ssrf_risk(url: str, resolve_dns: bool = False) -> bool:
    """Return True if the URL poses an SSRF risk.

    Args:
        url:         The URL to check.
        resolve_dns: If True, resolve the hostname and check the resulting IPs.
                     Set False in hot paths (web search query validation) where
                     latency matters. Set True for user-supplied URLs.
    """
    _load_policy()
    if not url:
        return False

    url_lower = url.lower().strip()
    parsed = urlparse(url_lower)

    # Scheme check
    scheme = parsed.scheme
    if scheme and scheme in _BLOCKED_SCHEMES:
        logger.warning("ssrf_blocked_scheme", scheme=scheme, url=url[:120])
        return True

    hostname = parsed.hostname or ""

    # Literal hostname check
    if hostname in _BLOCKED_HOSTNAMES:
        logger.warning("ssrf_blocked_hostname", hostname=hostname, url=url[:120])
        return True

    # Direct IP check
    if _ip_is_blocked(hostname):
        logger.warning("ssrf_blocked_ip", ip=hostname, url=url[:120])
        return True

    # DNS rebinding check (optional, adds latency)
    if resolve_dns and hostname:
        if _resolve_and_check(hostname):
            logger.warning("ssrf_blocked_dns_rebinding", hostname=hostname, url=url[:120])
            return True

    return False


def assert_not_ssrf(
    url: str,
    correlation_id: str = "",
    resolve_dns: bool = False,
) -> None:
    """Raise GuardrailBlocked if the URL is an SSRF risk."""
    if is_ssrf_risk(url, resolve_dns=resolve_dns):
        raise GuardrailBlocked(
            reason="ssrf_blocked",
            surface="input",
            guard_type="ssrf",
            correlation_id=correlation_id,
            detail=f"URL blocked: {url[:80]}",
        )


def get_blocked_domains_from_policy() -> Optional[List[str]]:
    """Return extra blocked domains from policy (for web_search blocklist)."""
    _load_policy()
    return _BLOCKED_HOSTNAMES
