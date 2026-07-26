"""SSRF guard — consolidated IP/scheme/hostname blocking.

Replaces the ad-hoc string-prefix matching in web_search.py and
api_routes.py with proper CIDR-based checking via Python stdlib ipaddress.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from app.guardrails.exceptions import GuardrailBlocked
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Populated from policies.yaml at module init via _load_policy()
_BLOCKED_CIDRS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
_BLOCKED_SCHEMES: list[str] = []
_BLOCKED_HOSTNAMES: list[str] = []
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
            ipaddress.ip_network(cidr, strict=False) for cidr in ssrf_cfg.get("blocked_cidrs", [])
        ]
        _BLOCKED_SCHEMES = [s.lower() for s in ssrf_cfg.get("blocked_schemes", [])]
        _BLOCKED_HOSTNAMES = [h.lower() for h in ssrf_cfg.get("blocked_hostnames", [])]
    except Exception as e:
        logger.warning("ssrf_policy_load_failed", error=str(e))
        # Safe defaults if policy file unavailable
        _BLOCKED_CIDRS = [
            ipaddress.ip_network(c, strict=False)
            for c in [
                "127.0.0.0/8",
                "10.0.0.0/8",
                "172.16.0.0/12",
                "192.168.0.0/16",
                "169.254.0.0/16",
                "0.0.0.0/8",
            ]
        ]
        _BLOCKED_SCHEMES = ["file", "ftp", "gopher", "dict", "ldap"]
        _BLOCKED_HOSTNAMES = [
            "localhost",
            "169.254.169.254",
            "metadata.google.internal",
            "fd00:ec2::254",  # AWS EC2 IMDSv1/v2 IPv6 metadata endpoint
        ]
    _INITIALIZED = True


def _ip_is_blocked(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in _BLOCKED_CIDRS)
    except ValueError:
        return False


_DNS_RESOLVE_TIMEOUT_SEC = 2.0


def _resolve_and_check(hostname: str) -> bool:
    """Resolve hostname to IPs and check each against blocked CIDRs.

    Guards against DNS rebinding: a hostname that resolves to a private IP
    is blocked even if the hostname itself looks innocent.

    Bounded with a short timeout so a slow/unresponsive DNS server can't be
    used to stall the calling request — the timeout is applied via a
    dedicated thread rather than socket.setdefaulttimeout() so it doesn't
    leak into concurrent, unrelated socket calls on other threads.
    """
    import concurrent.futures

    def _lookup() -> list:
        return socket.getaddrinfo(hostname, None)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            results = ex.submit(_lookup).result(timeout=_DNS_RESOLVE_TIMEOUT_SEC)
        for result in results:
            ip_str = result[4][0]
            if _ip_is_blocked(ip_str):
                return True
    except concurrent.futures.TimeoutError:
        logger.warning("ssrf_dns_resolve_timeout", hostname=hostname)
    except (socket.gaierror, OSError):
        pass
    return False


def is_ssrf_risk(url: str, resolve_dns: bool = True) -> bool:
    """Return True if the URL poses an SSRF risk.

    Args:
        url:         The URL to check.
        resolve_dns: If True (default), resolve the hostname and check the
                     resulting IPs — closes the DNS-rebinding gap where an
                     attacker-controlled hostname resolves to a private/
                     metadata IP. Bounded by _DNS_RESOLVE_TIMEOUT_SEC so it
                     cannot be used to stall a request. Callers on a very
                     hot, high-QPS path may pass False to skip the lookup,
                     but every current call site keeps the safe default.
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

    # Fallback: bare IPv6 without brackets (e.g. http://::1/path) or
    # bracketed IPv6 that urlparse failed to parse as hostname.
    if not hostname and parsed.netloc:
        # Strip port suffix and brackets; try raw netloc as an IP address
        raw = parsed.netloc.strip("[]")
        if ":" in raw:
            # Could be IPv6; try as-is first, then strip port
            candidates = [raw]
            if raw.count(":") == 1:
                candidates.append(raw.rsplit(":", 1)[0])  # strip :port for ipv4:port
            for candidate in candidates:
                try:
                    addr = ipaddress.ip_address(candidate)
                    if any(addr in net for net in _BLOCKED_CIDRS):
                        logger.warning(
                            "ssrf_blocked_bare_ipv6", netloc=parsed.netloc, url=url[:120]
                        )
                        return True
                    break
                except ValueError:
                    continue

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
    resolve_dns: bool = True,
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


def get_blocked_domains_from_policy() -> list[str] | None:
    """Return extra blocked domains from policy (for web_search blocklist)."""
    _load_policy()
    return _BLOCKED_HOSTNAMES
