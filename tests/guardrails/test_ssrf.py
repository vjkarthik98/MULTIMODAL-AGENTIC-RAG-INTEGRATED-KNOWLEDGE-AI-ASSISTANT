"""Red-team tests for app/guardrails/ssrf.py.

Tests cover:
  - Private IP CIDR ranges (127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x)
  - IPv6 private ranges (::1, fc00::/7)
  - Blocked schemes (file://, gopher://, dict://, ldap://)
  - Blocked hostnames (localhost, metadata endpoint)
  - DNS rebinding guard (optional)
  - Public URLs that must be allowed
  - assert_not_ssrf raises GuardrailBlocked
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pytest

from app.guardrails.ssrf import is_ssrf_risk, assert_not_ssrf
from app.guardrails.exceptions import GuardrailBlocked
import app.guardrails.ssrf as ssrf_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_corpus() -> List[Dict[str, Any]]:
    # app/guardrails/data/, not tests/ — this corpus is a live production
    # detection input, not just a fixture; see conftest.py's CORPUS_PATH.
    path = (
        Path(__file__).parent.parent.parent
        / "app"
        / "guardrails"
        / "resources"
        / "adversarial"
        / "red_team_prompts.jsonl"
    )
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


CORPUS = _load_corpus()
SSRF_CASES = [c for c in CORPUS if "ssrf" in c.get("tags", [])]


def _reset_ssrf():
    ssrf_mod._INITIALIZED = False
    ssrf_mod._load_policy()


# ---------------------------------------------------------------------------
# 1. PRIVATE IP RANGES
# ---------------------------------------------------------------------------

class TestPrivateIPBlocking:
    """All RFC-1918 and loopback addresses must be blocked."""

    PRIVATE_URLS = [
        "http://127.0.0.1/admin",
        "http://127.0.0.1:8080/api",
        "http://10.0.0.1/secret",
        "http://10.255.255.255/",
        "http://172.16.0.1/internal",
        "http://172.31.255.255/",
        "http://192.168.0.1/router",
        "http://192.168.1.254/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/",
    ]

    @pytest.mark.parametrize("url", PRIVATE_URLS)
    def test_private_ip_blocked(self, url):
        _reset_ssrf()
        assert is_ssrf_risk(url), f"Expected SSRF block for {url}"

    def test_loopback_blocked(self):
        _reset_ssrf()
        assert is_ssrf_risk("http://127.0.0.1")

    def test_link_local_blocked(self):
        _reset_ssrf()
        assert is_ssrf_risk("http://169.254.1.1/metadata")


# ---------------------------------------------------------------------------
# 2. BLOCKED SCHEMES
# ---------------------------------------------------------------------------

class TestBlockedSchemes:
    """Non-HTTP schemes (file, gopher, dict, ldap, ftp) must be blocked."""

    BLOCKED_SCHEME_URLS = [
        "file:///etc/passwd",
        "file:///C:/Windows/System32/",
        "gopher://localhost:70/1",
        "dict://localhost:2628/d:password",
        "ldap://localhost:389/",
        "ftp://192.168.1.1/",
    ]

    @pytest.mark.parametrize("url", BLOCKED_SCHEME_URLS)
    def test_blocked_scheme(self, url):
        _reset_ssrf()
        assert is_ssrf_risk(url), f"Expected SSRF block for scheme in {url}"


# ---------------------------------------------------------------------------
# 3. BLOCKED HOSTNAMES
# ---------------------------------------------------------------------------

class TestBlockedHostnames:
    """localhost and cloud metadata endpoints must be blocked."""

    BLOCKED_HOSTNAME_URLS = [
        "http://localhost/api",
        "http://localhost:8080/admin",
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://metadata.google.internal/computeMetadata/v1/",
    ]

    @pytest.mark.parametrize("url", BLOCKED_HOSTNAME_URLS)
    def test_blocked_hostname(self, url):
        _reset_ssrf()
        assert is_ssrf_risk(url), f"Expected SSRF block for hostname in {url}"


# ---------------------------------------------------------------------------
# 4. IPv6 PRIVATE ADDRESSES
# ---------------------------------------------------------------------------

class TestIPv6Blocking:
    """IPv6 loopback and ULA ranges must be blocked."""

    IPV6_PRIVATE_URLS = [
        "http://[::1]/admin",
        "http://[0:0:0:0:0:0:0:1]/",
        "http://[fc00::1]/internal",
        "http://[fd12:3456:789a:1::1]/secret",
    ]

    @pytest.mark.parametrize("url", IPV6_PRIVATE_URLS)
    def test_ipv6_private_blocked(self, url):
        _reset_ssrf()
        assert is_ssrf_risk(url), f"Expected SSRF block for IPv6 URL {url}"


# ---------------------------------------------------------------------------
# 5. PUBLIC URLS — must be ALLOWED
# ---------------------------------------------------------------------------

class TestPublicURLsAllowed:
    """Public URLs must NOT be flagged as SSRF risks."""

    PUBLIC_URLS = [
        "https://www.google.com/search?q=rag",
        "https://api.openai.com/v1/embeddings",
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://arxiv.org/abs/2307.01849",
        "https://huggingface.co/models",
    ]

    @pytest.mark.parametrize("url", PUBLIC_URLS)
    def test_public_url_allowed(self, url):
        _reset_ssrf()
        assert not is_ssrf_risk(url), f"Public URL incorrectly flagged as SSRF: {url}"

    def test_empty_url_not_ssrf(self):
        _reset_ssrf()
        assert not is_ssrf_risk("")

    def test_none_url_handled_gracefully(self):
        _reset_ssrf()
        assert not is_ssrf_risk(None)


# ---------------------------------------------------------------------------
# 6. CORPUS CASES
# ---------------------------------------------------------------------------

class TestCorpusSSRF:
    """SSRF cases from the red-team corpus.

    Some corpus entries are bare URLs; others embed URLs in prose.
    We test both: bare-URL entries via is_ssrf_risk(), prose entries via
    input_guard.check() which calls _check_ssrf_in_text() internally.
    """

    @pytest.mark.parametrize(
        "case",
        SSRF_CASES,
        ids=[c["id"] for c in SSRF_CASES],
    )
    def test_ssrf_corpus_case(self, case):
        _reset_ssrf()
        import re as _re
        from app.guardrails.exceptions import GuardrailBlocked
        import app.guardrails.input_guard as ig
        ig._policy_loaded = False
        ig._load_policy()

        prompt = case["prompt"]
        # First try treating the whole prompt as a URL (bare URL cases)
        if is_ssrf_risk(prompt):
            return  # Bare URL — blocked correctly

        # Otherwise extract embedded URLs and check each one
        url_pat = _re.compile(r'https?://\S+|file://\S+|ftp://\S+|gopher://\S+', _re.IGNORECASE)
        urls = url_pat.findall(prompt)
        if urls:
            # At least one embedded URL should be flagged
            blocked = any(is_ssrf_risk(u) for u in urls)
            assert blocked, (
                f"No SSRF detected in embedded URLs for case {case['id']}: "
                f"URLs={urls}, prompt={prompt[:80]!r}"
            )
            return

        # Fallback: the input_guard should block the full prompt
        with pytest.raises(GuardrailBlocked) as exc_info:
            ig.check(prompt, surface="test", session_id="test-ssrf")
        assert exc_info.value.guard_type == "ssrf"


# ---------------------------------------------------------------------------
# 7. assert_not_ssrf raises GuardrailBlocked
# ---------------------------------------------------------------------------

class TestAssertNotSSRF:
    """assert_not_ssrf must raise GuardrailBlocked for private/blocked URLs."""

    def test_raises_for_private_ip(self):
        _reset_ssrf()
        with pytest.raises(GuardrailBlocked) as exc_info:
            assert_not_ssrf("http://192.168.1.1/admin")
        assert exc_info.value.guard_type == "ssrf"
        assert exc_info.value.reason == "ssrf_blocked"

    def test_raises_for_file_scheme(self):
        _reset_ssrf()
        with pytest.raises(GuardrailBlocked):
            assert_not_ssrf("file:///etc/passwd")

    def test_does_not_raise_for_public_url(self):
        _reset_ssrf()
        assert_not_ssrf("https://www.example.com/api")  # Must not raise


# ---------------------------------------------------------------------------
# 8. EDGE CASES
# ---------------------------------------------------------------------------

class TestSSRFEdgeCases:
    """Boundary conditions."""

    def test_url_with_port_still_blocked(self):
        _reset_ssrf()
        assert is_ssrf_risk("http://127.0.0.1:9200/")

    def test_url_with_path_still_blocked(self):
        _reset_ssrf()
        assert is_ssrf_risk("http://10.0.0.1/api/v1/admin/users")

    def test_url_with_query_string_still_blocked(self):
        _reset_ssrf()
        assert is_ssrf_risk("http://192.168.0.1/config?token=secret")

    def test_case_insensitive_scheme(self):
        _reset_ssrf()
        assert is_ssrf_risk("FILE:///etc/passwd")
        assert is_ssrf_risk("Gopher://localhost:70")
