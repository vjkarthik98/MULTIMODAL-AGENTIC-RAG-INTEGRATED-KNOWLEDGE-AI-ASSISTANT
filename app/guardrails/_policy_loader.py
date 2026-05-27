"""Load and cache policies.yaml.

Single import point for all guardrail modules. Cached after first load.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_POLICY: Dict[str, Any] | None = None
_DEFAULT_PATH = Path(__file__).parent / "policies.yaml"


def get_policy(path: Path | None = None) -> Dict[str, Any]:
    global _POLICY
    if _POLICY is not None:
        return _POLICY

    policy_path = path or Path(
        os.environ.get("GUARDRAILS_POLICY_PATH", str(_DEFAULT_PATH))
    )
    with open(policy_path, "r", encoding="utf-8") as fh:
        _POLICY = yaml.safe_load(fh) or {}
    return _POLICY


def reload_policy(path: Path | None = None) -> Dict[str, Any]:
    global _POLICY
    _POLICY = None
    return get_policy(path)
