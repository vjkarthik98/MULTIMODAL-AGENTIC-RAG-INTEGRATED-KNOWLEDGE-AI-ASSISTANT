"""Uniform metric return type so the runner can aggregate without special-casing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MetricResult:
    name: str
    value: float
    n: int
    notes: str = ""
    sub: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls, name: str, reason: str) -> "MetricResult":
        return cls(name=name, value=float("nan"), n=0, notes=f"empty: {reason}")


@dataclass
class SuiteResult:
    suite: str
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    breached: Dict[str, str] = field(default_factory=dict)
    duration_sec: float = 0.0
    judge: Optional[str] = None
    dataset_version: Optional[str] = None

    def add(self, m: MetricResult) -> None:
        self.metrics[m.name] = m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "breached": self.breached,
            "duration_sec": self.duration_sec,
            "judge": self.judge,
            "dataset_version": self.dataset_version,
        }
