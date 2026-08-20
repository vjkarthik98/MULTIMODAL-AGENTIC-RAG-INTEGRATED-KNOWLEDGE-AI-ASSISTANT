"""Eval-only configuration. Reads pipeline settings via get_settings() — never hardcodes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import get_settings

_settings = get_settings()


EVAL_ROOT: Path = _settings.PROJECT_ROOT / "app" / "eval"
DATASETS_DIR: Path = EVAL_ROOT / "datasets"
GOLD_DIR: Path = DATASETS_DIR / "gold"
REPORTS_DIR: Path = EVAL_ROOT / "reports"
BASELINES_DIR: Path = EVAL_ROOT / "baselines"
THRESHOLDS_PATH: Path = EVAL_ROOT / "thresholds.yaml"
MANIFEST_PATH: Path = DATASETS_DIR / "manifest.yaml"
RAW_CORPUS_DIR: Path = _settings.PROJECT_ROOT / "data" / "raw" / "finance"

EVAL_USER_ID: str = (
    os.getenv("EVAL_USER_ID", _settings.PROJECT_ROOT and "eval_default") or "eval_default"
)
EVAL_SESSION_PREFIX: str = "eval"
# Provenance/metadata only (MLflow logging, SuiteResult.judge) — MAGIK has a
# single judge (app/eval/judges/qwen_judge.py), so this no longer selects
# between judges the way it once did. Kept as an env var for override/testing.
EVAL_JUDGE_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", "qwen2.5_7b")
EVAL_JUDGE_TEMPERATURE: float = float(os.getenv("EVAL_JUDGE_TEMPERATURE", "0.1"))
EVAL_FAIL_ON_MISSING_JUDGE: bool = (
    os.getenv("EVAL_FAIL_ON_MISSING_JUDGE", "false").lower() == "true"
)


SUITE_NAMES: list[str] = [
    "retrieval",
    "generation",
    "behavioral",
    "ocr",
    "audio",
    "video",
    "routing",
    "e2e",
    "multimodal",
    "regression",
    "full",
]


@dataclass
class WeakenSpec:
    """--weaken flag: artificial regressions to prove the gate."""

    top_k: int | None = None
    no_rerank: bool = False
    no_mmr: bool = False
    no_rrf: bool = False

    @classmethod
    def parse(cls, raw: str | None) -> WeakenSpec:
        spec = cls()
        if not raw:
            return spec
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                k, v = token.split("=", 1)
                if k.strip() == "top_k":
                    spec.top_k = int(v)
            elif token == "no_rerank":
                spec.no_rerank = True
            elif token == "no_mmr":
                spec.no_mmr = True
            elif token == "no_rrf":
                spec.no_rrf = True
        return spec

    def is_active(self) -> bool:
        return any([self.top_k is not None, self.no_rerank, self.no_mmr, self.no_rrf])

    def as_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "no_rerank": self.no_rerank,
            "no_mmr": self.no_mmr,
            "no_rrf": self.no_rrf,
        }


@dataclass
class EvalConfig:
    user_id: str = field(default=EVAL_USER_ID)
    session_prefix: str = field(default=EVAL_SESSION_PREFIX)
    judge_model: str = field(default=EVAL_JUDGE_MODEL)
    judge_temperature: float = field(default=EVAL_JUDGE_TEMPERATURE)
    gold_dir: Path = field(default=GOLD_DIR)
    reports_dir: Path = field(default=REPORTS_DIR)
    baselines_dir: Path = field(default=BASELINES_DIR)
    thresholds_path: Path = field(default=THRESHOLDS_PATH)
    manifest_path: Path = field(default=MANIFEST_PATH)
    raw_corpus_dir: Path = field(default=RAW_CORPUS_DIR)
    weaken: WeakenSpec = field(default_factory=WeakenSpec)
    # Optional single-modality filter (txt|pdf|docx|xlsx|image|audio|video).
    # When set, retrieval/generation/behavioral suites evaluate ONLY this modality.
    modality: str | None = None
    # When True, run_generation_suite posts to /rag/query/stream (the endpoint
    # the UI actually calls, incl. rag_pipeline.stream()'s post-verification
    # _conversational_rewrap pass) instead of /rag/query. See thresholds.yaml's
    # "KNOWN COVERAGE GAP" comment and app/eval/http_client.py::post_sse().
    live_path: bool = False
    # Set by `app/eval/run.py` from --baseline, read by
    # app/eval/runners/regression_runner.py via getattr(). It worked (plain
    # dataclass, so the attribute could be grafted on at runtime) but was an
    # undeclared, invisible contract between two modules — declared here so
    # the handshake is greppable and type-checked instead of implicit.
    _baseline_path: Path | None = None

    def ensure_dirs(self) -> None:
        for p in (self.gold_dir, self.reports_dir, self.baselines_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_config() -> EvalConfig:
    cfg = EvalConfig()
    cfg.ensure_dirs()
    return cfg
