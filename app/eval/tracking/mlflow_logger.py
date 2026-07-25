"""MLflow experiment tracking with file backend.

Each `python -m app.eval.run` invocation is one MLflow run.
Backend: local filesystem (mlruns/ in repo root, gitignored).
Phase 30 can point this at a hosted server with one MLFLOW_TRACKING_URI swap.

Logged params: git_sha, dataset_version, prompt_version, embedding_model,
               reranker_model, judge_model, top_k, hybrid_weights, mmr_lambda.
Logged metrics: every MetricResult from every suite.
Logged artifacts: rag_report.json, rag_report.md, thresholds.yaml.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.eval.metrics.base import SuiteResult

MLFLOW_TRACKING_URI = "mlruns"  # file-backend; override with env var MLFLOW_TRACKING_URI
EXPERIMENT_NAME = "multimodal-rag-eval"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _dataset_version(gold_dir: Path) -> str:
    manifest = gold_dir.parent / "manifest.yaml"
    if manifest.exists():
        import yaml  # type: ignore

        try:
            data = yaml.safe_load(manifest.read_text())
            return str(data.get("dataset_version", "unknown"))
        except Exception:
            pass
    return "unknown"


def _prompt_version() -> str:
    try:
        from app.prompt.prompt_builder import PROMPT_VERSION

        return PROMPT_VERSION
    except Exception:
        return "unknown"


def log_eval_run(
    suite_results: dict[str, SuiteResult],
    cfg: Any,
    report_path: Path | None = None,
    report_md_path: Path | None = None,
    exit_code: int = 0,
) -> str | None:
    """Log an eval run to MLflow. Returns the run_id, or None if MLflow unavailable."""
    try:
        import mlflow
    except ImportError:
        return None

    tracking_uri = __import__("os").environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # ── Parameters ──────────────────────────────────────────────────────
        mlflow.log_param("git_sha", _git_sha())
        mlflow.log_param("dataset_version", _dataset_version(cfg.gold_dir))
        mlflow.log_param("judge_model", cfg.judge_model)
        mlflow.log_param("user_id", cfg.user_id)
        mlflow.log_param("exit_code", exit_code)
        # So a prompt-template edit that shifts scores is distinguishable from
        # a genuine model/retrieval regression when comparing runs later.
        mlflow.log_param("prompt_version", _prompt_version())

        # Read pipeline config values from settings
        try:
            from app.core.config import get_settings

            s = get_settings()
            mlflow.log_param("embedding_model", getattr(s, "EMBEDDING_MODEL", "unknown"))
            mlflow.log_param(
                "top_k", getattr(s, "DEFAULT_TOP_K", getattr(s, "RAG_TOP_K", "unknown"))
            )
            mlflow.log_param("vector_top_k", getattr(s, "VECTOR_TOP_K", "unknown"))
        except Exception:
            pass

        if cfg.weaken.is_active():
            mlflow.log_param("weaken_spec", str(cfg.weaken))

        # ── Metrics ─────────────────────────────────────────────────────────
        for suite_name, suite_result in suite_results.items():
            for metric_name, metric in suite_result.metrics.items():
                v = metric.value
                if v != v:  # nan — MLflow can't store NaN
                    continue
                try:
                    mlflow.log_metric(f"{suite_name}/{metric_name}", float(v))
                except Exception:
                    pass

            if suite_result.duration_sec:
                mlflow.log_metric(f"{suite_name}/duration_sec", suite_result.duration_sec)

        # ── Artifacts ───────────────────────────────────────────────────────
        if report_path and report_path.exists():
            mlflow.log_artifact(str(report_path))
        if report_md_path and report_md_path.exists():
            mlflow.log_artifact(str(report_md_path))

        thresholds_path = Path(__file__).resolve().parents[1] / "thresholds.yaml"
        if thresholds_path.exists():
            mlflow.log_artifact(str(thresholds_path))

        return run.info.run_id
