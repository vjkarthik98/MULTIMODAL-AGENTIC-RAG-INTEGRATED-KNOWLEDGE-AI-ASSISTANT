"""Phase 25 CLI entrypoint.

Usage:
    python -m app.eval.run --suite retrieval
    python -m app.eval.run --suite full
    python -m app.eval.run --suite full --weaken top_k=1,no_rerank
    python -m app.eval.run --suite regression --baseline app/eval/baselines/rag_report_v1.json

Exit codes:
    0 — all thresholds passed (or no thresholds defined)
    1 — one or more thresholds breached
    2 — infrastructure / data error (Qdrant down, gold file missing, etc.)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on path when run as `python -m app.eval.run`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Prevent the eval subprocess from loading GGUF into VRAM — the live server
# already holds the model. The eval judge routes through HTTP (/rag/query).
os.environ.setdefault("EVAL_SKIP_LLM_WARMUP", "true")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 25 RAG Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.eval.run --suite retrieval
  python -m app.eval.run --suite full
  python -m app.eval.run --suite full --weaken top_k=1,no_rerank
  python -m app.eval.run --suite regression --baseline app/eval/baselines/rag_report_v1.json
""",
    )
    parser.add_argument(
        "--suite",
        required=True,
        choices=["retrieval", "generation", "hallucination", "ocr", "audio",
                 "video", "routing", "e2e", "multimodal", "regression", "full"],
        help="Eval suite to run",
    )
    parser.add_argument(
        "--weaken",
        default=None,
        help="Comma-separated weakening spec for gate-proof mode, e.g. top_k=1,no_rerank",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline JSON for regression comparison",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Override eval user_id (default: EVAL_USER_ID env var or eval_default)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing rag_report files",
    )
    args = parser.parse_args()

    # Build config
    from app.eval.config import EvalConfig, WeakenSpec, load_config
    cfg = load_config()

    if args.weaken:
        cfg.weaken = WeakenSpec.parse(args.weaken)
        print(f"[WEAKEN MODE] {cfg.weaken.as_dict()}")

    if args.user_id:
        cfg.user_id = args.user_id

    if args.baseline:
        cfg._baseline_path = Path(args.baseline)  # regression runner picks this up

    # Run
    from app.eval.runner import EvalRunner
    runner = EvalRunner(cfg)

    suites = [args.suite]
    try:
        results = runner.run(suites)
    except Exception as exc:
        print(f"[FATAL] Runner raised an exception: {exc}")
        return 2

    # Write reports
    if not args.no_report:
        try:
            from app.eval.report import write_reports
            write_reports(results, cfg=cfg, weaken_spec=args.weaken)
        except Exception as exc:
            print(f"[WARN] Could not write report: {exc}")

    # Check thresholds
    exit_code = runner.check_thresholds(results)

    # MLflow tracking (file backend; optional — fails gracefully if mlflow not installed)
    if not args.no_report:
        try:
            from app.eval.tracking.mlflow_logger import log_eval_run
            run_id = log_eval_run(
                suite_results=results,
                cfg=cfg,
                report_path=cfg.reports_dir / "rag_report.json",
                report_md_path=cfg.reports_dir / "rag_report.md",
                exit_code=exit_code,
            )
            if run_id:
                print(f"[MLFLOW] Run logged: {run_id}")
        except Exception as exc:
            print(f"[WARN] MLflow logging failed (non-fatal): {exc}")

    if cfg.weaken.is_active() and exit_code == 0:
        print(
            "\n[GATE PROOF] WARNING: Weakened pipeline passed all thresholds. "
            "This means thresholds are too loose or weaken spec had no effect. "
            "Expected exit code 1 for proper gate proof."
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
