"""Core evaluation orchestration: load gold → run suites → check thresholds → collect results.

Usage (programmatic):
    from app.eval.runner import EvalRunner
    from app.eval.config import EvalConfig, WeakenSpec
    runner = EvalRunner(EvalConfig())
    results = runner.run(["retrieval", "generation"])
    exit_code = runner.check_thresholds(results)

The runner itself is infra-free — individual suite runners import pipeline code.
"""

from __future__ import annotations

import math
import time
from typing import Any

import yaml

from app.eval.config import EvalConfig, load_config
from app.eval.metrics.base import MetricResult, SuiteResult


class EvalRunner:
    def __init__(self, cfg: EvalConfig | None = None):
        self.cfg = cfg or load_config()
        self._thresholds: dict[str, Any] | None = None

    def _load_thresholds(self) -> dict[str, Any]:
        if self._thresholds is not None:
            return self._thresholds
        if not self.cfg.thresholds_path.exists():
            return {}
        with open(self.cfg.thresholds_path) as f:
            self._thresholds = yaml.safe_load(f) or {}
        return self._thresholds

    def run(self, suites: list[str]) -> dict[str, SuiteResult]:
        """Run named suites and return {suite_name: SuiteResult}."""
        results: dict[str, SuiteResult] = {}

        for suite in suites:
            print(f"\n{'='*60}")
            print(f"  Running suite: {suite}")
            print(f"{'='*60}")
            t0 = time.time()

            try:
                suite_result = self._dispatch(suite)
            except Exception as exc:
                suite_result = SuiteResult(suite=suite)
                suite_result.breached["runner_exception"] = str(exc)
                print(f"  [ERROR] Suite '{suite}' raised: {exc}")

            suite_result.duration_sec = time.time() - t0
            results[suite] = suite_result

            # Print per-suite summary
            for name, m in suite_result.metrics.items():
                if not math.isnan(m.value):
                    print(
                        f"  {name:40s} = {m.value:.4f}  (n={m.n}) {m.notes[:60] if m.notes else ''}"
                    )
                else:
                    print(f"  {name:40s} = nan  [{m.notes}]")

            if suite_result.breached:
                for k, v in suite_result.breached.items():
                    print(f"  [BREACH/ERROR] {k}: {v}")

        return results

    def _dispatch(self, suite: str) -> SuiteResult:
        """Route suite name to its runner."""
        if suite == "retrieval":
            from app.eval.runners.retrieval_runner import run_retrieval_suite

            return run_retrieval_suite(self.cfg)

        elif suite == "generation":
            from app.eval.runners.generation_runner import run_generation_suite

            return run_generation_suite(self.cfg)

        elif suite == "hallucination":
            from app.eval.runners.generation_runner import run_hallucination_suite

            return run_hallucination_suite(self.cfg)

        elif suite == "ocr":
            from app.eval.runners.ocr_runner import run_ocr_suite

            return run_ocr_suite(self.cfg)

        elif suite == "audio":
            from app.eval.runners.audio_runner import run_audio_suite

            return run_audio_suite(self.cfg)

        elif suite == "video":
            from app.eval.runners.video_runner import run_video_suite

            return run_video_suite(self.cfg)

        elif suite == "routing":
            from app.eval.runners.routing_runner import run_routing_suite

            return run_routing_suite(self.cfg)

        elif suite == "e2e":
            from app.eval.runners.e2e_runner import run_e2e_suite

            return run_e2e_suite(self.cfg)

        elif suite == "behavioral":
            from app.eval.runners.behavioral_runner import run_behavioral_suite

            return run_behavioral_suite(self.cfg)

        elif suite == "multimodal":
            from app.eval.runners.multimodal_runner import run_multimodal_suite

            return run_multimodal_suite(self.cfg)

        elif suite == "regression":
            from app.eval.runners.regression_runner import run_regression_suite

            return run_regression_suite(self.cfg)

        elif suite == "full":
            # Full runs all core suites — multimodal and regression are added too
            combined = SuiteResult(suite="full")
            sub_suites = [
                "retrieval",
                "generation",
                "behavioral",
                "routing",
                "e2e",
                "ocr",
                "audio",
                "video",
                "regression",
            ]
            for s in sub_suites:
                try:
                    r = self._dispatch(s)
                    # Prefix metrics with suite name to avoid collisions
                    for name, m in r.metrics.items():
                        prefixed = MetricResult(
                            name=f"{s}.{name}",
                            value=m.value,
                            n=m.n,
                            notes=m.notes,
                            sub=m.sub,
                        )
                        combined.add(prefixed)
                    combined.breached.update({f"{s}.{k}": v for k, v in r.breached.items()})
                except Exception as exc:
                    combined.breached[f"{s}.runner_error"] = str(exc)
            return combined

        else:
            sr = SuiteResult(suite=suite)
            sr.breached["unknown_suite"] = f"Unknown suite '{suite}'. Valid: {self._valid_suites()}"
            return sr

    @staticmethod
    def _valid_suites() -> list[str]:
        return [
            "retrieval",
            "generation",
            "hallucination",
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

    def check_thresholds(self, results: dict[str, SuiteResult]) -> int:
        """Check all metrics against thresholds.yaml. Returns 0=pass, 1=breach, 2=error."""
        thresholds = self._load_thresholds()
        if not thresholds:
            print("[WARN] thresholds.yaml not found or empty — skipping threshold check")
            return 0
        if thresholds.get("gate_enabled") is False:
            print(
                "[GATE] gate_enabled: false — thresholds are informational only "
                "(pending baseline v4). Skipping gate. Exit code 0."
            )
            return 0

        breaches: list[str] = []
        errors: list[str] = []

        for suite_name, suite_result in results.items():
            if suite_result.breached:
                for k, v in suite_result.breached.items():
                    if "error" in k.lower():
                        errors.append(f"{suite_name}.{k}: {v}")
                    else:
                        breaches.append(f"{suite_name}.{k}: {v}")

            for metric_name, m in suite_result.metrics.items():
                if math.isnan(m.value):
                    continue  # skip uncomputed metrics
                # Strip suite prefix for threshold lookup (e.g. "full.retrieval.recall_at_5")
                bare_name = metric_name.split(".")[-1]
                threshold = self._find_threshold(thresholds, bare_name)
                if not threshold:
                    continue
                min_val = threshold.get("min")
                max_val = threshold.get("max")
                if min_val is not None and m.value < float(min_val):
                    msg = (
                        f"[BREACH] {metric_name}: {m.value:.4f} < min {min_val} "
                        f"({threshold.get('why', '')})"
                    )
                    breaches.append(msg)
                    print(msg)
                if max_val is not None and m.value > float(max_val):
                    msg = (
                        f"[BREACH] {metric_name}: {m.value:.4f} > max {max_val} "
                        f"({threshold.get('why', '')})"
                    )
                    breaches.append(msg)
                    print(msg)

        if errors:
            print(f"\n[GATE] Infrastructure/data errors: {len(errors)}")
            return 2
        if breaches:
            print(f"\n[GATE] {len(breaches)} threshold breach(es). Exit code 1.")
            return 1
        print("\n[GATE] All thresholds passed. Exit code 0.")
        return 0

    @staticmethod
    def _find_threshold(thresholds: dict[str, Any], metric_name: str) -> dict | None:
        """Recursively search the nested threshold dict for a metric entry."""
        # Direct lookup
        if metric_name in thresholds:
            v = thresholds[metric_name]
            if isinstance(v, dict) and ("min" in v or "max" in v):
                return v
        # Nested lookup (e.g. thresholds["retrieval"]["recall_at_5"])
        for section_val in thresholds.values():
            if isinstance(section_val, dict):
                if metric_name in section_val:
                    v = section_val[metric_name]
                    if isinstance(v, dict) and ("min" in v or "max" in v):
                        return v
                # One more level deep
                for v2 in section_val.values():
                    if isinstance(v2, dict) and metric_name in v2:
                        candidate = v2[metric_name]
                        if isinstance(candidate, dict) and (
                            "min" in candidate or "max" in candidate
                        ):
                            return candidate
        return None
