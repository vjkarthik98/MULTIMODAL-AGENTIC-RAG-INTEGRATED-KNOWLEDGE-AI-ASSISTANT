"""DeepEval suite — a second, independent OSS LLM-eval framework alongside Ragas.

Why a second framework: app/eval/metrics/generation.py already scores MAGIK
with Ragas (answer_relevancy, context_recall, faithfulness — see
app/eval/ragas_report.py for the dedicated Ragas report). DeepEval
(Apache-2.0, github.com/confident-ai/deepeval) computes faithfulness,
relevancy, and hallucination with a different methodology (G-Eval /
QAG-based, not Ragas's statement-decomposition approach), plus two metrics
Ragas doesn't have at all: bias and toxicity of the generated answer. Two
independent frameworks agreeing (or disagreeing) on the same gold queries is
a stronger, more portfolio-credible claim than either alone.

Judge: MAGIK's single dedicated eval judge (app/eval/judges/qwen_judge.py —
Qwen2.5-7B-Instruct), never OpenAI — this project's whole positioning is
100%-open-source and privacy-preserving (see CLAUDE.md), and DeepEval's
default judge is GPT-4o, which would silently break that claim.
MagikLocalLLM below is DeepEval's documented extension point for exactly
this (DeepEvalBaseLLM), calling qwen_judge.generate() directly. This used to
route through the live app server (`/rag/llm/generate`) and judge the RAG
model with itself — a real self-evaluation-bias concern, fixed as a side
effect of consolidating onto one dedicated, separately-loaded judge model
shared with the Tier-2 gate and Ragas report.

Mode (local vs live) is EVAL_SERVER_URL, same convention as the rest of
app/eval/ — see app/eval/ragas_report.py's docstring for the full rationale.
That variable still selects which deployed RAG system's *answers* get
graded; the judge itself always runs locally regardless of mode.

Usage:
    python -m app.eval.deepeval_suite                         # local, txt/pdf/docx gold
    python -m app.eval.deepeval_suite --modality txt --limit 10
    EVAL_SERVER_URL=https://magik.vk-ai.online \\
        python -m app.eval.deepeval_suite --limit 5           # explicit, capped live run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from app.eval.config import EvalConfig, load_config
from app.eval.runners.generation_runner import (
    _SERVER_URL,
    _full_contexts,
    _load_eval_rows,
    _make_full_context_retriever,
    _query_via_server,
    _server_available,
    _strip_verification_hedge,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "quality-reports" / "deepeval"


def _mode_tag(server_url: str) -> str:
    host = urlparse(server_url).hostname or ""
    return "local" if host in ("127.0.0.1", "localhost") else "live"


def get_deepeval_llm(temperature: float = 0.0):
    """Build a DeepEvalBaseLLM subclass wrapping MAGIK's dedicated eval judge.

    Built lazily inside this function (deepeval is an optional `[quality]`
    extra) so `python -m app.eval.run` and every other eval entrypoint keep
    working without deepeval installed.
    """
    from deepeval.models import DeepEvalBaseLLM

    from app.eval.judges.qwen_judge import _extract_json_from_text
    from app.eval.judges.qwen_judge import generate as _judge_generate

    class _MagikLocalLLM(DeepEvalBaseLLM):
        def __init__(self, temperature: float = 0.0):
            self.temperature = temperature

        def load_model(self):
            return self

        def generate(self, prompt: str, schema=None) -> object:
            raw = _judge_generate(
                prompt,
                system="You are a strict JSON-only evaluator. Output ONLY raw JSON "
                "matching the schema requested in the prompt — no preamble, no "
                "markdown fences, no explanation.",
                temperature=self.temperature,
            )
            extracted = _extract_json_from_text(raw)
            if schema is not None:
                try:
                    return schema(**json.loads(extracted))
                except Exception:
                    # Let the metric's own trimAndLoadJson parsing have a shot
                    # at the raw text rather than hard-failing the whole run.
                    return extracted
            return extracted

        async def a_generate(self, prompt: str, schema=None) -> object:
            return await asyncio.to_thread(self.generate, prompt, schema)

        def get_model_name(self) -> str:
            return "qwen2.5-7b-instruct (local, no OpenAI)"

    return _MagikLocalLLM(temperature=temperature)


# Metrics that need only (input, actual_output) — run on every answer.
_SAFETY_METRICS = ("bias", "toxicity")
# Metrics that need retrieval context / expected_output too.
_QUALITY_METRICS = ("faithfulness", "answer_relevancy", "hallucination", "contextual_recall")
ALL_METRICS = _SAFETY_METRICS + _QUALITY_METRICS


def _build_metrics(names: list[str], model, threshold: float = 0.5) -> dict:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BiasMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        HallucinationMetric,
        ToxicityMetric,
    )

    factory = {
        "faithfulness": lambda: FaithfulnessMetric(
            threshold=threshold, model=model, include_reason=False
        ),
        "answer_relevancy": lambda: AnswerRelevancyMetric(
            threshold=threshold, model=model, include_reason=False
        ),
        "hallucination": lambda: HallucinationMetric(
            threshold=threshold, model=model, include_reason=False
        ),
        "contextual_recall": lambda: ContextualRecallMetric(
            threshold=threshold, model=model, include_reason=False
        ),
        "bias": lambda: BiasMetric(threshold=threshold, model=model, include_reason=False),
        "toxicity": lambda: ToxicityMetric(threshold=threshold, model=model, include_reason=False),
    }
    return {name: factory[name]() for name in names if name in factory}


async def _build_eval_rows(cfg: EvalConfig, limit: int | None) -> tuple[list[dict], list[str]]:
    if not _server_available():
        raise RuntimeError(
            f"MAGIK API not reachable at {_SERVER_URL} — start it first "
            f"(local: `docker compose up -d api qdrant redis mongo`; live: set "
            f"EVAL_SERVER_URL to the real deployed URL)."
        )

    gold_rows = _load_eval_rows(cfg)
    if limit:
        gold_rows = gold_rows[:limit]
    if not gold_rows:
        raise RuntimeError(
            "No curated gold rows with reference answers for the requested modality."
        )

    import os

    access_token = os.getenv("EVAL_ACCESS_TOKEN", "")
    full_ctx_retriever = _make_full_context_retriever()
    eval_rows: list[dict] = []
    errors: list[str] = []

    for row in gold_rows:
        query = row["query"]
        session_id = f"{cfg.session_prefix}_deepeval_{row['id']}"
        try:
            result = _query_via_server(
                query=query,
                session_id=session_id,
                user_id=cfg.user_id,
                access_token=access_token,
            )
        except Exception as exc:
            errors.append(f"{row['id']}: {exc}")
            continue

        answer = _strip_verification_hedge(result.get("answer") or result.get("response") or "")
        sources = result.get("sources") or []
        context_texts = _full_contexts(full_ctx_retriever, query, cfg.user_id, session_id)
        if not context_texts:
            context_texts = [s.get("text") or "" for s in sources if isinstance(s, dict)]

        ref = row.get("reference_answer") or ""
        eval_rows.append(
            {
                "row_id": row["id"],
                "query": query,
                "answer": answer,
                "contexts": context_texts or ["(no retrieved context available)"],
                "reference_answer": ref if ref and ref != "TODO" else None,
            }
        )

    return eval_rows, errors


def _run_metrics(eval_rows: list[dict], metric_names: list[str]) -> dict[str, dict]:
    from deepeval.test_case import LLMTestCase

    model = get_deepeval_llm()
    results: dict[str, list[float]] = {name: [] for name in metric_names}
    per_row: list[dict] = []

    for row in eval_rows:
        test_case = LLMTestCase(
            input=row["query"],
            actual_output=row["answer"],
            retrieval_context=row["contexts"],
            context=row["contexts"],
            expected_output=row["reference_answer"],
        )
        applicable = list(metric_names)
        if row["reference_answer"] is None:
            applicable = [m for m in applicable if m != "contextual_recall"]

        metrics = _build_metrics(applicable, model)
        row_scores = {}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                score = metric.score
                results[name].append(score)
                row_scores[name] = score
            except Exception as exc:
                row_scores[name] = f"error: {exc}"
        per_row.append({"row_id": row["row_id"], "scores": row_scores})

    summary = {
        name: {
            "mean": round(sum(vals) / len(vals), 4) if vals else None,
            "n": len(vals),
        }
        for name, vals in results.items()
    }
    return {"summary": summary, "per_row": per_row}


def _write_report(payload: dict, mode: str, errors: list[str]) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    date_tag = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    full = {
        "generated_at": timestamp,
        "mode": mode,
        "server_url": _SERVER_URL,
        "judge": "magik-mistral-7b-gguf (local, no OpenAI)",
        "errors": errors,
        **payload,
    }

    json_path = REPORTS_DIR / f"{date_tag}-{mode}.json"
    md_path = REPORTS_DIR / f"{date_tag}-{mode}.md"

    with open(json_path, "w") as f:
        json.dump(full, f, indent=2, default=str)

    lines = [
        "# DeepEval Report",
        "",
        f"**Mode:** `{mode}` ({_SERVER_URL})  ",
        f"**Generated:** {timestamp}  ",
        "**Judge:** local Mistral-7B GGUF (no OpenAI)  ",
        "",
        "| Metric | Mean | n |",
        "|---|---|---|",
    ]
    for name, s in sorted(full["summary"].items()):
        mean = f"{s['mean']:.4f}" if s["mean"] is not None else "n/a"
        lines.append(f"| `{name}` | {mean} | {s['n']} |")
    lines.append("")
    if errors:
        lines += ["## Errors", ""] + [f"- {e}" for e in errors] + [""]
    lines += [
        "---",
        "",
        "Copy-paste summary for README / portfolio:",
        "",
        "```",
        f"DeepEval ({mode}, {timestamp[:10]}): "
        + ", ".join(
            f"{name}={s['mean']:.3f}"
            for name, s in sorted(full["summary"].items())
            if s["mean"] is not None
        ),
        "```",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DeepEval suite — second-opinion LLM eval, local judge"
    )
    parser.add_argument(
        "--modality",
        default=None,
        choices=["txt", "pdf", "docx", "xlsx", "image", "audio", "video"],
    )
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap the number of gold rows evaluated"
    )
    parser.add_argument(
        "--metrics",
        default=",".join(ALL_METRICS),
        help=f"Comma-separated subset of: {','.join(ALL_METRICS)}",
    )
    args = parser.parse_args()

    try:
        import deepeval  # noqa: F401
    except ImportError:
        print(
            '[deepeval-suite] FATAL: deepeval not installed — pip install "MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT[quality]"'
        )
        return 2

    cfg = load_config()
    if args.modality:
        cfg.modality = args.modality
    if args.user_id:
        cfg.user_id = args.user_id

    mode = _mode_tag(_SERVER_URL)
    print(f"[deepeval-suite] mode={mode} server={_SERVER_URL}")
    if mode == "live":
        print(
            "[deepeval-suite] LIVE MODE — querying the real deployed server. "
            "This wakes the wake-on-demand AWS box if asleep."
        )

    try:
        eval_rows, errors = asyncio.run(_build_eval_rows(cfg, args.limit))
    except RuntimeError as exc:
        print(f"[deepeval-suite] FATAL: {exc}")
        return 2

    if not eval_rows:
        print("[deepeval-suite] FATAL: no queries succeeded: " + "; ".join(errors))
        return 2

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    payload = _run_metrics(eval_rows, metric_names)

    json_path, md_path = _write_report(payload, mode, errors)
    print(f"[deepeval-suite] JSON: {json_path}")
    print(f"[deepeval-suite] MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
