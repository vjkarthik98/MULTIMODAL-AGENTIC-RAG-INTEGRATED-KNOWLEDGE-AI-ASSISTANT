"""Standalone RAGAS report exporter — portfolio-facing, always the real library.

app/eval/metrics/generation.py:compute_generation_metrics_ragas() already
computes real Ragas metrics (ragas==0.1.21, pinned in pyproject.toml):
answer_relevancy + context_recall via ragas.evaluate() against MAGIK's single
judge (app/eval/judges/qwen_judge.py — Qwen2.5-7B-Instruct), plus faithfulness
via a direct NLI call (Ragas's own faithfulness "decompose" step truncates
long contexts — see that function's docstring for why it's computed
separately). This script produces a dedicated, always-Ragas report: "graded
by the real Ragas library, here is exactly how MAGIK's answers scored" is a
specific, reusable claim worth its own artifact for the README / portfolio.

No retrieval/query logic is duplicated here — this reuses the exact same
gold-row loading, live-server querying, and full-context grading helpers
generation_runner.py already uses, so the numbers in this report and in
rag_report.md are always computed the same way.

Mode (local vs live) is entirely a function of EVAL_SERVER_URL, the same env
var generation_runner.py already reads:
  - unset / http://127.0.0.1:8000 (default) -> local mode, docker-compose stack
  - set to the real deployed URL             -> live mode
There is no separate "live" code path to maintain — pointing EVAL_SERVER_URL
at the real URL IS the live run. Nothing in this repo does that by default.

Usage:
    python -m app.eval.ragas_report                        # local, txt/pdf/docx gold
    python -m app.eval.ragas_report --modality txt
    EVAL_SERVER_URL=https://magik.vk-ai.online \\
        python -m app.eval.ragas_report                    # explicit live run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from app.eval.config import EvalConfig, load_config
from app.eval.http_client import EvalAuth
from app.eval.metrics.generation import compute_generation_metrics_ragas
from app.eval.metrics.hallucination import compute_finance_fidelity, hallucination_rate
from app.eval.runners.generation_runner import (
    _SERVER_URL,
    _full_contexts,
    _load_eval_rows,
    _make_full_context_retriever,
    _query_via_server,
    _server_available,
    _strip_verification_hedge,
    release_full_context_models,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "quality-reports" / "ragas"


def _mode_tag(server_url: str) -> str:
    """Which deployment the graded ANSWERS came from — free-form, not an enum.

    EVAL_MODE_TAG overrides the URL sniff, because the URL stopped being a
    reliable proxy for it. quality-report.yml runs on the staging box's own
    self-hosted runner and reaches the staging container over
    `http://127.0.0.1:8000` (the same loopback call tier2-eval.yml makes) — a
    real measurement of a deployed image that the sniff below would stamp
    `local`, on the report filename AND on the public shields.io badge. The
    transport says nothing about what was measured; only the caller knows, so
    the caller may say. That workflow passes "staging" rather than "live" on
    purpose: it grades the promoted image on a box cloned from production, not
    production itself, and the badge should not claim otherwise.
    """
    override = os.getenv("EVAL_MODE_TAG", "").strip()
    if override:
        return override
    host = urlparse(server_url).hostname or ""
    return "local" if host in ("127.0.0.1", "localhost") else "live"


async def _build_eval_rows(
    cfg: EvalConfig, limit: int | None = None
) -> tuple[list[dict], list[str]]:
    """Query the live server for each gold row. Returns (eval_rows, errors).

    `limit` caps how many gold rows are queried, the same way
    deepeval_suite.py's `--limit` already does, and for a concrete reason:
    cd.yml runs this as a SECOND python process inside the running
    `magik-current` container (`docker exec`), so its own model_loader loads a
    second copy of the text embedder and SigLIP alongside everything the live
    app already has resident — and those are exactly the models
    model_loader._EVICTABLE_MODELS deliberately refuses to evict, being
    core query-path models. Meanwhile every row drives a full RAG query
    through the app process in the same container.

    Unbounded, that meant ~17 minutes (v1.0.0-rc3, 98 rows) of two model
    stacks coexisting on a 32GB box. v1.0.0-rc4's run did not survive it: the
    kernel OOM-killer took the eval process at 8m17s (`exit code 137`, roughly
    half way through the query phase), and the cascade knocked the self-hosted
    runner offline, cancelling the job before any report was written.

    Capping the row count is the only lever that actually applies here —
    eviction frees nothing (the loaded models are non-evictable by design) and
    the memory is per-process, not per-request. Fewer rows means a shorter
    window in which both stacks are resident.

    Rows are taken in gold-set order, so a capped run is deterministic and
    comparable between releases rather than a random sample.
    """
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
            "No curated gold rows with reference answers for the requested modality — "
            "run app.eval.datasets.build_gold_set --ingest and review TODO rows first."
        )

    # EvalAuth, not a raw EVAL_ACCESS_TOKEN string. `_query_via_server` takes
    # `auth=` and has for some time; this module still passed `access_token=`,
    # so EVERY row raised
    #     TypeError: got an unexpected keyword argument 'access_token'
    # straight into `errors` and the report was written with zero rows. It went
    # unnoticed because the module crashed earlier still, at import, on
    # Settings.validate() (see quality-live.yml's placeholder env block) — the
    # import crash masked this one. Matches behavioral_runner.py:45,63 and
    # generation_runner.py:378-383, the two callers that were already correct.
    eval_auth = EvalAuth(cfg.user_id)
    full_ctx_retriever = _make_full_context_retriever()
    eval_rows: list[dict] = []
    errors: list[str] = []

    for row in gold_rows:
        query = row["query"]
        session_id = f"{cfg.session_prefix}_ragas_{row['id']}"
        # /rag/query 400s on an empty scope (api_routes.py's FILE SCOPE REQUIRED
        # gate), so an unscoped row never gets an answer at all. 133 of the 164
        # gold rows carry the scope needed to satisfy it.
        _row_sources = row.get("relevant_doc_ids") or (
            [row["source_file"]] if row.get("source_file") else None
        )
        try:
            result = _query_via_server(
                query=query,
                session_id=session_id,
                user_id=cfg.user_id,
                auth=eval_auth,
                sources=_row_sources,
            )
        except Exception as exc:
            errors.append(f"{row['id']}: {exc}")
            continue

        answer = _strip_verification_hedge(result.get("answer") or result.get("response") or "")
        sources = result.get("sources") or []
        # Grading context must mirror the scope the answer was generated under —
        # see _full_contexts' docstring for the live case where unscoped grading
        # context pulled in a different file entirely and faithfulness was scored
        # against contamination.
        context_texts = _full_contexts(
            full_ctx_retriever, query, cfg.user_id, session_id, sources=_row_sources
        )
        if not context_texts:
            context_texts = [s.get("text") or "" for s in sources if isinstance(s, dict)]

        eval_rows.append(
            {
                "query": query,
                "answer": answer,
                "contexts": context_texts,
                "reference_answer": row.get("reference_answer"),
                "retrieved_docs": sources,
                "row_id": row["id"],
                "modality": row.get("modality"),
            }
        )

    # Free the query-phase stack before any judging begins — see
    # release_full_context_models() for why this is both safe (separate
    # process from the serving app) and necessary (the OOM that killed the
    # production runner in rc4 and rc5).
    release_full_context_models()

    return eval_rows, errors


def _write_report(
    eval_rows: list[dict],
    ragas_metrics: dict,
    errors: list[str],
    mode: str,
    server_url: str,
) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    date_tag = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    finance_fidelity_scores = [
        compute_finance_fidelity(r["answer"], r["contexts"]) for r in eval_rows
    ]
    avg_fidelity = (
        sum(finance_fidelity_scores) / len(finance_fidelity_scores)
        if finance_fidelity_scores
        else None
    )
    hallucination = hallucination_rate(eval_rows) if eval_rows else None

    payload = {
        "generated_at": timestamp,
        "mode": mode,
        "server_url": server_url,
        "ragas_version": "0.1.21",
        "judge": "qwen2.5-7b-instruct (direct NLI for faithfulness; ragas.evaluate() for "
        "answer_relevancy/context_recall)",
        "n_queries": len(eval_rows),
        "n_errors": len(errors),
        "errors": errors,
        "metrics": {
            name: m.to_dict() if hasattr(m, "to_dict") else vars(m)
            for name, m in ragas_metrics.items()
        },
        "finance_fidelity": avg_fidelity,
        "hallucination_rate": hallucination.value if hallucination else None,
    }

    json_path = REPORTS_DIR / f"{date_tag}-{mode}.json"
    md_path = REPORTS_DIR / f"{date_tag}-{mode}.md"

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# RAGAS Evaluation Report",
        "",
        f"**Mode:** `{mode}` ({server_url})  ",
        f"**Generated:** {timestamp}  ",
        "**Ragas version:** 0.1.21  ",
        "**Judge:** Qwen2.5-7B-Instruct (local, no OpenAI)  ",
        f"**Queries evaluated:** {len(eval_rows)} ({len(errors)} errors)",
        "",
        "| Metric | Value | n | Notes |",
        "|---|---|---|---|",
    ]
    for name, m in sorted(ragas_metrics.items()):
        lines.append(f"| `{name}` | {m.value:.4f} | {m.n} | {m.notes or ''} |")
    if avg_fidelity is not None:
        lines.append(
            f"| `finance_fidelity` | {avg_fidelity:.4f} | {len(finance_fidelity_scores)} | verbatim number match |"
        )
    if hallucination is not None:
        lines.append(f"| `hallucination_rate` | {hallucination.value:.4f} | {hallucination.n} | |")
    lines.append("")

    if errors:
        lines += ["## Errors", ""] + [f"- {e}" for e in errors] + [""]

    lines += [
        "---",
        "",
        "Copy-paste summary for README / portfolio:",
        "",
        "```",
        f"RAGAS ({mode}, {timestamp[:10]}): "
        + ", ".join(f"{name}={m.value:.3f}" for name, m in sorted(ragas_metrics.items())),
        "```",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone RAGAS report exporter")
    parser.add_argument(
        "--modality",
        default=None,
        choices=["txt", "pdf", "docx", "xlsx", "image", "audio", "video"],
        help="Evaluate only this modality (default: txt/pdf/docx, same as generation_runner)",
    )
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the number of gold rows queried (deterministic, gold-set order). "
            "Mirrors deepeval_suite.py's --limit; see _build_eval_rows for why the "
            "in-container run needs a bound."
        ),
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.modality:
        cfg.modality = args.modality
    if args.user_id:
        cfg.user_id = args.user_id

    mode = _mode_tag(_SERVER_URL)
    print(f"[ragas-report] mode={mode} server={_SERVER_URL}")
    if mode == "live":
        print(
            "[ragas-report] LIVE MODE — querying the real deployed server. "
            "This wakes the wake-on-demand AWS box if asleep."
        )

    try:
        eval_rows, errors = asyncio.run(_build_eval_rows(cfg, args.limit))
    except RuntimeError as exc:
        print(f"[ragas-report] FATAL: {exc}")
        return 2

    if not eval_rows:
        print("[ragas-report] FATAL: no queries succeeded:")
        for e in errors:
            print(f"[ragas-report]   {e}")
        return 2

    ragas_metrics = asyncio.run(compute_generation_metrics_ragas(eval_rows))

    json_path, md_path = _write_report(eval_rows, ragas_metrics, errors, mode, _SERVER_URL)
    print(f"[ragas-report] JSON: {json_path}")
    print(f"[ragas-report] MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
