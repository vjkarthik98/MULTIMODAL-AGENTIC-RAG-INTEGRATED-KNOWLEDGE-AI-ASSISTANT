"""
PDF and DOCX retrieval eval — runs Retriever.retrieval() on PDF and DOCX gold rows,
scores retrieval metrics, and merges results into rag_report.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EVAL_SKIP_LLM_WARMUP", "true")

from app.eval.config import EvalConfig
from app.eval.datasets.gold_loader import load_gold
from app.eval.metrics.retrieval import aggregate_retrieval_metrics
from app.eval.metrics.latency import latency_stats

REPORT_FILE = ROOT / "app/eval/reports/rag_report.json"


def run_retrieval_for_modality(modality: str, cfg: EvalConfig):
    from app.retrieval.retriever import Retriever
    retriever = Retriever()

    gold_rows = load_gold(modality, gold_dir=cfg.gold_dir)
    # Filter: only rows with relevant_chunk_ids (not search-required)
    gold_rows = [r for r in gold_rows if r.get("relevant_chunk_ids") and "SEARCH_REQUIRED" not in r.get("reference_answer", "")]

    if not gold_rows:
        print(f"[{modality.upper()}] No rows with relevant_chunk_ids — skipping retrieval.")
        return None

    eval_results = []
    latencies = []

    for row in gold_rows:
        query = row["query"]
        relevant_ids = row.get("relevant_chunk_ids", [])
        if isinstance(relevant_ids, str):
            relevant_ids = [relevant_ids]

        session_id = f"eval_retrieval_{modality}_{row['id']}"
        q_start = time.time()
        try:
            retrieved = retriever.retrieval(
                query=query,
                session_id=session_id,
                top_k=10,
                user_id=cfg.user_id,
                use_mmr=True,
                use_rrf=True,
            )
        except Exception as e:
            print(f"  [{row['id']}] retrieval error: {e}")
            continue
        elapsed = time.time() - q_start
        latencies.append(elapsed)

        # Build retrieved chunk IDs same way retrieval_runner does
        retrieved_ids = []
        for doc in retrieved:
            meta = doc.metadata if hasattr(doc, "metadata") else (doc.get("metadata") or {})
            source = meta.get("source", "")
            cid = meta.get("chunk_id")
            if source and cid is not None:
                retrieved_ids.append(f"{source}::chunk_{cid}")

        eval_results.append({
            "query_id": row["id"],
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "latency": elapsed,
        })
        n_hit = len(set(retrieved_ids) & set(relevant_ids))
        print(f"  [{row['id']}] retrieved={len(retrieved_ids)} relevant_hit={n_hit}/{len(relevant_ids)} latency={elapsed:.2f}s")

    if not eval_results:
        return None

    metrics = aggregate_retrieval_metrics(eval_results, k5=5, k10=10)
    lat = latency_stats(latencies, prefix="retrieval_")

    n = len(eval_results)
    print(f"\n[{modality.upper()} RETRIEVAL] n={n}")
    for name, mr in metrics.items():
        print(f"  {name} = {mr.value:.4f}")

    return {"metrics": metrics, "latency": lat, "n": n}


def main():
    cfg = EvalConfig()

    if REPORT_FILE.exists():
        with open(REPORT_FILE) as f:
            report = json.load(f)
    else:
        report = {"suites": {}}

    def metric(name, value, n_, notes=""):
        return {"name": name, "value": value, "n": n_, "notes": notes, "sub": {}}

    for modality in ["pdf", "docx"]:
        print(f"\n{'='*50}")
        print(f"Running retrieval for: {modality.upper()}")
        print('='*50)
        result = run_retrieval_for_modality(modality, cfg)
        if result is None:
            print(f"[{modality.upper()}] No retrieval results.")
            continue

        suite_key = f"retrieval_{modality}"
        suite_metrics = {}
        for name, mr in result["metrics"].items():
            suite_metrics[name] = metric(name, mr.value, mr.n, mr.notes)
        for name, mr in result["latency"].items():
            suite_metrics[name] = metric(name, mr.value, mr.n, mr.notes)

        report["suites"][suite_key] = {
            "suite": suite_key,
            "modality": modality,
            "metrics": suite_metrics,
        }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Updated: {REPORT_FILE}")


if __name__ == "__main__":
    main()
