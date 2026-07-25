"""
Retrieval eval for XLSX, image, audio, video modalities.
Runs HybridRetriever.search() and scores metrics, merges into rag_report.json.
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

REPORT_FILE = ROOT / "app/eval/reports/rag_report.json"


def run_retrieval_for_modality(modality: str, cfg: EvalConfig):
    from app.core.infra_registry import infra
    from app.core.model_loader import model_loader
    from app.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(
        bm25=infra.get_bm25(),
        vector_store=infra.get_vector_store(),
        embedder=model_loader.get_embedder(),
    )

    gold_rows = load_gold(modality, gold_dir=cfg.gold_dir)
    gold_rows = [r for r in gold_rows if r.get("relevant_chunk_ids") and "SEARCH_REQUIRED" not in r.get("reference_answer", "")]

    if not gold_rows:
        print(f"[{modality.upper()}] No retrievable rows — skipping.")
        return None

    eval_results = []
    latencies = []

    for row in gold_rows:
        query = row["query"]
        relevant_ids = row.get("relevant_chunk_ids", [])
        if isinstance(relevant_ids, str):
            relevant_ids = [relevant_ids]

        session_id = f"eval_retrieval_{modality}_{row['id']}"
        t0 = time.time()
        try:
            retrieved = retriever.search(
                query=query,
                session_id=session_id,
                top_k=10,
                user_id=cfg.user_id,
            )
        except Exception as e:
            print(f"  [{row['id']}] error: {e}")
            continue
        elapsed = time.time() - t0
        latencies.append(elapsed)

        retrieved_ids = []
        for doc in retrieved:
            meta = doc.metadata if hasattr(doc, "metadata") else (doc.get("metadata") or {})
            source = meta.get("source", "")
            cid = meta.get("chunk_id")
            if source and cid is not None:
                retrieved_ids.append(f"{source}::chunk_{cid}")

        n_hit = len(set(retrieved_ids) & set(relevant_ids))
        print(f"  [{row['id']}] retrieved={len(retrieved_ids)} hit={n_hit}/{len(relevant_ids)} lat={elapsed:.2f}s")

        eval_results.append({
            "query_id": row["id"],
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "latency": elapsed,
        })

    if not eval_results:
        return None

    metrics = aggregate_retrieval_metrics(eval_results, k5=5, k10=10)
    n = len(eval_results)
    print(f"[{modality.upper()}] n={n}")
    for name, mr in metrics.items():
        print(f"  {name} = {mr.value:.4f}")

    def metric(name, value, n_, notes=""):
        return {"name": name, "value": value, "n": n_, "notes": notes, "sub": {}}

    suite_metrics = {}
    for name, mr in metrics.items():
        suite_metrics[name] = metric(name, mr.value, mr.n, mr.notes)

    if latencies:
        s = sorted(latencies)
        p50 = s[int(len(s)*0.50)]
        p95 = s[min(int(len(s)*0.95), len(s)-1)]
        suite_metrics["retrieval_p50_sec"] = metric("retrieval_p50_sec", p50, n)
        suite_metrics["retrieval_p95_sec"] = metric("retrieval_p95_sec", p95, n)

    return {"suite": f"retrieval_{modality}", "modality": modality, "metrics": suite_metrics}


def main():
    cfg = EvalConfig()

    if REPORT_FILE.exists():
        with open(REPORT_FILE) as f:
            report = json.load(f)
    else:
        report = {"suites": {}}

    for modality in ["xlsx", "image", "audio", "video"]:
        print(f"\n{'='*50}")
        print(f"Running retrieval for: {modality.upper()}")
        print('='*50)
        result = run_retrieval_for_modality(modality, cfg)
        if result:
            report["suites"][result["suite"]] = result

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Updated: {REPORT_FILE}")


if __name__ == "__main__":
    main()
