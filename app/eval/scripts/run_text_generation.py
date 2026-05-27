"""
Text-modality generation eval via HTTP server (no second GGUF).

Calls POST /rag/query for each text gold row, collects answers + sources,
computes generation metrics using the lexical judge, then merges results
into the existing rag_report.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import httpx

SERVER_URL = os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8000")
GOLD_FILE = ROOT / "app/eval/datasets/gold/text_gold.jsonl"
REPORT_FILE = ROOT / "app/eval/reports/rag_report.json"
USER_ID = "eval_default"
SESSION_PREFIX = "eval_gen_txt"


def load_text_gold():
    rows = []
    with open(GOLD_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref = row.get("reference_answer", "")
            # Skip TODO, injection probe, search-required
            if not ref or ref in ("TODO", "") or "SEARCH_REQUIRED" in ref or "INJECTION_PROBE" in ref:
                continue
            rows.append(row)
    return rows


def query_server(query: str, session_id: str) -> dict:
    payload = {"query": query, "session_id": session_id, "user_id": USER_ID}
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{SERVER_URL}/rag/query", json=payload)
        resp.raise_for_status()
        return resp.json()


def lexical_score(answer: str, reference: str, context_texts: list[str]) -> dict:
    """Simple lexical overlap metrics as fallback judge."""
    import re

    def tokens(text: str) -> set:
        return set(re.findall(r"\b\w+\b", text.lower()))

    ans_tokens = tokens(answer)
    ref_tokens = tokens(reference)
    ctx_tokens = set()
    for c in context_texts:
        ctx_tokens |= tokens(c)

    # Answer relevancy: overlap between answer and reference
    if not ref_tokens:
        answer_relevancy = 0.0
    else:
        answer_relevancy = len(ans_tokens & ref_tokens) / len(ref_tokens)

    # Context recall: ref tokens found in context
    if not ref_tokens:
        context_recall = 0.0
    else:
        context_recall = len(ref_tokens & ctx_tokens) / len(ref_tokens)

    # Faithfulness: answer tokens found in context
    if not ans_tokens:
        faithfulness = 0.0
    else:
        faithfulness = len(ans_tokens & ctx_tokens) / len(ans_tokens)

    # Template leak: look for common prompt artifacts
    leak_patterns = [r"\[sic\]", r"Sources Used:", r"<\|im_end\|>", r"\[INST\]", r"\[\/INST\]"]
    import re
    template_leak = any(re.search(p, answer) for p in leak_patterns)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
        "template_leak": template_leak,
    }


def main():
    print(f"[TEXT GEN EVAL] Loading gold rows from {GOLD_FILE}")
    rows = load_text_gold()
    print(f"[TEXT GEN EVAL] {len(rows)} rows to evaluate")

    results = []
    latencies = []
    errors = []

    for row in rows:
        qid = row["id"]
        query = row["query"]
        reference = row["reference_answer"]
        session_id = f"{SESSION_PREFIX}_{qid}"

        print(f"  [{qid}] querying...", flush=True)
        t0 = time.time()
        try:
            resp = query_server(query, session_id)
        except Exception as e:
            print(f"  [{qid}] ERROR: {e}")
            errors.append({"id": qid, "error": str(e)})
            continue
        elapsed = time.time() - t0
        latencies.append(elapsed)

        answer = resp.get("answer") or resp.get("response") or ""
        sources = resp.get("sources") or []
        context_texts = []
        for s in sources:
            if isinstance(s, dict):
                context_texts.append(s.get("text") or s.get("content") or "")

        scores = lexical_score(answer, reference, context_texts)
        results.append({
            "id": qid,
            "query": query,
            "answer": answer[:200],
            "scores": scores,
            "latency": elapsed,
        })
        print(f"  [{qid}] done in {elapsed:.1f}s | faith={scores['faithfulness']:.3f} rel={scores['answer_relevancy']:.3f}")

    if not results:
        print("[TEXT GEN EVAL] No results — all queries failed.")
        sys.exit(1)

    # Aggregate
    n = len(results)
    avg = lambda key: sum(r["scores"][key] for r in results) / n
    faithfulness = avg("faithfulness")
    answer_relevancy = avg("answer_relevancy")
    context_recall = avg("context_recall")
    hallucination_rate = sum(1 for r in results if r["scores"]["faithfulness"] < 0.3) / n
    template_leak_rate = sum(1 for r in results if r["scores"]["template_leak"]) / n

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0
    p99 = sorted_lat[min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)] if sorted_lat else 0

    print(f"\n[TEXT GEN EVAL] Results (n={n}, errors={len(errors)}):")
    print(f"  faithfulness       = {faithfulness:.4f}")
    print(f"  answer_relevancy   = {answer_relevancy:.4f}")
    print(f"  context_recall     = {context_recall:.4f}")
    print(f"  hallucination_rate = {hallucination_rate:.4f}")
    print(f"  template_leak_rate = {template_leak_rate:.4f}")
    print(f"  latency p50={p50:.2f}s p95={p95:.2f}s p99={p99:.2f}s")

    # Merge into existing report
    if REPORT_FILE.exists():
        with open(REPORT_FILE) as f:
            report = json.load(f)
    else:
        report = {"suites": {}}

    def metric(name, value, n_, notes=""):
        return {"name": name, "value": value, "n": n_, "notes": notes, "sub": {}}

    report["suites"]["generation_txt"] = {
        "suite": "generation_txt",
        "judge": "lexical_fallback",
        "modality": "txt",
        "metrics": {
            "faithfulness": metric("faithfulness", faithfulness, n, "lexical judge"),
            "answer_relevancy": metric("answer_relevancy", answer_relevancy, n, "lexical judge"),
            "context_recall": metric("context_recall", context_recall, n, "lexical judge"),
            "hallucination_rate": metric("hallucination_rate", hallucination_rate, n),
            "template_leak_rate": metric("template_leak_rate", template_leak_rate, n),
            "gen_p50_sec": metric("gen_p50_sec", p50, n),
            "gen_p95_sec": metric("gen_p95_sec", p95, n),
            "gen_p99_sec": metric("gen_p99_sec", p99, n),
        },
        "errors": errors,
        "per_query": results,
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[TEXT GEN EVAL] Report updated: {REPORT_FILE}")


if __name__ == "__main__":
    main()
