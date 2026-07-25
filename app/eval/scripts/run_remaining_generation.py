"""
Generation eval via HTTP for XLSX, image, audio, video modalities.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import httpx

SERVER_URL = os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8000")
REPORT_FILE = ROOT / "app/eval/reports/rag_report.json"
USER_ID = "eval_default"

GOLD_FILES = {
    "xlsx": ROOT / "app/eval/datasets/gold/xlsx_gold.jsonl",
    "image": ROOT / "app/eval/datasets/gold/image_gold.jsonl",
    "audio": ROOT / "app/eval/datasets/gold/audio_gold.jsonl",
    "video": ROOT / "app/eval/datasets/gold/video_gold.jsonl",
}


def load_gold(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref = row.get("reference_answer", "")
            if not ref or ref in ("TODO", "") or "SEARCH_REQUIRED" in ref or "INJECTION_PROBE" in ref:
                continue
            rows.append(row)
    return rows


def query_server(query: str, session_id: str) -> dict:
    payload = {"query": query, "session_id": session_id, "user_id": USER_ID}
    with httpx.Client(timeout=180) as client:
        resp = client.post(f"{SERVER_URL}/rag/query", json=payload)
        resp.raise_for_status()
        return resp.json()


def lexical_score(answer: str, reference: str, context_texts: list[str]) -> dict:
    def tokens(text: str) -> set:
        return set(re.findall(r"\b\w+\b", text.lower()))

    ans_tokens = tokens(answer)
    ref_tokens = tokens(reference)
    ctx_tokens = set()
    for c in context_texts:
        ctx_tokens |= tokens(c)

    answer_relevancy = len(ans_tokens & ref_tokens) / len(ref_tokens) if ref_tokens else 0.0
    context_recall = len(ref_tokens & ctx_tokens) / len(ref_tokens) if ref_tokens else 0.0
    faithfulness = len(ans_tokens & ctx_tokens) / len(ans_tokens) if ans_tokens else 0.0

    leak_patterns = [r"\[sic\]", r"Sources Used:", r"<\|im_end\|>", r"\[INST\]", r"\[\/INST\]"]
    template_leak = any(re.search(p, answer) for p in leak_patterns)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall": context_recall,
        "template_leak": template_leak,
    }


def run_modality(modality: str):
    gold_path = GOLD_FILES[modality]
    rows = load_gold(gold_path)
    print(f"\n[{modality.upper()} GEN EVAL] {len(rows)} rows")

    results = []
    latencies = []
    errors = []

    for row in rows:
        qid = row["id"]
        query = row["query"]
        reference = row["reference_answer"]
        session_id = f"eval_gen_{modality}_{qid}"

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
        context_texts = [s.get("text") or s.get("content") or "" for s in sources if isinstance(s, dict)]

        scores = lexical_score(answer, reference, context_texts)
        results.append({"id": qid, "scores": scores, "latency": elapsed})
        print(f"  [{qid}] done in {elapsed:.1f}s | faith={scores['faithfulness']:.3f} rel={scores['answer_relevancy']:.3f}")

    if not results:
        return None

    n = len(results)
    avg = lambda key: sum(r["scores"][key] for r in results) / n
    faithfulness = avg("faithfulness")
    answer_relevancy = avg("answer_relevancy")
    context_recall = avg("context_recall")
    hallucination_rate = sum(1 for r in results if r["scores"]["faithfulness"] < 0.3) / n
    template_leak_rate = sum(1 for r in results if r["scores"]["template_leak"]) / n

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0
    p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat)-1)] if sorted_lat else 0

    print(f"\n[{modality.upper()} GEN] faith={faithfulness:.4f} rel={answer_relevancy:.4f} ctx_recall={context_recall:.4f} halluc={hallucination_rate:.4f}")

    def metric(name, value, n_, notes=""):
        return {"name": name, "value": value, "n": n_, "notes": notes, "sub": {}}

    return {
        "suite": f"generation_{modality}",
        "judge": "lexical_fallback",
        "modality": modality,
        "metrics": {
            "faithfulness": metric("faithfulness", faithfulness, n, "lexical judge"),
            "answer_relevancy": metric("answer_relevancy", answer_relevancy, n, "lexical judge"),
            "context_recall": metric("context_recall", context_recall, n, "lexical judge"),
            "hallucination_rate": metric("hallucination_rate", hallucination_rate, n),
            "template_leak_rate": metric("template_leak_rate", template_leak_rate, n),
            "gen_p50_sec": metric("gen_p50_sec", p50, n),
            "gen_p95_sec": metric("gen_p95_sec", p95, n),
        },
        "errors": errors,
    }


def main():
    if REPORT_FILE.exists():
        with open(REPORT_FILE) as f:
            report = json.load(f)
    else:
        report = {"suites": {}}

    for modality in ["xlsx", "image", "audio", "video"]:
        result = run_modality(modality)
        if result:
            report["suites"][f"generation_{modality}"] = result

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] Updated: {REPORT_FILE}")


if __name__ == "__main__":
    main()
