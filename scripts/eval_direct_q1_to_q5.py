"""
Direct evaluation harness — File 1 (aapl_10k_2023.txt), Q1–Q5.
Calls ingestion and RAG pipelines directly (no HTTP / auth overhead).

Usage:
    cd /home/ubuntu/multimodal-rag-assistant-1
    source rag_env/bin/activate
    python scripts/eval_direct_q1_to_q5.py [--query N] [--skip-ingest]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

TXT_FILE   = "data/raw/finance/txt/aapl_10k_2023.txt"
SESSION_ID = "eval_file1_direct"
USER_ID    = "eval_user_direct"

# ── Expected answers ──────────────────────────────────────────────────────────

EXPECTED = {
    1: {
        "query": (
            "What were Apple's total net sales for fiscal year 2023, "
            "and how did they compare to fiscal year 2022?"
        ),
        "answer": (
            "Apple's total net sales for FY2023 were $383,285 million — a decrease of $11,043 million "
            "(−2.8%) from FY2022's $394,328 million. Products net sales were $298,085M (vs. $316,199M "
            "in FY2022); Services net sales were $85,200M (vs. $78,129M in FY2022). Services grew while "
            "all major hardware categories declined."
        ),
    },
    2: {
        "query": (
            "What was Apple's net income and earnings per share (diluted) for fiscal year 2023?"
        ),
        "answer": (
            "Apple's net income for FY2023 was $96,995 million. Diluted EPS was $6.13 (basic EPS $6.16). "
            "The diluted share count was approximately 15,812,547 thousand shares. Apple declared "
            "dividends of $0.94 per share during FY2023."
        ),
    },
    3: {
        "query": (
            "Which Apple product category had the largest year-over-year revenue decline in FY2023, "
            "and by how much?"
        ),
        "answer": (
            "Mac had the largest decline: net sales fell 27% (−$10.8 billion) in FY2023 vs. FY2022, "
            "primarily due to lower laptop sales. iPhone −2% (−$4.9B); iPad −3% (−$1.0B); "
            "Wearables −3% (−$1.4B). Services was the only category with growth (+$7.1B, +9.1% YoY)."
        ),
    },
    4: {
        "query": (
            "How much cash and cash equivalents did Apple report on its balance sheet at the end "
            "of fiscal 2023?"
        ),
        "answer": (
            "Apple reported cash and cash equivalents of $29,965 million ($~30.0B) at September 30, "
            "2023. Total cash, cash equivalents, and restricted cash was $30,737 million. During FY2023, "
            "Apple repurchased $76.6 billion of common stock and paid $15.0 billion in dividends."
        ),
    },
    5: {
        "query": (
            "What is Apple's latest/most recent stock price and trailing twelve-month revenue? "
            "Search the internet for current data."
        ),
        "answer": "Web search — current Apple stock price and TTM revenue (dynamic, no fixed expected)",
    },
}

# ── Scoring ───────────────────────────────────────────────────────────────────

_KEY_VALUES = {
    "383,285": "FY2023 net sales $383,285M",
    "394,328": "FY2022 net sales $394,328M",
    "11,043":  "decline $11,043M",
    "2.8":     "−2.8%",
    "298,085": "Products $298,085M",
    "85,200":  "Services $85,200M",
    "96,995":  "Net income $96,995M",
    "6.13":    "Diluted EPS $6.13",
    "6.16":    "Basic EPS $6.16",
    "27":      "Mac −27%",
    "10.8":    "Mac −$10.8B",
    "laptop":  "reason: lower laptop sales",
    "29,965":  "Cash $29,965M",
    "30,737":  "Total cash $30,737M",
    "76.6":    "Buyback $76.6B",
    "15.0":    "Dividends $15.0B",
}

_Q_KEYS = {
    1: ["383,285", "394,328", "11,043", "2.8", "298,085", "85,200"],
    2: ["96,995", "6.13", "6.16"],
    3: ["27", "10.8", "laptop"],
    4: ["29,965", "30,737", "76.6", "15.0"],
}

PASS_THRESHOLD = 0.85


def _score_answer(q_num: int, answer: str) -> tuple[float, list[str], list[str]]:
    if q_num not in _Q_KEYS:
        return 1.0, ["web-search (dynamic)"], []
    keys    = _Q_KEYS[q_num]
    # Normalize: remove commas for number matching
    norm    = answer.replace(",", "")
    found   = [k for k in keys if k.replace(",", "") in norm]
    missing = [k for k in keys if k not in found]
    score   = len(found) / len(keys)
    return score, [_KEY_VALUES[k] for k in found], [_KEY_VALUES[k] for k in missing]


def _check_citation(q_num: int, resp: dict) -> tuple[bool, str]:
    """Verify source attribution in the response."""
    if q_num == 5:
        return True, "web search — no file citation required"

    sources = resp.get("sources", []) or []
    raw     = json.dumps(sources).lower()
    if "aapl_10k_2023" in raw or "aapl_10k" in raw:
        return True, f"aapl_10k_2023.txt found in sources ({len(sources)} total)"
    # Also check answer text for source reference
    answer = (resp.get("answer") or "").lower()
    if "aapl_10k_2023" in answer or "10-k" in answer or "annual report" in answer:
        return True, "Source referenced in answer text"
    return False, f"aapl_10k_2023.txt NOT in sources — got: {str(sources)[:300]}"


def _print_report(q_num: int, resp: dict, elapsed: float) -> tuple[bool, bool]:
    answer = resp.get("answer", "") or ""
    score, found, missing = _score_answer(q_num, answer)
    cite_ok, cite_msg     = _check_citation(q_num, resp)
    answer_pass           = score >= PASS_THRESHOLD

    print(f"\n{'='*70}")
    print(f"  Q{q_num} — {'PASS ✓' if answer_pass else 'FAIL ✗'} | "
          f"Score {score*100:.0f}%  ({len(found)}/{len(found)+len(missing)} key values) | "
          f"{elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"\n[QUERY]\n{EXPECTED[q_num]['query']}")
    print(f"\n[EXPECTED]\n{textwrap.fill(EXPECTED[q_num]['answer'], 80)}")
    print(f"\n[ACTUAL]")
    if answer:
        print(textwrap.fill(answer[:2000], 80))
    else:
        print("(empty — no answer returned)")

    print(f"\n[KEY VALUES FOUND]  ({len(found)}/{len(found)+len(missing)})")
    for f_ in found:
        print(f"  ✓ {f_}")
    if missing:
        print(f"[KEY VALUES MISSING]")
        for m in missing:
            print(f"  ✗ {m}")

    print(f"\n[CITATION] {'✓' if cite_ok else '✗'} {cite_msg}")
    if resp.get("sources"):
        print(f"[SOURCES]")
        for s in (resp["sources"] or [])[:5]:
            print(f"  • {str(s)[:120]}")

    if not answer_pass and len(answer) > 2000:
        print(f"\n[FULL ANSWER — extra]\n{answer[2000:4000]}")

    return answer_pass, cite_ok


# ── Pipeline calls ────────────────────────────────────────────────────────────

def do_ingest() -> dict:
    """Ingest the 10-K TXT file directly using IngestionPipeline."""
    print(f"\n[INGEST] Loading pipeline …")
    from app.pipeline.ingestion_pipeline import IngestionPipeline
    pipeline = IngestionPipeline()

    print(f"[INGEST] Processing {TXT_FILE} …")
    t0     = time.time()
    result = pipeline.process_file(TXT_FILE, session_id=SESSION_ID, user_id=USER_ID)
    elapsed = time.time() - t0

    status = result.get("status", "?")
    chunks = result.get("chunks_stored", result.get("chunks", "?"))
    print(f"[INGEST] Done — status={status}, chunks={chunks}, time={elapsed:.1f}s")
    if status not in ("success", "duplicate"):
        print(f"[INGEST] Full result: {json.dumps(result, default=str)[:600]}")
    return result


def do_query(query: str) -> dict:
    """Run a query through RAGPipeline directly."""
    from app.pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    t0       = time.time()
    resp     = pipeline.run(query=query, session_id=SESSION_ID)
    elapsed  = time.time() - t0
    return resp, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, default=0,
                        help="Run only query N (1-5). Default: all.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion (file already in Qdrant).")
    args = parser.parse_args()

    if not args.skip_ingest:
        ingest_result = do_ingest()
        if ingest_result.get("status") not in ("success", "duplicate"):
            print("[ABORT] Ingestion failed.")
            sys.exit(1)
        print("[WAIT] Giving Qdrant + BM25 2 s to settle …")
        time.sleep(2)

    queries_to_run = [args.query] if args.query else list(range(1, 6))
    results = {}

    for q_num in queries_to_run:
        print(f"\n{'─'*70}")
        print(f"  Running Q{q_num} …")

        try:
            resp, elapsed = do_query(EXPECTED[q_num]["query"])
        except Exception as e:
            print(f"[Q{q_num}] EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results[q_num] = (False, False)
            continue

        if isinstance(resp, dict) and resp.get("error"):
            print(f"[Q{q_num}] PIPELINE ERROR: {resp['error']}")
            results[q_num] = (False, False)
            continue

        a_pass, c_pass = _print_report(q_num, resp, elapsed)
        results[q_num] = (a_pass, c_pass)

    # Summary
    print(f"\n\n{'='*70}")
    print("  FINAL SUMMARY — File 1 (aapl_10k_2023.txt)")
    print(f"{'='*70}")
    print(f"  {'Q':<6} {'Answer':>10}   {'Citation':>10}")
    print(f"  {'─'*34}")
    all_ans_pass = True
    for q_n, (ap, cp) in sorted(results.items()):
        sa = "PASS ✓" if ap else "FAIL ✗"
        sc = "PASS ✓" if cp else "FAIL ✗"
        print(f"  Q{q_n:<5} {sa:>10}   {sc:>10}")
        if not ap:
            all_ans_pass = False
    print(f"{'─'*70}")
    print(f"  Overall: {'ALL PASS ✓' if all_ans_pass else 'NEEDS WORK ✗'}")


if __name__ == "__main__":
    main()
