"""
Evaluation harness — File 1 (aapl_10k_2023.txt), Q1–Q5.

Runs each query through the live RAG API, scores accuracy vs.
expected answers, and prints a per-query report.

Usage:
    python scripts/eval_file1_q1_to_q5.py [--query N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import requests

BASE_URL  = os.getenv("EVAL_API_URL", "http://localhost:8000")
SESSION_ID = "eval_file1_session"
USER_ID    = "eval_user"
TXT_FILE   = "data/raw/finance/txt/aapl_10k_2023.txt"

# ── Auth helpers ─────────────────────────────────────────────────────────────

def _get_token() -> str:
    """Register + login to obtain a JWT bearer token."""
    # Try login first (user may already exist)
    r = requests.post(f"{BASE_URL}/auth/login",
                      json={"email": "eval@eval.local", "password": "EvalPass123!"},
                      timeout=30)
    if r.status_code == 200:
        return r.json().get("access_token", "")

    # Register then login
    requests.post(f"{BASE_URL}/auth/register",
                  json={"email": "eval@eval.local", "password": "EvalPass123!",
                        "name": "Eval Bot"},
                  timeout=30)
    r = requests.post(f"{BASE_URL}/auth/login",
                      json={"email": "eval@eval.local", "password": "EvalPass123!"},
                      timeout=30)
    return r.json().get("access_token", "") if r.status_code == 200 else ""


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_file(token: str) -> dict:
    """Upload aapl_10k_2023.txt to the ingestion endpoint."""
    file_path = Path(TXT_FILE)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    print(f"\n[INGEST] Uploading {file_path.name} …")
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/rag/ingest",
            headers=_headers(token),
            files={"file": (file_path.name, f, "text/plain")},
            data={"session_id": SESSION_ID},
            timeout=300,
        )

    if r.status_code not in (200, 201):
        print(f"[INGEST] ERROR {r.status_code}: {r.text[:600]}")
        return {}

    result = r.json()
    print(f"[INGEST] Done — chunks={result.get('chunks_stored', result.get('chunks', '?'))}, "
          f"modality={result.get('modality', '?')}, "
          f"status={result.get('status', '?')}")
    return result


# ── Query ─────────────────────────────────────────────────────────────────────

def run_query(query: str, token: str, session_id: str = SESSION_ID) -> dict:
    """Send a chat query and return the response dict."""
    payload = {
        "query": query,
        "session_id": session_id,
        "user_id": USER_ID,
        "no_cache": True,
    }
    r = requests.post(
        f"{BASE_URL}/rag/query",
        headers={**_headers(token), "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:600]}"}
    return r.json()


# ── Scoring ───────────────────────────────────────────────────────────────────

_KEY_VALUES = {
    # Q1
    "383,285": "FY2023 net sales $383,285M",
    "394,328": "FY2022 net sales $394,328M",
    "11,043":  "decline $11,043M",
    "2.8":     "−2.8%",
    "298,085": "Products $298,085M",
    "85,200":  "Services $85,200M",
    # Q2
    "96,995":  "Net income $96,995M",
    "6.13":    "Diluted EPS $6.13",
    "6.16":    "Basic EPS $6.16",
    # Q3
    "27":      "Mac −27%",
    "10.8":    "Mac −$10.8B",
    "laptop":  "reason: lower laptop sales",
    # Q4
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


def _score_answer(q_num: int, answer: str) -> tuple[float, list[str], list[str]]:
    """Return (score 0-1, found_keys, missing_keys)."""
    if q_num not in _Q_KEYS:
        return 0.0, [], []
    keys = _Q_KEYS[q_num]
    found   = [k for k in keys if k.replace(",", "") in answer.replace(",", "")]
    missing = [k for k in keys if k not in found]
    score   = len(found) / len(keys) if keys else 0.0
    return score, [_KEY_VALUES[k] for k in found], [_KEY_VALUES[k] for k in missing]


def _citation_ok(q_num: int, resp: dict) -> tuple[bool, str]:
    """Check that source is aapl_10k_2023.txt in the sources list."""
    sources = resp.get("sources", []) or resp.get("citations", []) or []
    raw = json.dumps(sources).lower()
    if "aapl_10k_2023" in raw or "aapl_10k" in raw:
        return True, "aapl_10k_2023.txt found in sources"
    return False, f"aapl_10k_2023.txt NOT in sources — got: {str(sources)[:200]}"


# ── Expected answers (reference) ─────────────────────────────────────────────

EXPECTED = {
    1: {
        "query": "What were Apple's total net sales for fiscal year 2023, and how did they compare to fiscal year 2022?",
        "answer": (
            "Apple's total net sales for FY2023 were $383,285 million — a decrease of $11,043 million "
            "(−2.8%) from FY2022's $394,328 million. Products net sales were $298,085M (vs. $316,199M "
            "in FY2022); Services net sales were $85,200M (vs. $78,129M in FY2022). Services grew while "
            "all major hardware categories declined."
        ),
    },
    2: {
        "query": "What was Apple's net income and earnings per share (diluted) for fiscal year 2023?",
        "answer": (
            "Apple's net income for FY2023 was $96,995 million. Diluted EPS was $6.13 (basic EPS: $6.16). "
            "The diluted share count was approximately 15,812,547 thousand shares. Apple declared dividends "
            "of $0.94 per share during FY2023."
        ),
    },
    3: {
        "query": "Which Apple product category had the largest year-over-year revenue decline in FY2023, and by how much?",
        "answer": (
            "Mac had the largest decline: net sales fell 27% (−$10.8 billion) in FY2023 vs. FY2022, "
            "primarily due to lower laptop sales. iPhone −2% (−$4.9B); iPad −3% (−$1.0B); "
            "Wearables −3% (−$1.4B). Services was the only category with growth (+$7.1B, +9.1% YoY)."
        ),
    },
    4: {
        "query": "How much cash and cash equivalents did Apple report on its balance sheet at the end of fiscal 2023?",
        "answer": (
            "Apple reported cash and cash equivalents of $29,965 million ($~30.0B) at September 30, 2023. "
            "Total cash, cash equivalents, and restricted cash was $30,737 million. During FY2023, Apple "
            "repurchased $76.6 billion of common stock and paid $15.0 billion in dividends."
        ),
    },
    5: {
        "query": "What is Apple's latest/most recent stock price and trailing twelve-month revenue? Search the internet for current data.",
        "answer": "Web search — current Apple stock price and TTM revenue",
    },
}


# ── Report printer ────────────────────────────────────────────────────────────

PASS_THRESHOLD = 0.85

def print_report(q_num: int, resp: dict, elapsed: float) -> tuple[bool, bool]:
    """Print detailed per-query report. Returns (answer_pass, citation_pass)."""
    answer      = resp.get("answer", resp.get("response", ""))
    score, found, missing = _score_answer(q_num, answer)
    cite_ok, cite_msg     = _citation_ok(q_num, resp)

    answer_pass = score >= PASS_THRESHOLD

    print(f"\n{'='*70}")
    print(f"  Q{q_num} — {'PASS ✓' if answer_pass else 'FAIL ✗'} | "
          f"Score {score*100:.0f}% | {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"\n[QUERY]\n{EXPECTED[q_num]['query']}")
    print(f"\n[EXPECTED ANSWER SUMMARY]\n{textwrap.fill(EXPECTED[q_num]['answer'], 80)}")
    print(f"\n[ACTUAL ANSWER]\n{textwrap.fill(answer[:1200] if answer else '(empty)', 80)}")
    print(f"\n[KEY VALUES FOUND]  ({len(found)}/{len(found)+len(missing)})")
    for f_ in found:
        print(f"  ✓ {f_}")
    if missing:
        print(f"[KEY VALUES MISSING]")
        for m in missing:
            print(f"  ✗ {m}")
    print(f"\n[CITATION] {'✓' if cite_ok else '✗'} {cite_msg}")
    print(f"\n[SOURCES RAW]\n{json.dumps(resp.get('sources', [])[:3], indent=2)[:600]}")

    if not answer_pass:
        print(f"\n[FULL ANSWER]\n{answer[:3000]}")

    return answer_pass, cite_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def wait_for_server(max_wait: int = 180) -> bool:
    """Poll /health until the server is ready."""
    print(f"[WAIT] Waiting for server at {BASE_URL} …", end="", flush=True)
    for _ in range(max_wait):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(" ready.")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" TIMEOUT")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=int, default=0,
                        help="Run only query N (1-5). Default: run all 1-5.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion (file already in Qdrant).")
    args = parser.parse_args()

    if not wait_for_server():
        sys.exit(1)

    token = _get_token()
    if not token:
        print("[AUTH] WARNING: No token obtained — proceeding without auth.")

    if not args.skip_ingest:
        ingest_file(token)
        print("[WAIT] Giving Qdrant + BM25 3 s to settle …")
        time.sleep(3)

    queries_to_run = [args.query] if args.query else list(range(1, 6))

    results = {}
    for q_num in queries_to_run:
        print(f"\n{'─'*70}")
        print(f"  Running Q{q_num} …")
        t0     = time.time()
        resp   = run_query(EXPECTED[q_num]["query"], token)
        elapsed = time.time() - t0

        if "error" in resp:
            print(f"[Q{q_num}] API ERROR: {resp['error']}")
            results[q_num] = (False, False)
            continue

        a_pass, c_pass = print_report(q_num, resp, elapsed)
        results[q_num] = (a_pass, c_pass)

    # Summary table
    print(f"\n\n{'='*70}")
    print("  FINAL SUMMARY — File 1 (aapl_10k_2023.txt)")
    print(f"{'='*70}")
    print(f"  {'Q':<6} {'Answer':>10}   {'Citation':>10}")
    print(f"  {'─'*30}")
    all_pass = True
    for q, (ap, cp) in sorted(results.items()):
        status_a = "PASS ✓" if ap else "FAIL ✗"
        status_c = "PASS ✓" if cp else "FAIL ✗"
        print(f"  Q{q:<5} {status_a:>10}   {status_c:>10}")
        if not ap:
            all_pass = False
    print(f"{'─'*70}")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'NEEDS WORK ✗'}")


if __name__ == "__main__":
    main()
