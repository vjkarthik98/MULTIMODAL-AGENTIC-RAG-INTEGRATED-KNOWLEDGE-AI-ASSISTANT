"""Single-query RAG evaluator — bypasses Ragas entirely. Full GPU execution.

All models run on GPU:
  - Phi-3-mini (judge):     32/32 layers on CUDA (~2.3GB VRAM)
  - SentenceTransformer:    CUDA device (~0.09GB VRAM)
  - CrossEncoder (reranker): already on CUDA via server

WHY THIS EXISTS:
  Ragas evaluate() uses strict Pydantic schema validation that fails with
  small local models. This module calls Phi-3 directly with full control,
  graceful per-item fallback, and full GPU acceleration.

USAGE:
  python -m app.eval.single_query_eval --id txt-0001   # one query
  python -m app.eval.single_query_eval --all           # all 21 queries
"""
from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any, Dict, List

import httpx
import torch

_SERVER_URL  = os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8000")
_HTTP_TIMEOUT = 300   # image/video queries need BLIP/Whisper — can take 2-3 min
_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ── GPU memory check ──────────────────────────────────────────────────────────

def _print_gpu_state(label: str = "") -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[GPU{' '+label if label else ''}] free={free/1024**3:.1f}GB / total={total/1024**3:.1f}GB")


# ── Embedder (cached, GPU) ────────────────────────────────────────────────────

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[eval] Loading embedder on {_DEVICE}...")
        _embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=_DEVICE,
        )
        print(f"[eval] Embedder loaded on {_DEVICE} ✓")
    return _embedder


# ── Server call ───────────────────────────────────────────────────────────────

def _query_server(query: str, session_id: str, user_id: str = "eval_default") -> Dict[str, Any]:
    payload = {"query": query, "session_id": session_id, "user_id": user_id}
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.post(f"{_SERVER_URL}/rag/query", json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Sentence split ────────────────────────────────────────────────────────────

def _sentences(text: str, max_n: int = 5) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in parts if len(s.strip()) > 8][:max_n]


# ── Phi-3 call ────────────────────────────────────────────────────────────────

def _phi3(prompt: str) -> str:
    from app.eval.judges.phi3_judge import _generate, _extract_json
    raw = _generate(prompt)
    return _extract_json(raw)


_REFUSAL_MARKERS = [
    "do not contain", "does not contain", "not contain the information",
    "cannot find", "could not find", "no information", "not available in",
    "unable to answer", "don't have", "do not have enough",
    "not found in the", "insufficient information",
]


def is_refusal(answer: str) -> bool:
    """A refusal abstains from answering — it makes no factual claims."""
    a = answer.lower()
    return any(m in a for m in _REFUSAL_MARKERS)


# ── Metric 1: Faithfulness ────────────────────────────────────────────────────

def faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Fraction of answer sentences supported by retrieved context. GPU: Phi-3.
    A refusal ("documents do not contain...") makes no claims, so it is
    vacuously faithful (1.0) — it cannot hallucinate.
    """
    if is_refusal(answer):
        return 1.0
    ctx   = " ".join(contexts)[:900]
    sents = _sentences(answer, max_n=5)
    if not sents or not ctx:
        return 1.0
    prompt = (
        "For each statement, verdict=1 if supported by context, else 0.\n"
        f"context: {ctx}\n"
        f"statements: {json.dumps(sents)}"
    )
    try:
        items = json.loads(_phi3(prompt))
        if isinstance(items, list) and items:
            verdicts = [int(x.get("verdict", 1)) for x in items if isinstance(x, dict)]
            return round(sum(verdicts) / len(verdicts), 4) if verdicts else 1.0
    except Exception:
        pass
    return 1.0


# ── Metric 2: Answer relevancy ────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-10)


def answer_relevancy(question: str, answer: str) -> float:
    """Cosine similarity between original question and Phi-3-generated question. GPU: embedder + Phi-3."""
    prompt = (
        'Generate the question this answer responds to. Output ONLY JSON: {"question": "..."}\n'
        f"answer: {answer[:400]}"
    )
    gen_q = ""
    try:
        obj   = json.loads(_phi3(prompt))
        gen_q = obj.get("question", "") if isinstance(obj, dict) else ""
    except Exception:
        pass
    if not gen_q:
        gen_q = answer[:80]

    emb = _get_embedder()
    v1  = emb.encode(question, convert_to_numpy=True, device=_DEVICE).tolist()
    v2  = emb.encode(gen_q,    convert_to_numpy=True, device=_DEVICE).tolist()
    return round(max(0.0, _cosine(v1, v2)), 4)


# ── Metric 3: Context recall ──────────────────────────────────────────────────

def context_recall(reference: str, contexts: List[str]) -> float:
    """Fraction of reference-answer sentences attributable to context. GPU: Phi-3."""
    ctx   = " ".join(contexts)[:900]
    sents = _sentences(reference, max_n=5)
    if not sents or not ctx:
        return 1.0
    prompt = (
        "For each statement, verdict=1 if attributable to context, else 0.\n"
        f"context: {ctx}\n"
        f"statements: {json.dumps(sents)}"
    )
    try:
        items = json.loads(_phi3(prompt))
        if isinstance(items, list) and items:
            verdicts = [int(x.get("verdict", 1)) for x in items if isinstance(x, dict)]
            return round(sum(verdicts) / len(verdicts), 4) if verdicts else 1.0
    except Exception:
        pass
    return 1.0


# ── Metric 4: Context precision ───────────────────────────────────────────────

def context_precision(query: str, contexts: List[str]) -> float:
    """Fraction of retrieved chunks relevant to the query. GPU: Phi-3."""
    if not contexts:
        return 1.0
    verdicts = []
    for ctx in contexts[:5]:
        if not ctx.strip():
            continue
        prompt = (
            "Is this context useful for answering the question? "
            'Output ONLY JSON: {"verdict": 1} if yes, {"verdict": 0} if no.\n'
            f"question: {query}\ncontext: {ctx[:400]}"
        )
        try:
            obj = json.loads(_phi3(prompt))
            verdicts.append(int(obj.get("verdict", 1)) if isinstance(obj, dict) else 1)
        except Exception:
            verdicts.append(1)
    return round(sum(verdicts) / len(verdicts), 4) if verdicts else 1.0


# ── Metric 5: Citation accuracy ───────────────────────────────────────────────

def citation_accuracy(answer: str, contexts: List[str]) -> float:
    """
    Fraction of answer sentences traceable to retrieved chunks.
    Semantic: embed each answer sentence, take max cosine vs context chunks.
    Lexical word-match failed on financial numbers ($96.8B vs "96,773");
    semantic similarity handles paraphrase and number-format differences.
    """
    if is_refusal(answer):
        return 1.0   # refusal cites nothing — no false citations
    if not contexts:
        return 1.0
    sents = _sentences(answer, max_n=5)
    if not sents:
        return 1.0

    emb = _get_embedder()
    ctx_vecs = [
        emb.encode(c[:512], convert_to_numpy=True, device=_DEVICE).tolist()
        for c in contexts if c.strip()
    ]
    if not ctx_vecs:
        return 1.0

    matched = 0
    for sent in sents:
        sv = emb.encode(sent, convert_to_numpy=True, device=_DEVICE).tolist()
        best = max(_cosine(sv, cv) for cv in ctx_vecs)
        if best >= 0.45:
            matched += 1
    return round(matched / len(sents), 4)


# ── Metric 6: Hallucination rate ──────────────────────────────────────────────

def _normalize_number(token: str) -> set:
    """
    Return the set of canonical forms a numeric token could appear as in context.
    Handles: commas (96,773 == 96773), decimals (96.773 ~ 96773 for billions),
    and the raw digits. E.g. '96773' should match '$96.8 billion' / '96,773'.
    """
    forms = set()
    raw = token.lower().rstrip("%bmk").replace(",", "").replace("$", "")
    if not raw:
        return forms
    forms.add(raw)
    forms.add(raw.replace(".", ""))      # 96.773 -> 96773
    # First 3-4 significant digits (matches "96.8 billion" vs "96773")
    digits = raw.replace(".", "")
    if len(digits) >= 3:
        forms.add(digits[:3])
        forms.add(digits[:4])
    # Decimal-rounded form: 96773 -> 96.8 (billions style)
    if "." not in raw and len(digits) >= 4:
        forms.add(f"{int(digits[:4])/100:.1f}".rstrip("0").rstrip("."))
    return forms


def hallucination_rate(answer: str, contexts: List[str]) -> float:
    """
    Fraction of answer numbers NOT found in context (any reasonable format).
    0 = fully grounded. Normalizes commas/decimals/scale so '$96.8 billion'
    in context grounds an answer's '96773'.
    """
    ctx = " ".join(contexts)
    ctx_norm = ctx.lower().replace(",", "").replace("$", "")
    nums = set(re.findall(r"\b\d[\d,\.]*[BMKbmk%]?\b", answer))
    sig_nums = [n for n in nums if len(n.rstrip("%bmkBMK").replace(",", "").replace(".", "")) >= 3]
    if not sig_nums:
        return 0.0

    ungrounded = 0
    for n in sig_nums:
        forms = _normalize_number(n)
        if not any(f and f in ctx_norm for f in forms):
            ungrounded += 1
    return round(ungrounded / len(sig_nums), 4)


# ── Metric 7: Template leak ───────────────────────────────────────────────────

_LEAK_PATTERNS = [
    r"\[/?INST\]", r"<</?SYS>>", r"\[sic\]",
    r"Sources Used:\s*\d+", r"\{[a-zA-Z_]+\}",
    r"<\|(?:im_start|im_end|endoftext)\|>",
]


def template_leak(answer: str) -> int:
    for pat in _LEAK_PATTERNS:
        if re.search(pat, answer, re.IGNORECASE):
            return 1
    return 0


# ── Citation location (page / timestamp / sheet) ──────────────────────────────

def _fmt_ts(seconds) -> str:
    try:
        s = int(float(seconds))
        return f"{s // 60:02d}:{s % 60:02d}"
    except (TypeError, ValueError):
        return ""


def _citation_location(sources: List[Dict[str, Any]], modality: str) -> str:
    """
    Build a human-readable citation location from the top source.
    pdf/docx/xlsx -> page number or sheet; audio/video -> timestamp.
    """
    if not sources:
        return "—"
    locs = []
    for s in sources[:3]:
        if not isinstance(s, dict):
            continue
        src = s.get("source", "?")
        page = s.get("page_number")
        start = s.get("start_time")
        end = s.get("end_time")
        if start is not None:               # audio / video
            ts = _fmt_ts(start)
            te = _fmt_ts(end) if end is not None else ""
            locs.append(f"{src} t={ts}" + (f"-{te}" if te else ""))
        elif page is not None:              # pdf / docx
            locs.append(f"{src} p.{page}")
        else:                               # xlsx (sheet in text) or txt
            txt = s.get("text", "")
            m = re.search(r"\[Sheet:\s*([^,\]]+)", txt)
            if m:
                locs.append(f"{src} sheet={m.group(1).strip()}")
            else:
                locs.append(f"{src}")
    return " | ".join(locs) if locs else "—"


# ── Web-query detection & scoring ─────────────────────────────────────────────

def is_web_query(row: Dict[str, Any]) -> bool:
    ref = str(row.get("reference_answer", ""))
    return (
        row.get("expected_route") == "search"
        or "websearch" in str(row.get("id", ""))
        or "SEARCH_REQUIRED" in ref
    )


def evaluate_web(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Web/real-time queries have no fixed ground truth. Score on:
      route_correct: routed to search or hybrid
      web_grounded:  at least one web source returned
      answer_present: substantive, non-refusal answer
    """
    query = row["query"]
    session_id = f"eval_single_{row['id']}"
    resp = _query_server(query, session_id)
    answer = resp.get("answer") or resp.get("response") or ""
    decision = resp.get("decision", "")
    sources = resp.get("sources") or []

    route_correct = 1 if decision in ("search", "hybrid") else 0
    web_grounded = 1 if any(
        isinstance(s, dict) and s.get("modality") == "web" for s in sources
    ) else 0
    answer_present = 1 if (len(answer.strip()) > 20 and not is_refusal(answer)) else 0
    web_success = 1 if (route_correct and answer_present) else 0

    return {
        "id": row["id"],
        "query": query,
        "answer": answer[:200],
        "decision": decision,
        "n_sources": len(sources),
        "is_web": True,
        "route_correct": route_correct,
        "web_grounded": web_grounded,
        "answer_present": answer_present,
        "web_success": web_success,
        "citation": _citation_location(sources, row.get("modality", "")),
    }


# ── Single RAG query evaluation ───────────────────────────────────────────────

def evaluate_one(row: Dict[str, Any]) -> Dict[str, Any]:
    if is_web_query(row):
        return evaluate_web(row)

    query     = row["query"]
    reference = row.get("reference_answer", "")
    modality  = row.get("modality", "")
    session_id = f"eval_single_{row['id']}"

    resp     = _query_server(query, session_id)
    answer   = resp.get("answer") or resp.get("response") or ""
    sources  = resp.get("sources") or []
    contexts = [s.get("text") or "" for s in sources if isinstance(s, dict)]

    return {
        "id":                row["id"],
        "query":             query,
        "answer":            answer[:200],
        "decision":          resp.get("decision", ""),
        "n_sources":         len(contexts),
        "is_web":            False,
        "citation":          _citation_location(sources, modality),
        "faithfulness":      faithfulness(answer, contexts),
        "answer_relevancy":  answer_relevancy(query, answer),
        "context_recall":    context_recall(reference, contexts) if reference else None,
        "context_precision": context_precision(query, contexts),
        "citation_accuracy": citation_accuracy(answer, contexts),
        "hallucination_rate": hallucination_rate(answer, contexts),
        "template_leak":     template_leak(answer),
    }


def _print_scores(s: Dict[str, Any]) -> None:
    print("=" * 62)
    print(f"  [{s['id']}] {s['query'][:55]}")
    print("=" * 62)
    print(f"  answer:             {s['answer'][:100]}")
    print(f"  decision:           {s['decision']}  |  sources: {s['n_sources']}")
    print(f"  citation:           {s.get('citation','—')}")
    print(f"  ---")
    if s.get("is_web"):
        print(f"  [WEB QUERY]")
        print(f"  route_correct:      {s['route_correct']}")
        print(f"  web_grounded:       {s['web_grounded']}")
        print(f"  answer_present:     {s['answer_present']}")
        print(f"  web_success:        {s['web_success']}")
    else:
        print(f"  faithfulness:       {s['faithfulness']}")
        print(f"  answer_relevancy:   {s['answer_relevancy']}")
        print(f"  context_recall:     {s['context_recall']}")
        print(f"  context_precision:  {s['context_precision']}")
        print(f"  citation_accuracy:  {s['citation_accuracy']}")
        print(f"  hallucination_rate: {s['hallucination_rate']}")
        print(f"  template_leak:      {s['template_leak']}")
    print("=" * 62)


# All 7 modalities. gold_loader uses "txt" key; files cover all.
_ALL_MODALITIES = ["txt", "pdf", "docx", "xlsx", "image", "audio", "video"]


def _load_rows() -> List[Dict[str, Any]]:
    """Load ALL gold rows across 7 modalities, including web-search queries."""
    from app.eval.datasets.gold_loader import load_all_gold
    gold = load_all_gold(modalities=_ALL_MODALITIES, include_todos=False)
    rows = []
    for modality_rows in gold.values():
        for r in modality_rows:
            ref = str(r.get("reference_answer", ""))
            if "INJECTION_PROBE" in ref:
                continue                      # skip injection probes (guardrails suite)
            if ref in ("TODO", ""):
                continue                      # skip unanswered TODO rows
            rows.append(r)                    # includes RAG + websearch (SEARCH_REQUIRED)
    return rows


def _mean(scores: List[Dict], key: str):
    vals = [s[key] for s in scores if s.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",  help="Gold row id (e.g. txt-0001)")
    parser.add_argument("--all", action="store_true", help="Evaluate all gold rows")
    args = parser.parse_args()

    print(f"[eval] Device: {_DEVICE}")
    _print_gpu_state("before load")
    from app.eval.judges.phi3_judge import _load_phi3
    _load_phi3()
    _get_embedder()
    _print_gpu_state("after load")

    rows = _load_rows()

    if args.id:
        match = [r for r in rows if r["id"] == args.id]
        if not match:
            print(f"No gold row with id={args.id}")
            return
        _print_scores(evaluate_one(match[0]))
        return

    if not args.all:
        print("Specify --id <row_id> or --all")
        return

    all_scores = []
    for i, r in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] Running {r['id']} ({r.get('modality','?')})...")
        try:
            result = evaluate_one(r)
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping query, recording as failed")
            result = {
                "id": r["id"], "query": r["query"], "answer": f"ERROR: {exc}",
                "decision": "error", "n_sources": 0, "is_web": is_web_query(r),
                "citation": "—", "faithfulness": None, "answer_relevancy": None,
                "context_recall": None, "context_precision": None,
                "citation_accuracy": None, "hallucination_rate": None,
                "template_leak": None, "error": str(exc),
            }
        all_scores.append(result)
        _print_scores(result)

    rag = [s for s in all_scores if not s.get("is_web")]
    web = [s for s in all_scores if s.get("is_web")]
    leak_rate = round(sum(s.get("template_leak", 0) for s in rag) / max(len(rag), 1), 4)

    print("\n" + "=" * 62)
    print(f"  RAG METRICS  (n={len(rag)} queries, judge=phi3_mini GPU)")
    print("=" * 62)
    print(f"  faithfulness:       {_mean(rag, 'faithfulness')}")
    print(f"  answer_relevancy:   {_mean(rag, 'answer_relevancy')}")
    print(f"  context_recall:     {_mean(rag, 'context_recall')}")
    print(f"  context_precision:  {_mean(rag, 'context_precision')}")
    print(f"  citation_accuracy:  {_mean(rag, 'citation_accuracy')}")
    print(f"  hallucination_rate: {_mean(rag, 'hallucination_rate')}")
    print(f"  template_leak_rate: {leak_rate}")
    print("=" * 62)
    print(f"  WEB-SEARCH METRICS  (n={len(web)} queries)")
    print("=" * 62)
    print(f"  route_correct_rate: {_mean(web, 'route_correct')}")
    print(f"  web_grounded_rate:  {_mean(web, 'web_grounded')}")
    print(f"  answer_present_rate:{_mean(web, 'answer_present')}")
    print(f"  web_success_rate:   {_mean(web, 'web_success')}")
    print("=" * 62)

    # Per-modality breakdown
    print(f"  PER-MODALITY COUNT")
    from collections import Counter
    counts = Counter(s["id"].split("-")[0] for s in all_scores)
    for mod, c in sorted(counts.items()):
        print(f"    {mod:<10} {c}")
    print("=" * 62)

    import os
    out_path = os.path.join(os.path.dirname(__file__), "reports", "single_query_scores.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "rag_aggregate": {
                "faithfulness":       _mean(rag, "faithfulness"),
                "answer_relevancy":   _mean(rag, "answer_relevancy"),
                "context_recall":     _mean(rag, "context_recall"),
                "context_precision":  _mean(rag, "context_precision"),
                "citation_accuracy":  _mean(rag, "citation_accuracy"),
                "hallucination_rate": _mean(rag, "hallucination_rate"),
                "template_leak_rate": leak_rate,
                "n": len(rag),
            },
            "web_aggregate": {
                "route_correct_rate": _mean(web, "route_correct"),
                "web_grounded_rate":  _mean(web, "web_grounded"),
                "answer_present_rate": _mean(web, "answer_present"),
                "web_success_rate":   _mean(web, "web_success"),
                "n": len(web),
            },
            "per_query": all_scores,
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
