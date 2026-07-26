"""Generation quality metrics: faithfulness, answer_relevancy, context_recall,
citation_accuracy, template_leak_rate.

Uses Ragas with local GGUF judge when available, lexical fallback otherwise.
Judge availability is recorded in metric notes so reports are never misleading.
"""

from __future__ import annotations

import re
from typing import Any

from app.eval.metrics.base import MetricResult

# Prompt-template artifact patterns (P1-7: template leakage)
_TEMPLATE_LEAK_PATTERNS = [
    r"\[sic\]",
    r"Sources Used: \d+",
    r"\{[a-zA-Z_]+\}",  # unfilled template variable
    r"<context>",
    r"</context>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]
_TEMPLATE_LEAK_RE = re.compile("|".join(_TEMPLATE_LEAK_PATTERNS), re.IGNORECASE)


def _extract_context_texts(contexts: list[Any]) -> list[str]:
    """Normalize contexts to list of strings."""
    result = []
    for c in contexts:
        if isinstance(c, str):
            result.append(c)
        elif isinstance(c, dict):
            result.append(c.get("text") or c.get("content") or str(c))
    return result


def citation_accuracy_single(
    answer: str,
    retrieved_docs: list[dict],
) -> float:
    """Fraction of [filename] citations in the answer that appear in retrieved_docs.

    Catches Phase-26-scope gap: fabricated citations like [b62c7383...valid_document.txt].
    Even though this is a Phase 26 full-fix, we MEASURE it here as a gap indicator.
    """
    if not answer or not retrieved_docs:
        return float("nan")

    # Extract [filename.ext] patterns from the answer
    citation_pattern = re.compile(r"\[([^\]]+\.\w{2,5})\]")
    cited = set(citation_pattern.findall(answer))
    if not cited:
        return 1.0  # no citations = no wrong citations

    # Build set of valid source file names
    valid_sources = set()
    for doc in retrieved_docs:
        meta = doc.get("metadata") or {}
        fname = meta.get("filename") or meta.get("source") or meta.get("doc_id")
        if fname:
            valid_sources.add(str(fname).split("/")[-1])

    if not valid_sources:
        return float("nan")  # can't validate without source info

    valid_citations = sum(1 for c in cited if any(v in c or c in v for v in valid_sources))
    return valid_citations / len(cited)


def citation_locator_accuracy(eval_rows: list[dict]) -> MetricResult | None:
    """Per-modality citation correctness: does a retrieved source match the row's
    expected_citation {source, locator_type, locator}? Scores source match + the
    modality-appropriate locator (page/sheet/timestamp/image_title). Rows whose
    expected locator is None (not yet filled) are skipped, not counted as wrong.
    """
    scored = []
    for row in eval_rows:
        exp = row.get("expected_citation") or {}
        lt = exp.get("locator_type")
        if lt in (None, "none", "web"):
            continue  # nothing to cite (refusal / websearch)
        exp_src = exp.get("source")
        exp_loc = exp.get("locator")
        sources = row.get("retrieved_docs") or []
        if not sources:
            continue
        # source match: any retrieved source basename equals expected source
        src_ok = exp_src is not None and any(
            (s.get("source") or "").split("/")[-1] == str(exp_src).split("/")[-1]
            for s in sources
            if isinstance(s, dict)
        )
        # locator match (only if we have a filled expected locator)
        loc_ok = True
        if exp_loc is not None:
            key = {"page": "page_number", "sheet": "sheet_name", "image_title": "image_title"}.get(
                lt
            )
            if lt in ("timestamp", "timestamp+frame", "frame"):
                loc_ok = any(
                    abs(
                        float(s.get("start_time") or s.get("timestamp_start") or -1e9)
                        - float(exp_loc)
                    )
                    <= 15.0
                    for s in sources
                    if isinstance(s, dict)
                    and (s.get("start_time") or s.get("timestamp_start")) is not None
                )
            elif key:
                loc_ok = any(
                    str(s.get(key)) == str(exp_loc) for s in sources if isinstance(s, dict)
                )
        scored.append(1.0 if (src_ok and loc_ok) else 0.0)
    if not scored:
        return None
    return MetricResult(
        name="citation_locator_accuracy",
        value=round(sum(scored) / len(scored), 4),
        n=len(scored),
        notes="source + per-modality locator match vs expected_citation",
    )


def template_leak_rate(answers: list[str]) -> MetricResult:
    """Fraction of answers containing prompt-template artifacts (P1-7 detection)."""
    if not answers:
        return MetricResult.empty("template_leak_rate", "no answers")
    leaky = sum(1 for a in answers if a and _TEMPLATE_LEAK_RE.search(a))
    return MetricResult(
        name="template_leak_rate",
        value=leaky / len(answers),
        n=len(answers),
        notes=f"leaky={leaky}/{len(answers)}",
    )


def compute_generation_metrics_lexical(
    eval_rows: list[dict],
    judge_label: str = "lexical_fallback",
) -> dict[str, MetricResult]:
    """Compute generation metrics using lexical fallback (no LLM judge required).

    eval_rows: [{"query", "answer", "contexts": [...], "reference_answer", "retrieved_docs"}]
    """
    from app.eval.judges.lexical_judge import (
        lexical_answer_relevancy,
        lexical_context_recall,
        lexical_faithfulness,
    )

    faithfulnesses, relevancies, recalls, cit_accs = [], [], [], []
    answers_for_leak = []

    for row in eval_rows:
        answer = row.get("answer") or ""
        query = row.get("query") or ""
        contexts = _extract_context_texts(row.get("contexts") or [])
        reference = row.get("reference_answer") or ""
        retrieved_docs = row.get("retrieved_docs") or []

        answers_for_leak.append(answer)

        if answer and contexts:
            faithfulnesses.append(lexical_faithfulness(answer, contexts))
        if answer and query:
            relevancies.append(lexical_answer_relevancy(answer, query))
        if contexts and reference and reference not in ("TODO", ""):
            recalls.append(lexical_context_recall(contexts, reference))
        ca = citation_accuracy_single(answer, retrieved_docs)
        if not (isinstance(ca, float) and ca != ca):  # skip nan
            cit_accs.append(ca)

    def _mean(lst: list[float], name: str, n_total: int) -> MetricResult:
        if not lst:
            return MetricResult.empty(name, "insufficient data")
        return MetricResult(
            name=name,
            value=sum(lst) / len(lst),
            n=len(lst),
            notes=f"judge={judge_label}",
        )

    n = len(eval_rows)
    return {
        "faithfulness": _mean(faithfulnesses, "faithfulness", n),
        "answer_relevancy": _mean(relevancies, "answer_relevancy", n),
        "context_recall": _mean(recalls, "context_recall", n),
        "citation_accuracy": _mean(cit_accs, "citation_accuracy", n),
        "template_leak_rate": template_leak_rate(answers_for_leak),
    }


# specific facts = numbers with >=2 digits, decimals, or percentages — reliable to match
_CR_SPECIFIC = re.compile(r"\d[\d,]*\.?\d*\s?%?|\d+\.\d+")


def _deterministic_context_recall(reference: str, contexts: list[str]) -> float | None:
    """Fraction of the reference answer's specific facts (numbers/percentages) that
    appear in the retrieved context. Replaces the mis-framed Prometheus context_recall
    (which graded the raw context as an 'answer'). None when unmeasurable."""
    if not reference or not contexts:
        return None
    facts = [
        f.strip()
        for f in _CR_SPECIFIC.findall(reference)
        if any(ch.isdigit() for ch in f) and len(f.strip()) >= 2
    ]
    facts = [f for f in facts if len(re.sub(r"\D", "", f)) >= 2]  # >=2 digits → specific
    if not facts:
        return None
    ctx = " ".join(contexts)
    ctx_nc = ctx.replace(",", "")

    def _present(f: str) -> bool:
        if f in ctx or f.replace(",", "") in ctx_nc:
            return True
        # numeric core: strip %/commas so "46.2%" matches "46.2 percent", and
        # "391,035" matches "391035" — the digit content is what must be recoverable
        core = f.rstrip("%").replace(",", "").strip()
        return bool(core) and core in ctx_nc

    hits = sum(1 for f in facts if _present(f))
    return round(hits / len(facts), 4)


def compute_generation_metrics_prometheus(
    eval_rows: list[dict],
) -> dict[str, MetricResult]:
    """Compute generation metrics with the Prometheus-2-7B purpose-built judge.

    Prometheus scores each row 1-5 against a per-metric rubric (native Direct
    Assessment contract); scores are normalized to 0..1. Rows the judge cannot
    parse are skipped, not counted as 0, so a parser miss never fakes a regression.
    """
    from app.eval.judges import prometheus_judge

    # context_recall is computed DETERMINISTICALLY (see _deterministic_context_recall):
    # the Prometheus framing graded the raw context-wall as an "answer", scoring ~0
    # even when the reference facts were present. faithfulness/relevancy/correctness
    # stay on Prometheus (verified deterministic + discriminating on clean inputs).
    metric_names = ["faithfulness", "answer_relevancy", "answer_correctness"]
    buckets: dict[str, list[float]] = {m: [] for m in metric_names}
    recall_vals: list[float] = []

    for row in eval_rows:
        answer = row.get("answer") or ""
        query = row.get("query") or ""
        contexts = _extract_context_texts(row.get("contexts") or [])
        reference = row.get("reference_answer") or ""
        graded_row = {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "reference_answer": reference,
        }
        for m in metric_names:
            # faithfulness needs context; the reference-based metrics need a reference.
            if m == "faithfulness" and not (answer and contexts):
                continue
            if m == "answer_correctness" and not (reference and reference not in ("TODO", "")):
                continue
            if m == "answer_relevancy" and not (answer and query):
                continue
            try:
                val = prometheus_judge.grade_metric(m, graded_row)
            except Exception:
                val = None
            if val is not None:
                buckets[m].append(val)

        # deterministic context_recall: fraction of the reference's specific facts
        # (numbers/percentages) recoverable from the retrieved context.
        cr = _deterministic_context_recall(reference, contexts)
        if cr is not None:
            recall_vals.append(cr)

    if recall_vals:
        buckets["context_recall"] = recall_vals
    metric_names = metric_names + ["context_recall"]

    metrics_out: dict[str, MetricResult] = {}
    for m in metric_names:
        vals = buckets[m]
        if vals:
            metrics_out[m] = MetricResult(
                name=m,
                value=round(sum(vals) / len(vals), 4),
                n=len(vals),
                notes=(
                    "deterministic (reference facts recoverable from context)"
                    if m == "context_recall"
                    else "judge=prometheus_2_7b"
                ),
            )

    # Heuristic metrics Prometheus does not cover (kept identical to other paths).
    answers = [r.get("answer") or "" for r in eval_rows]
    metrics_out["template_leak_rate"] = template_leak_rate(answers)

    cite_loc = citation_locator_accuracy(eval_rows)
    if cite_loc is not None:
        metrics_out["citation_locator_accuracy"] = cite_loc

    cit_accs = []
    for row in eval_rows:
        ca = citation_accuracy_single(row.get("answer") or "", row.get("retrieved_docs") or [])
        if not (isinstance(ca, float) and ca != ca):
            cit_accs.append(ca)
    if cit_accs:
        metrics_out["citation_accuracy"] = MetricResult(
            name="citation_accuracy",
            value=sum(cit_accs) / len(cit_accs),
            n=len(cit_accs),
            notes="judge=heuristic",
        )

    return metrics_out


async def compute_generation_metrics_ragas(
    eval_rows: list[dict],
) -> dict[str, MetricResult]:
    """Compute generation metrics using Ragas + local GGUF judge.

    Falls back to lexical judge if GGUF unavailable or Ragas fails.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import HuggingfaceEmbeddings
        from ragas.metrics import (
            answer_relevancy,
            context_recall,
        )
        from ragas.run_config import RunConfig

        from app.core.config import settings as app_settings
        from app.eval.judges.phi3_judge import get_judge

        # Phi-3-mini-4k-instruct judge — dedicated eval judge, outputs strict JSON.
        # Loads ~2.3GB alongside Mistral 7B on GPU.
        RunConfig(max_workers=1, timeout=300)
        judge = get_judge()
        embeddings = HuggingfaceEmbeddings(model_name=app_settings.EMBEDDING_MODEL)

        # Build ragas-compatible dataset (use ground_truth, not deprecated ground_truths)
        data: dict[str, list] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        for row in eval_rows:
            data["question"].append(row.get("query") or "")
            data["answer"].append(row.get("answer") or "")
            ctx = _extract_context_texts(row.get("contexts") or [])
            data["contexts"].append(ctx if ctx else [""])
            ref = row.get("reference_answer") or ""
            data["ground_truth"].append(ref if ref and ref != "TODO" else "")

        dataset = Dataset.from_dict(data)
        # Run answer_relevancy and context_recall via Ragas (no decompose step)
        result = evaluate(
            dataset,
            metrics=[answer_relevancy, context_recall],
            llm=judge,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=1, timeout=600),
        )

        metrics_out: dict[str, MetricResult] = {}
        for key in ("answer_relevancy", "context_recall"):
            val = result.get(key)
            if val is not None:
                metrics_out[key] = MetricResult(
                    name=key,
                    value=float(val),
                    n=len(eval_rows),
                    notes="judge=phi3_mini",
                )

        # Faithfulness — direct Phi-3 NLI (skip Ragas decompose step which truncates)
        try:
            import json as _json
            import re as _re

            from app.eval.judges.phi3_judge import _extract_json, _generate

            faith_scores = []
            for row in eval_rows:
                answer = row.get("answer") or ""
                contexts = row.get("contexts") or []
                ctx_text = " ".join(contexts)[:800] if contexts else ""
                # Split answer into simple sentences
                sentences = [
                    s.strip() for s in _re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10
                ][:4]
                if not sentences or not ctx_text:
                    faith_scores.append(1.0)
                    continue
                stmts_str = str(sentences)
                nli_prompt = (
                    f"Judge faithfulness. For each statement, verdict=1 if supported by context, 0 if not.\n"
                    f"context: {ctx_text}\n"
                    f"statements: {stmts_str}"
                )
                raw = _generate(nli_prompt)
                extracted = _extract_json(raw)
                try:
                    items = _json.loads(extracted)
                    if isinstance(items, list) and items:
                        verdicts = [int(x.get("verdict", 0)) for x in items if isinstance(x, dict)]
                        faith_scores.append(sum(verdicts) / len(verdicts) if verdicts else 1.0)
                    else:
                        faith_scores.append(1.0)
                except Exception:
                    faith_scores.append(1.0)
            if faith_scores:
                metrics_out["faithfulness"] = MetricResult(
                    name="faithfulness",
                    value=round(sum(faith_scores) / len(faith_scores), 4),
                    n=len(faith_scores),
                    notes="judge=phi3_mini_direct_nli",
                )
        except Exception as _fe:
            pass

        # Add metrics Ragas doesn't compute
        answers = [r.get("answer") or "" for r in eval_rows]
        metrics_out["template_leak_rate"] = template_leak_rate(answers)

        cit_accs = []
        for row in eval_rows:
            ca = citation_accuracy_single(row.get("answer") or "", row.get("retrieved_docs") or [])
            if not (isinstance(ca, float) and ca != ca):
                cit_accs.append(ca)
        if cit_accs:
            metrics_out["citation_accuracy"] = MetricResult(
                name="citation_accuracy",
                value=sum(cit_accs) / len(cit_accs),
                n=len(cit_accs),
                notes="judge=heuristic",
            )

        return metrics_out

    except Exception as exc:
        # Fall back to lexical judge, clearly labelled
        fallback = compute_generation_metrics_lexical(
            eval_rows, judge_label=f"lexical_fallback (ragas_error: {exc})"
        )
        return fallback


def compute_generation_metrics(
    eval_rows: list[dict],
    prefer_ragas: bool = True,
) -> dict[str, MetricResult]:
    """Synchronous entry point.

    Judge selection follows EVAL_JUDGE_MODEL:
      - "prometheus" / "prometheus_2_7b" → purpose-built Prometheus-2 judge
      - anything else with prefer_ragas   → Ragas + phi3 judge (legacy path)
      - lexical fallback if the chosen judge fails.
    """
    # Single source of truth — app/eval/config.py owns this env var's default.
    # This previously re-read EVAL_JUDGE_MODEL directly with its own, DIFFERENT
    # fallback ("gguf_mistral" here vs "prometheus_2_7b" there). While .env
    # happened to set the variable explicitly the two agreed by luck; once the
    # settings migration dropped it from .env, they silently diverged — the
    # report's `judge` field (from EvalConfig) claimed prometheus_2_7b while
    # this function actually dispatched the legacy Ragas/phi3 path. That makes
    # every generation number attest to a judge that never ran, which is
    # exactly the provenance thresholds.yaml's v3 retirement depends on.
    from app.eval.config import EVAL_JUDGE_MODEL

    judge_model = EVAL_JUDGE_MODEL.lower()

    if judge_model.startswith("prometheus"):
        try:
            from app.eval.judges import prometheus_judge

            if prometheus_judge.is_available():
                return compute_generation_metrics_prometheus(eval_rows)
            print("[eval] Prometheus GGUF not found — falling back to lexical judge")
        except Exception as exc:
            print(f"[eval] Prometheus judge failed ({exc}) — falling back to lexical judge")
        return compute_generation_metrics_lexical(
            eval_rows, judge_label="lexical_fallback (prometheus_unavailable)"
        )

    if prefer_ragas:
        try:
            import asyncio

            loop = asyncio.new_event_loop()
            metrics = loop.run_until_complete(compute_generation_metrics_ragas(eval_rows))
            loop.close()
            return metrics
        except Exception:
            pass  # fall through to lexical

    return compute_generation_metrics_lexical(eval_rows)
