"""Generation quality metrics: faithfulness, answer_relevancy, context_recall,
citation_accuracy, template_leak_rate.

Uses Ragas with local GGUF judge when available, lexical fallback otherwise.
Judge availability is recorded in metric notes so reports are never misleading.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.eval.metrics.base import MetricResult


# Prompt-template artifact patterns (P1-7: template leakage)
_TEMPLATE_LEAK_PATTERNS = [
    r"\[sic\]",
    r"Sources Used: \d+",
    r"\{[a-zA-Z_]+\}",        # unfilled template variable
    r"<context>",
    r"</context>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]
_TEMPLATE_LEAK_RE = re.compile("|".join(_TEMPLATE_LEAK_PATTERNS), re.IGNORECASE)


def _extract_context_texts(contexts: List[Any]) -> List[str]:
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
    retrieved_docs: List[Dict],
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


def template_leak_rate(answers: List[str]) -> MetricResult:
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
    eval_rows: List[Dict],
    judge_label: str = "lexical_fallback",
) -> Dict[str, MetricResult]:
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

    def _mean(lst: List[float], name: str, n_total: int) -> MetricResult:
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


async def compute_generation_metrics_ragas(
    eval_rows: List[Dict],
) -> Dict[str, MetricResult]:
    """Compute generation metrics using Ragas + local GGUF judge.

    Falls back to lexical judge if GGUF unavailable or Ragas fails.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from app.eval.judges.gguf_judge import get_judge
        from app.eval.config import EVAL_JUDGE_TEMPERATURE

        judge = get_judge(temperature=EVAL_JUDGE_TEMPERATURE)

        # Build ragas-compatible dataset
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truths": [],
        }
        for row in eval_rows:
            data["question"].append(row.get("query") or "")
            data["answer"].append(row.get("answer") or "")
            ctx = _extract_context_texts(row.get("contexts") or [])
            data["contexts"].append(ctx if ctx else [""])
            ref = row.get("reference_answer") or ""
            data["ground_truths"].append([ref] if ref and ref != "TODO" else [""])

        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=judge,
        )

        metrics_out: Dict[str, MetricResult] = {}
        for key in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
            val = result.get(key)
            if val is not None:
                metrics_out[key] = MetricResult(
                    name=key,
                    value=float(val),
                    n=len(eval_rows),
                    notes="judge=gguf_mistral",
                )

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
        fallback = compute_generation_metrics_lexical(eval_rows, judge_label=f"lexical_fallback (ragas_error: {exc})")
        return fallback


def compute_generation_metrics(
    eval_rows: List[Dict],
    prefer_ragas: bool = True,
) -> Dict[str, MetricResult]:
    """Synchronous entry point. Uses Ragas if available, lexical fallback otherwise."""
    if prefer_ragas:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            metrics = loop.run_until_complete(compute_generation_metrics_ragas(eval_rows))
            loop.close()
            return metrics
        except Exception as exc:
            pass  # fall through to lexical

    return compute_generation_metrics_lexical(eval_rows)
