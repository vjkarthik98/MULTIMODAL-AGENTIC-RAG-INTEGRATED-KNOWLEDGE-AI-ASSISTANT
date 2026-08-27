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

# Context budget for the direct-NLI faithfulness call in
# compute_generation_metrics_ragas(). Was 800 chars — roughly an eighth of what
# the production model itself was given (settings.MAX_CONTEXT_CHARS=16000), so
# any claim supported only by later context looked unfaithful purely because the
# judge could not see its own evidence. That is the identical failure
# qwen_judge.grade_metric documents at length for its own `_CTX_CHAR_BUDGET`,
# which live measurement (2026-08-17) settled at 6000 against the same
# n_ctx=8192 judge. Matched to it here.
#
# 6000 is comfortable rather than tight in this path: grade_metric has to insert
# its context TWICE (once as the instruction, once as the graded reference,
# because faithfulness has no separate gold answer), whereas this prompt embeds
# it once alongside at most _NLI_MAX_STATEMENTS short sentences.
#
# Raising this cannot regress a published baseline, because this code has never
# executed: it sits after the HuggingfaceEmbeddings construction that raised on
# NumPy >= 2, so every run to date fell through to the lexical judge before
# reaching it.
_NLI_CTX_CHAR_BUDGET = 6000
_NLI_MAX_STATEMENTS = 4


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
# (_CR_SPECIFIC removed 2026-08-13 — _deterministic_context_recall now reuses
# hallucination._parse_numbers, which classifies years/identifiers correctly
# instead of treating any >=2-digit token as a fact.)
#
# _CR_DATE_RE moved to app/eval/metrics/hallucination.py (2026-08-17) — the
# SAME date-embedded-number bug it fixes here on the reference-answer side
# (context_recall) was independently found unfixed on the answer/fabrication
# side (_numbers_grounded, which drives the GATED fabrication_rate metric);
# hallucination.py already had the lower-level _parse_numbers this reuses, so
# the regex lives there now and both modules share the one definition instead
# of drifting into two copies. See hallucination.py's docstring on it for the
# full reproduction (the day-of-month in "September 18, 2024" independently
# flagging 3 separate audio rows as fabrication in one suite run).


def _deterministic_context_recall(reference: str, contexts: list[str]) -> float | None:
    """Fraction of the reference answer's specific facts (numbers/percentages) that
    appear in the retrieved context. Replaces the mis-framed Prometheus context_recall
    (which graded the raw context as an 'answer'). None when unmeasurable.

    CALENDAR YEARS AND IDENTIFIERS ARE EXCLUDED (2026-08-13, per-modality
    quality pass). The old ">=2 digits is a specific fact" rule counted "2024"
    as a fact to be recovered, which systematically punished modalities whose
    CONTEXT does not restate the year even though it is on-topic. Measured on
    audio (FOMC press conferences), where speakers say "this year"/"in August"
    while the gold reference writes the full "September 18, 2024": 8 of the 10
    measurable audio rows were missing ONLY a year or date fragment
    (['2024'], ['2024,'], ['2024.'], ['18,', '2024']), holding audio's
    context_recall at 0.583 — the worst of all 7 modalities — for a date
    convention rather than any retrieval failure. Document modalities are
    unaffected because a 10-K's text repeats the year everywhere, so this was
    an apples-to-oranges comparison across modalities.

    Reuses app/eval/metrics/hallucination.py's `_parse_numbers`, whose _Num
    already carries the correct `is_year` (bare 4-digit 1900-2099, no unit/
    decimal/%) and `is_id` (>=7-digit bare integer, e.g. SEC accession)
    classification — the same exclusions that module's own numeric-grounding
    checks apply. Deliberately does NOT reuse its stricter
    `_is_material_figure`, which additionally requires a unit/percent/decimal
    and would drop genuine bare figures like "116,000" payroll jobs.
    """
    if not reference or not contexts:
        return None
    from app.eval.metrics.hallucination import _CR_DATE_RE, _parse_numbers

    _ref_no_dates = _CR_DATE_RE.sub(" ", reference)
    facts = [
        n.raw.strip()
        for n in _parse_numbers(_ref_no_dates)
        if not n.is_year and not n.is_id and len(re.sub(r"\D", "", n.raw)) >= 2
    ]
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


def compute_generation_metrics_rubric(
    eval_rows: list[dict],
) -> dict[str, MetricResult]:
    """Compute generation metrics via MAGIK's single judge (app/eval/judges/qwen_judge.py),
    using its Direct-Assessment rubric interface.

    The judge scores each row 1-5 against a per-metric rubric; scores are
    normalized to 0..1. Rows the judge cannot parse are skipped, not counted
    as 0, so a parser miss never fakes a regression.
    """
    from app.eval.judges import qwen_judge

    # context_recall is computed DETERMINISTICALLY (see _deterministic_context_recall):
    # grading the raw context-wall as an "answer" scores ~0 even when the reference
    # facts are present. faithfulness/relevancy/correctness stay on the judge
    # (verified deterministic + discriminating on clean inputs under Prometheus;
    # same rubric contract, unaffected by the judge swap).
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
                val = qwen_judge.grade_metric(m, graded_row)
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
                    else "judge=qwen2.5_7b"
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


def _build_ragas_embeddings():
    """Ragas-compatible embeddings backed by MAGIK's OWN resident encoder.

    NOT `ragas.embeddings.HuggingfaceEmbeddings`, and not as a style
    preference — that class cannot be constructed here at all. Its
    `__post_init__` decides bi-encoder vs cross-encoder with

        self.is_cross_encoder = bool(np.intersect1d(
            list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
            config.architectures))

    For any plain bi-encoder — `BAAI/bge-large-en-v1.5`'s architecture is
    `BertModel`, which is not a sequence-classification head — that
    intersection is EMPTY, and `bool()` of an empty NumPy array raises
    `ValueError: The truth value of an empty array is ambiguous` on NumPy
    >= 2. Ragas declares the class with `pydantic.dataclasses.dataclass`, so
    it surfaces as "1 validation error for HuggingfaceEmbeddings". It is
    deterministic: the embedding model this project uses can never construct
    it, on any run.

    `compute_generation_metrics_ragas` caught that and fell through to the
    lexical judge — silently. Every metric in v1.0.0-rc3's quality report
    (quality-reports/ragas/20260827-064744-live.json) carries
    `judge=lexical_fallback (ragas_error: ...)` in its notes while the badge
    read "ragas faithfulness 0.87" in bright green. 98 rows, 0 errors, real
    numbers — from a crude lexical scorer, not the Qwen-judge-backed Ragas
    evaluation the label claimed.

    Reusing MAGIK's own embedder is strictly better than repairing the Ragas
    class would have been:
      * `model_loader.get_embedder()` is a singleton that
        `generation_runner._make_full_context_retriever()` has ALREADY loaded
        in this process to build grading context, so this costs no extra
        VRAM — whereas HuggingfaceEmbeddings would load a second, independent
        SentenceTransformer copy of the same weights.
      * Relevancy is then measured in the exact embedding space production
        retrieval ranks in (same model, same device placement, same
        finance-number normalisation), rather than a parallel one.

    Length is preserved 1:1 on purpose. `TextEmbedder.embed_texts` DROPS any
    entry it cannot sanitise or encode (it appends nothing in that branch),
    and Ragas's `AnswerRelevancy.calculate_similarity` does
    `np.asarray(embed_documents(qs)).reshape(len(qs), -1)` — a short list is
    a hard ValueError there, so every input must map to exactly one vector.
    """
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.run_config import RunConfig

    from app.core.config import settings as app_settings
    from app.core.model_loader import model_loader

    # Ragas can generate an empty question for a degenerate answer. Embedding
    # a placeholder keeps the 1:1 contract above with a real vector; a zero
    # vector would divide by a zero norm in calculate_similarity and turn the
    # WHOLE metric's mean into NaN over one bad row.
    _EMPTY_PLACEHOLDER = "(empty)"

    class _MagikRagasEmbeddings(BaseRagasEmbeddings):  # type: ignore[misc]
        def __init__(self) -> None:
            self._embedder = model_loader.get_embedder()
            self._dim = getattr(self._embedder, "expected_dim", app_settings.TEXT_EMBEDDING_DIM)
            self.set_run_config(RunConfig(max_workers=1, timeout=600))

        def _embed_one(self, text: str) -> list[float]:
            try:
                return self._embedder.embed_text((text or "").strip() or _EMPTY_PLACEHOLDER)
            except Exception:
                # Never propagate: one unembeddable string must degrade that
                # single comparison, not abort the whole report.
                return [0.0] * self._dim

        def embed_query(self, text: str) -> list[float]:
            return self._embed_one(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._embed_one(t) for t in texts]

    return _MagikRagasEmbeddings()


async def compute_generation_metrics_ragas(
    eval_rows: list[dict],
) -> dict[str, MetricResult]:
    """Compute generation metrics using the real Ragas library + MAGIK's judge.

    Backs app/eval/ragas_report.py's standalone, always-real-Ragas-library
    portfolio report. Falls back to the lexical judge if the judge is
    unavailable or Ragas itself fails.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_recall,
        )
        from ragas.run_config import RunConfig

        from app.eval.judges.qwen_judge import get_ragas_judge

        judge = get_ragas_judge()
        embeddings = _build_ragas_embeddings()

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
                    notes="judge=qwen2.5_7b",
                )

        # Faithfulness — direct NLI call (skip Ragas decompose step which truncates)
        try:
            import json as _json
            import re as _re

            from app.eval.judges.qwen_judge import _extract_json_from_text
            from app.eval.judges.qwen_judge import generate as _judge_generate

            # A row the judge could not grade is NOT a faithful row.
            #
            # Every failure path here used to append 1.0 — a perfect score for
            # "no statements to check", "no context", "the judge returned
            # something unparseable", and "an exception was raised" alike. That
            # is the same class of bug as the lexical fallback above (a number
            # that does not measure what it claims), pointing at the same
            # public badge, and it matters more now than it ever has: this
            # whole block sits AFTER the HuggingfaceEmbeddings construction
            # that used to raise, so until that was fixed none of it had ever
            # executed in production — rc3's report scored faithfulness with
            # the lexical judge, not this. Turning the Ragas path back on turns
            # this on with it, so an ungradeable row is now excluded from the
            # mean and counted in `notes` instead of silently inflating it.
            faith_scores = []
            faith_unscored = 0
            for row in eval_rows:
                answer = row.get("answer") or ""
                contexts = row.get("contexts") or []
                ctx_text = " ".join(contexts)[:_NLI_CTX_CHAR_BUDGET] if contexts else ""
                # First _NLI_MAX_STATEMENTS sentences only. A deliberate cap on
                # judge cost per row, not an oversight — but it does mean
                # `faithfulness` describes the opening of each answer rather
                # than all of it, which is worth knowing when reading the number.
                sentences = [
                    s.strip() for s in _re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10
                ][:_NLI_MAX_STATEMENTS]
                if not sentences or not ctx_text:
                    faith_unscored += 1
                    continue
                stmts_str = str(sentences)
                nli_prompt = (
                    f"Judge faithfulness. For each statement, verdict=1 if supported by context, 0 if not.\n"
                    f"context: {ctx_text}\n"
                    f"statements: {stmts_str}"
                )
                raw = _judge_generate(
                    nli_prompt,
                    system="You are a JSON-only evaluator. Output ONLY a JSON array.",
                )
                extracted = _extract_json_from_text(raw)
                try:
                    items = _json.loads(extracted)
                    verdicts = (
                        [int(x.get("verdict", 0)) for x in items if isinstance(x, dict)]
                        if isinstance(items, list)
                        else []
                    )
                    if verdicts:
                        faith_scores.append(sum(verdicts) / len(verdicts))
                    else:
                        faith_unscored += 1
                except Exception:
                    faith_unscored += 1
            if faith_scores:
                metrics_out["faithfulness"] = MetricResult(
                    name="faithfulness",
                    value=round(sum(faith_scores) / len(faith_scores), 4),
                    n=len(faith_scores),
                    notes=(
                        "judge=qwen2.5_7b_direct_nli"
                        + (
                            f" ({faith_unscored} rows ungradeable, excluded)"
                            if faith_unscored
                            else ""
                        )
                    ),
                )
            elif faith_unscored:
                metrics_out["faithfulness"] = MetricResult.empty(
                    "faithfulness", f"judge returned nothing gradeable on all {faith_unscored} rows"
                )
        except Exception:
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
        return compute_generation_metrics_lexical(
            eval_rows, judge_label=f"lexical_fallback (ragas_error: {exc})"
        )


def compute_generation_metrics(
    eval_rows: list[dict],
) -> dict[str, MetricResult]:
    """Synchronous entry point — MAGIK's single judge (qwen_judge), rubric
    interface, with lexical fallback if the judge is unavailable or fails.

    Judge history: this used to branch on EVAL_JUDGE_MODEL between multiple
    LLM judges (Prometheus, and before that Phi-3 via Ragas). Simplified
    2026-08-01 to one judge, one path — see app/eval/judges/qwen_judge.py's
    module docstring for why. app/eval/ragas_report.py calls
    compute_generation_metrics_ragas() directly for its own real-Ragas-library
    report; that path is deliberately not branched into here.
    """
    from app.eval.judges import qwen_judge

    try:
        if qwen_judge.ensure_available():
            return compute_generation_metrics_rubric(eval_rows)
    except Exception as exc:
        print(f"[eval] Qwen judge failed ({exc}) — falling back to lexical judge")
    return compute_generation_metrics_lexical(
        eval_rows, judge_label="lexical_fallback (qwen_judge_unavailable)"
    )
