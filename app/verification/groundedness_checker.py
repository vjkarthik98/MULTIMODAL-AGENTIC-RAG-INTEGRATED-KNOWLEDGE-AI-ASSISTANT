"""GroundednessChecker — Responsibility 2 (docs/Phase_32_Agentic_Answer_Verification.md).

Wraps app/reasoning/reasoning_engine.py's existing numeric-faithfulness guard
(_hallucination_guard, _unsupported_numbers) rather than duplicating it — that
guard already runs a one-shot hardened retry inside generate_answer(). This
module adds the 0-100 confidence score VerificationLoop needs on top of the
guard's own (bool, float) / (List[str]) outputs.

NLI PASS (hallucination-reduction initiative, Phase 3, 2026-08-13): the
lexical `_hallucination_guard` support_score is pure word-overlap (>=25% of a
sentence's significant words as raw substrings in context) — a well-grounded
paraphrase can score as unsupported, and a sentence that merely shares
keywords with an unrelated chunk can score as grounded. When
settings.NLI_GROUNDEDNESS_ENABLED, a real GPU entailment model
(cross-encoder/nli-deberta-v3-base via model_loader.get_nli_model(), verified
locally: id2label={0:'contradiction',1:'entailment',2:'neutral'}) scores each
answer sentence against its single best-matching context chunk.

DESIGN NOTE — this does NOT replace support_score with raw entailment
probability, despite that being the original plan. Empirically verified
(2026-08-13, live probe against real generated answers) that entailment
probability is unreliable for this purpose: a genuinely correct restatement
that adds attribution framing ("According to the FOMC...") or shifts pronoun
("our" -> "their") scores as NEUTRAL, not entailment (0.9996 neutral prob in
the reproduced case), because NLI models check strict logical entailment, not
topical correctness. Contradiction probability, by contrast, is cleanly
separable in every case tested: 0.0001-0.003 for correct/neutral answers vs.
0.9999 for a fabricated number. So NLI here is used ONLY as an additive
CONTRADICTION penalty layered on the unchanged lexical support_score/
is_hallucinated verdict (same pattern as the existing numeric_penalty below)
— it can only ADD a hallucination flag (a wrong-number-embedded-in-high-
word-overlap-text case the lexical+numeric checks might individually miss or
under-weight), never REMOVE one the lexical guard already raised. This is a
deliberate, narrower scope than "recognize correct paraphrases the lexical
check misses" — single-premise sentence-level NLI cannot reliably distinguish
that case from a genuinely unrelated/fabricated claim (both produce the same
low-contradiction, high-neutral signature); a real fix would need a
differently-trained model or multi-premise aggregation, not attempted here.

Bounded by settings.NLI_MAX_SENTENCE_PAIRS_PER_CALL sentences per call
(CLAUDE.md's bounded-loop discipline) and matched one-doc-per-sentence, not an
all-pairs sweep, to keep the GPU cost per answer small and predictable. Any
failure (model load, OOM, empty input) contributes no penalty — an optional
accuracy upgrade must never turn into an availability regression.

Imports reasoning_engine/model_loader lazily (function-scoped), matching this
codebase's existing convention for cross-module calls between app/reasoning/,
app/pipeline/, app/agents/, and app/core/ — there are no module-level
cross-imports among those packages today, and this file must not become the
first one.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger
from app.verification.verification_schema import GroundednessResult

logger = get_logger(__name__)

# xlsx retrieval chunks pack many rows (often 40-60+ countries/tickers/etc.)
# into one ~2000-char chunk, each row's first cell being the entity/row label
# (e.g. "Canada | AAA | AA+ | Aaa" or the colon-style variant some sheets use,
# "Country: Canada, S&P Rating: AAA, ..."). Matched below in _narrow_tabular_premise.
_XLSX_PIPE_ROW_RE = re.compile(r"^([^|:\n]+?)\s*\|")
_XLSX_COLON_ROW_RE = re.compile(r"^Country:\s*([^,]+),")

# Generic table-structure words that can never be a real entity/row label,
# for sheets whose "Columns:" line is too vague (a free-text description
# rather than actual pipe-delimited column names) for the column-name
# exclusion below to catch a malformed metadata row echoing these terms.
_XLSX_GENERIC_ROW_LABELS = frozenset(
    {
        "country",
        "total",
        "region",
        "sheet",
        "columns",
        "notes",
        "date",
        "source",
        "average",
        "median",
        "rating",
        "moody's rating",
    }
)

# Credit-rating grades (S&P style "AA+"/"BBB-"/"NR", Moody's style "Aaa"/"Baa2")
# — short enough, and reused as a row KEY in generic ratings-lookup sheets
# (e.g. "Rating | Default Spread"), that they collide with an answer simply
# STATING an entity's rating as an attribute. Live-reproduced: an answer
# about Switzerland's Aaa rating word-boundary-matched a row labeled "Aaa" in
# an unrelated rating-to-spread lookup table (score 0.39 contradiction,
# premise about the wrong table entirely) — a real word-boundary match, just
# not the entity the answer is actually about. _find_tabular_row_premise
# tries excluding these first so a genuine entity-name match (a country,
# company, ticker — not a grade) always wins when both exist.
_XLSX_RATING_CODE_RE = re.compile(r"^(?:AAA|AA|A|BBB|BB|B|CCC|CC|C|D|NR|N/A)[+-]?$|^[ABC]a{0,2}[123]?$", re.I)


def _narrow_tabular_premise(text: str, sentence: str, exclude_rating_codes: bool = False) -> str:
    """Narrow a dense xlsx sheet chunk to just the row(s) the sentence is
    actually about, keeping the header line(s) for column-name context.

    Live-reproduced (2026-08-17, xlsx grounding follow-up): passing a whole
    ~2000-char, 58-country "Sovereign Ratings" chunk as the NLI premise for a
    single-country sentence about Canada — a verbatim-correct claim — measured
    contradiction=0.87. Isolating just Canada's row (+ header) for the SAME
    sentence measured contradiction=0.0003. The NLI model isn't wrong about
    THIS row; it's overwhelmed by dozens of other entities' similarly-shaped
    but different rating tokens sharing the premise. Falls back to the
    unmodified text whenever the chunk doesn't look tabular or no row's label
    is mentioned in the sentence — this can only ever shrink an already-
    selected premise to a more relevant subset, never change WHICH doc is used.
    """
    lines = text.split("\n")
    header_lines = [ln for ln in lines if ln.startswith("Sheet:") or ln.startswith("Columns:")]
    if not header_lines:
        return text

    # Column names themselves (declared in the "Columns:" line) never count
    # as a row label — live-reproduced: some sheets have a malformed
    # instructions/metadata row baked into the data area whose first cell is
    # literally "Country" (echoing the column header, not naming a country),
    # e.g. "Country | Africa | Moody's rating | ... | Has to be sorted in
    # ascending order". Since "country" is a generic word present in nearly
    # every answer about country data, that row matched spuriously and was
    # used as premise in place of the real entity's row. Excluding any label
    # equal to a declared column name closes this without a hand-maintained
    # denylist — it generalizes to every sheet's own column vocabulary.
    column_names: set[str] = set()
    for hl in header_lines:
        if hl.startswith("Columns:"):
            column_names |= {c.strip().lower() for c in hl[len("Columns:") :].split("|")}

    sentence_lower = sentence.lower()
    matched = []
    for ln in lines:
        m = _XLSX_PIPE_ROW_RE.match(ln) or _XLSX_COLON_ROW_RE.match(ln)
        if not m:
            continue
        label = m.group(1).strip()
        label_lower = label.lower()
        if len(label) < 3 or label_lower in column_names or label_lower in _XLSX_GENERIC_ROW_LABELS:
            continue
        if exclude_rating_codes and _XLSX_RATING_CODE_RE.match(label):
            continue
        # Word-boundary match, not a bare substring check: this domain is
        # full of short alphanumeric rating codes that are substrings of each
        # other ("Aa1" inside "Caa1", "A1" inside "Baa1"). Live-reproduced: a
        # row labeled "Aa1" in an unrelated ratings-lookup sheet matched an
        # answer stating Argentina's Moody's rating as "Caa1" purely because
        # "aa1" is a contiguous substring of "caa1" — a false match with no
        # relation to the actual claim.
        if re.search(rf"\b{re.escape(label_lower)}\b", sentence_lower):
            matched.append(ln)

    if not matched:
        return text
    return "\n".join(header_lines + matched)


def _find_tabular_row_premise(answer: str, docs: list[dict[str, Any]]) -> str | None:
    """Scan every retrieved doc (not just _best_matching_doc_text's single
    word-overlap pick) for a tabular row whose label the answer names.

    Live-reproduced (2026-08-17): _best_matching_doc_text's raw hit-COUNT
    heuristic favors longer/denser chunks over the one actually containing
    the named entity — a 48-doc xlsx retrieval had Switzerland's row at rank
    5 (score 0.30) while an unrelated, longer chunk with more incidental
    keyword overlap ranked 1st and was selected as the NLI premise, so
    _narrow_tabular_premise correctly found no match IN THAT chunk and fell
    back to it unchanged — still noisy, still 0.54 contradiction. Searching
    every candidate doc directly for the row, independent of which one
    word-overlap preferred, finds Switzerland's actual row regardless of its
    retrieval rank. Returns None (caller falls back to the existing
    per-sentence word-overlap selection) when no doc yields a match — this
    can only ever supply a MORE targeted premise than the fallback, never a
    worse one.
    """
    texts = [str(d.get("text") or "") if isinstance(d, dict) else "" for d in (docs or [])]

    def _best_of(candidates: list[str]) -> str | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # The same entity (e.g. "India") commonly has a row in SEVERAL sheets
        # at different granularity (a detailed per-country lookup vs. a
        # regional weighted-AVERAGE summary) — live-reproduced: the first
        # entity match found was a 3-column regional-average row missing most
        # of the answer's own stated figures, while the correct 9-column
        # per-country sheet ranked lower in the doc list. Preferring whichever
        # candidate contains the most of the answer's own numeric tokens
        # picks the sheet the answer was actually drawn from.
        answer_numbers = set(re.findall(r"\d+\.?\d*%?", answer))
        if not answer_numbers:
            return candidates[0]
        return max(candidates, key=lambda c: len(answer_numbers & set(re.findall(r"\d+\.?\d*%?", c))))

    # Pass 1: entity-name matches only (rating codes excluded) — a genuine
    # country/company/ticker row always wins over an unrelated table whose
    # row key happens to equal a rating grade mentioned in the answer.
    candidates = [
        narrowed
        for text in texts
        if text
        for narrowed in [_narrow_tabular_premise(text, answer, exclude_rating_codes=True)]
        if narrowed is not text
    ]
    best = _best_of(candidates)
    if best is not None:
        return best

    # Pass 2: fall back to allowing rating-code labels too, for answers that
    # are themselves genuinely about a rating-to-spread lookup (no entity
    # name involved at all, e.g. "what spread does a Ba1 rating carry").
    candidates = [
        narrowed
        for text in texts
        if text
        for narrowed in [_narrow_tabular_premise(text, answer)]
        if narrowed is not text
    ]
    return _best_of(candidates)


def _best_matching_doc_text(sentence: str, docs: list[dict[str, Any]]) -> str:
    """Cheapest (CPU, word-overlap) doc whose text is the best NLI premise
    for `sentence` — avoids an all-pairs sentence x doc NLI sweep."""
    words = [w.lower() for w in sentence.split() if len(w) > 4]
    texts = [str(d.get("text") or "") for d in docs]
    if not words:
        return texts[0] if texts else ""
    best_text, best_hits = "", -1
    for text in texts:
        low = text.lower()
        hits = sum(1 for w in words if w in low)
        if hits > best_hits:
            best_hits, best_text = hits, text
    return best_text


def _nli_contradiction_score(answer: str, docs: list[dict[str, Any]]) -> float | None:
    """Mean contradiction probability across up to NLI_MAX_SENTENCE_PAIRS_PER_CALL
    answer sentences, each scored against its best-matching doc. Returns None
    (not 0.0 — "no signal", not "clean") on any failure, so callers apply no
    penalty rather than treating an unrelated failure (e.g. GPU OOM) as either
    a pass or a hallucination finding."""
    try:
        from app.reasoning.reasoning_engine import _split_answer_sentences

        sentences = _split_answer_sentences(answer)[: settings.NLI_MAX_SENTENCE_PAIRS_PER_CALL]
        if not sentences or not docs:
            return None

        from app.core.model_loader import model_loader

        nli = model_loader.get_nli_model()
        contra_idx = 0
        try:
            id2label = nli.model.config.id2label
            contra_idx = next(i for i, label in id2label.items() if label == "contradiction")
        except Exception:
            pass  # verified default (index 0) for cross-encoder/nli-deberta-v3-base

        # Match row entities against the WHOLE answer, not just the current
        # sentence — a multi-sentence answer commonly names its subject once
        # ("China carries a Moody's rating of A1.") then continues without
        # repeating it ("Rating-based approach: default spread 0.599%...").
        # Search every retrieved doc directly for the matching row FIRST
        # (_find_tabular_row_premise), since _best_matching_doc_text's raw
        # word-overlap hit count can rank a longer, unrelated chunk above the
        # one actually containing the named entity's row. Only fall back to
        # the per-sentence word-overlap pick when no doc yields a row match
        # anywhere (non-tabular content, or a genuinely unnamed entity).
        shared_premise = _find_tabular_row_premise(answer, docs)
        pairs = [
            (shared_premise or _narrow_tabular_premise(_best_matching_doc_text(s, docs), answer), s)
            for s in sentences
        ]
        import torch

        raw_scores = nli.predict(pairs)
        probs = torch.softmax(torch.tensor(raw_scores, dtype=torch.float32), dim=1)
        contra_probs = probs[:, contra_idx].tolist()
        return sum(contra_probs) / len(contra_probs)
    except Exception as exc:
        logger.warning(event="nli_groundedness_failed", error=str(exc))
        return None


def lexical_and_nli_verdict(
    answer: str, docs: list[dict[str, Any]], deterministic_answer: bool = False
) -> tuple[bool, bool, float, float, bool]:
    """The lexical word-overlap check PLUS the additive NLI contradiction
    penalty, WITHOUT numeric grounding — the exact piece that was duplicated
    across three places before the hallucination-reduction initiative's
    Phase 4 consolidation (2026-08-13): reasoning_engine.generate_answer()
    called `_hallucination_guard` directly, GroundednessChecker.check() wrapped
    it, and output_guard._check_groundedness() called a THIRD, independent
    implementation (app.eval.metrics.hallucination.hallucination_flag_single).
    All three now call this single function (either directly, in
    generate_answer()'s case, or via GroundednessChecker.check() below).

    Deliberately excludes numeric grounding: reasoning_engine.generate_answer()
    has its own more sophisticated numeric-mismatch retry-then-citation-bypass
    flow (a hardened one-shot LLM retry, then a citation-based override) that
    must not be duplicated or second-guessed here — see that function's own
    numeric-handling block, which stays untouched by this consolidation.
    GroundednessChecker.check() runs its own separate numeric check
    (_unsupported_numbers) after calling this function.

    `deterministic_answer=True` marks an answer that was MECHANICALLY DERIVED
    from the retrieved context rather than generated — currently only
    rag_pipeline._synthesize_image_chart_answer(), which parses figures
    straight out of the context's own "CHART VALUES" block. Such an answer
    cannot fabricate by construction, so the NLI *semantic-entailment* judgment
    is the wrong instrument for it (the same category error as NLI-checking a
    SQL result against the database it came from). Measured: single-premise NLI
    returned contradiction >0.3 on 11 of 14 image rows — multi-clause
    ranking/comparison answers scored against one chart chunk listing several
    series' values — capping the penalty and pinning grounding at 40.0 despite
    lexical support=1.0 and zero unsupported numbers. This is the
    single-premise limitation already documented in this module's docstring.
    The lexical AND numeric checks still run for these answers (the numeric one
    is the real safety property: every figure must appear in context, which
    would also catch a bug in the synthesizer itself) — only the NLI penalty is
    skipped.

    Returns (is_hallucinated, lexical_hallucinated, support_score,
    nli_contradiction, nli_contradicted) — `lexical_hallucinated` is the
    lexical-only verdict, kept separate from the combined `is_hallucinated`
    so callers can report which signal actually fired instead of a merged,
    potentially misleading message.
    """
    from app.reasoning.reasoning_engine import _hallucination_guard

    lexical_hallucinated, support_score = _hallucination_guard(answer, docs)

    nli_contradiction = 0.0
    if settings.NLI_GROUNDEDNESS_ENABLED and not deterministic_answer:
        score_or_none = _nli_contradiction_score(answer, docs)
        if score_or_none is not None:
            nli_contradiction = score_or_none
    nli_contradicted = nli_contradiction > settings.NLI_CONTRADICTION_MAX
    is_hallucinated = lexical_hallucinated or nli_contradicted

    return is_hallucinated, lexical_hallucinated, support_score, nli_contradiction, nli_contradicted


class GroundednessChecker:
    """Every important statement in the answer must be supported by retrieved docs."""

    def check(
        self,
        answer: str,
        docs: list[dict[str, Any]],
        query: str = "",
        deterministic_answer: bool = False,
    ) -> GroundednessResult:
        """See lexical_and_nli_verdict() for `deterministic_answer` semantics."""
        if not answer:
            return GroundednessResult(score=0.0, is_hallucinated=True)

        # REFUSALS ARE NOT HALLUCINATIONS (2026-08-13, per-modality quality
        # pass). A refusal asserts no factual claim, so there is nothing to
        # ground and nothing that can be ungrounded. Scoring one as maximally
        # ungrounded was BOTH a metric bug and a user-facing one:
        #   * metric — confirmed live on xlsx, where 2 of 3 sampled gold rows
        #     returned the "I couldn't generate a reliable answer." fallback
        #     and one returned "No relevant information was found in your
        #     knowledge base…"; each scored grounding=0.0 with NLI
        #     contradiction 0.99, dragging xlsx's grounding_success_rate to
        #     0.417 for reasons unrelated to answer quality.
        #   * user-facing — verified=False then made VerificationLoop append
        #     "…treat the figures above with caution" to a refusal containing
        #     no figures whatsoever.
        # score=100 (nothing unsupported was asserted) rather than excluding
        # the row from the metric: a modality that refuses often still shows
        # up plainly in answer_correctness/faithfulness, which score a refusal
        # near zero against a real gold answer — so counting refusals as
        # "grounded" here hides nothing.
        from app.reasoning.reasoning_engine import is_refusal_answer

        if is_refusal_answer(answer):
            return GroundednessResult(score=100.0, is_hallucinated=False)

        if not docs:
            # Empty-retrieval path is handled upstream (explicit "no information"
            # answer) — a groundedness check with no evidence to check against
            # is meaningless, not automatically passing.
            return GroundednessResult(
                score=0.0,
                is_hallucinated=True,
                unsupported_claims=["no retrieved evidence to verify against"],
            )

        from app.reasoning.reasoning_engine import _unsupported_numbers

        is_hallucinated, lexical_hallucinated, support_score, nli_contradiction, nli_contradicted = (
            lexical_and_nli_verdict(answer, docs, deterministic_answer=deterministic_answer)
        )
        bad_numbers = _unsupported_numbers(answer, docs, query=query)

        # Blend sentence-level support (0-1) with a numeric-fidelity penalty
        # and (when NLI flags contradiction) an NLI penalty of the same shape:
        # each is a harder failure than a paraphrase miss, since fabricated
        # claims are the domain's "sacred" correctness bar
        # (engineering-standards.md §4.1).
        numeric_penalty = min(len(bad_numbers) * 0.15, 0.6)
        nli_penalty = min(nli_contradiction, 0.6) if nli_contradicted else 0.0
        raw_score = max(0.0, support_score - numeric_penalty - nli_penalty)
        score = round(raw_score * 100.0, 2)

        unsupported_claims: list[str] = []
        if lexical_hallucinated:
            unsupported_claims.append(
                f"answer support score {support_score:.2f} below hallucination threshold"
            )
        if nli_contradicted:
            unsupported_claims.append(
                f"NLI contradiction score {nli_contradiction:.2f} exceeds threshold"
            )

        return GroundednessResult(
            score=score,
            is_hallucinated=is_hallucinated or bool(bad_numbers),
            unsupported_claims=unsupported_claims,
            unsupported_numbers=bad_numbers,
        )
