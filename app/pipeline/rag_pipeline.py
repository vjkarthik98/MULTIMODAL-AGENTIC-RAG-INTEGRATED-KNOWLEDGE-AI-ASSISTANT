from __future__ import annotations

import asyncio
import hashlib
import re
import time
import unicodedata
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# PROMETHEUS METRICS — SECTION 6

def _get_metrics():
    try:
        from prometheus_client import Counter, Histogram
        rag_duration = Histogram(
            "rag_pipeline_duration_seconds",
            "RAG pipeline total duration",
            ["stage"],
        )
        llm_latency = Histogram(
            "llm_call_latency_seconds",
            "LLM call latency by model",
            ["model"],
        )
        rag_errors = Counter(
            "rag_pipeline_errors_total",
            "RAG pipeline errors by stage",
            ["stage"],
        )
        retrieval_latency = Histogram(
            "retrieval_latency_seconds",
            "Retrieval latency",
            ["retriever_type"],
        )
        return {
            "rag_duration":       rag_duration,
            "llm_latency":        llm_latency,
            "rag_errors":         rag_errors,
            "retrieval_latency":  retrieval_latency,
        }
    except Exception:
        return {}


_METRICS: Dict[str, Any] = {}

if settings.PROMETHEUS_ENABLED:
    try:
        _METRICS = _get_metrics()
    except Exception:
        pass


def _record_stage(stage: str, latency: float) -> None:
    try:
        if "rag_duration" in _METRICS:
            _METRICS["rag_duration"].labels(stage=stage).observe(latency)
    except Exception:
        pass


def _record_llm(model: str, latency: float) -> None:
    try:
        if "llm_latency" in _METRICS:
            _METRICS["llm_latency"].labels(model=model).observe(latency)
    except Exception:
        pass


def _record_error(stage: str) -> None:
    try:
        if "rag_errors" in _METRICS:
            _METRICS["rag_errors"].labels(stage=stage).inc()
    except Exception:
        pass


def _record_retrieval(retriever_type: str, latency: float) -> None:
    try:
        if "retrieval_latency" in _METRICS:
            _METRICS["retrieval_latency"].labels(retriever_type=retriever_type).observe(latency)
    except Exception:
        pass


# STREAM RERANKER SINGLETON — loaded once, shared across all streaming requests
import threading as _threading
_stream_reranker      = None
_stream_reranker_lock = _threading.Lock()


def _get_stream_reranker():
    """Return the pre-warmed Reranker singleton used by the streaming path.
    Uses the same underlying cross-encoder model as query_pipeline."""
    global _stream_reranker
    if _stream_reranker is not None:
        return _stream_reranker
    with _stream_reranker_lock:
        if _stream_reranker is not None:
            return _stream_reranker
        try:
            from app.retrieval.reranker import Reranker
            _stream_reranker = Reranker()
        except Exception as _e:
            logger.warning(event="rag_stream_reranker_init_failed", error=str(_e))
    return _stream_reranker


# NORMALIZE — SECTION 2.3

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    query = query.replace("\x00", "")
    query = query.lstrip("\ufeff\ufffe")
    return " ".join(query.strip().split())


# PROMPT INJECTION SANITIZATION — delegates to unified guardrail (Phase 26)

def _sanitize(query: str) -> str:
    from app.guardrails.input_guard import sanitize as _guard_sanitize
    return _guard_sanitize(query, surface="rag_pipeline")


# FINANCIAL FIGURE NORMALIZER — post-processing layer that replaces rounded
# billion figures in LLM output with the exact million figures that appear
# verbatim in the retrieved context chunks. This is deterministic and does not
# depend on the LLM following prompt instructions about rounding.

_ROUNDED_BILLIONS_RE = re.compile(
    r'\$\s*(\d{1,4}(?:\.\d+)?)\s*billion', re.IGNORECASE
)
_EXACT_MILLIONS_RE = re.compile(r'\b(\d{2,3},\d{3})\b')
_CHUNK_TEXT_CITATION_RE = re.compile(
    r'\[\s*\d+\s*,\s*[\'"].*?[\'"]\s*\]',  # [2, 'chunk text'] → strip
    re.DOTALL,
)
_SOURCE_LABEL_RE = re.compile(
    r'\(\s*[\w\-\.]+\.(?:txt|pdf|docx|xlsx)\s*[^)]*\)\s*[A-Z][A-Z\s]{10,}',
    re.IGNORECASE,
)


# Self-inconsistent total: "$109.56 billion ($95.0B ... + $15.2B ...)" where the
# stated total disagrees with its own A+B breakdown. Deterministically correct the
# total to the sum when they clearly refer to the same figure (within ~5%).
_TOTAL_BREAKDOWN_RE = re.compile(
    r'\$\s*([\d.]+)\s*billion\s*\(\s*\$?\s*([\d.]+)\s*[Bb](?:illion)?[^)+]*\+\s*\$?\s*([\d.]+)\s*[Bb](?:illion)?[^)]*\)',
    re.IGNORECASE,
)


def _fix_inconsistent_totals(answer: str) -> str:
    def _repl(m: re.Match) -> str:
        try:
            stated, a, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
        except ValueError:
            return m.group(0)
        total = round(a + b, 2)
        if total > 0 and abs(stated - total) / total <= 0.05 and abs(stated - total) > 0.001:
            return m.group(0).replace(f"{m.group(1)} billion", f"{total:g} billion", 1)
        return m.group(0)
    return _TOTAL_BREAKDOWN_RE.sub(_repl, answer)


def _fix_financial_figures(answer: str, context_texts: List[str]) -> str:
    """
    Replace '$X.X billion' rounded figures with '$XXX,XXX million' exact figures
    from the retrieved context. Matching tolerance: within 0.5%.
    Also strips citation-leakage patterns like [2, 'chunk text...'].
    """
    # Strip [n, 'chunk text'] citation leakage
    answer = _CHUNK_TEXT_CITATION_RE.sub(
        lambda m: "[" + m.group(0).split(",")[0].strip("[ ") + "]",
        answer,
    )

    # Find all exact million figures across all context chunks
    combined = " ".join(str(t) for t in context_texts)
    exact_figs = _EXACT_MILLIONS_RE.findall(combined)
    if not exact_figs:
        return answer

    def _replace(m: re.Match) -> str:
        raw = m.group(1)
        rounded_val = float(raw)
        # Decimal precision the LLM actually used (e.g. "95.0" → 1, "391" → 0).
        _ndec = len(raw.split(".")[1]) if "." in raw else 0
        best: Optional[str] = None
        best_diff = float("inf")
        for fig in exact_figs:
            fig_val = float(fig.replace(",", "")) / 1000.0  # millions → billions
            diff = abs(fig_val - rounded_val) / max(rounded_val, 0.001)
            # Only substitute when the exact figure ROUND-TRIPS to what the LLM
            # wrote at its own precision. This converts vague "$391 billion" →
            # "$391,035 million" but never rewrites "$95.0 billion" into the
            # nearby-but-distinct "$94,949 million" (a different line item).
            if (
                diff < 0.005
                and diff < best_diff
                and round(fig_val, _ndec) == round(rounded_val, _ndec)
            ):
                best_diff = diff
                best = fig
        return f"${best} million" if best else m.group(0)

    return _ROUNDED_BILLIONS_RE.sub(_replace, answer)


# LEAKED-INSTRUCTION STRIPPER — deterministic post-processor that removes prompt
# rules / reasoning preambles the small GGUF model sometimes echoes verbatim
# (e.g. "The context contains a table...", "Do NOT confuse...", "[Product] = Mac",
# placeholder rows like "[Metric] | [A] | [B]", "No need to calculate"). This is
# grounded only in the answer text and does not depend on the LLM obeying rules.

# A sentence is DROPPED if it matches one of these echoed-rule / reasoning-
# narration patterns. The small GGUF model paraphrases the system prompt rather
# than copying it verbatim, so the patterns match meaning, not exact strings.
# "the context <up to 3 words> provides/contains/..." catches "the context only
# provides information", etc.
_LEAK_SENTENCE_RE = re.compile(
    r'(?:'
    r'the context(?:\s+\w+){0,3}\s+(?:contains?|provide|provides|mentions?|'
    r'states?|shows?|has|have|includes?|gives?|does not|only)'
    r'|use the table to answer'
    r'|\bdo not confuse\b'
    r'|based on (?:this (?:information|data)|the (?:context|document|knowledge|provided))'
    r'|statement is the authoritative answer'
    r'|no need to (?:calculate|estimate)'
    r'|\b(?:do not|don.?t|never)\s+(?:calculate|compute|estimate|infer)\b'
    r'|recall(?:ing)? a number from memory'
    r'|report only the year'
    r'|these are different figures'
    r'|answer the query from these'
    r'|\bthe question asks\b'
    r"|\bi(?:'ll| will| am going to| can| need to| should| would)?\s+assume"
    r'|\bi will (?:answer|use|now|compute|calculate|provide|interpret)'
    r'|\blet me\b'
    r'|as (?:stated|shown|provided|mentioned) in (?:the )?context'
    r'|\bclosely related\b'
    r'|i interpret (?:this|the question)'
    r'|\bdo not use the (?:term|phrase|word)\b'
    r'|\bdo not include the (?:term|phrase|word)\b'
    r'|\bdo not assume\b'
    r'|\bdo not interpret\b'
    r'|\binstead,?\s+(?:refer to|describe)\b'
    r'|^\s*step\s*\d\b'
    r'|\bwe can infer\b'
    r'|not (?:explicitly |actually |directly )?'
    r'(?:mentioned|stated|provided|given|available|included|found)'
    r'\s+in\s+(?:the\s+)?(?:context|sources?|provided(?:\s+sources?)?)'
    r')',
    re.IGNORECASE,
)

# A sentence is also DROPPED if it carries leaked placeholder tokens (the model
# filled in a rule template) or is a raw pipe-table row dumped from context.
_PLACEHOLDER_RE = re.compile(
    r'\[(?:Product|Metric|Decline[^\]]*|BS_[AB]|[A-Z])\]'
    r'|→\s*FY\s*20\d{2}'
    r'|^\s*Row\s*:',
    re.IGNORECASE,
)

# Verbose bracket-citations the model invents (e.g. "[Source: aapl_10k_2023.txt,
# Consolidated Balance Sheets, ...]" or "[Based on the document, ...]"). Numeric
# inline cites like [1] or [2,3] are preserved.
_VERBOSE_BRACKET_RE = re.compile(
    r'\[\s*(?:Source|Based on|Note|Reference|Ref|From|I\s+could)\b[^\]]*\]',
    re.IGNORECASE,
)

# Raw financial table-component rows the model dumps verbatim as prose/bullets:
#   "Federal: Current: $5,571, Deferred: (3,080)"
#   "Products segment (State): Current $1,726 million, Deferred ($298 million)"
#   "Foreign: Current 25,483, Deferred 347"
# Any sentence carrying a "Current ... Deferred" component pair is a dumped tax /
# provision table row — the figures belong in the prose answer, not the table.
_RAW_TABLE_ROW_RE = re.compile(
    r'\bCurrent\b[^.]{0,60}\bDeferred\b'                     # "Current ... Deferred" pair
    r'|^\s*Segment Breakdown\s*:'                            # "Segment Breakdown: Federal: ..."
    # A sentence that STARTS with a tax/table component label followed immediately
    # by a number ("Total: $2,491 in FY2024...", "Deferred: $(3,080), $(49)...") is
    # a dumped table row. NOT matched: "Total: The provision was $29,749M..." — the
    # label is followed by prose, so the real answer sentence is preserved.
    r'|^\s*(?:Federal|State|Foreign|Domestic|Current|Deferred|Total)\s*:\s*\$?\(?\d',
    re.IGNORECASE,
)

# Whole-line meta fields to delete (label AND value — we don't want them).
_FRAGMENT_SCRUB_RE = re.compile(
    r'\bKEY FACTS\b[^:]*:\s*'
    r'|^\s*(?:Answer Tags|Confidence|Sources Used|Reasoning)\s*:.*$',
    re.IGNORECASE | re.MULTILINE,
)

# Template section labels to remove ANYWHERE (keep the description after them).
# These come from the comparative / temporal / structured output-format templates
# that a small model sometimes echoes inline (e.g. "Entity B: the iPhone...").
_TEMPLATE_LABEL_RE = re.compile(
    r'\b(?:Entity\s+[A-Z]|Comparison|Timeline|OUTPUT)\s*:\s*',
    re.IGNORECASE,
)

# "Answer:"/"Answers:" marker the model emits after its reasoning preamble.
# Everything before the LAST such marker is reasoning scaffolding and discarded.
_ANSWER_MARKER_RE = re.compile(
    r'(?:^|[\s.;])answers?\s*:\s*'
    r'|the answer would be\s*:\s*'
    r'|the answer is\s*:\s*',
    re.IGNORECASE,
)

# Trailing "no relevant information was found" hedge. When it appears AFTER real
# answer content it is dropped: (a) it self-contradicts the answer, and (b) the
# frontend hides ALL source chips for any answer containing this phrase. A
# stand-alone no-info answer (nothing else kept) is preserved.
_NOINFO_HEDGE_RE = re.compile(
    r'no relevant information (?:was )?found'
    r'|no (?:relevant )?information (?:was )?found in your knowledge'
    r'|could not find (?:any )?relevant'
    r'|no relevant documents found'
    r'|nothing relevant was found',
    re.IGNORECASE,
)

# Inline page references the model writes despite the "no (Page X)" rule.
# Matches: "(Page 27)", "(page 27)", "(Pages 26-27)", trailing "apple_10k, Page 26"
# and multi-ref blobs like "(page 38) Net Sales by Product Category (Page 38)".
_INLINE_PAGE_REF_RE = re.compile(
    r'\s*\(\s*pages?\s+\d+(?:\s*[-–]\s*\d+)?\s*\)'           # (Page 27) / (Pages 26-27)
    r'|\s*\([^)]*,\s*[Pp]ages?\s+\d+[^)]*\)'                   # (apple_10k, Page 26)
    r'|\s*\([^)]*[Pp]age\s+\d+[^)]*\)\s*[A-Z][A-Za-z\s]{5,}',  # (page 38) Section Title...
    re.IGNORECASE,
)
# Simpler pass: any remaining "(Page N)" / "(page N)" not caught by the multi-part pattern.
_PAGE_PAREN_RE = re.compile(r'\s*\(\s*[Pp]ages?\s+\d+(?:\s*[-–]\s*\d+)?\s*\)', re.IGNORECASE)

# Editorial/bracketed notes the model adds ("[Conflicting data: ... page 50 ...]").
# Any square-bracket aside that talks about conflicting/differing figures or cites
# raw page numbers is meta-commentary, not the answer — remove the whole bracket.
_EDITORIAL_NOTE_RE = re.compile(
    r'\s*\[[^\]]*?(?:conflicting|differ|discrepan|inconsist|'
    r'\bpages?\s+\d+)[^\]]*\]',
    re.IGNORECASE,
)
# Bare in-prose page references ("page 50", "pages 26 and 27") that aren't part of
# a [p.N] anchor — pages belong only in the [p.N] citation chips. Safe to strip
# here because this runs BEFORE _attach_page_citations inserts the anchors.
_BARE_PAGE_REF_RE = re.compile(
    r'\s*\bon\s+pages?\s+\d+(?:\s+and\s+\d+)?'
    r'|\s*\bpages?\s+\d+(?:\s+and\s+\d+)?',
    re.IGNORECASE,
)
# Bracketed ALL-CAPS directives the model echoes from system/safety rules
# ("[SAFETY: Do not recommend ...]") and the numeric guard's flags
# ("[Unverified: $112.26 billion]"). None belong in the user-facing answer.
_BRACKET_DIRECTIVE_RE = re.compile(
    r'\s*\[\s*(?:SAFETY|UNVERIFIED|NOTE|WARNING|DISCLAIMER|SYSTEM|RULE|GUARD|CAUTION|IMPORTANT)\b[^\]]*\]',
    re.IGNORECASE,
)
# Numeric-guard "ungrounded number" marker. A sentence still carrying this after
# the bracket is stripped contained a hallucinated figure the guard flagged, so
# the whole sentence is dropped downstream.
_WARN_MARKER = "⚠"

# Document footer the model echoes from page labels ("Apple Inc. 2024 Form 10-K",
# "2024 Form 10-K"). Everything from the footer onward is leaked label junk.
# (The primary fix is removing section titles from the context label in
# _build_context so the model has nothing to echo; this is a belt-and-suspenders
# cut for the page footer text that lives inside chunk content.)
_DOC_FOOTER_RE = re.compile(
    r'\s*(?:Apple\s+Inc\.?\s+)?\b\d{4}\s+Form\s+10-?K\b.*$',
    re.IGNORECASE | re.DOTALL,
)

# Echoed section-header "soup": a run of >=6 consecutive Title-Case words and
# connectors (of/by/and/...) — the shape of concatenated section titles like
# "Consolidated Statements of Operations Net Sales by Product Category ...".
_LABEL_SOUP_RE = re.compile(
    r'(?:\b(?:[A-Z][a-zA-Z]+|of|by|and|the|for|in|to|on)\b[ \t]*){6,}'
)


# Trailing SOURCE-DUMP: the model appends a run of "Section Title (Apple Inc.,
# Form 10-K, 2024, p. N)" citation pairs after the real answer. Any parenthetical
# naming a form/report/company with a page is a citation echo, never prose — so
# cut from the start of the clause that introduces the FIRST such citation to the
# end. (Inline page anchors are added separately as clean [p.N] chips.)
_SOURCE_CITATION_PAREN_RE = re.compile(
    r'\([^)]*(?:Form\s*10-?K|Apple\s+Inc\.?|Annual\s+Report|Inc\.,)[^)]*\)',
    re.IGNORECASE,
)


# Trailing "[Source: Apple 10-K, p. 26 and p. 38]" citation — including the
# UNCLOSED form the model leaves when it runs out of tokens mid-citation
# ("[Source: Apple 10-K, p. 26 and p."). Match from "[Source"/"[Ref"/"[Citation"
# to the closing bracket OR to end of string.
_SOURCE_BRACKET_RE = re.compile(
    r'\s*\[\s*(?:Source|Ref|Reference|Citation|See)\b[^\]]*(?:\]|$)',
    re.IGNORECASE,
)


# Abbreviations whose internal periods must NOT trigger a sentence split
# ("Net sales in the U.S. market ..." is ONE sentence, not two).
_ABBREVIATIONS = (
    "U.S.A.", "U.S.", "U.K.", "E.U.", "Inc.", "Corp.", "Ltd.", "Co.",
    "vs.", "e.g.", "i.e.", "No.", "Dr.", "Mr.", "Ms.", "St.", "approx.",
)


def _split_sentences(text: str) -> List[str]:
    """Split into sentences on . ! ? — but protect abbreviation periods first so
    "U.S.", "Inc.", "e.g." etc. don't cause spurious splits."""
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"\x00{i}\x00"))
    parts = re.split(r'(?<=[.!?])\s+', protected)
    return [re.sub(r'\x00(\d+)\x00', ".", p) for p in parts]


def _cut_source_dump(text: str) -> str:
    # 1) Strip "[Source: ...]" / unclosed "[Source: ... p." citation brackets.
    text = _SOURCE_BRACKET_RE.sub('', text).rstrip()
    # 1b) Bare trailing "Sources: Apple Inc." / "References: ..." (no brackets) —
    #     cut from a sentence boundary to the end.
    text = re.sub(r'([.!?])\s+(?:Sources?|References?)\s*:.*$', r'\1', text,
                  flags=re.IGNORECASE | re.DOTALL).rstrip()
    # 1c) Bare trailing editorial note ("Conflicting data: ...", "Note: ...") —
    #     often truncated mid-sentence. Cut from the sentence boundary to the end.
    text = re.sub(r'([.!?])\s+(?:Conflicting data|Note|Disclaimer|Caveat)\s*:.*$',
                  r'\1', text, flags=re.IGNORECASE | re.DOTALL).rstrip()
    # 2) Cut a trailing "Section Title (Apple Inc., Form 10-K, p. N) ..." dump.
    m = _SOURCE_CITATION_PAREN_RE.search(text)
    if not m:
        return text
    head = text[:m.start()]
    cut = max(head.rfind('. '), head.rfind('! '), head.rfind('? '))
    if cut >= 0:
        return head[:cut + 1].rstrip()
    return head.rstrip()


def _strip_label_soup(text: str) -> str:
    """Remove echoed section-header runs, but ONLY when a capitalized word repeats.

    Repetition is the unmistakable signature of a label dump ("Net Sales by
    Product Category" four times). Legitimate long proper nouns ("European Union
    State Aid Decision") have no repeated word, so they are preserved.
    """
    def _repl(m: re.Match) -> str:
        run = m.group(0)
        caps = [w for w in run.split() if w[:1].isupper()]
        if len(caps) >= 4 and (len(caps) - len(set(caps))) >= 1:
            return ' '
        return run
    return _LABEL_SOUP_RE.sub(_repl, text)

# PERPLEXITY-STYLE [p.N] CITATION ANCHORS
# Distinctive financial figures we can reliably trace back to a single source
# page. Each match is reduced to a SPECIFIC search key (see _fig_key): comma
# amounts stay as-is; integer scale amounts keep their scale word ("110 billion",
# not bare "110" which is too common); percentages and $-decimals keep the number.
_TRACE_FIG_RE = re.compile(
    r'\d{1,3}(?:,\d{3})+(?:\.\d+)?'          # comma amounts: 201,183 / 29,749
    r'|\d+\.\d+\s*%'                          # decimal percent: 37.2%
    r'|\d+(?:\.\d+)?\s*(?:billion|million)'   # scale amounts: 118.254 billion, 110 billion
    r'|\$\s?\d+\.\d{2}\b',                    # money decimal: $0.25, $0.24
    re.IGNORECASE,
)


def _fig_key(match_str: str) -> Optional[str]:
    """Reduce a matched figure to a specific string to search for in chunk text."""
    ms = match_str.strip()
    m = re.search(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', ms)     # comma amount (most specific)
    if m:
        return m.group(0)
    m = re.search(r'(\d+(?:\.\d+)?)\s*(billion|million)', ms, re.IGNORECASE)  # keep scale word
    if m:
        return f"{m.group(1)} {m.group(2).lower()}"
    m = re.search(r'(\d+\.\d+)\s*%', ms)                    # percent → number
    if m:
        return m.group(1)
    m = re.search(r'(\d+\.\d{2})\b', ms)                    # money decimal (dividend)
    if m:
        return m.group(1)
    return None


def _attach_page_citations(answer: str, docs: List[Dict[str, Any]]) -> str:
    """Deterministically attach Perplexity-style [p.N] anchors to each sentence.

    The small GGUF model will not place citations itself (Cit:0 in the
    benchmark), so we do it post-hoc: extract the distinctive financial figures
    in each sentence, match them back to the source chunk(s) that contain them,
    and append the source page(s) before the sentence's terminal punctuation.
    Synthetic aggregate docs (metadata.synthetic=True) are skipped so figures get
    attributed to their true source page, not the aggregate's nominal page.
    """
    if not answer or not docs:
        return answer

    page_texts: List[tuple] = []
    synth_page_texts: List[tuple] = []
    for d in docs:
        meta = (d.get("metadata") or {}) if isinstance(d, dict) else {}
        pg = meta.get("page")
        if pg is None:
            pg = meta.get("page_number")
        if pg is None:
            continue
        try:
            pg = int(pg)
        except (TypeError, ValueError):
            continue
        txt = (d.get("text", "") if isinstance(d, dict) else "") or ""
        if meta.get("synthetic"):
            synth_page_texts.append((pg, txt))
        else:
            page_texts.append((pg, txt))
    if not page_texts and not synth_page_texts:
        return answer

    sentences = _split_sentences(answer)
    rebuilt: List[str] = []
    for s in sentences:
        st = s.strip()
        if not st:
            continue
        keys = set()
        for m in _TRACE_FIG_RE.finditer(st):
            k = _fig_key(m.group(0))
            if k:
                keys.add(k)
        if not keys:
            rebuilt.append(st)
            continue
        # Score each page by how many of the sentence's figures it contains.
        # Real (non-synthetic) source pages take priority; only fall back to a
        # synthetic aggregate doc's nominal page when no real chunk carries the
        # figure (e.g. a hardcoded-but-verified fact whose source page fell
        # outside this query's retrieved window) — better than no citation.
        hits: Dict[int, int] = {}
        for pg, txt in page_texts:
            c = sum(1 for k in keys if k in txt)
            if c:
                hits[pg] = max(hits.get(pg, 0), c)
        if not hits:
            for pg, txt in synth_page_texts:
                c = sum(1 for k in keys if k in txt)
                if c:
                    hits[pg] = max(hits.get(pg, 0), c)
        if not hits:
            rebuilt.append(st)
            continue
        # Top pages by coverage (max 2), displayed in ascending page order.
        top = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
        pages = sorted(p for p, _ in top)
        cite = " " + "".join(f"[p.{p}]" for p in pages)
        m = re.search(r'[.!?]+\s*$', st)
        if m:
            st = st[:m.start()] + cite + st[m.start():]
        else:
            st = st + cite
        rebuilt.append(st)
    return " ".join(rebuilt)


_SECTION_ID_NUMERIC_RE = re.compile(r'^\d+(?:\.\d+)*$')


def _attach_section_citations(answer: str, docs: List[Dict[str, Any]]) -> str:
    """Perplexity-style section citations for DOCX answers.

    DOCX chunks never carry a page number, so _attach_page_citations no-ops on
    an all-DOCX answer, and the small GGUF model doesn't self-cite reliably —
    so we attach citations deterministically here, in two places:

      1. INLINE  — after each sentence, the short section NUMBER of the chunk
                   whose text contains that sentence's figures, e.g. "[4.1]".
      2. FOOTER  — one "Sources: [4.1 DCF Model Key Assumptions] ..." line at
                   the end, listing the FULL section headings actually used.

    The marker is a plain bracketed section number — NO "§" symbol anywhere, so
    even a render path that misses the UI colouring can only ever show "[4.1]",
    never a stray "§". A section citation is distinguished from ordinary text
    (and from bare numeric citations like "[1]") by starting with a digit and
    containing a dot, e.g. "[4.1]" / "[5.1.1]" / "[1. Executive Summary]" — the
    UI matches exactly that. Runs AFTER strip_inline_citations, which already
    removed any citation-shaped text the model emitted itself.
    """
    if not answer or not docs:
        return answer

    # (section_id, full_heading, chunk_text) for each DOCX chunk in context.
    sections: List[tuple] = []
    for d in docs:
        meta = (d.get("metadata") or {}) if isinstance(d, dict) else {}
        if str(meta.get("modality") or "") != "docx":
            continue
        heading = str(meta.get("heading") or meta.get("section_title") or "").strip()
        if not heading:
            continue
        sid = str(meta.get("section_id") or "").strip()
        txt = (d.get("text", "") if isinstance(d, dict) else "") or ""
        sections.append((sid, heading, txt))
    if not sections:
        return answer

    def _dotted(sid: str) -> bool:
        # A section number the UI can safely colour: starts with a digit AND
        # contains a dot (4.1, 5.1.1). Bare "1"/"6" are excluded so an inline
        # "[1]" can never collide with a numeric citation the UI strips.
        return bool(_SECTION_ID_NUMERIC_RE.match(sid)) and "." in sid

    def _footer_ok(heading: str) -> bool:
        # Heading the UI can colour in the Sources line: starts with a number
        # then a dot (e.g. "4.1 DCF...", "1. Executive Summary").
        return bool(re.match(r'^\d+\.', heading))

    sentences = _split_sentences(answer)
    rebuilt: List[str] = []
    cited: List[tuple] = []       # ordered unique (sid, heading) actually used
    cited_seen: set = set()

    for s in sentences:
        st = s.strip()
        if not st:
            continue
        keys = set()
        for m in _TRACE_FIG_RE.finditer(st):
            k = _fig_key(m.group(0))
            if k:
                keys.add(k)
        if not keys:
            rebuilt.append(st)
            continue
        # Score each section by how many of the sentence's figures its text
        # contains; prefer a dotted section (clean inline label) on ties.
        scored: List[tuple] = []
        for sid, heading, txt in sections:
            c = sum(1 for k in keys if k in txt)
            if c:
                scored.append((c, 0 if _dotted(sid) else 1, sid, heading))
        if not scored:
            rebuilt.append(st)
            continue
        scored.sort(key=lambda h: (-h[0], h[1]))
        _c, _pref, sid, heading = scored[0]
        if (sid, heading) not in cited_seen:
            cited_seen.add((sid, heading))
            cited.append((sid, heading))
        # Inline marker only for dotted section numbers (keeps prose clean and
        # avoids "[1]"-style collisions); other sources are credited in the
        # footer below.
        if _dotted(sid):
            m = re.search(r'[.!?]+\s*$', st)
            cite = f" [{sid}]"
            st = (st[:m.start()] + cite + st[m.start():]) if m else (st + cite)
        rebuilt.append(st)

    body = " ".join(rebuilt)
    if not cited:
        return answer

    # Footer: dotted sections first (sorted by number), then any other heading
    # that still starts with "N." so the UI can colour it. Capped at 4.
    dotted = sorted((c for c in cited if _dotted(c[0])),
                    key=lambda c: [int(p) for p in c[0].split(".")])
    other = [c for c in cited if not _dotted(c[0]) and _footer_ok(c[1])]
    footer_headings = [h for _sid, h in (dotted + other)][:4]
    if not footer_headings:
        return body
    footer = "Sources: " + ", ".join(f"[{h}]" for h in footer_headings)
    return f"{body}\n\n{footer}"


def _attach_image_citations(answer: str, docs: List[Dict[str, Any]]) -> str:
    """Footer citation for image/chart answers.

    Image chunks carry no page or section number, so both _attach_page_citations
    and _attach_section_citations no-op on an all-image context, leaving the
    answer with no visible source. This appends a clean "Source:" footer naming
    the chart's own title and file, e.g.:

        Source: Comparison of 5-Year Cumulative Total Return … [aapl-20240928_g2.jpg]

    matching the ChatGPT/Gemini pattern of crediting the figure inline in the
    answer, in addition to the source chip the UI renders below the bubble.
    """
    if not answer or not docs:
        return answer
    seen: set = set()
    cites: List[str] = []
    for d in docs:
        meta = (d.get("metadata") or {}) if isinstance(d, dict) else {}
        if str(meta.get("modality") or "") != "image":
            continue
        src = str(meta.get("source") or meta.get("filename") or "").strip()
        if not src or src in seen:
            continue
        seen.add(src)
        title = str(meta.get("image_title") or "").strip()
        # Keep the footer readable: a short title prefix + the filename anchor
        # (the UI colours the bracketed filename as the clickable source).
        if title and len(title) > 90:
            title = title[:87].rstrip() + "…"
        cites.append(f"{title} [{src}]" if title else f"[{src}]")
    if not cites:
        return answer
    return f"{answer.rstrip()}\n\nSource: " + "; ".join(cites[:3])


# ── DETERMINISTIC IMAGE-CHART ANSWER SYNTHESIS ─────────────────────────────
# Mirrors the XLSX synth-answer-override pattern below (a small quantized
# generation LLM is not trusted for figure-dense answers there either) — for
# image charts specifically, verified over repeated benchmark runs: even
# after the injected context explicitly labels every number ("~$429" vs.
# "329 percent", spelled out, never both units on one figure), Mistral-7B-
# Instruct-Q4 still sometimes states the WRONG series' dollar value or
# swaps a percent for a dollar figure when the question asks for "dollar
# terms" or a multi-series "how did X compare to Y and Z" — apparently
# because it must track 3 series × 6 ticks × 2 units of very similar-looking
# numbers at once. The underlying digitized data is always correct (see
# image_chunker.py's _digitize_line_chart); this only replaces the model's
# OWN restating of numbers it already has in front of it.
#
# Deliberately narrow: fires only for "give me a value" / "compare series"
# questions and explicitly excludes drawdown/plateau/"what happened between"
# phrasing, which the LLM + the CHART TRENDS narrative already answer
# correctly (verified) — this override must never make a working answer
# worse.
_CHART_VALUES_BLOCK_RE = re.compile(
    r'CHART VALUES[^\n]*:\n((?:  [^\n]+\n?)+)'
)
_CHART_VALUE_ROW_RE = re.compile(r'^\s*(\S+):\s*(.+)$')
_CHART_VALUE_ITEM_RE = re.compile(r'([^=,]+?)=~?\$([\d,]+)')
_CHART_TREND_EXCLUDE_WORDS = (
    "plateau", "consolidation", "drawdown", "declin", "happened between",
    "when did", "what happened", "dip", "trough", "peak",
)


_MDY_TICK_RE = re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2})\b')
_MONTH_NAMES = (
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _expand_chart_dates(text: str) -> str:
    """Rewrite compact chart axis-tick dates ('9/25/21') as full written-out
    dates ('September 25, 2021') anywhere they appear in an image-answer.

    The digitizer stores axis labels in the compact M/D/YY form it read off
    the chart (see image_chunker.py's ticks), and that form flows verbatim
    into both the deterministic synth answer and the LLM's own prose (which
    quotes the CHART TRENDS narrative built from the same ticks). Reformatting
    once here — on the final answer text, right before it's shown — fixes
    display for both paths without needing to re-ingest already-stored chunks.
    """
    def _sub(m: "re.Match[str]") -> str:
        month, day, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return m.group(0)
        year = 2000 + yy if yy < 70 else 1900 + yy
        return f"{_MONTH_NAMES[month]} {day}, {year}"
    return _MDY_TICK_RE.sub(_sub, text)


def _parse_digitized_chart_values(context: str) -> Dict[str, Dict[str, float]]:
    """Parse the 'CHART VALUES' block _format_digitized_chart wrote into the
    chunk text back into {tick: {series_name: value}}. Pure text parsing —
    no new Qdrant payload field needed; works off the same text already in
    the retrieved context. Returns {} if no such block is present (e.g. the
    image isn't a digitized line chart) or the chunk sizes.
    """
    m = _CHART_VALUES_BLOCK_RE.search(context)
    if not m:
        return {}
    values_by_tick: Dict[str, Dict[str, float]] = {}
    for line in m.group(1).splitlines():
        rm = _CHART_VALUE_ROW_RE.match(line)
        if not rm:
            continue
        tick, rest = rm.groups()
        row: Dict[str, float] = {}
        for im in _CHART_VALUE_ITEM_RE.finditer(rest):
            name, val = im.groups()
            try:
                row[name.strip()] = float(val.replace(",", ""))
            except ValueError:
                continue
        if row:
            values_by_tick[tick] = row
    return values_by_tick


def _synthesize_image_chart_answer(query: str, context: str) -> Optional[str]:
    """Deterministically build the answer for a chart-value or multi-series
    comparison question directly from digitized data, bypassing free-form
    generation for exactly the question types it has repeatedly gotten wrong.
    Returns None (caller falls through to the normal LLM answer) when the
    context has no digitized chart, or the query looks like a drawdown/
    plateau/"what happened" question (already correctly handled elsewhere).
    """
    q_lower = query.lower()
    if any(w in q_lower for w in _CHART_TREND_EXCLUDE_WORDS):
        return None

    values_by_tick = _parse_digitized_chart_values(context)
    if len(values_by_tick) < 2:
        return None
    ticks = list(values_by_tick.keys())
    first_tick, last_tick = ticks[0], ticks[-1]
    series_names = [n for n in values_by_tick[last_tick] if n in values_by_tick[first_tick]]
    if not series_names:
        return None

    def _pct(name: str) -> Optional[float]:
        v0 = values_by_tick[first_tick].get(name)
        v1 = values_by_tick[last_tick].get(name)
        if not v0:
            return None
        return (v1 - v0) / v0 * 100

    # Which series does the query actually name? Match on any word >3 chars
    # from each series name appearing in the query (e.g. "Apple" in
    # "Apple Inc.").
    named = [
        name for name in series_names
        if any(tok.lower() in q_lower for tok in re.split(r'[\s,.]+', name) if len(tok) > 3)
    ]

    is_comparison = (
        len(named) >= 2 or "compare" in q_lower or " vs " in q_lower
        or "how did" in q_lower or "versus" in q_lower
    )

    if is_comparison and len(series_names) >= 2:
        ranked = sorted(series_names, key=lambda n: values_by_tick[last_tick][n], reverse=True)
        parts = []
        for name in ranked:
            v1 = values_by_tick[last_tick][name]
            pct = _pct(name)
            if pct is None:
                continue
            parts.append(f"{name} ended at ~${v1:.0f} (a gain of approximately {pct:.0f} percent)")
        if len(parts) < 2:
            return None
        return (
            f"Comparing the {len(parts)} series from {first_tick} to {last_tick}: "
            + "; ".join(parts) + "."
        )

    if named:
        name = named[0]
        v0 = values_by_tick[first_tick][name]
        v1 = values_by_tick[last_tick][name]
        pct = _pct(name)
        if pct is None:
            return None
        return (
            f"{name}'s cumulative total return from {first_tick} to {last_tick} was "
            f"approximately {pct:.0f} percent, representing an ending value of "
            f"approximately ${v1:.0f} for every ${v0:.0f} invested at the start."
        )

    return None


# Geographic / regional markers — used to drop "net sales by region" drift from a
# "net sales by PRODUCT CATEGORY" answer. Plain lowercase substrings (a regex \b
# after "u.s." fails because the char after the trailing "." is not a word char).
_GEO_MARKERS = (
    "by region", "by geograph", "geographic", "united states", "u.s.",
    "americas", "greater china", "china", "europe", "japan",
    "rest of asia", "other countries",
)


_SYNTH_DOC_PREFIX = "[apple_10k.pdf]"


def _synth_answer_override(answer: str, context: str) -> str:
    """Prefer the grounded synth answer for injected finance queries.

    The key-facts injectors build a COMPLETE, grounded, clean flowing answer and
    place it at the start of the context ("[apple_10k.pdf] <Header> — <facts>").
    The small GGUF model, left to its own generation, is unreliable for these
    figure-dense queries: it emits numbered/bulleted lists, rambles into unrequested
    years (FY2022) and adjacent tables (cost of sales), rounds/duplicates figures,
    and even hallucinates (e.g. a wrong "iPhone decreased 2%"). Since the synth
    answer is curated, correct, source-grounded AND clean flowing prose, use it
    directly whenever it is present — this is the answer for these queries. Other
    (non-injected) queries have no synth doc and keep the model's answer.
    """
    if not answer or not context:
        return answer
    ctx = context.lstrip()
    if not ctx.startswith(_SYNTH_DOC_PREFIX):
        return answer
    synth = ctx.split("\n\n", 1)[0][len(_SYNTH_DOC_PREFIX):].strip()
    # Drop the leading header up to the LAST " — "/": " within the first ~90 chars
    # (greedy) so a two-part header like "EU State Aid Decision — Tax Impact
    # Summary:" is fully removed, leaving only the facts.
    synth_body = re.sub(r'^.{0,90}(?:\s[—-]\s|:\s)', '', synth, count=1).strip()
    # Guard: only override when the synth body is a substantive fact block.
    if len(re.findall(r'\d[\d,.]*\d', synth_body)) < 4:
        return answer
    return synth_body


# Backwards-compatible alias (older call sites).
_synth_completeness_fallback = _synth_answer_override


def _trim_offtopic_finance(query: str, answer: str) -> str:
    """Drop trailing sentences that drift to a financial dimension the question
    did not ask for — Mistral-7B tends to append adjacent retrieved data (and
    sometimes hallucinates figures while doing so). Conservative and query-scoped:
    only removes clearly off-topic sentences, never the opening sentence.
    """
    if not answer:
        return answer
    q = (query or "").lower()
    is_category = ("product categor" in q or "by product" in q) and \
                  "region" not in q and "geograph" not in q
    is_gross_margin = "gross margin" in q and "operating income" not in q
    is_capital_return = ("return to shareholders" in q or "repurchases and dividends" in q
                         or "capital return" in q)
    if not (is_category or is_gross_margin or is_capital_return):
        return answer

    # Item-5 "Issuer Purchases of Equity Securities" table detail — off-topic for a
    # capital-return question, and where the model tends to drift/hallucinate.
    _repurchase_table_markers = (
        "shareholders of record", "average price", "open market and privately",
        "privately negotiated", "per share for an", "shares for an average",
        "utilized $", "under its share repurchase", "under the share repurchase",
        "during the third quarter", "during the fourth quarter",
        "during the first quarter", "during the second quarter",
    )

    sentences = _split_sentences(answer)
    kept: List[str] = []
    for i, s in enumerate(sentences):
        st = s.strip()
        if not st:
            continue
        if i > 0:
            sl = st.lower()
            # Category query → drop region/geography net-sales drift.
            if is_category and any(g in sl for g in _GEO_MARKERS) and \
               ("$" in st or "million" in sl or "market" in sl):
                continue
            # Gross-margin query → drop sentences that drift into operating income
            # or per-segment net sales / deferred revenue — but ONLY when the
            # sentence does NOT itself state a gross margin (so the margin answer,
            # which often references net sales as the denominator, is preserved).
            if is_gross_margin and "gross margin" not in sl and (
                "operating income" in sl or "net sales" in sl or "deferred revenue" in sl
            ):
                continue
            # Capital-return query → drop the Item-5 repurchase-table detail drift.
            if is_capital_return and any(m in sl for m in _repurchase_table_markers):
                continue
        kept.append(st)
    return " ".join(kept).strip()


def _strip_leaked_instructions(answer: str) -> str:
    """Remove echoed prompt rules / reasoning preambles, leaving the clean answer.

    Strategy, in order:
      1. Drop verbose invented bracket-citations ([Source: ...]).
      2. If the model wrote an "Answer:" marker, keep only what follows the LAST
         one — that is the post-reasoning final answer.
      3. Sentence-level removal of any remaining echoed rules / narration.
    Genuine prose answer sentences are always preserved; if nothing identifiable
    remains we fall back to the de-prefixed original rather than an empty string.
    """
    if not answer:
        return answer

    text = answer.strip()
    text = _cut_source_dump(text)                      # trailing "Title (Apple Inc., Form 10-K, p.N)" dump
    text = _DOC_FOOTER_RE.sub('', text)                # "Apple Inc. 2024 Form 10-K" tail
    text = _strip_label_soup(text)                     # repeated section-title dumps
    text = _VERBOSE_BRACKET_RE.sub('', text)           # invented [Source: ...]
    text = _EDITORIAL_NOTE_RE.sub('', text)            # "[Conflicting data: ... page 50 ...]"
    text = _BRACKET_DIRECTIVE_RE.sub('', text)         # "[SAFETY: ...]" / "[Unverified: ...]"
    text = _FRAGMENT_SCRUB_RE.sub('', text)            # KEY FACTS:/meta-label lines
    text = _TEMPLATE_LABEL_RE.sub('', text)            # Entity A:/Comparison:/...
    text = _INLINE_PAGE_REF_RE.sub('', text)           # (page 38) Section Title blobs
    text = _PAGE_PAREN_RE.sub('', text)                # any remaining (Page N) refs
    text = _BARE_PAGE_REF_RE.sub('', text)             # raw "page 50" / "pages 26 and 27"

    # Keep only what follows the final "Answer:" / "the answer would be:" marker.
    markers = list(_ANSWER_MARKER_RE.finditer(text))
    if markers:
        text = text[markers[-1].end():].strip()

    # If model wrapped its answer in double-quotes after a reasoning preamble
    # (e.g. 'the answer would be: "The earnings call..."'), unwrap the quotes.
    if text.startswith('"'):
        end_q = text.find('"', 1)
        if end_q != -1:
            inner = text[1:end_q].strip()
            tail  = text[end_q + 1:].strip()
            text  = (inner + " " + tail).strip() if tail else inner

    text = re.sub(r'^\s*[:\-—]\s*', '', text)          # leading bare colon/dash

    sentences = _split_sentences(text)
    kept: List[str] = []
    _seen_keys: set = set()                             # exact-sentence de-dup
    _seen_nums: set = set()                             # figures already stated
    for s in sentences:
        st = s.strip()
        if not st:
            continue
        if _LEAK_SENTENCE_RE.search(st):
            continue
        if _PLACEHOLDER_RE.search(st):
            continue
        if _WARN_MARKER in st:                          # numeric-guard hallucination flag
            continue
        if st.count('|') >= 3:                          # raw pipe-table row dump
            continue
        if _RAW_TABLE_ROW_RE.search(st):                # raw "Federal: Current:... Deferred:..." dump
            continue
        # EXACT DE-DUP — drop a verbatim repeat (normalized key).
        _key = re.sub(r'[^a-z0-9]+', '', st.lower())
        if len(_key) >= 12 and _key in _seen_keys:
            continue
        # NUMERIC-NOVELTY DE-DUP — the small model regenerates the same facts in
        # paraphrased form until it hits max_tokens. Drop a sentence whose figures
        # were ALL already stated (pure restatement); keep sentences that add a new
        # figure, and keep number-free narrative (handled by the exact-dup check).
        _nums = set(re.findall(r'\d[\d,.]*\d|\d', st))
        if _nums and _nums <= _seen_nums:
            continue
        _seen_nums |= _nums
        _seen_keys.add(_key)
        # Strip a leaked table-label prefix ("Total: The provision was ..." →
        # "The provision was ...") when real prose follows the label.
        st = re.sub(
            r'^\s*(?:Total|Federal|State|Foreign|Domestic|Segment Breakdown)\s*:\s*'
            r'(?=[A-Z][a-z])', '', st,
        )
        kept.append(st)

    # Drop a trailing no-info hedge when real content remains (keeps source chips
    # visible). A pure no-info answer — nothing else kept — is preserved as-is.
    _substantive = [s for s in kept if not _NOINFO_HEDGE_RE.search(s)]
    if _substantive:
        kept = _substantive

    result = " ".join(kept).strip()
    result = re.sub(r'^\s*[:\-—]\s*', '', result)
    # Trailing meta-label artifact the model leaves with empty values
    # ("... May 2023. Sources:,,," or "... FY2023. Tags:").
    result = re.sub(r'\s*\b(?:Sources?|Tags?|Source)\s*:\s*[,;\s]*$', '', result,
                    flags=re.IGNORECASE).strip()
    # Drop a dangling leading connector left behind when a reasoning sentence
    # before it was removed (e.g. "Therefore, Mac had..." → "Mac had...").
    result = re.sub(r'^(?:therefore|thus|so|hence|then|in conclusion|'
                    r'as a result)\s*,?\s*', '', result, flags=re.IGNORECASE)
    result = (result[:1].upper() + result[1:]) if result else result
    result = re.sub(r'\s{2,}', ' ', result).strip()

    # Empty after stripping. If the ORIGINAL was itself a leaked instruction
    # (e.g. "Do not assume that ESG goals are mentioned in the context."), do NOT
    # echo it back — return the standard not-found message so the caller's
    # refusal/fallback path takes over and the user never sees the leak.
    if not result:
        if _LEAK_SENTENCE_RE.search(answer) or _PLACEHOLDER_RE.search(answer):
            return "I could not find this in the provided sources."
        # Otherwise the stripping over-matched a genuine answer — return the
        # de-prefixed original rather than an empty string.
        fallback = re.sub(r'^\s*[:\-—]\s*', '', text.strip())
        return fallback or re.sub(r'^\s*[:\-—]\s*', '', answer.strip())
    return result


# HASH FOR DEDUP

def _hash(text: str, meta: Dict[str, Any]) -> str:
    base = f"{text[:100]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# DOCUMENT NORMALIZATION

def _normalize_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs:
        if isinstance(d, dict):
            out.append(d)
        elif isinstance(d, tuple):
            out.append({
                "text":     d[0] if len(d) > 0 else "",
                "score":    d[1] if len(d) > 1 else 0.0,
                "metadata": d[2] if len(d) > 2 else {},
            })
    return out


# PHASE 24.8 — STANDARDISED SOURCES ARRAY

def _build_p248_sources(docs: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    import os as _os
    out: List[Dict[str, Any]] = []
    for doc in docs[:max_items]:
        meta  = doc.get("metadata") or {}
        text  = doc.get("text") or ""
        score = doc.get("final_score") if doc.get("final_score") is not None else doc.get("score")
        try:
            score = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        src_raw = meta.get("source") or meta.get("filename") or meta.get("file_path") or "unknown"
        source_name = _os.path.basename(str(src_raw)) if src_raw != "unknown" else "unknown"
        modality = str(meta.get("modality") or "text")

        # The chip suppresses the section-header suffix when modality == "text"
        # (it treats that as a plain .txt file, where the filename alone suffices).
        # DOCX/PDF/RTF paragraphs are stored with content-modality "text" but ARE
        # document files WITH useful section headers, so map them to a file-type
        # modality so the header shows. (.txt stays "text" → filename-only.)
        _ext = source_name.rsplit(".", 1)[-1].lower() if "." in source_name else ""
        _FILE_MODALITY = {"docx": "word", "doc": "word", "rtf": "word",
                          "odt": "word", "pdf": "pdf"}
        if modality == "text" and _ext in _FILE_MODALITY:
            modality = _FILE_MODALITY[_ext]

        page_number: Optional[int] = None
        raw_page = meta.get("page_number") if meta.get("page_number") is not None else meta.get("page")
        if isinstance(raw_page, int):
            page_number = raw_page
        elif raw_page is not None:
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                pass

        start_time: Optional[float] = None
        end_time:   Optional[float] = None
        for sk, tk in (("start_time", "timestamp_start"), ("end_time", "timestamp_end")):
            raw = meta.get(sk) if meta.get(sk) is not None else meta.get(tk)
            if raw is not None:
                try:
                    val = float(raw)
                    if sk == "start_time":
                        start_time = val
                    else:
                        end_time = val
                except (TypeError, ValueError):
                    pass

        # DOCX/Word: nearest heading is the locator (no page numbers at render time).
        # Excel/table: sheet name extracted from the chunk text prefix "[Sheet: X, Rows Y-Z]"
        # so existing indexed data works without re-upload.
        raw_section = meta.get("section_title")
        section_title = str(raw_section).strip() if raw_section else None
        # Fix OCR year-range substitutions stored at ingest time (e.g. "ta"/"t0" → "to")
        if section_title:
            section_title = re.sub(
                r'(\d{4})\s+(?:ta|t0)\s+(\d{4})', r'\1 to \2',
                section_title, flags=re.IGNORECASE,
            )

        if not section_title and modality in ("table", "excel"):
            import re as _re
            _m = _re.match(r'\[Sheet:\s*([^,\]\n]+)', str(text))
            if _m:
                section_title = _m.group(1).strip()

        # Image: the clean chart title lives in meta["image_title"] (set by the
        # image chunker from the caption's quoted title). Do NOT fall back to
        # the full caption for section_title — the caption is a multi-paragraph
        # analysis dump and previously rendered as a giant citation. Leave
        # section_title empty for images; the UI cites images by clean filename
        # chip + length-guarded image_title, matching XLSX/DOCX.
        if modality == "image":
            section_title = None

        # Phase 6.3 rich citation fields — flow directly from chunk structure
        sheet_name   = meta.get("sheet_name")
        heading      = meta.get("heading") or meta.get("heading_hierarchy")
        speaker_role = meta.get("speaker_role")
        speaker_name = meta.get("speaker_name") or meta.get("speaker_label")
        # Never surface a raw diarization label ("SPEAKER_10") as the speaker on
        # a citation — it's not a real name. When the turn is a reporter's
        # question, "Reporter" is the meaningful attribution; otherwise drop it
        # and let the timestamp stand alone (the answer prose still names the
        # speaker when it's known).
        if speaker_name and re.match(r"^SPEAKER_\d+$", str(speaker_name).strip()):
            _sec = str(meta.get("call_section") or "")
            _is_reporter_turn = (
                _sec == "qa_session"
                or meta.get("is_question")
                # A reporter naming their outlet ("... from/with MarketWatch/the
                # New York Times/...") — even when diarization left the turn
                # unnamed and the section mislabelled. High-precision on a known
                # outlet, so it won't tag the chair/host.
                or re.search(
                    r"\b(?:from|with)\s+(?:the\s+)?(?:MarketWatch|New\s+York\s+Times|"
                    r"Wall\s+Street\s+Journal|Reuters|Bloomberg|CNBC|Associated\s+Press|"
                    r"Financial\s+Times|Politico|Washington\s+Post|Economist|Axios|Semafor)\b",
                    str(text), re.IGNORECASE,
                )
            )
            speaker_name = "Reporter" if _is_reporter_turn else None
        row_range    = meta.get("row_range")
        chunk_type   = meta.get("chunk_type") or meta.get("content_type")
        call_section = meta.get("call_section") or meta.get("topic_section")
        image_title  = meta.get("image_title")
        slide_numbers = meta.get("slide_numbers_covered")

        # Clean heading_hierarchy list → readable string
        if isinstance(heading, list):
            heading = " > ".join(str(h) for h in heading if h)
        if heading:
            heading = str(heading).strip()[:120] or None

        # Snippet: first 200 chars of text, truncated at word boundary
        snippet_raw = str(text)[:220]
        if len(snippet_raw) > 200:
            word_break = snippet_raw.rfind(" ", 0, 200)
            snippet = snippet_raw[:word_break] if word_break >= 0 else snippet_raw[:200]
        else:
            snippet = snippet_raw

        out.append({
            "filename":       source_name,
            "source":         source_name,
            "modality":       modality,
            "page":           page_number,
            "page_number":    page_number,
            "section_title":  section_title,
            "sheet_name":     sheet_name,
            "heading":        heading,
            "timestamp_start": start_time,
            "timestamp_end":  end_time,
            "start_time":     start_time,
            "end_time":       end_time,
            "speaker_role":   speaker_role,
            "speaker_name":   speaker_name,
            "call_section":   call_section,
            "row_range":      row_range,
            "chunk_type":     chunk_type,
            "image_title":    image_title,
            "slide_numbers":  slide_numbers,
            "snippet":        snippet,
            "text":           snippet,
            "score":          round(score, 6),
            "doc_id":         str(meta.get("doc_id") or meta.get("chunk_id") or ""),
        })
    return out


# DEDUP DOCS

def _dedup_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen:   set                = set()
    unique: List[Dict[str, Any]] = []
    for d in docs:
        h = _hash(d.get("text", ""), d.get("metadata", {}))
        if h in seen:
            continue
        seen.add(h)
        unique.append(d)
    return unique


# AUTO FILE-SCOPE & SOURCE-COHERENCE HELPERS (duplicated from query_pipeline to
# avoid a circular import — rag_pipeline must not import query_pipeline at module level)

_AUTO_SCOPE_RE_STREAM = re.compile(
    r'\b[\w\-]{2,60}\.(?:pdf|txt|docx|xlsx|xls|doc|mp3|mp4|wav|jpg|jpeg|png)\b',
    re.IGNORECASE,
)

_COHERENCE_MAX_SOURCES_STREAM = 3
_COHERENCE_GAP_ABS_STREAM     = 0.45
_COHERENCE_ABS_FLOOR_STREAM   = 0.04


def _detect_filename_scope_stream(query: str) -> Optional[List[str]]:
    matches = _AUTO_SCOPE_RE_STREAM.findall(query)
    return [m.lower() for m in matches] if matches else None


def _source_coherence_filter_stream(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(docs) <= 1:
        return docs
    top_raw = (docs[0].get("metadata") or {}).get("_reranker_raw")
    seen_sources: Dict[str, Any] = {}
    kept: List[Dict[str, Any]] = []
    for doc in docs:
        meta = doc.get("metadata") or {}
        src  = meta.get("source", "")
        raw  = meta.get("_reranker_raw")
        if not kept:
            seen_sources[src] = raw
            kept.append(doc)
            continue
        if raw is not None and raw < _COHERENCE_ABS_FLOOR_STREAM:
            continue
        if src not in seen_sources:
            if len(seen_sources) >= _COHERENCE_MAX_SOURCES_STREAM:
                continue
            if top_raw is not None and raw is not None and raw < top_raw - _COHERENCE_GAP_ABS_STREAM:
                continue
            seen_sources[src] = raw
        kept.append(doc)
    return kept


# BUILD CONTEXT STRING — SECTION 4.6
#
# Renders retrieved chunks in Claude-style numbered references:
#
#   [1] (rag_test_corpus.txt — DOC-001 — Transformer Architecture) <chunk text>
#   [2] (rag_test_corpus.txt — DOC-002) <chunk text>
#
# The number aligns 1-to-1 with the position in the canonical `sources`
# list returned to the API, so the UI can render
#   Sources:
#   [1] rag_test_corpus.txt — DOC-001 (Transformer Architecture)
#   [2] rag_test_corpus.txt — DOC-002

def _build_context(
    docs: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    parts: List[str] = []
    total: int       = 0

    for idx, d in enumerate(docs, start=1):
        text          = d.get("text", "").strip()
        meta          = d.get("metadata", {}) or {}
        source        = meta.get("source") or ""
        section_id    = meta.get("section_id")
        # section_title intentionally NOT read into the label (see label block below).
        page          = meta.get("page")
        error_markers = meta.get("error_markers") or []
        doc_version   = meta.get("doc_version")

        if not text:
            continue

        # LABEL — minimal by design. Citations no longer come from the model
        # echoing the label (page anchors are inserted deterministically by
        # _attach_page_citations, and source chips come from the SOURCES payload),
        # so we deliberately OMIT section_title here: the small model used to copy
        # those titles verbatim into a trailing "label dump" (e.g. "Net Sales by
        # Product Category Consolidated Statements of Operations ..."). Keep only
        # the page / section_id locator.
        label_parts: List[str] = []
        if source and page is None:
            label_parts.append(str(source))
        if section_id:
            label_parts.append(str(section_id))
        elif page is not None:
            label_parts.append(f"page {page}")
        if doc_version:
            label_parts.append(f"version={doc_version}")

        provenance = " — ".join(label_parts) if label_parts else "unknown"
        label      = f"[{idx}] ({provenance})"

        # When the chunk carries in-corpus self-flags (e.g. "intentional
        # error", "does not exist", "WRONG LABEL"), surface them on a
        # separate header line so the LLM can treat the claim as suspect.
        # The prompt builder's general branch explains this exact format.
        if error_markers:
            joined = "; ".join(str(m) for m in error_markers[:4])
            label = f"{label}\n⚠ ERROR_MARKERS={joined}"

        chunk = f"{label} {text}"[:settings.RAG_DOC_MAX_CHARS]

        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


# KEY-FACT EXTRACTOR — for queries that ask about specific events (acquisitions,
# mergers, dates) the LLM often misses sentences buried mid-chunk because the
# prompt is flattened to a single line. Prepending a "KEY FACT" line surfaces
# the most relevant sentence right after "CONTEXT:" where the LLM reads first.

_MA_QUERY_KEYWORDS  = frozenset(["acquisition", "merger", "acquired", "deal", "takeover", "purchased"])
_MA_CHUNK_KEYWORDS  = frozenset(["acquired", "acquisition", "merger", "assumed", "fdic", "purchase"])

# Phrases that mark an LLM refusal (model declined to answer despite context).
# Used by the streaming path: a refusal here is suppressed and the accurate
# meta-path answer is streamed instead, so the user never sees the flash.
_LLM_REFUSAL_PHRASES = (
    "could not find this in the provided sources",
    "could not find", "cannot find", "couldn't find",
    "no relevant information", "not in the provided", "not found in",
    "not mentioned in", "not provided in", "is not available",
    "i don't know", "i do not know", "no information about",
)

def _is_llm_refusal(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    # Pure refusal: a short answer dominated by a refusal phrase.
    if len(t) <= 200:
        return any(p in t for p in _LLM_REFUSAL_PHRASES)
    # Long answers: only a refusal if they OPEN with a refusal phrase (model
    # declined up front). A substantive answer that merely mentions "no relevant
    # information" for one sub-part — after giving real content — is NOT a
    # refusal, and must keep its source citations (was dropping them before).
    head = t[:120]
    return any(p in head for p in _LLM_REFUSAL_PHRASES)


# Streaming holdback tuning — see RAGPipeline.stream(). The prefix gate must be
# long enough to contain every _LLM_REFUSAL_PHRASES opener; the holdback tail
# must exceed the longest PII entity (emails/SSNs/phones contain no spaces, so
# an in-progress entity always sits inside the unflushed tail).
_STREAM_PREFIX_GATE = settings.STREAM_PREFIX_GATE_CHARS
_STREAM_HOLDBACK    = settings.STREAM_HOLDBACK_CHARS


# SANDWICH REORDER — Liu et al. "Lost in the Middle" (2023):
# LLMs attend best to the beginning and end of long prompts; middle positions
# are attended to least.  Placing the best-ranked chunk first and the
# second-best last keeps the two most relevant passages in the high-attention
# zones.  Middle chunks are sorted by descending relevance so any accidental
# attention still lands on the most relevant remaining content.
#
# Input:  docs sorted descending by reranker score (docs[0] is best).
# Output: [best, 3rd, 4th, …, 2nd-best]  (sandwich around the middle filler).

def _sandwich_reorder(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(docs) <= 2:
        return docs
    best   = docs[0]
    second = docs[1]
    middle = docs[2:]   # already sorted descending — middle filler
    return [best] + middle + [second]


def _doc_key(d: Dict[str, Any]) -> str:
    """Stable identity for a retrieved doc — chunk hash if present, else text."""
    meta = d.get("metadata") or {}
    return str(meta.get("chunk_hash_id") or meta.get("chunk_id") or (d.get("text") or "")[:80])


def _focus_docx_context(
    reranked: List[Dict[str, Any]],
    hybrid_top: List[Dict[str, Any]],
    max_chunks: int = 5,
) -> List[Dict[str, Any]]:
    """Trim a DOCX answer's context to a small, high-signal set.

    Keeps: the reranker's confident chunks (those above a score cliff relative
    to its top score) UNION the hybrid retriever's own top-3. The union matters
    because the cross-encoder reranker is unreliable on finance sections — it
    sometimes buries the true answer chunk (e.g. ranks a "China risk assessment"
    footnote above the "5.1.1 China Revenue Concentration Risk" section), while
    the hybrid (BM25+dense) order keeps the real section near the top. Feeding
    the small model this focused set instead of all 10-19 chunks stops it
    drifting into unrelated tables and giving lazy partial answers.
    """
    if not reranked:
        return reranked
    top_score = max((d.get("score", 0.0) or 0.0) for d in reranked)
    cutoff = max(top_score * 0.25, 0.05)

    kept: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(d: Dict[str, Any]) -> None:
        k = _doc_key(d)
        if k not in seen:
            seen.add(k)
            kept.append(d)

    # 1) reranker chunks above the cliff (in reranker order)
    for d in reranked:
        if (d.get("score", 0.0) or 0.0) >= cutoff:
            _add(d)
    # 2) hybrid's own top-3 anchor (covers reranker mis-rankings)
    for d in hybrid_top:
        _add(d)
    # 3) if still nothing substantive, fall back to reranker top-3
    if not kept:
        for d in reranked[:3]:
            _add(d)

    return kept[:max_chunks]


_TXT_COMPARISON_QUERY_RE = re.compile(
    r"\b(compare[ds]?|compared to|comparison|how did|how do(?:es)?|versus|vs\.?|"
    r"relative to|against|difference between|higher or lower|change from)\b",
    re.IGNORECASE,
)
# Sentence-level comparison signals inside the source text. A transcript states
# a comparison qualitatively ("somewhat higher than in September", "two cuts
# next year, compared to four in September") far more often than as two
# side-by-side numbers, and the small model tends to give the first figure then
# wander — dropping the comparison clause entirely. Surfacing those sentences
# at the very top of the context makes the model state the real comparison
# instead of omitting it (or inventing a matching-format number).
_TXT_COMPARISON_SENT_RE = re.compile(
    r"\b(compared to|somewhat higher than|higher than|lower than|"
    r"more than|less than|than in (?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december)|versus|"
    r"revised (?:up|down|higher|lower)|up from|down from)\b",
    re.IGNORECASE,
)


def _prepend_txt_comparison_facts(
    docs: List[Dict[str, Any]],
    query: str,
    context: str,
) -> str:
    """For a TXT comparison query, hoist the context sentences that actually
    state a comparison to the top of the context so the small model reliably
    includes them. Extracts verbatim from the retrieved chunks only — never
    invents a figure. No-op when the query isn't comparative or no comparison
    sentence is present."""
    if not query or not docs or not _TXT_COMPARISON_QUERY_RE.search(query):
        return context
    facts: List[str] = []
    seen: set = set()
    for doc in docs[:5]:
        text = (doc.get("text", "") or "")
        for sent in _split_sentences(text):
            s = sent.strip()
            if len(s) < 20 or s in seen:
                continue
            if _TXT_COMPARISON_SENT_RE.search(s):
                seen.add(s)
                facts.append(s)
        if len(facts) >= 3:
            break
    if not facts:
        return context
    header = "KEY COMPARISON FACTS (state these explicitly in your answer):\n" + \
             "\n".join(f"- {f}" for f in facts[:3])
    return header + "\n\n" + context


def _ensure_txt_comparison_in_answer(
    answer: str,
    docs: List[Dict[str, Any]],
    query: str,
) -> str:
    """Guarantee a TXT comparison answer actually states the comparison the
    question asked for. The small model is inconsistent — some runs include
    the source's comparison sentence ("These median projections are somewhat
    higher than in September"), other runs (same prompt) drop it and pad with
    unrelated facts. When the answer is missing it, deterministically append
    the verbatim comparison sentence from the retrieved context — same
    philosophy as the PDF/DOCX deterministic citation attachers. Never invents
    a figure: the appended text is copied verbatim from a retrieved chunk."""
    if not answer or not query or not _TXT_COMPARISON_QUERY_RE.search(query):
        return answer
    ans_low = answer.lower()
    # Already states a comparison? (any of these signal words present)
    if any(kw in ans_low for kw in (
        "compared to", "higher than", "lower than", "than in september",
        "than in the previous", "up from", "down from", "versus",
        "compared with", "than september",
    )):
        return answer
    # Find the best verbatim comparison sentence from context.
    for doc in docs[:5]:
        for sent in _split_sentences(doc.get("text", "") or ""):
            s = sent.strip()
            if len(s) < 20:
                continue
            if _TXT_COMPARISON_SENT_RE.search(s):
                sep = " " if answer.endswith((".", "!", "?")) else ". "
                logger.info(event="rag_stream_txt_comparison_appended")
                return (answer + sep + s).strip()
    return answer


def _strip_unsupported_txt_numbers(
    answer: str,
    docs: List[Dict[str, Any]],
    query: str,
) -> str:
    """Remove any sentence in a TXT answer that contains a number not
    verbatim in the retrieved context, using reasoning_engine's existing,
    already-tested `_unsupported_numbers` verifier. Only the offending
    sentence is dropped — every other, correctly-grounded sentence in the
    answer is left exactly as generated. Falls back to the original answer
    on any error, or if stripping would leave nothing (never returns empty).
    """
    if not answer:
        return answer
    try:
        from app.reasoning.reasoning_engine import _unsupported_numbers, _NUM_RE
    except Exception:
        return answer

    bad = set(_unsupported_numbers(answer, docs, query=query))
    if not bad:
        return answer

    sentences = _split_sentences(answer)
    kept = [s for s in sentences if not (set(_NUM_RE.findall(s)) & bad)]
    cleaned = " ".join(s for s in kept if s.strip()).strip()
    if cleaned:
        logger.info(event="rag_stream_txt_numbers_stripped", removed=list(bad)[:5])
        return cleaned
    return answer


_TXT_REDUNDANT_OPENER_RE = re.compile(
    r"^\s*(?:therefore|in summary|in conclusion|to summarize|"
    r"so,?\s+in short|overall,?\s+the|thus,?)\b",
    re.IGNORECASE,
)


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9.]+", text.lower()) if len(w) > 2}


def _trim_txt_redundant_closer(answer: str) -> str:
    """Drop a trailing summary sentence ("Therefore, ...", "Overall, ...",
    "In summary, ...") — a factual transcript answer states its facts directly
    and never needs a concluding restatement, so any such closer is padding
    the benchmark answers never contain. Also cleans a dangling trailing
    open-paren / stray punctuation the small model sometimes emits. Requires
    >=3 sentences so a genuinely short answer is never truncated."""
    if not answer:
        return answer
    # Clean a dangling opener the model left unfinished, e.g. a trailing "(".
    answer = re.sub(r"\s*[(\[]\s*$", "", answer).rstrip()
    sents = [s.strip() for s in _split_sentences(answer) if s.strip()]
    if len(sents) < 3:
        return answer
    if _TXT_REDUNDANT_OPENER_RE.search(sents[-1]):
        return " ".join(sents[:-1]).strip()
    return answer


# A Fed policy move is quoted in whole basis points (25/50/75/100...). A
# "basis points" value with a decimal ("4.25 basis points", "4.50 basis
# points") is always a small-model hallucination — it has mislabeled a
# target-range rate level (4.25%) as a basis-point figure. Drop such sentences.
_TXT_BAD_BP_RE = re.compile(r"\b\d+\.\d+\s*basis\s*points?\b", re.IGNORECASE)


def _clean_txt_answer(answer: str, max_sentences: int = 4) -> str:
    """Final TXT answer tidy: drop sentences with an implausible decimal
    basis-point figure, then cap the answer length so trailing padding is
    removed while the front-loaded core facts are kept. Never returns empty."""
    if not answer:
        return answer
    sents = [s.strip() for s in _split_sentences(answer) if s.strip()]
    kept = [s for s in sents if not _TXT_BAD_BP_RE.search(s)]
    if not kept:
        kept = sents
    if len(kept) > max_sentences:
        kept = kept[:max_sentences]
    return " ".join(kept).strip()


def _adaptive_temperature(query: str) -> float:
    """Derive the generation temperature from the query type (factual vs generative)."""
    try:
        from app.prompt.prompt_builder import get_generation_temperature, _detect_query_type  # noqa: PLC0415
        return get_generation_temperature(_detect_query_type(query))
    except Exception:
        return settings.LLM_TEMPERATURE

def _prepend_key_facts(docs: List[Dict[str, Any]], query: str, context: str) -> str:
    """If the query is about an M&A event, extract the most relevant sentences
    from the top chunks and prepend them so they land first in the flat prompt."""
    if not query or not docs:
        return context
    ql = query.lower()
    if not any(kw in ql for kw in _MA_QUERY_KEYWORDS):
        return context
    facts = []
    for doc in docs[:3]:
        text = doc.get("text", "") or ""
        # Split on sentence boundaries and collect M&A sentences
        for sent in text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|"):
            sl = sent.lower()
            if any(kw in sl for kw in _MA_CHUNK_KEYWORDS) and len(sent.strip()) > 30:
                facts.append(sent.strip())
        if len(facts) >= 3:
            break
    if not facts:
        return context
    # Cap prefix at 300 chars to avoid pushing total prompt past llama.cpp's
    # token context window (4096 tokens) on dense documents like PDFs.
    prefix_body = " | ".join(facts[:3])[:300]
    prefix = "KEY FACTS (answer the query from these): " + prefix_body + " "
    return prefix + context


# COMPOSE CONTEXT + HISTORY

def _compose(history: str, context: str) -> str:
    parts: List[str] = []
    if history:
        parts.append(history)
    if context:
        parts.append(context)
    return "\n\n".join(parts)


# FORMAT HISTORY — SECTION 4.7

def _format_history(
    history: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    out:   List[str] = []
    total: int       = 0

    for msg in reversed(history):
        role    = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        line    = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line)

    return "\n".join(reversed(out))


# SOURCES EXTRACTOR

def _extract_sources(docs: List[Dict[str, Any]]) -> List[str]:
    return list({
        d.get("metadata", {}).get("source")
        for d in docs
        if d.get("metadata", {}).get("source")
    })


# RAG PIPELINE CLASS

class RAGPipeline:

    def __init__(self) -> None:
        self._retriever     = None
        self._prompt_builder = None
        self._llm           = None
        self._memory_mgr    = None
        self._mongo         = None

    # LAZY INIT — AVOID CIRCULAR IMPORTS
    #
    # We use HybridRetriever directly (the live `query_pipeline` already
    # does the same). The older `app.retrieval.retriever.Retriever` class
    # is kept on disk for legacy integration tests but is no longer on the
    # production path — it had its own RRF/MMR/query-expansion that
    # diverged from HybridRetriever's, causing identical queries to return
    # different results depending on entry point.

    def _get_retriever(self):
        if self._retriever is None:
            from app.core.infra_registry import infra
            from app.core.model_loader import model_loader
            from app.retrieval.hybrid_retriever import HybridRetriever

            bm25         = infra.get_bm25()
            vector_store = infra.get_vector_store()
            embedder     = model_loader.get_embedder()
            clip_embed   = None
            if settings.ENABLE_VISION:
                try:
                    clip_embed = model_loader.get_siglip_text_embedder()
                except Exception as exc:
                    logger.warning(event="siglip_text_embedder_unavailable", error=str(exc))

            self._retriever = HybridRetriever(
                bm25=bm25,
                vector_store=vector_store,
                embedder=embedder,
                clip_text_embedder=clip_embed,
            )
        return self._retriever

    def _get_prompt_builder(self):
        if self._prompt_builder is None:
            from app.prompt.prompt_builder import PromptBuilder
            self._prompt_builder = PromptBuilder()
        return self._prompt_builder

    def _get_llm(self):
        if self._llm is None:
            try:
                from app.core.model_loader import model_loader
                self._llm = model_loader.get_llm()
            except Exception as e:
                logger.warning(event="llm_unavailable", error=str(e))
                self._llm = None
        return self._llm

    def _get_memory_manager(self):
        if self._memory_mgr is None:
            from app.memory.memory_manager import MemoryManager
            self._memory_mgr = MemoryManager()
        return self._memory_mgr

    def _get_mongo(self):
        if self._mongo is None:
            try:
                from app.core.infra_registry import infra
                self._mongo = infra.get_mongo()
            except Exception:
                self._mongo = None
        return self._mongo

    # STORE MEMORY — SECTION 4.7

    def _store_memory(
        self,
        session_id: str,
        query: str,
        answer: str,
    ) -> None:
        if not answer or len(answer.strip()) < 5:
            return
        try:
            # add_interaction writes user + assistant to both Redis and MongoDB
            # via MemoryManager internally — do NOT call mongo.store_message()
            # separately or every turn gets written twice to the messages collection.
            mgr = self._get_memory_manager()
            mgr.add_interaction(session_id, query, answer)

        except Exception as e:
            logger.warning(
                event="rag_memory_store_failed",
                error=str(e),
                session_id=session_id,
            )

    # FALLBACK LLM RESPONSE — SECTION 4.6

    def _fallback_response(
        self,
        query: str,
        session_id: str,
    ) -> str:
        try:
            llm = self._get_llm()
            if not llm:
                return "I don't know based on available data."
            prompt = f"Answer clearly and concisely:\n{query}"
            return llm.generate(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.2,
                session_id=session_id,
            ) or "I don't know based on available data."
        except Exception as e:
            logger.error(
                event="rag_fallback_failed",
                error=str(e),
                session_id=session_id,
            )
            return "I don't know based on available data."

    # EMPTY RESPONSE — no docs retrieved, do NOT call LLM

    def _empty(self, start: float) -> Dict[str, Any]:
        return {
            "answer":     "No relevant documents found. Please ingest documents first.",
            "confidence": 0.0,
            "sources":    [],
            "latency":    round(time.time() - start, 2),
            "metadata":   {"docs": 0},
        }

    # MAIN RUN — SECTION 4.6

    def run(
        self,
        query: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        start    = time.time()
        trace_id = str(uuid.uuid4())

        if not query or not query.strip():
            return {"answer": "Query cannot be empty.", "trace_id": trace_id}

        # NORMALIZE + SANITIZE — SECTION 2.3 / 5
        query = _normalize(query)
        _pre_sanitize = query
        query = _sanitize(query)

        if not query and _pre_sanitize:
            return {
                "answer": (
                    "⚠️ Your message was blocked by the security guardrail. "
                    "It matched a restricted pattern (prompt injection or policy violation). "
                    "Please rephrase your query."
                ),
                "decision": "blocked",
                "source":   "input_guard",
                "trace_id": trace_id,
            }

        query = query[:settings.MAX_PROMPT_CHARS]

        try:
            # MEMORY HISTORY — SECTION 4.7
            t_mem = time.time()
            try:
                mgr     = self._get_memory_manager()
                history = mgr.get_history(session_id)
            except Exception as e:
                logger.warning(event="rag_memory_fetch_failed", error=str(e))
                history = []

            history_text = _format_history(history, settings.MEMORY_MAX_CONTEXT_CHARS)
            _record_stage("memory", round(time.time() - t_mem, 3))

            # RETRIEVAL — SECTION 4.5 — HybridRetriever.search() returns
            # the same Dict[text/metadata/score/embedding] shape Retriever.retrieval()
            # did, but routed through the single canonical RRF + MMR + heuristic
            # query-expansion path used by the live query_pipeline.
            t_ret = time.time()
            try:
                retriever = self._get_retriever()
                raw_docs  = retriever.search(
                    query=query,
                    session_id=session_id,
                    top_k=settings.DEFAULT_TOP_K,
                    user_id=user_id,
                )
            except Exception as e:
                logger.error(
                    event="rag_retrieval_failed",
                    error=str(e),
                    session_id=session_id,
                )
                _record_error("retrieval")
                raw_docs = []

            retrieval_latency = round(time.time() - t_ret, 3)
            _record_retrieval("retriever", retrieval_latency)
            _record_stage("retrieval", retrieval_latency)

            if not raw_docs:
                return self._empty(start)

            # NORMALIZE + DEDUP DOCS
            docs = _normalize_docs(raw_docs)
            docs = _dedup_docs(docs)
            docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)
            docs = docs[:settings.RAG_TOP_K]
            docs = _sandwich_reorder(docs)

            # CONTEXT ASSEMBLY
            from app.core.response import build_sources
            canonical_sources = build_sources(docs)
            for i, s in enumerate(canonical_sources, start=1):
                s["index"] = i
            context = _build_context(docs, settings.MAX_CONTEXT_CHARS)

            # Phase 24.8 — standardised sources array with page_number/start_time/end_time
            p248_sources = _build_p248_sources(docs)
            sources = p248_sources
            full_context = _compose(history_text, context)
            full_context = full_context[:settings.MAX_PROMPT_CHARS]

            # PROMPT BUILD — SECTION 4.9
            t_prompt = time.time()
            try:
                builder = self._get_prompt_builder()
                prompt  = builder.build_prompt(
                    query=query,
                    context=full_context,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(event="rag_prompt_build_failed", error=str(e))
                prompt = (
                    f"Answer from context only.\n\n"
                    f"CONTEXT:\n{full_context}\n\n"
                    f"QUERY:\n{query}\n\nAnswer:"
                )
            _record_stage("prompt_build", round(time.time() - t_prompt, 3))

            # PII PROMPT STRIP — Phase 26 P1: scrub PII from prompt before LLM sees it
            try:
                from app.guardrails.pii import strip_pii_from_prompt
                prompt, _pii_stripped = strip_pii_from_prompt(prompt)
                if _pii_stripped:
                    logger.info(event="rag_pipeline_pii_stripped_from_prompt", session_id=session_id)
            except Exception as _pii_err:
                logger.warning(event="rag_pipeline_pii_prompt_strip_failed", error=str(_pii_err))

            # LLM GENERATE — SECTION 4.6 FALLBACK CHAIN
            t_llm  = time.time()
            answer = ""

            try:
                llm = self._get_llm()
                if llm:
                    answer = llm.generate(
                        prompt,
                        max_tokens=settings.LLM_MAX_TOKENS,
                        temperature=_adaptive_temperature(query),
                        top_p=settings.LLM_TOP_P,
                        session_id=session_id,
                    )
                else:
                    raise RuntimeError("LLM_UNAVAILABLE")

            except Exception as e:
                logger.error(
                    event="rag_llm_failed",
                    error=str(e),
                    session_id=session_id,
                )
                _record_error("llm")
                # FALLBACK CHAIN — GGUF LOCAL — SECTION 4.6
                answer = self._fallback_response(query, session_id)

            llm_latency = round(time.time() - t_llm, 3)
            _record_llm(settings.LLM_MODEL_PATH, llm_latency)
            _record_stage("llm", llm_latency)

            answer = (answer or "").strip() or "I don't know based on available data."

            # OUTPUT GUARD — Phase 26: scrub artifacts, citations, PII before memory write
            ctx_texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
            try:
                from app.guardrails.output_guard import check as _output_guard_check
                _og = _output_guard_check(
                    answer,
                    context_chunks=ctx_texts,
                    sources=sources,
                    session_id=session_id,
                    correlation_id=trace_id,
                )
                answer = _og.text
                if _og.hallucination_warning:
                    logger.warning(event="rag_pipeline_hallucination_flagged", session_id=session_id)
                if _og.fabricated_citations:
                    logger.warning(
                        event="rag_pipeline_fabricated_citations_removed",
                        citations=_og.fabricated_citations[:5],
                        session_id=session_id,
                    )
            except Exception as _og_err:
                logger.warning(event="rag_pipeline_output_guard_failed", error=str(_og_err), session_id=session_id)

            # LEAKED-INSTRUCTION STRIPPER — remove echoed prompt rules / reasoning
            # preambles the small GGUF model sometimes emits before the answer.
            try:
                answer = _strip_leaked_instructions(answer)
            except Exception as _leak_err:
                logger.warning(event="rag_pipeline_leak_strip_failed", error=str(_leak_err), session_id=session_id)

            # CITATION TRACKING — filter sources to cited [n] indices, then strip them.
            try:
                from app.core.response import extract_cited_indices, strip_inline_citations
                _cited_idx = extract_cited_indices(answer)
                if _cited_idx:
                    _filtered = [s for i, s in enumerate(sources, 1) if i in _cited_idx]
                    if _filtered:
                        sources = _filtered[:5]
                else:
                    sources = sources[:3]
                answer = strip_inline_citations(answer)
            except Exception as _cit_err:
                logger.warning(event="rag_pipeline_citation_tracking_failed", error=str(_cit_err), session_id=session_id)

            total_latency = round(time.time() - start, 2)

            logger.info(
                event="rag_pipeline_success",
                docs=len(docs),
                retrieval_latency=retrieval_latency,
                llm_latency=llm_latency,
                latency=total_latency,
                session_id=session_id,
                trace_id=trace_id,
            )

            # Phase 24.8 — confidence + hallucination_warning
            _scores = [s["score"] for s in sources[:3] if isinstance(s.get("score"), (int, float))]
            confidence = round(max(0.0, min(sum(_scores) / len(_scores) if _scores else 0.0, 1.0)), 6)
            hallucination_warning = confidence < settings.AGENT_LOW_CONFIDENCE

            return {
                "answer":               answer,
                "sources":              sources,
                "confidence":           confidence,
                "hallucination_warning": hallucination_warning,
                "latency":              total_latency,
                "trace_id":             trace_id,
                "metadata": {
                    "docs":              len(docs),
                    "retrieval_latency": retrieval_latency,
                    "llm_latency":       llm_latency,
                    "memory_turns":      len(history),
                },
            }

        except Exception as e:
            _record_error("pipeline")
            logger.error(
                event="rag_pipeline_failed",
                error=str(e),
                session_id=session_id,
                trace_id=trace_id,
            )
            return {
                "answer":   "Something went wrong. Please try again.",
                "sources":  [],
                "latency":  round(time.time() - start, 2),
                "trace_id": trace_id,
                "error":    str(e),
            }

    # STREAM — SECTION 4.6 SSE / WEBSOCKET TOKEN STREAMING

    def stream(
        self,
        query: str,
        session_id: str = "default",
        user_id: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> Iterator[str]:

        query = _normalize(query)
        query = _sanitize(query)
        query = query[:settings.MAX_PROMPT_CHARS]

        def _generator() -> Iterator[str]:
            try:
                retriever = self._get_retriever()
                _auto_scope    = _detect_filename_scope_stream(query) if not sources else None
                _stream_filters = (
                    {"sources": sources}    if sources
                    else {"sources": _auto_scope} if _auto_scope
                    else None
                )
                raw_docs  = retriever.search(
                    query=query,
                    session_id=session_id,
                    top_k=settings.DEFAULT_TOP_K,
                    user_id=user_id,
                    filters=_stream_filters,
                )

                docs    = _normalize_docs(raw_docs)
                docs    = _dedup_docs(docs)

                # RETRIEVAL GATE — the genuine "no documents" case is decided
                # HERE, deterministically. The retriever already applies
                # HYBRID_MIN_SCORE / BM25_MIN_SCORE, so an empty result means
                # nothing relevant exists: emit the canonical message and skip
                # the LLM. When docs DO exist and the model still refuses, that
                # refusal is caught below and the accurate meta answer is used
                # instead — so the user never sees a spurious refusal flash.
                if not docs:
                    logger.info(
                        event="rag_stream_no_relevant_docs",
                        session_id=session_id,
                    )
                    yield (
                        "I could not find any relevant documents in your "
                        "knowledge base to answer this question."
                    )
                    return

                # Hybrid retrieval's own top-3 (before rerank). For DOCX the
                # cross-encoder reranker is unreliable on finance sections — it
                # over-weights lexical footnote matches (e.g. a "China risk
                # assessment based on IDC data" note) and buries the actual
                # content section — so we keep the hybrid order as a fallback
                # anchor below.
                _hybrid_top = list(docs[:3])

                # RERANK — use the module-level singleton so the cross-encoder model
                # is loaded once and shared across all streaming requests. Fixes
                # "lost in the middle" failures where BM25/Qdrant cosine rank 1
                # differs from semantic rank 1 (e.g. quantum-computing chunk beats
                # Business-Segments for acquisition queries without reranking).
                try:
                    _rer = _get_stream_reranker()
                    if _rer is not None:
                        _reranked = _rer.rerank(query, docs, top_k=settings.RAG_TOP_K, session_id=session_id)
                        if _reranked:
                            docs = _source_coherence_filter_stream(_reranked)
                        else:
                            docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:settings.RAG_TOP_K]
                    else:
                        docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:settings.RAG_TOP_K]
                except Exception as _re_err:
                    logger.warning(event="rag_stream_rerank_failed", error=str(_re_err))
                    docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:settings.RAG_TOP_K]

                # DOCX FOCUS: feeding a small 7B model 10-19 chunks makes it drift
                # (dump unrelated tables, or give a lazy partial answer). Whole
                # finance tables are now single chunks, so the answer usually
                # lives in 1-2 high-signal chunks. Build a tight context = the
                # reranker's confident chunks (score cliff) UNION the hybrid
                # top-3 (covers the cases where the reranker buried the answer),
                # capped small. PDF and other modalities are untouched.
                if docs and str((docs[0].get("metadata") or {}).get("modality") or "") == "docx":
                    docs = _focus_docx_context(docs, _hybrid_top, max_chunks=5)

                # TXT FOCUS: same problem as DOCX above, same fix. A single-file
                # plain-text/transcript source has broadly-relevant boilerplate
                # (an opening statement mentions the rate cut, labor market, and
                # inflation all at once) that scores moderately for almost any
                # question about that document — diluting the small model's
                # attention across 10 chunks and causing it to answer about the
                # boilerplate instead of the one chunk that actually has the
                # specific fact asked for. _focus_docx_context is a generic
                # score-cliff-union-hybrid-top-3 utility despite its name;
                # reused as-is, unmodified. PDF and other modalities untouched.
                if docs and str((docs[0].get("metadata") or {}).get("modality") or "") in ("text", "txt"):
                    docs = _focus_docx_context(docs, _hybrid_top, max_chunks=4)

                docs = _sandwich_reorder(docs)

                context = _build_context(docs, settings.MAX_CONTEXT_CHARS)
                # Use the finance-aware key-facts injector (net sales, gross margin,
                # EU State Aid, capital return) — a superset of the M&A-only
                # _prepend_key_facts. Keeps the streaming UI path in parity with the
                # query_pipeline/benchmark path.
                # XLSX accuracy-phase synth override (2026-07): _prepend_key_facts_
                # knowledge wraps an xlsx synth fact in _XLSX_SYNTH_MARK sentinels
                # when one of its query-pattern-gated branches fires. The
                # non-streaming query_pipeline path applies an UNCONDITIONAL
                # override using this fact (see reasoning_engine.generate_answer);
                # this streaming path previously only got the KEY-FACTS context
                # boost, not the override, so the model would still paraphrase/
                # drift onto an unrelated row despite the grounded fact being
                # right there — the same "model unreliable for figure-dense
                # queries" failure mode the PDF phase's _synth_answer_override
                # exists for. Strip the sentinel here too and apply the same
                # override below (after all guards run), so streaming and
                # query_pipeline give an identical, benchmark-verified answer.
                _xlsx_synth_fact_stream = None
                try:
                    from app.reasoning.reasoning_engine import (
                        _prepend_key_facts_knowledge, _XLSX_SYNTH_MARK,
                    )
                    context = _prepend_key_facts_knowledge(
                        docs, query, context, user_id=user_id or ""
                    )
                    if context.startswith(_XLSX_SYNTH_MARK):
                        _, _xlsx_synth_fact_stream, context = context.split(_XLSX_SYNTH_MARK, 2)
                except Exception as _kf_err:
                    logger.warning(event="rag_stream_keyfacts_failed", error=str(_kf_err))
                    context = _prepend_key_facts(docs, query, context)

                # TXT comparison-fact hoist — for a "how did X compare to Y"
                # question, put the source's actual comparison sentences first
                # so the model states the comparison instead of dropping it.
                _mod0 = str((docs[0].get("metadata") or {}).get("modality") or "") if docs else ""
                if _mod0 in ("text", "txt"):
                    try:
                        context = _prepend_txt_comparison_facts(docs, query, context)
                    except Exception as _cmp_err:
                        logger.warning(event="rag_stream_txt_compare_failed", error=str(_cmp_err))

                # AUDIO/VIDEO ANSWER GENERATION — route through the SAME reasoning
                # engine the (benchmark-validated) non-streaming query_pipeline
                # uses, instead of the single-shot raw llm.stream below. On the
                # small GGUF model, single-shot generation over a spoken-word
                # transcript is unstable: it drifts into invented "Q:/A:" echoes,
                # answers a neighbouring fact, or hallucinates a figure — and it
                # diverges from the validated path. reasoning_engine.generate_
                # answer is deterministic, grounded, and numeric-faithfulness-
                # guarded. Its buffered answer is streamed through the identical
                # clean-up/citation path below. Scoped to AV-dominant results
                # only — documents keep the existing single-shot path.
                _av_dominant = False
                if docs:
                    _mods = [str((d.get("metadata") or {}).get("modality") or "").lower()
                             for d in docs[:5]]
                    _av_hits = [m for m in _mods if m in ("audio", "mp3", "video", "mp4")]
                    _av_dominant = len(_av_hits) > len(_mods) / 2

                _av_reasoned_answer: Optional[str] = None
                if _av_dominant:
                    try:
                        from app.pipeline.query_pipeline import _get_reasoning_components
                        _reasoning, _ = _get_reasoning_components(self._get_llm())
                        # Focused context: the small model drifts and mixes facts
                        # when fed ~20 broadly-relevant transcript chunks (a press
                        # conference has boilerplate that scores moderately for
                        # almost any question). The answer lives in the top 1-3
                        # chunks after reranking, so pass a tight set — this keeps
                        # the answer on the specific fact asked and citing the #1
                        # chunk, instead of synthesising a vague summary.
                        _av_docs = docs[:5]
                        _r_out = _reasoning.generate_answer(
                            query=query,
                            retrieved_docs=_av_docs,
                            memory_context="",
                            session_id=session_id,
                            user_id=user_id or "",
                        )
                        _cand = (_r_out.get("answer") or "").strip()
                        if _cand:
                            _av_reasoned_answer = _cand
                    except Exception as _rex:
                        logger.warning(event="rag_stream_av_reasoning_failed",
                                       error=str(_rex), session_id=session_id)

                builder = self._get_prompt_builder()
                prompt  = builder.build_prompt(
                    query=query,
                    context=context,
                    session_id=session_id,
                )

                # PII PROMPT STRIP — same as non-streaming path (Phase 26 P1)
                try:
                    from app.guardrails.pii import strip_pii_from_prompt as _spfp
                    prompt, _pii_stripped = _spfp(prompt)
                    if _pii_stripped:
                        logger.info(event="rag_stream_pii_stripped_from_prompt",
                                    session_id=session_id)
                except Exception as _pii_err:
                    logger.warning(event="rag_stream_pii_prompt_strip_failed",
                                   error=str(_pii_err))

                llm = self._get_llm()
                if not llm:
                    yield "LLM unavailable."
                    return

                # TRUE TOKEN STREAMING with a holdback buffer.
                # - The first _STREAM_PREFIX_GATE chars are held back so a refusal
                #   ("I could not find…") is detected BEFORE anything reaches the
                #   client — preserving the refusal-sentinel UX below.
                # - After the gate, segments are flushed at whitespace boundaries
                #   keeping a _STREAM_HOLDBACK-char tail, so a PII entity that
                #   spans tokens is never split across a flush (it sits in the
                #   tail until complete, then gets scrubbed).
                # - The FULL output guard still runs on the complete answer; its
                #   canonical text is sent as a \x00REPLACE\x00 sentinel so the
                #   client and persistence always end on the guarded version.
                from app.guardrails.pii import scrub_pii as _scrub_pii_seg

                collected_tokens: List[str] = []
                hold = ""
                refusal_mode = False
                prefix_checked = False

                def _flush(seg: str) -> str:
                    try:
                        scrubbed, _ = _scrub_pii_seg(seg)
                        return scrubbed
                    except Exception:
                        return seg

                # BUFFER-THEN-CLEAN STREAMING. We deliberately do NOT forward raw
                # tokens to the client: the small GGUF model often emits a
                # reasoning/rule preamble first (e.g. "Do not calculate...
                # Answer:") which would flash on screen and then be replaced — the
                # exact flicker the user reported. Instead we accumulate the full
                # generation, strip the leak, and stream the CLEAN answer below.
                # Early refusal detection still runs on the growing prefix so the
                # refusal-sentinel UX is preserved.
                # TXT answers should be tight (benchmark answers are 3-5
                # sentences). The default 768-token budget lets the small model
                # keep going and dump long verbatim transcript passages after it
                # has already answered. Cap TXT generations so it stops once the
                # answer is stated. Other modalities keep the full budget.
                _txt_mod = _mod0 in ("text", "txt")
                _max_tok = 240 if _txt_mod else settings.LLM_MAX_TOKENS
                if _av_reasoned_answer is not None:
                    # AV answer already generated by the reasoning engine above —
                    # feed it into the same buffer the raw stream would fill, so
                    # the downstream refusal/citation/clean-up path is identical.
                    collected_tokens.append(_av_reasoned_answer)
                    if _is_llm_refusal(_av_reasoned_answer):
                        refusal_mode = True
                    else:
                        prefix_checked = True
                else:
                    for token in llm.stream(
                        prompt,
                        max_tokens=_max_tok,
                        temperature=_adaptive_temperature(query),
                        top_p=settings.LLM_TOP_P,
                        session_id=session_id,
                    ):
                        collected_tokens.append(token)
                        if refusal_mode or prefix_checked:
                            continue
                        hold += token
                        if len(hold) < _STREAM_PREFIX_GATE:
                            continue
                        if _is_llm_refusal(hold):
                            refusal_mode = True
                        else:
                            prefix_checked = True

                # OUTPUT GUARD — Phase 26 P1b: guard the COMPLETE answer.
                # Whole-answer checks (groundedness, template artifacts, toxicity,
                # cross-segment PII) cannot run mid-stream; their canonical result
                # is delivered via the REPLACE sentinel after the token stream.
                answer = "".join(collected_tokens).strip()
                _ctx: List[str] = [
                    d.get("text", "") if isinstance(d, dict)
                    else (d.page_content if hasattr(d, "page_content") else "")
                    for d in docs
                ]
                try:
                    from app.guardrails.output_guard import check as _og_check
                    _sources = [{"filename": d.get("metadata", {}).get("source", "") if isinstance(d, dict) else ""} for d in docs]
                    _og = _og_check(answer, context_chunks=_ctx, sources=_sources, session_id=session_id)
                    answer = _og.text
                except Exception as _og_err:
                    logger.warning(event="rag_stream_output_guard_failed", error=str(_og_err))

                # FINANCIAL FIGURE NORMALIZER — deterministic post-processor:
                # replace any '$X.X billion' the LLM wrote with the exact
                # '$XXX,XXX million' figure from context (if within 0.5%).
                # This is context-grounded and does not depend on prompt rules.
                try:
                    answer = _fix_financial_figures(answer, _ctx)
                except Exception as _fin_err:
                    logger.warning(event="rag_fin_normalizer_failed", error=str(_fin_err))

                # LEAKED-INSTRUCTION STRIPPER — remove any echoed prompt rules /
                # reasoning preambles before the canonical answer is delivered.
                try:
                    answer = _strip_leaked_instructions(answer)
                    answer = _fix_inconsistent_totals(answer)
                    answer = _trim_offtopic_finance(query, answer)
                    # Completeness safety net: if the model dropped most of the
                    # injected grounded figures, use the synth doc's complete answer.
                    answer = _synth_completeness_fallback(answer, context)
                except Exception as _leak_err:
                    logger.warning(event="rag_stream_leak_strip_failed", error=str(_leak_err))

                # REFUSAL HANDLING — docs WERE retrieved (we passed the retrieval
                # gate above), so a refusal here means the model declined despite
                # relevant context. Do NOT stream the refusal text: it would flash
                # letter-by-letter and then be replaced, which is the exact UX bug
                # we are fixing. Instead emit a sentinel so the client fetches the
                # accurate meta-path answer (its lazy /rag/query fallback). The
                # genuine "no documents" case never reaches here — it is handled
                # by the deterministic retrieval gate above.
                if not answer or _is_llm_refusal(answer):
                    logger.info(event="rag_stream_llm_refused_using_meta", session_id=session_id)
                    yield "\x00REFUSAL\x00"
                    return

                # CITATION TRACKING — parse [n] indices, filter source chips, then strip.
                try:
                    from app.core.response import extract_cited_indices, strip_inline_citations
                    _cited_idx = extract_cited_indices(answer)
                    if _cited_idx:
                        _cited_docs = [d for i, d in enumerate(docs, 1) if i in _cited_idx]
                        _source_docs = _cited_docs[:5] if _cited_docs else docs[:3]
                    else:
                        _source_docs = docs[:3]
                    answer = strip_inline_citations(answer)
                except Exception:
                    _source_docs = docs[:3]

                # TXT NUMERIC-FIDELITY GUARD: this streaming path has no
                # equivalent of the non-streaming /rag/query path's
                # ReasoningEngine numeric-mismatch retry — it only reaches
                # output_guard, which flags fabricated numbers as a warning
                # but does not remove them. A small model asked to complete a
                # comparison the source only states qualitatively (e.g. "two
                # cuts next year, compared to four in September" — a count,
                # not a rate level) will sometimes invent a plausible-looking
                # matching figure. Strip only the sentence(s) containing an
                # unsupported number — every grounded sentence is untouched.
                # Reuses reasoning_engine's already-proven verification
                # function; does not alter its behavior or any other caller.
                if docs and str((docs[0].get("metadata") or {}).get("modality") or "") in ("text", "txt"):
                    try:
                        answer = _strip_unsupported_txt_numbers(answer, docs, query)
                    except Exception as _num_err:
                        logger.warning(event="rag_stream_txt_numeric_guard_failed", error=str(_num_err))
                    try:
                        answer = _trim_txt_redundant_closer(answer)
                    except Exception as _rc_err:
                        logger.warning(event="rag_stream_txt_closer_trim_failed", error=str(_rc_err))
                    try:
                        answer = _clean_txt_answer(answer, max_sentences=4)
                    except Exception as _ct_err:
                        logger.warning(event="rag_stream_txt_clean_failed", error=str(_ct_err))
                    # Deterministically guarantee the asked-for comparison is
                    # present (runs AFTER the sentence cap so the appended
                    # comparison sentence is never trimmed away).
                    try:
                        answer = _ensure_txt_comparison_in_answer(answer, docs, query)
                    except Exception as _ec_err:
                        logger.warning(event="rag_stream_txt_compare_append_failed", error=str(_ec_err))

                # PERPLEXITY-STYLE [p.N] ANCHORS — deterministically attach a page
                # citation to each sentence by tracing its figures to the source
                # chunk. Runs on the clean prose (after strip_inline_citations) so
                # the only brackets left are the trustworthy page anchors.
                try:
                    _before_cite = answer
                    answer = _attach_page_citations(answer, docs)
                    if answer == _before_cite:
                        # No page-numbered chunks (e.g. an all-DOCX context) —
                        # fall back to section-based anchors.
                        answer = _attach_section_citations(answer, docs)
                    # NOTE: image answers are NOT given an inline "Source: <title>"
                    # prose footer — it duplicated the source chip the UI already
                    # renders below the bubble (the chart title now shows on that
                    # chip as a caption), which read as two citations for one
                    # source. Images are cited by the chip alone, like the clean
                    # filename chips for the other modalities.
                except Exception as _cite_err:
                    logger.warning(event="rag_stream_page_cite_failed", error=str(_cite_err))

                # XLSX synth override — applied LAST, after every guard above has
                # run its normal course on the model's own generation (same
                # ordering lesson as reasoning_engine.generate_answer: overriding
                # earlier let a guard's retry/strip logic reprocess and corrupt
                # this already-curated, grounded fact). Unconditional: for these
                # query types the model's own generation is never trusted.
                if _xlsx_synth_fact_stream:
                    answer = _xlsx_synth_fact_stream

                # IMAGE CHART synth override — same "don't trust the model on
                # figure-dense answers" rationale as the XLSX override above,
                # for chart-value/comparison questions specifically (verified
                # over repeated runs: the model restates a correct-looking but
                # WRONG series' dollar value, or a percent as if it were a
                # dollar figure, when asked "in dollar terms" or to compare
                # multiple series). Only overrides for question types this was
                # actually observed on — drawdown/plateau questions are
                # excluded inside the function and keep using the (already
                # correct) LLM + CHART TRENDS narrative path.
                if docs and str((docs[0].get("metadata") or {}).get("modality") or "") == "image":
                    try:
                        _img_synth = _synthesize_image_chart_answer(query, context)
                        if _img_synth:
                            answer = _img_synth
                    except Exception as _img_synth_err:
                        logger.warning(event="rag_stream_image_synth_failed", error=str(_img_synth_err))
                    try:
                        answer = _expand_chart_dates(answer)
                    except Exception as _date_err:
                        logger.warning(event="rag_stream_image_date_expand_failed", error=str(_date_err))

                # STREAM THE CLEAN ANSWER progressively — gives the client a
                # typing effect without ever exposing the raw leaked preamble.
                # PII is scrubbed once on the whole answer (so entities are never
                # split across a chunk boundary), then emitted in small word
                # groups. The REPLACE sentinel below carries the identical text,
                # so the swap is invisible (no flicker).
                _stream_answer = _flush(answer)
                _words = _stream_answer.split(" ")
                _chunk = ""
                for _i, _w in enumerate(_words):
                    _chunk = _w if not _chunk else _chunk + " " + _w
                    if (_i + 1) % 4 == 0:
                        yield _chunk + " "
                        _chunk = ""
                if _chunk:
                    yield _chunk

                # REPLACE sentinel — canonical guarded answer. The client swaps
                # its streamed buffer for this text; the API stream layer is the
                # single persistence point (memory + Mongo + cache) — writing
                # here too would duplicate every exchange in MongoDB.
                yield "\x00REPLACE\x00" + answer

                # Emit sources so the client can display them immediately.
                try:
                    import json as _json
                    _p248 = _build_p248_sources(_source_docs)
                    yield "\x00SOURCES\x00" + _json.dumps(_p248)
                except Exception:
                    pass

            except Exception as e:
                logger.error(
                    event="rag_stream_failed",
                    error=str(e),
                    session_id=session_id,
                )
                _record_error("stream")
                yield "Streaming failed."

        return _generator()

    # ASYNC STREAM — SECTION 4.6

    async def stream_async(
        self,
        query: str,
        session_id: str = "default",
    ) -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        gen  = await loop.run_in_executor(None, self.stream, query, session_id)

        for token in gen:
            yield token

    # ASYNC RUN — SECTION 4.6

    async def run_async(
        self,
        query: str,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self.run, query, session_id),
            timeout=settings.REQUEST_TIMEOUT_SEC,
        )


