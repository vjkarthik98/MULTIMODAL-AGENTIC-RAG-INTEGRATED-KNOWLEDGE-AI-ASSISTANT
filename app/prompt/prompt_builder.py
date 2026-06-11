import asyncio
import hashlib
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_prompt_duration = Histogram(
    "prompt_builder_duration_seconds",
    "Prompt builder duration",
    ["status"],
)
_prompt_errors = Counter(
    "prompt_builder_errors_total",
    "Prompt builder errors by type",
    ["error_type"],
)
_prompt_length = Histogram(
    "prompt_builder_length_chars",
    "Final prompt length in characters",
    ["prompt_type"],
)
_injection_detections = Counter(
    "prompt_injection_detections_total",
    "Number of prompt injection attempts detected",
)

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)

# BUDGET RATIOS — applied to the REMAINING capacity after fixed components
# (system prompt + query block + format block) are reserved.
_MEM_RATIO   = 0.25  # memory share of the variable budget
_CTX_RATIO   = 0.75  # context share of the variable budget

# Appended to every prompt's output-format block. Keeps inline [n] citations
# (needed to map the answer to source chips) but forbids the trailing
# "Sources:/Confidence:/Reasoning:/Answer:" labels and reasoning dumps the model
# otherwise leaks — so the user-facing prose stays clean.
_ANSWER_ONLY_RULE = (
    "\nIMPORTANT OUTPUT FORMAT RULES:\n"
    "1. Reply with ONLY your answer as plain prose.\n"
    "2. Cite facts with inline [n] numbers like [1] or [2,3]. "
    "NEVER write [n, 'text'] or quote any chunk content in citations.\n"
    "3. NEVER copy filenames, section titles, or text from the [n] (...) "
    "source labels into your answer.\n"
    "4. Do NOT add 'Sources:', 'Confidence:', 'Reasoning:', 'Answer:', or "
    "any other section labels.\n"
    "5. Do not restate these instructions in your reply.\n"
    "6. Do NOT show reasoning steps, working, or intermediate calculations. "
    "Do NOT reproduce context rows, table rows, or examples from the instructions. "
    "Start your answer DIRECTLY with the factual answer sentence.\n"
    "7. Do NOT output raw data rows (e.g. 'Date | Value | ...' or "
    "'Row: ...' or '→ FY20XX=...'). Report only the final answer value.\n"
)

# PROMPT INJECTION PATTERNS — consolidated into app/guardrails/policies.yaml (Phase 26)

# STRUCTURED KEYWORDS
_STRUCTURED_KEYWORDS = [
    "table", "row", "column", "page number",
    "which page", "toc", "section", "cell",
    "extract", "list all", "enumerate",
]

# FINANCIAL KEYWORDS — for detecting numeric-comparison / accounting queries.
# Two or more of these in the query triggers the financial-specific system prompt
# that includes arithmetic direction rules and a CoT worked example.
_FINANCIAL_EXACT_PHRASES = frozenset({
    "net sales", "net revenue", "gross revenue", "total revenue",
    "cash flow", "year over year", "year-over-year", "yoy", "fiscal year",
    "operating income", "cost of goods", "gross margin", "operating margin",
    "balance sheet", "earnings per share", "annual report", "compared to",
})
_FINANCIAL_TOKENS = frozenset({
    "revenue", "profit", "loss", "income", "earnings", "margin", "eps",
    "ebitda", "ebit", "dividend", "shares", "fiscal", "quarter", "fy",
    "compare", "versus", "vs", "increased", "decreased", "change", "growth",
    "decline", "assets", "liabilities", "equity", "capex", "depreciation",
    "amortization", "goodwill", "impairment", "acquisition", "segment",
    "guidance", "outlook", "forecast",
})

# MULTIMODAL KEYWORDS
_IMAGE_KEYWORDS = {"image", "photo", "diagram", "figure", "chart", "screenshot", "picture", "visual"}
_AUDIO_KEYWORDS = {"audio", "sound", "speech", "transcript", "recording", "voice", "spoken"}
_VIDEO_KEYWORDS = {"video", "clip", "footage", "scene", "frame", "watch", "recording"}

# CODE KEYWORDS
_CODE_KEYWORDS = {"code", "function", "class", "implement", "script", "snippet", "syntax", "debug", "algorithm"}

# COMPARATIVE KEYWORDS
_COMPARATIVE_KEYWORDS = {"compare", "difference", "vs", "versus", "contrast", "better", "worse", "pros", "cons"}

# TEMPORAL KEYWORDS
_TEMPORAL_KEYWORDS = {"when", "before", "after", "since", "latest", "recent", "current", "timeline", "history"}


# NORMALIZE TEXT

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# TRUNCATE WITH LIMIT

def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[:max(limit, 0)]


# SHA-256 HASH

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# PROMPT INJECTION DETECTION AND SANITIZATION — delegates to unified guardrail (Phase 26)

def _sanitize_query(query: str) -> Tuple[str, bool]:
    """Returns (sanitized_query, was_injection_detected)."""
    from app.guardrails.input_guard import sanitize as _guard_sanitize
    original = query
    query = _guard_sanitize(query, surface="prompt_builder")
    detected = query != original
    if detected:
        _injection_detections.inc()
        logger.warning("prompt_injection_detected", query_prefix=original[:80])
    return query, detected


# QUERY MODE DETECTION

def _is_structured(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _STRUCTURED_KEYWORDS)


def _is_code(query: str) -> bool:
    tokens = set(query.lower().split())
    return bool(tokens & _CODE_KEYWORDS)


def _is_comparative(query: str) -> bool:
    tokens = set(query.lower().split())
    return bool(tokens & _COMPARATIVE_KEYWORDS)


def _is_temporal(query: str) -> bool:
    tokens = set(query.lower().split())
    return bool(tokens & _TEMPORAL_KEYWORDS)


def _is_financial(query: str) -> bool:
    """True when the query is about financial metrics or numeric comparisons.

    Activates on any exact phrase match OR on 2+ financial token matches.
    The higher token bar prevents single-word false positives like "change" or
    "income" from triggering the financial prompt on unrelated queries.
    """
    q_lower = query.lower()
    if any(phrase in q_lower for phrase in _FINANCIAL_EXACT_PHRASES):
        return True
    tokens = set(q_lower.split())
    return len(tokens & _FINANCIAL_TOKENS) >= 2


def _detect_modality(query: str, context: str) -> Optional[str]:
    combined = (query + " " + context).lower()
    tokens   = set(combined.split())
    if tokens & _IMAGE_KEYWORDS:
        return "image"
    if tokens & _AUDIO_KEYWORDS:
        return "audio"
    if tokens & _VIDEO_KEYWORDS:
        return "video"
    return None


# DETECT QUERY TYPE FOR PROMPT ROUTING
# Financial is checked before comparative/temporal because many financial
# queries ("compare FY2023 to FY2022") would otherwise fall into "comparative"
# and miss the arithmetic direction rules.

def _detect_query_type(query: str) -> str:
    if _is_code(query):
        return "code"
    if _is_financial(query):
        return "financial"
    if _is_comparative(query):
        return "comparative"
    if _is_temporal(query):
        return "temporal"
    if _is_structured(query):
        return "structured"
    modality = _detect_modality(query, "")
    if modality:
        return modality
    return "general"


# QUERY-ADAPTIVE GENERATION TEMPERATURE
# Factual / extraction tasks need low temperature (deterministic, accurate).
# Generative / analytical tasks benefit from slightly higher temperature.

_FACTUAL_QUERY_TYPES    = frozenset({"financial", "structured", "code", "temporal"})
_GENERATIVE_QUERY_TYPES = frozenset({"comparative", "image", "audio", "video"})


def get_generation_temperature(query_type: str) -> float:
    """Return the optimal sampling temperature for the given query type."""
    if query_type in _FACTUAL_QUERY_TYPES:
        return settings.LLM_TEMPERATURE_FACTUAL
    if query_type in _GENERATIVE_QUERY_TYPES:
        return settings.LLM_TEMPERATURE_GENERATIVE
    return settings.LLM_TEMPERATURE


# DEDUP OVERLAP — REMOVE MEMORY CONTENT ALREADY IN CONTEXT

def _deduplicate_context(memory: str, context: str) -> Tuple[str, str]:
    if not memory or not context:
        return memory, context
    key = memory[:200].strip()
    if key and key in context:
        context = context.replace(key, "").strip()
    return memory, context


# PII SCRUB BEFORE PROMPT INJECTION

def _scrub_pii(text: str) -> str:
    if not settings.PII_DETECTION_ENABLED:
        return text
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        entities   = getattr(settings, "PII_ENTITIES", [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
            "US_SSN", "CREDIT_CARD", "LOCATION", "IP_ADDRESS",
        ])
        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results    = analyzer.analyze(text=text, entities=entities, language="en")
        if results:
            text = anonymizer.anonymize(text=text, analyzer_results=results).text
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pii_scrub_failed", error=str(exc))
    return text


# SYSTEM PROMPT SELECTION — QUERY TYPE AWARE

def _system_prompt(
    query_type: str,
    structured: bool,
    is_code: bool,
    modality: Optional[str],
) -> str:

    if structured:
        return (
            "You are a strict extraction system.\n"
            "RULES:\n"
            "- Use ONLY the provided context\n"
            "- Return the exact requested value\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- If sources disagree, surface the disagreement and cite each"
            " conflicting source rather than picking one silently\n"
            "- If a source is flagged with an error marker (e.g."
            " 'intentional error', 'does not exist', 'WRONG'), treat its"
            " specific claim as suspect and prefer the agreeing sources\n"
            "- No explanation or padding\n"
            "- Do NOT expand abbreviations or acronyms unless the context"
            " explicitly defines them\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n\n"
        )

    if is_code:
        return (
            "You are a precise code assistant.\n"
            "RULES:\n"
            "- Use ONLY the provided context\n"
            "- Return clean, working code\n"
            "- After each non-trivial claim, cite the source number in"
            " square brackets, e.g. [1] or [2,3]\n"
            "- If sources disagree, surface the disagreement and cite each"
            " conflicting source\n"
            "- If a source is flagged with an error marker, treat its"
            " claim as suspect\n"
            "- No hallucination. If the answer is not in the context,"
            " reply exactly: \"I could not find this in the provided sources.\"\n\n"
        )

    if query_type == "financial":
        return (
            "You are a precise financial analyst.\n"
            "Answer ONLY using the provided CONTEXT chunks. Each chunk is\n"
            "labelled [1], [2], [3]... — cite every fact with its number.\n"
            "\n"
            "EXACT FIGURES RULE (HIGHEST PRIORITY — non-negotiable):\n"
            "  Financial statements express figures IN MILLIONS.\n"
            "  A number like '383,285' in a table means $383,285 million — report it as '$383,285 million'.\n"
            "  NEVER round: e.g. '$383,285 million' must NOT become '$383.3 billion'.\n"
            "  NEVER round: e.g. '$96,995 million' must NOT become '$97.0 billion'.\n"
            "  If a table row has columns [FY2023 | FY2022 | FY2021], the FIRST number is FY2023.\n"
            "\n"
            "COLUMN ORDER RULE (critical for multi-year tables):\n"
            "  Tables in 10-K filings list years left-to-right: most-recent first.\n"
            "  Row: '[Metric] | [A] | [B] | [C]'  →  FY2023=[A], FY2022=[B], FY2021=[C]\n"
            "  Do NOT confuse columns. Report ONLY the year that was asked about.\n"
            "\n"
            "ROW IDENTIFICATION RULE (critical for EPS tables):\n"
            "  'Basic' and 'Diluted' are DIFFERENT rows — never swap them.\n"
            "  If asked for DILUTED EPS, read ONLY the 'Diluted' row. Ignore the 'Basic' row.\n"
            "  Example table:\n"
            "    Basic   | [X] | [Y] | [Z]    ← BASIC EPS rows\n"
            "    Diluted | [A] | [B] | [C]    ← DILUTED EPS rows\n"
            "  FY2022 diluted EPS = [B]  (NOT [Y] which is the Basic row)\n"
            "\n"
            "NO-CALCULATION RULE:\n"
            "  When the context provides an EXPLICIT dollar change (e.g., 'decreased $X billion'),\n"
            "  use that exact figure. NEVER calculate the dollar change by applying a percentage to\n"
            "  another number — percentages and base values in different rows create rounding errors.\n"
            "  CORRECT: use the verbatim '[Product] net sales decreased Y% or $Z billion'\n"
            "  WRONG:   calculate Y% of some base number and report a different amount\n"
            "\n"
            "PRODUCT SEGMENT DECLINE RULE (for 'largest decline' questions):\n"
            "  When a context chunk explicitly states '[Product] net sales decreased Y% or $Z', that IS\n"
            "  the authoritative answer. Do NOT compute. Do NOT estimate. Do NOT invent a figure.\n"
            "  Step 1 — Scan ALL context chunks for lines matching: '[Product] net sales decreased X% or $Y'\n"
            "  Step 2 — Compare those STATED percentages to find the largest decline.\n"
            "  Step 3 — Report the product, its stated % decline, AND its stated $ amount verbatim.\n"
            "  The $ amount MUST come from the explicit text ('or $Z billion') — NEVER compute it.\n"
            "\n"
            "ARITHMETIC RULE FOR COMPARISONS (ONLY when question explicitly asks to compare):\n"
            "  Apply this rule ONLY when the question uses words like 'compare', 'change',\n"
            "  'year-over-year', 'YoY', 'how did X compare', 'difference', 'more or less than'.\n"
            "  If the question asks for a SINGLE year's figure, skip this rule entirely.\n"
            "  When comparison IS asked:\n"
            "  Step 1 — State both values: 'FY2023: $X million, FY2022: $Y million'\n"
            "  Step 2 — Compute: delta = FY2023 − FY2022\n"
            "  Step 3 — Direction: delta > 0 → INCREASED; delta < 0 → DECREASED\n"
            "  Step 4 — State: 'decreased/increased by $|delta| million (X.X%)'\n"
            "  NEVER reverse the sign.\n"
            "\n"
            "ARITHMETIC EXAMPLE (do NOT copy into answer — internal reference only):\n"
            "  Table: '[Metric] | [A] | [B]'  →  FY2023=[A]M, FY2022=[B]M\n"
            "  delta = [A] − [B]. If negative → DECREASED; if positive → INCREASED.\n"
            "  State the delta as an absolute value with a direction word.\n"
            "\n"
            "COMPLETENESS RULE:\n"
            "  If the question asks 'how did X compare' or 'how did they compare', include BOTH\n"
            "  the current year AND the comparison year figures for ALL major line items.\n"
            "  For net sales with Products/Services breakdown, include BOTH breakdowns.\n"
            "  If asked about ALL product categories or declines, include ALL 5 categories:\n"
            "  Mac, iPhone, iPad, Wearables/Home/Accessories, AND Services.\n"
            "  For cash/liquidity questions, include: (1) balance sheet cash equivalents,\n"
            "  (2) total restricted cash if mentioned, (3) capital return info (buybacks+dividends)\n"
            "  IF those figures appear in the context.\n"
            "\n"
            "BALANCE SHEET CASH RULE:\n"
            "  'Cash and cash equivalents' (balance sheet line) ≠ 'cash, cash equivalents and\n"
            "  restricted cash' (cash flow statement line). These are DIFFERENT figures.\n"
            "  Balance sheet: 'Cash and cash equivalents | [BS_A] | [BS_B]'\n"
            "    → FY2023 balance sheet cash = $[BS_A]M\n"
            "    → FY2022 balance sheet cash = $[BS_B]M\n"
            "  Cash flow statement ending balance is a DIFFERENT line item — do NOT mix it.\n"
            "  Report ONLY the specific line item the question asks about.\n"
            "\n"
            "ANTI-HALLUCINATION RULE:\n"
            "  Report ONLY figures that appear verbatim in the context chunks.\n"
            "  Do NOT derive, infer, or calculate numbers not in the context.\n"
            "  Do NOT report beginning balances, net changes, or YoY comparisons unless the\n"
            "  question explicitly asks for them AND the figures are in the context.\n"
            "  If a context chunk contains multiple years' data, report ONLY the year asked about\n"
            "  unless the question specifically asks for comparison.\n"
            "\n"
            "SCOPE RULE (strictly enforce — violations lose points):\n"
            "  Answer EXACTLY what is asked — nothing more.\n"
            "  If the question asks for FY2023 figures only, do NOT volunteer FY2022 figures.\n"
            "  If the question asks for one balance sheet line, do NOT add YoY change calculations.\n"
            "  Adding UNREQUESTED prior-year comparisons is a hallucination error even if the\n"
            "  numbers seem plausible — they may be from the WRONG row (e.g. Basic vs Diluted EPS).\n"
            "  WRONG: Question asks 'diluted EPS for FY2023' → model says '$[X], decreased from $[Y]'\n"
            "    — volunteering FY2022 is wrong AND $[Y] may be the BASIC row, not DILUTED.\n"
            "  CORRECT: Answer only the specific value asked for.\n"
            "\n"
            "TIME-SERIES DATA RULE (for S&P 500 / XLSX queries):\n"
            "  If context contains a [COMPUTED SUMMARY] block, extract answers DIRECTLY from it.\n"
            "  IGNORE all raw table rows — use ONLY the [COMPUTED SUMMARY] values:\n"
            "  - 'closing value at end of year YYYY': read 'Year YYYY: close=X (YYYY-MM-DD)'\n"
            "  - 'highest value in dataset': read 'Overall maximum: X on DATE'\n"
            "  - 'average for year YYYY': read 'Year YYYY: avg=X'\n"
            "  - 'percentage change from start of YYYY1 to end of YYYY2': read\n"
            "    'Cross-period: start of YYYY1 (..., V1) to end of YYYY2 (..., V2) = PCT%'\n"
            "  Do NOT recalculate from raw rows. Do NOT use rounded figures like '3,840'.\n"
            "  Report exact 2-decimal figures as shown in the summary.\n"
            "  CRITICAL: NEVER output raw data rows (e.g. 'Date | Value | ...' lines) in your answer.\n"
            "  Your answer must be ONE clean sentence using the extracted value.\n"
            "\n"
            "ADDITIONAL RULES:\n"
            "1. Report figures EXACTLY as in the source — millions = millions, not billions.\n"
            "2. Never round a figure that appears as an exact integer in the source.\n"
            "3. Do NOT add unrequested calculations, ratios, or YoY comparisons.\n"
            "4. If the answer is not in the context, reply: 'I could not find this in the provided sources.'\n\n"
        )

    if query_type == "comparative":
        return (
            "You are an analytical assistant specializing in comparisons.\n"
            "RULES:\n"
            "- Use ONLY the provided context\n"
            "- Structure: describe A, describe B, then compare\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- Be objective and factual; if sources disagree, surface the"
            " disagreement and cite EVERY conflicting source — do not"
            " silently pick one\n"
            "- If a source is flagged with an error marker (e.g."
            " 'intentional error', 'does not exist', 'WRONG'), treat its"
            " specific claim as suspect and prefer the agreeing sources\n"
            "- Do NOT expand abbreviations or acronyms unless the context"
            " explicitly defines them\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n\n"
        )

    if query_type == "temporal":
        return (
            "You are a temporal reasoning assistant.\n"
            "RULES:\n"
            "- Use ONLY the provided context\n"
            "- Preserve chronological order; note time periods explicitly\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- If sources give different dates or time ranges for the same"
            " event, surface the disagreement and cite EVERY conflicting"
            " source — do not silently pick one date\n"
            "- If a source is flagged with an error marker, treat its"
            " specific claim as suspect and prefer the agreeing sources\n"
            "- Do NOT expand abbreviations or acronyms unless the context"
            " explicitly defines them\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n\n"
        )

    if modality == "image":
        return (
            "You are an expert in visual reasoning.\n"
            "RULES:\n"
            "- Use ONLY the provided visual context and captions\n"
            "- Describe visual content accurately\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- If sources disagree, surface the disagreement and cite each\n"
            "- If a source is flagged with an error marker, treat its"
            " claim as suspect\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n\n"
        )

    if modality == "audio":
        return (
            "You are an expert in audio and speech understanding.\n"
            "RULES:\n"
            "- Use ONLY the provided transcripts and context\n"
            "- Reference speaker labels if available\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- If sources disagree, surface the disagreement and cite each\n"
            "- If a source is flagged with an error marker, treat its"
            " claim as suspect\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n\n"
        )

    if modality == "video":
        return (
            "You are an expert in video understanding.\n"
            "RULES:\n"
            "- Use ONLY the provided frames, captions and transcripts\n"
            "- Reference timestamps if available\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- If sources disagree, surface the disagreement and cite each\n"
            "- If a source is flagged with an error marker, treat its"
            " claim as suspect\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n\n"
        )

    return (
        "You are a precise knowledge assistant.\n"
        "Answer ONLY using the provided CONTEXT chunks below. Each chunk is\n"
        "labelled with a number like [1], [2], [3] at the start. Some chunks\n"
        "may carry inline markers in their header such as\n"
        "  ⚠ ERROR_MARKERS=intentional error; does not exist\n"
        "These markers come from the source text itself and indicate that\n"
        "the specific claim in that chunk is suspect.\n"
        "Rules:\n"
        "1. Use ONLY information present in the provided context.\n"
        "2. After every fact in your answer, cite the source number in\n"
        "   square brackets, e.g. \"... is the answer [1]\" or \"both A and B\n"
        "   are valid [1,3]\" for multi-source claims.\n"
        "3. If sources disagree, surface the disagreement and cite EVERY\n"
        "   conflicting source — do not silently pick one. Phrase it as\n"
        "   \"Sources differ: X according to [1], Y according to [2]\".\n"
        "4. If a chunk carries an ERROR_MARKERS header, treat its specific\n"
        "   claim as suspect: prefer the agreeing sources, and if asked\n"
        "   directly about that claim, flag the error rather than repeating it.\n"
        "5. Never add information not present in the context. Do not use\n"
        "   prior knowledge.\n"
        "6. If the answer is not in the context, reply exactly:\n"
        '   "I could not find this in the provided sources."\n'
        "7. Be concise and direct.\n"
        "8. Do NOT expand abbreviations or acronyms unless the context\n"
        "   explicitly defines them. Use the term exactly as it appears.\n"
        "8b. MULTI-PERIOD FIGURES: Financial tables often list figures for\n"
        "   several years or quarters side by side. When the question names a\n"
        "   specific year or period, report ONLY the figure belonging to that\n"
        "   exact year/period column — never a neighbouring year's value.\n"
        "   Copy figures exactly as written; do not convert units.\n"
        "9. ACQUISITIONS: If the context uses words like 'acquired', 'acquisition',\n"
        "   'assumed liabilities', or any M&A language, that IS a corporate\n"
        "   acquisition. Report it as such. NEVER say 'no acquisition occurred'\n"
        "   or 'did not complete' if the context describes one.\n\n"
    )


# OUTPUT FORMAT SELECTION

def _output_format(
    structured: bool,
    is_code: bool,
    query_type: str,
) -> str:

    if structured:
        return "OUTPUT:\n<exact answer>"

    if is_code:
        return "OUTPUT:\n```\n<code here>\n```"

    if query_type == "financial":
        return (
            "Answer the question using ONLY the figures the question asks for. "
            "Do NOT add prior-year comparisons or YoY changes unless the question "
            "explicitly asks for comparison. Cite sources with [n] inline.\n"
        )

    if query_type == "comparative":
        return (
            "Write your answer below. Start with a complete sentence.\n"
            "Entity A: [description]\n"
            "Entity B: [description]\n"
            "Comparison: [key differences and similarities]"
        )

    if query_type == "temporal":
        return (
            "Write your answer below. Start with a complete sentence.\n"
            "Timeline: [chronological summary]\n"
            "Answer: [direct answer]"
        )

    return "Write your complete answer below. Begin with a full sentence:\n"


class PromptBuilder:

    def __init__(self) -> None:
        self.max_chars = settings.MAX_PROMPT_CHARS

    # MAIN BUILD PROMPT

    def build_prompt(
        self,
        query: str,
        context: str,
        memory: str = "",
        session_id: str = "default",
        scrub_pii: bool = True,
    ) -> str:

        start = time.time()

        with tracer.start_as_current_span("prompt_builder") as span:
            span.set_attribute("session.id", session_id)

            try:
                # CLEAN INPUTS
                query   = _clean(query)
                context = _clean(context)
                memory  = _clean(memory)

                if not query:
                    raise ValueError("EMPTY_QUERY")

                # INJECTION SANITIZATION
                query, was_injected = _sanitize_query(query)

                if not query:
                    raise ValueError("QUERY_FULLY_SANITIZED_INJECTION_DETECTED")

                span.set_attribute("injection.detected", was_injected)

                # PII SCRUB BEFORE PROMPT INJECTION
                if scrub_pii:
                    context = _scrub_pii(context)
                    memory  = _scrub_pii(memory)

                # QUERY TYPE AND MODALITY DETECTION
                query_type = _detect_query_type(query)
                structured = _is_structured(query)
                is_code    = _is_code(query)
                modality   = _detect_modality(query, context)

                span.set_attribute("query.type", query_type)
                span.set_attribute("query.modality", modality or "text")

                # NO-CONTEXT GUARD — do not build prompt when retrieval was empty
                if not context:
                    raise ValueError("EMPTY_CONTEXT_NO_DOCUMENTS_RETRIEVED")

                # DEDUP OVERLAP BETWEEN MEMORY AND CONTEXT
                memory, context = _deduplicate_context(memory, context)

                # SYSTEM PROMPT + FORMAT BLOCK (fixed overhead — computed first)
                system     = _system_prompt(query_type, structured, is_code, modality)
                output_fmt = _output_format(structured, is_code, query_type) + _ANSWER_ONLY_RULE

                # QUERY BLOCK — hard cap at 15% of total budget
                query_budget = int(self.max_chars * 0.15)
                query_text   = _truncate(query, query_budget)
                query_block  = (
                    f"TASK:\n{query_text}\n\n"
                    if structured
                    else f"QUERY:\n{query_text}\n\n"
                )

                # RESIDUAL BUDGET — everything left after fixed components.
                # This prevents the system prompt (which varies 400–900 chars)
                # from silently compressing the context window.
                fixed_len = len(system) + len(output_fmt) + len(query_block) + 100
                remaining = max(0, self.max_chars - fixed_len)

                mem_budget = int(remaining * _MEM_RATIO)
                ctx_budget = remaining - mem_budget

                memory  = _truncate(memory,  mem_budget)
                context = _truncate(context, ctx_budget)

                # BLOCK ASSEMBLY
                mem_block = f"MEMORY:\n{memory}\n\n" if memory else ""
                ctx_block = f"CONTEXT:\n{context}\n\n" if context else ""

                prompt = (
                    system
                    + mem_block
                    + ctx_block
                    + query_block
                    + output_fmt
                )

                # OVERFLOW GUARD — PRESERVE SYSTEM + QUERY + FORMAT
                if len(prompt) > self.max_chars:
                    fixed   = system + query_block + output_fmt
                    allowed = self.max_chars - len(fixed) - 20
                    middle  = _truncate(mem_block + ctx_block, allowed)
                    prompt  = system + middle + query_block + output_fmt

                    logger.warning(
                        "prompt_truncated",
                        original_size=len(
                            system + mem_block
                            + ctx_block + query_block + output_fmt
                        ),
                        final_size=len(prompt),
                        session_id=session_id,
                    )

                latency = round(time.time() - start, 3)

                _prompt_duration.labels(status="success").observe(latency)
                _prompt_length.labels(prompt_type=query_type).observe(len(prompt))

                span.set_attribute("prompt.length", len(prompt))
                span.set_attribute("mem.chars", len(memory))
                span.set_attribute("ctx.chars", len(context))
                span.set_attribute("query.chars", len(query))
                span.set_status(Status(StatusCode.OK))

                logger.debug(
                    "prompt_built",
                    size=len(prompt),
                    mem_chars=len(memory),
                    ctx_chars=len(context),
                    query_chars=len(query),
                    structured=structured,
                    is_code=is_code,
                    query_type=query_type,
                    modality=modality,
                    injection_detected=was_injected,
                    latency=latency,
                    session_id=session_id,
                )

                return prompt

            except Exception as exc:
                latency    = round(time.time() - start, 3)
                error_type = type(exc).__name__

                _prompt_duration.labels(status="error").observe(latency)
                _prompt_errors.labels(error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "prompt_build_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )
                raise

    # BATCH BUILD — BUILD PROMPTS FOR MULTIPLE QUERIES

    def build_batch(
        self,
        queries: List[str],
        context: str,
        memory: str = "",
        session_id: str = "default",
    ) -> List[str]:
        prompts: List[str] = []
        for q in queries:
            try:
                prompt = self.build_prompt(q, context, memory, session_id)
                prompts.append(prompt)
            except Exception as exc:
                logger.warning(
                    "prompt_batch_item_failed",
                    query_prefix=q[:50],
                    error=str(exc),
                    session_id=session_id,
                )
        return prompts

    # ASYNC BUILD

    async def build_prompt_async(
        self,
        query: str,
        context: str,
        memory: str = "",
        session_id: str = "default",
        scrub_pii: bool = True,
    ) -> str:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.build_prompt(query, context, memory, session_id, scrub_pii),
            )
