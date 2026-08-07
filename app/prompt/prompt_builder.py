import asyncio
import hashlib
import time
import unicodedata

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from app.core.config import settings
from app.llm.regeneration import REGENERATE_DIRECTIVE as _REGENERATE_DIRECTIVE

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMPT VERSION — bump on any meaningful change to the instruction text
# below (grounding/abstention rules, citation format, answer-format rules,
# etc.), not on pure refactors. This has no runtime effect on its own; it
# exists so app/eval/tracking/mlflow_logger.py can log it alongside every
# eval run's metrics — otherwise a prompt edit that shifts eval scores is
# indistinguishable from a genuine model/retrieval regression after the fact.
PROMPT_VERSION = "1.0.0"

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
_MEM_RATIO = 0.25  # memory share of the variable budget
_CTX_RATIO = 0.75  # context share of the variable budget

# Appended to every prompt's output-format block. Keeps inline [n] citations
# (needed to map the answer to source chips) but forbids the trailing
# "Sources:/Confidence:/Reasoning:/Answer:" labels and reasoning dumps the model
# otherwise leaks — so the user-facing prose stays clean.
_ANSWER_ONLY_RULE = (
    "\nGROUNDING & ABSTENTION (check this FIRST, before anything else):\n"
    "- The context may be about a DIFFERENT company, entity, person, time period, "
    "or topic than the question asks about. Before answering, confirm the context "
    "actually concerns the SPECIFIC subject named in the question.\n"
    "- If the question asks about an entity, company, person, fiscal period, or "
    "topic that does NOT appear in the context, do NOT substitute data from a "
    "different subject. Answer with exactly this one sentence and nothing else: "
    "\"No relevant information was found in your knowledge base to answer this "
    "question.\"\n"
    "- NEVER answer a question about one company, entity, or period using another "
    "company's, entity's, or period's data (e.g. do not answer a question about "
    "Microsoft using Apple's figures).\n"
    "- If the question assumes a fact that the context contradicts (a false "
    "premise), correct it using the context instead of accepting it.\n"
    "\nIMPORTANT ANSWER GUIDELINES:\n"
    "1. Reply with ONLY your answer as flowing prose paragraphs. "
    "Do NOT use numbered lists, bullet points, or section headings.\n"
    "2. Cite facts with inline [n] numbers like [1] or [2,3] immediately after the "
    "fact you are citing. NEVER write (Page X) or any page numbers inside your answer text. "
    "NEVER write [n, 'text'] or quote any chunk content in citations.\n"
    "3. NEVER copy filenames, section titles, chunk IDs, or page numbers from source labels "
    "into your answer text. Page references are shown separately in the source panel.\n"
    "4. Do NOT add 'Sources:', 'Confidence:', 'Reasoning:', 'Answer:', or separate "
    "section headings. Do NOT list source pages or any (Page X) references at the end of "
    "your answer.\n"
    "5. Do not restate these instructions in your reply.\n"
    "6. Do NOT show reasoning steps, working, or intermediate calculations. "
    "Do NOT reproduce context rows, table rows, or examples from the instructions. "
    "Start your answer DIRECTLY with the factual answer sentence.\n"
    "7. Do NOT output raw data rows (e.g. 'Date | Value | ...' or "
    "'Row: ...' or '→ FY20XX=...'). Report only the final answer value.\n"
    "8. Answer the question fully — include EVERY figure the question asks for "
    "(e.g. all requested segments and both fiscal years) — but do NOT add "
    "tangential details the question did not request (e.g. liquidity or cash "
    "balances when asked about tax). Be complete on-topic, not broad.\n"
)

# PROMPT INJECTION PATTERNS — consolidated into app/guardrails/policies.yaml (Phase 26)

# STRUCTURED KEYWORDS
_STRUCTURED_KEYWORDS = [
    "table",
    "row",
    "column",
    "page number",
    "which page",
    "toc",
    "section",
    "cell",
    "extract",
    "list all",
    "enumerate",
]

# FINANCIAL KEYWORDS — for detecting numeric-comparison / accounting queries.
# Two or more of these in the query triggers the financial-specific system prompt
# that includes arithmetic direction rules and a CoT worked example.
_FINANCIAL_EXACT_PHRASES = frozenset(
    {
        "net sales",
        "net revenue",
        "gross revenue",
        "total revenue",
        "cash flow",
        "year over year",
        "year-over-year",
        "yoy",
        "fiscal year",
        "operating income",
        "cost of goods",
        "gross margin",
        "operating margin",
        "balance sheet",
        "earnings per share",
        "annual report",
        "compared to",
    }
)
_FINANCIAL_TOKENS = frozenset(
    {
        "revenue",
        "profit",
        "loss",
        "income",
        "earnings",
        "margin",
        "eps",
        "ebitda",
        "ebit",
        "dividend",
        "shares",
        "fiscal",
        "quarter",
        "fy",
        "compare",
        "versus",
        "vs",
        "increased",
        "decreased",
        "change",
        "growth",
        "decline",
        "assets",
        "liabilities",
        "equity",
        "capex",
        "depreciation",
        "amortization",
        "goodwill",
        "impairment",
        "acquisition",
        "segment",
        "guidance",
        "outlook",
        "forecast",
        # Investment-research vocabulary — a question like "what are the three
        # pillars of the investment thesis" has none of the above tokens, so it
        # fell through to the "general" catch-all template ("one or two
        # sentences"), which this small model does not reliably follow — observed
        # it ignoring the instruction and dumping an unfocused numbered list of
        # every fact in context instead. The "financial" template's explicit
        # RULES block is what actually keeps this model grounded and complete.
        "thesis",
        "pillars",
        "pillar",
        "rating",
        "recommendation",
        "upside",
        "downside",
        "valuation",
        "consensus",
        "analyst",
        "target",
        "risk",
        "risks",
        "exposure",
        "concentration",
        "headwind",
        "tailwind",
    }
)

# MULTIMODAL KEYWORDS
_IMAGE_KEYWORDS = {
    "image",
    "photo",
    "diagram",
    "figure",
    "chart",
    "screenshot",
    "picture",
    "visual",
}
_AUDIO_KEYWORDS = {"audio", "sound", "speech", "transcript", "recording", "voice", "spoken"}
_VIDEO_KEYWORDS = {"video", "clip", "footage", "scene", "frame", "watch", "recording"}

# CODE KEYWORDS
_CODE_KEYWORDS = {
    "code",
    "function",
    "class",
    "implement",
    "script",
    "snippet",
    "syntax",
    "debug",
    "algorithm",
}

# COMPARATIVE KEYWORDS
_COMPARATIVE_KEYWORDS = {
    "compare",
    "difference",
    "vs",
    "versus",
    "contrast",
    "better",
    "worse",
    "pros",
    "cons",
}

# TEMPORAL KEYWORDS
_TEMPORAL_KEYWORDS = {
    "when",
    "before",
    "after",
    "since",
    "latest",
    "recent",
    "current",
    "timeline",
    "history",
}


# NORMALIZE TEXT


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# TRUNCATE WITH LIMIT


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[: max(limit, 0)]


# SHA-256 HASH


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# PROMPT INJECTION DETECTION AND SANITIZATION — delegates to unified guardrail (Phase 26)


def _sanitize_query(query: str) -> tuple[str, bool]:
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
    if bool(tokens & _COMPARATIVE_KEYWORDS):
        return True
    q_lower = query.lower()
    # Tax rate and state aid questions are inherently comparative (need YoY context).
    if "effective tax rate" in q_lower or ("state aid" in q_lower and "tax" in q_lower):
        return True
    return False


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


def _detect_modality(query: str, context: str) -> str | None:
    # DOCX chunks are cited by section (e.g. "[report.docx 4.1 DCF Model Key
    # Assumptions]"), never by page — checked first so the financial/comparative
    # templates below can swap their PDF-style "(Page N)" citation instruction
    # for a section-based one without touching the PDF-serving branches at all.
    if ".docx" in context.lower():
        return "docx"
    # Plain-text/transcript sources (FOMC press conferences, earnings-call txt
    # uploads) are speaker-turn documents with no page/section locator — cited
    # as "[source — SPEAKER]" rather than "(Page N)". Checked before the
    # image/audio/video keyword scan so a transcript that happens to mention
    # "video" or "call" isn't misrouted to the AV prompt branches.
    if any(ext in context.lower() for ext in (".txt", ".md", ".rst", ".csv", ".log")):
        return "text"
    combined = (query + " " + context).lower()
    tokens = set(combined.split())
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
    # Tax rate and state-aid impact questions need YoY comparison before financial routing.
    _ql = query.lower()
    if "effective tax rate" in _ql or ("state aid" in _ql and "tax" in _ql):
        return "comparative"
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

_FACTUAL_QUERY_TYPES = frozenset({"financial", "structured", "code", "temporal"})
_GENERATIVE_QUERY_TYPES = frozenset({"comparative", "image", "audio", "video"})


def get_generation_temperature(query_type: str) -> float:
    """Return the optimal sampling temperature for the given query type."""
    if query_type in _FACTUAL_QUERY_TYPES:
        return settings.LLM_TEMPERATURE_FACTUAL
    if query_type in _GENERATIVE_QUERY_TYPES:
        return settings.LLM_TEMPERATURE_GENERATIVE
    return settings.LLM_TEMPERATURE


# DEDUP OVERLAP — REMOVE MEMORY CONTENT ALREADY IN CONTEXT


def _deduplicate_context(memory: str, context: str) -> tuple[str, str]:
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

        entities = getattr(
            settings,
            "PII_ENTITIES",
            [
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "CREDIT_CARD",
                "LOCATION",
                "IP_ADDRESS",
            ],
        )
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text, entities=entities, language="en")
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
    modality: str | None,
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

    # DOCX sources are cited by section, not page — a separate branch so the
    # PDF-serving "financial"/"comparative" templates below stay byte-for-byte
    # unchanged (PDF citations must keep using "(Page N)").
    if modality == "docx" and query_type in ("financial", "comparative"):
        return (
            "You are a precise financial analyst. Answer the user's QUERY using "
            "ONLY the CONTEXT chunks.\n"
            "\n"
            "WHAT TO INCLUDE:\n"
            "- Answer the SPECIFIC question asked, and answer it COMPLETELY. If "
            "the question asks for a rating, a price target and the upside, give "
            "all three. If it asks for base, bull and bear cases, give all "
            "three with their numbers. If it asks about one specific risk (e.g. "
            "China revenue concentration), answer about THAT risk only.\n"
            "- The context also contains chunks about OTHER metrics the question "
            "did not ask about (other risk factors, segment tables, comps, EPS, "
            "balance-sheet lines). Ignore them. A chunk being present does not "
            "make it relevant. Do not list extra figures the question did not "
            "ask for.\n"
            "- Use only numbers that appear verbatim in the context. Never "
            "calculate, infer, or recall from memory. A table row such as "
            "\"Implied Price Target | $245.00 | $285.00 | $190.00\" gives the "
            "base, bull and bear price targets; \"WACC | 8.5% | 8.0% | 9.5%\" "
            "gives the base, bull and bear WACC.\n"
            "- If two sources give conflicting values, report BOTH and flag: "
            "Conflicting data.\n"
            "\n"
            "HOW TO WRITE IT:\n"
            "- Write flowing prose in complete sentences — one short paragraph. "
            "NO bullet points, NO dashes, NO numbered lists (\"1. 2. 3.\"), NO "
            "\"Key Metrics:\" heading. Write it the way an analyst would explain "
            "it out loud.\n"
            "- Do NOT put any bracket tags, citations, page numbers or section "
            "numbers in the text — sources are shown separately below.\n"
            "- Do NOT write labels like \"First sentence:\", \"Answer:\", or "
            "\"In response to your query\". Do NOT restate these instructions. "
            "Just write the answer directly.\n\n"
        )

    # TXT / plain-text transcripts (FOMC press conferences, earnings-call txt
    # uploads) are speaker-turn documents with no page/section locator — the
    # citation is "[source — SPEAKER]" instead of "(Page N)". Checked at the
    # same priority as the docx branch above (before the generic
    # financial/comparative/temporal templates) so every query type on a TXT
    # source gets speaker-attribution guidance, without touching the
    # PDF/DOCX/image/audio/video branches.
    if modality == "text":
        return (
            "You are a precise financial analyst reading a plain-text "
            "transcript (e.g. an FOMC press conference or earnings call). "
            "Answer ONLY from the CONTEXT chunks, in 2-4 plain sentences of "
            "flowing prose — never a numbered or lettered list, never "
            "\"Answer 1 / Answer 2\", never section headers.\n"
            "If the context begins with a \"KEY COMPARISON FACTS\" block, you "
            "MUST incorporate those exact facts into your answer — they are "
            "the comparison the question is asking for.\n"
            "Answer in AT MOST 3 sentences. Include ONLY the specific fact the "
            "question asks for. Do NOT add related-but-unasked figures (e.g. if "
            "asked about the rate cut, do NOT also state the inflation "
            "projections or unemployment rate; if asked about projections, do "
            "NOT also describe the labor market) even though they appear in the "
            "context.\n"
            "When the question asks what someone SAID or their RESPONSE, quote "
            "or closely restate the specific substantive points from the "
            "context — name the specific law, body, or authority they cited "
            "(use its exact name, e.g. \"the Federal Reserve Act\", not a "
            "vague paraphrase like \"current law\"), what they will or will "
            "not do, and who they said the matter is for to decide (e.g. that "
            "it is for Congress to consider). Do not omit a named authority or "
            "caveat that appears in the context.\n"
            "Do NOT end with a sentence that restates or summarizes what you "
            "already said (no \"Therefore, ...\", no \"In summary, ...\"). "
            "Stop as soon as the question is fully answered.\n"
            "EXAMPLE of a question with a comparison clause — \"What was the "
            "Q3 figure, and how did it compare to Q2?\" — a correct answer "
            "states BOTH parts: \"The Q3 figure was $50M. This was up from "
            "Q2's $45M.\" A one-sentence answer that only gives the Q3 "
            "figure and stops is INCOMPLETE and wrong, even if that "
            "sentence is itself accurate — the same applies whenever the "
            "question says \"and how did/does X compare to Y\", \"versus\", "
            "or asks for two time periods.\n"
            "CRITICAL — if the context states the comparison in different "
            "terms than the question expects, report it in THOSE terms; "
            "never invent a matching-format number to fill the gap. "
            "Example: the question asks for Q2's dollar figure, but the "
            "context only says \"Q2 was $10M lower than Q3\" or \"the count "
            "was higher in Q2\" — report exactly that sentence (the "
            "difference or the qualitative comparison), and do NOT compute "
            "or state a specific Q2 dollar figure, since that number does "
            "not appear verbatim anywhere in the context.\n"
            "Use only numbers and quotes that appear verbatim in the "
            "context; never calculate or recall a figure from memory. Basis "
            "points, percentage points, and percentages are three different "
            "units — never substitute one for another. If the question asks "
            "you to compare two periods or two figures, state the "
            "comparison itself as its own sentence, not just the two raw "
            "numbers. Never combine a figure from one part of the context "
            "with a figure from an unrelated part into one new number or "
            "range — report each verbatim, separately. Attribute a "
            "statement only to the speaker whose name introduces the exact "
            "sentence it came from. Answer ONLY what the question asks. The "
            "context contains many other speakers and topics that are not "
            "relevant to this question — do not mention, name, summarize, or "
            "list any speaker, journalist, or question that is not the one "
            "being asked about, even if they appear in the same context. Stop "
            "as soon as the question is answered; do not keep describing the "
            "rest of the meeting. Do not quote more than one short phrase. Do not repeat "
            "the source file name, speaker names, or a tag list anywhere in "
            "your answer — the source is already shown to the reader "
            "separately. Do not invent a title, date, or citation that is "
            "not in the context. If the context does not contain the "
            "answer, reply exactly: \"I could not find this in the "
            "provided sources.\"\n\n"
        )

    if query_type == "financial":
        return (
            "You are a precise financial analyst. Answer ONLY from the CONTEXT "
            "chunks, each labelled [1], [2], [3].\n"
            "\n"
            "RULES:\n"
            "- Use only figures that appear verbatim in the context. Never "
            "calculate, infer, or recall a number from memory.\n"
            "- Report each figure exactly as written, with its unit (a value of "
            "383,285 in a millions table is \"$383,285 million\").\n"
            "- 10-K tables list years newest-first, left to right. Report only "
            "the year the question asks about.\n"
            "- Basic and Diluted EPS are different rows — use the one asked for.\n"
            "- When the context states an explicit change (e.g. \"decreased 27% "
            "or $10.8 billion\"), quote that figure; never recompute it.\n"
            "- If the context has a line beginning [COMPUTED SUMMARY] or [AUDIO "
            "SUMMARY], take the answer from that line and ignore raw table rows.\n"
            "- Answer only what is asked. Do not add prior-year figures unless "
            "the question asks to compare.\n"
            "- If two sources give conflicting values, report BOTH with citations "
            "and flag: ⚠ Conflicting data.\n"
            "- If the figure is not in the context, reply exactly: \"I could not "
            "find this in the provided sources.\"\n"
            "\n"
            "FINANCE ARITHMETIC RULES:\n"
            "- Change % = (current − prior) / |prior| × 100; never round intermediate values.\n"
            "- Convert to common units before arithmetic; report in source units.\n"
            "- Diluted EPS unless explicitly asked for basic.\n"
            "- 'Margin' = gross margin unless prefixed operating/net/EBITDA.\n"
            "- 'Growth' = YoY unless QoQ specified.\n"
            "- If context states the change explicitly, quote it; never recompute.\n"
            "\n"
            "CITATION RULES:\n"
            "- After every figure or fact, add the source citation [n] inline.\n"
            "- Also state the page number explicitly: e.g. \"(Page 26)\" or "
            "\"per p.28\". Use the page shown in the context label (p.NN).\n"
            "- If multiple pages are cited, list all: \"(Pages 28-29)\".\n"
            "\n"
            "ANSWER FORMAT:\n"
            "- State ALL key metrics, numbers, and segment breakdowns found in context.\n"
            "- For multi-part questions (revenue by category, margin by segment), list "
            "each component on its own line with the [n] citation.\n"
            "- Do NOT truncate after one sentence — cover every sub-question asked.\n"
            "- Do NOT restate these rules or show your reasoning.\n\n"
        )

    if query_type == "comparative":
        return (
            "You are a precise financial analyst specializing in comparisons.\n"
            "CRITICAL RULES — FOLLOW EXACTLY:\n"
            "- Use ONLY numbers and facts that appear verbatim in the provided context chunks.\n"
            "  Do NOT use your training knowledge. If a number is not in the context, do not cite it.\n"
            "- SEGMENT BREAKDOWN RULE: If the question asks for a breakdown BY SEGMENT (Products vs Services,"
            " by category, etc.), you MUST list EACH segment separately — a total alone is INCOMPLETE.\n"
            "  Example: 'Products gross margin: $109,633M (37.2%) (Page 27). Services gross margin:"
            " $71,050M (73.9%) (Page 27). Total gross margin: $180,683M (46.2%) (Page 27).'\n"
            "- PERCENTAGES RULE: When context contains percentage metrics alongside dollar amounts, report"
            " BOTH for each segment. Never report only the total dollar amount for a margin question.\n"
            "- COMPARISON RULE: When the question asks FY2024 versus FY2023, report BOTH fiscal years for"
            " every line item listed. List each category with its FY2024 figure then FY2023 figure.\n"
            "- ANNUAL vs QUARTERLY RULE: Always report annual FY totals first. If context says 'During 2024,"
            " the Company repurchased $95.0 billion and paid dividends of $15.2 billion', report those"
            " annual figures (Page N) before any quarterly breakdown.\n"
            "- PAGE NUMBERS ARE MANDATORY: After EVERY fact, write the page in format: (Page N).\n"
            "  Use the page shown in the context header label like 'apple_10k.pdf p.27'.\n"
            "- COMPLETENESS: Cover EVERY part of the question. Do NOT truncate after the first metric.\n"
            "- 'Growth' = YoY unless QoQ is specified; 'Margin' = gross unless qualified.\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n"
            "- Do NOT restate these rules. Do NOT truncate after one sentence.\n\n"
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

    if modality in ("audio", "mp3"):
        return (
            "You are an expert in earnings call transcripts and speech understanding.\n"
            "RULES:\n"
            "- Use ONLY the provided transcripts and context\n"
            "- Attribute every statement to its speaker using this format: "
            "'[CFO, Q3 2024 earnings call]: ...' or '[CEO]: ...'\n"
            "- Reference speaker labels and timestamps if available\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- If sources disagree, surface the disagreement and flag: ⚠ Conflicting data\n"
            "- If a source is flagged with an error marker, treat its"
            " claim as suspect\n"
            "- If the answer is not in the context, reply exactly:"
            " \"I could not find this in the provided sources.\"\n\n"
        )

    if modality in ("video", "mp4"):
        return (
            "You are an expert in financial presentation video understanding.\n"
            "RULES:\n"
            "- Use ONLY the provided frames, captions and transcripts\n"
            "- Attribute statements to their speaker and slide number when visible\n"
            "- Reference timestamps if available (e.g. 'at 28:30, the CFO stated...')\n"
            "- After each fact, cite the source number in square brackets,"
            " e.g. [1] or [2,3] for multi-source claims\n"
            "- If sources disagree, surface the disagreement and flag: ⚠ Conflicting data\n"
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

    if is_code:
        return "OUTPUT:\n```\n<code here>\n```"

    if structured:
        return (
            "Reply with only the exact answer requested, as plain text. "
            "Do NOT add labels, headings, or reasoning.\n"
        )

    if query_type == "financial":
        return (
            "Answer the question using ONLY the figures the question asks for. "
            "Do NOT add prior-year comparisons or YoY changes unless the question "
            "explicitly asks for comparison. Cite sources with [n] inline.\n"
        )

    if query_type == "comparative":
        return (
            "Write one clean paragraph that describes each item and then states "
            "the key differences and similarities. Cite sources with [n] inline. "
            "Do NOT use 'Entity A:', 'Comparison:', or any other section labels.\n"
        )

    if query_type == "temporal":
        return (
            "Write one clean answer that keeps events in chronological order. "
            "Cite sources with [n] inline. Do NOT use 'Timeline:' or 'Answer:' "
            "labels.\n"
        )

    # Default — covers image / audio / video / general modalities.
    return (
        "Write a clean, direct answer in one or two sentences using only the "
        "context. Cite sources with [n] inline. Do NOT show reasoning, restate "
        "these rules, or add section labels.\n"
    )


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
        regenerate: bool = False,
    ) -> str:

        start = time.time()

        with tracer.start_as_current_span("prompt_builder") as span:
            span.set_attribute("session.id", session_id)

            try:
                # CLEAN INPUTS
                query = _clean(query)
                context = _clean(context)
                memory = _clean(memory)

                if not query:
                    raise ValueError("EMPTY_QUERY")

                # INJECTION SANITIZATION
                query, was_injected = _sanitize_query(query)

                if not query:
                    raise ValueError("QUERY_FULLY_SANITIZED_INJECTION_DETECTED")

                span.set_attribute("injection.detected", was_injected)

                # MODALITY DETECTION — on the pre-scrub context. Presidio's URL
                # recognizer false-positives on "*.docx" filenames (".do" is a
                # valid ccTLD), mangling them into "report_<URL>cx" — after PII
                # scrub the ".docx" substring this check looks for is gone.
                modality_context = context

                # PII SCRUB BEFORE PROMPT INJECTION
                if scrub_pii:
                    context = _scrub_pii(context)
                    memory = _scrub_pii(memory)

                # QUERY TYPE AND MODALITY DETECTION
                query_type = _detect_query_type(query)
                structured = _is_structured(query)
                is_code = _is_code(query)
                modality = _detect_modality(query, modality_context)

                span.set_attribute("query.type", query_type)
                span.set_attribute("query.modality", modality or "text")

                # NO-CONTEXT GUARD — do not build prompt when retrieval was empty
                if not context:
                    raise ValueError("EMPTY_CONTEXT_NO_DOCUMENTS_RETRIEVED")

                # DEDUP OVERLAP BETWEEN MEMORY AND CONTEXT
                memory, context = _deduplicate_context(memory, context)

                # SYSTEM PROMPT + FORMAT BLOCK (fixed overhead — computed first)
                system = _system_prompt(query_type, structured, is_code, modality)
                output_fmt = _output_format(structured, is_code, query_type) + _ANSWER_ONLY_RULE

                # QUERY BLOCK — hard cap at 15% of total budget
                query_budget = int(self.max_chars * 0.15)
                query_text = _truncate(query, query_budget)
                query_block = (
                    f"TASK:\n{query_text}\n\n" if structured else f"QUERY:\n{query_text}\n\n"
                )

                # REGENERATE BLOCK — only present when the user explicitly
                # asked for a different answer. Sits between the context and
                # the query (an instruction about how to answer THIS query),
                # never after the answer cue in output_fmt, where the model
                # would continue the directive text instead of obeying it.
                regen_block = _REGENERATE_DIRECTIVE if regenerate else ""

                # RESIDUAL BUDGET — everything left after fixed components.
                # This prevents the system prompt (which varies 400–900 chars)
                # from silently compressing the context window.
                fixed_len = (
                    len(system) + len(output_fmt) + len(query_block) + len(regen_block) + 100
                )
                remaining = max(0, self.max_chars - fixed_len)

                mem_budget = int(remaining * _MEM_RATIO)
                ctx_budget = remaining - mem_budget

                memory = _truncate(memory, mem_budget)
                context = _truncate(context, ctx_budget)

                # BLOCK ASSEMBLY
                mem_block = f"MEMORY:\n{memory}\n\n" if memory else ""
                ctx_block = f"CONTEXT:\n{context}\n\n" if context else ""

                prompt = system + mem_block + ctx_block + regen_block + query_block + output_fmt

                # OVERFLOW GUARD — PRESERVE SYSTEM + QUERY + FORMAT
                if len(prompt) > self.max_chars:
                    fixed = system + regen_block + query_block + output_fmt
                    allowed = self.max_chars - len(fixed) - 20
                    middle = _truncate(mem_block + ctx_block, allowed)
                    prompt = system + middle + regen_block + query_block + output_fmt

                    logger.warning(
                        "prompt_truncated",
                        original_size=len(
                            system + mem_block + ctx_block + regen_block + query_block + output_fmt
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
                latency = round(time.time() - start, 3)
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
        queries: list[str],
        context: str,
        memory: str = "",
        session_id: str = "default",
    ) -> list[str]:
        prompts: list[str] = []
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
        regenerate: bool = False,
    ) -> str:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.build_prompt(
                    query, context, memory, session_id, scrub_pii, regenerate
                ),
            )
