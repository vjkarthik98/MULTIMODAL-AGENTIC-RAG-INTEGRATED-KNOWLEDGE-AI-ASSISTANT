import asyncio
import hashlib
import math
import re
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_reasoning_duration = Histogram(
    "reasoning_engine_duration_seconds",
    "Reasoning engine duration",
    ["status"],
)
_reasoning_errors = Counter(
    "reasoning_engine_errors_total",
    "Reasoning engine errors by type",
    ["error_type"],
)
_hallucination_flags = Counter(
    "reasoning_hallucination_flags_total",
    "Answers flagged as potentially hallucinated",
)
_llm_call_duration = Histogram(
    "llm_call_latency_seconds",
    "LLM call latency",
    ["model"],
)

# SEMAPHORE — lazy init to avoid missing event loop at import time
_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_lock = threading.Lock()


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        with _semaphore_lock:
            if _semaphore is None:
                _semaphore = asyncio.Semaphore(5)
    return _semaphore

# REASONING STEP TYPES
STEP_RETRIEVE  = "retrieve"
STEP_REASON    = "reason"
STEP_VERIFY    = "verify"
STEP_SYNTHESIZE = "synthesize"


# SHA-256 HASH FOR DEDUP

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# NORMALIZE TEXT

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# CITATION TAG HANDLING
# Tags look like [foo.pdf p.4], [bar.mp4 t=01:23], [baz.xlsx sheet=Q1], [qux.docx].
# The closed-set guarantee is enforced by SUBSTRING scanning the answer text for
# each known cite_key — we never accept a tag that wasn't built by build_sources().

# Used only by the hallucination guard to strip bracketed annotations before
# scoring. Conservative: only short [..] groups (citation-shaped).
_CITE_TAG_STRIP_RE = re.compile(r"\[[^\[\]\n]{1,160}\]")


def _strip_cite_tags(text: str) -> str:
    if not text:
        return ""
    return _CITE_TAG_STRIP_RE.sub(" ", text)


def _extract_cite_tags(text: str, valid: List[str]) -> List[str]:
    """Return the cite_keys from `valid` that actually appear as substrings of `text`.
    Closed-set: a tag the LLM invented (not in `valid`) cannot pass."""
    if not text or not valid:
        return []
    out: List[str] = []
    seen: set = set()
    # Preserve order in which keys first appear in the answer.
    spans: List[Tuple[int, str]] = []
    for key in valid:
        if not key or key in seen:
            continue
        idx = text.find(key)
        if idx >= 0:
            spans.append((idx, key))
            seen.add(key)
    spans.sort(key=lambda t: t[0])
    for _, key in spans:
        out.append(key)
    return out


# HALLUCINATION GUARD
# CROSS-CHECKS ANSWER AGAINST RETRIEVED CHUNKS
# FLAGS CLAIMS NOT SUPPORTED BY ANY CHUNK

def _hallucination_guard(
    answer: str,
    docs: List[Dict],
    threshold: float = None,
) -> Tuple[bool, float]:
    """
    RETURNS (IS_HALLUCINATED, SUPPORT_SCORE).
    SUPPORT_SCORE = FRACTION OF ANSWER SENTENCES SUPPORTED BY DOCS.
    LOW SUPPORT → POTENTIAL HALLUCINATION.
    """
    if not answer or not docs:
        return False, 1.0

    # Default to a looser global threshold (paraphrases lose surface overlap).
    # Caller may override; settings.HALLUCINATION_THRESHOLD still wins if set lower.
    thr = threshold
    if thr is None:
        thr = float(getattr(settings, "HALLUCINATION_THRESHOLD", 0.4) or 0.4)

    # STRIP CITATION TAGS BEFORE SCORING — they bias overlap and aren't claims.
    cleaned = _strip_cite_tags(answer)

    # COLLECT ALL DOC TEXT
    all_doc_text = " ".join(
        str(d.get("text", "") or "").lower()
        for d in docs
        if d.get("text")
    )

    if not all_doc_text.strip():
        return False, 1.0

    # SENTENCE-LEVEL SUPPORT CHECK
    sentences = [
        s.strip()
        for s in cleaned.replace("!", ".").replace("?", ".").split(".")
        if len(s.strip()) > 20
    ]

    if not sentences:
        return False, 1.0

    # Per-sentence threshold lowered to 0.25 (paraphrases keep ~25-40% of >4-char tokens).
    per_sentence_thr = 0.25

    supported = 0
    for sentence in sentences:
        words = [w.lower() for w in sentence.split() if len(w) > 4]
        if not words:
            supported += 1
            continue
        # CHECK IF SUFFICIENT FRACTION OF SIGNIFICANT WORDS APPEAR IN DOCS
        hits = sum(1 for w in words if w in all_doc_text)
        if hits / max(len(words), 1) >= per_sentence_thr:
            supported += 1

    support_score  = supported / max(len(sentences), 1)
    is_hallucinated = support_score < thr

    return is_hallucinated, round(support_score, 3)


# NUMERIC FAITHFULNESS CHECK
# Extracts numeric tokens from the answer and verifies each one appears in
# the retrieved knowledge. Catches the case where the LLM substitutes a
# parametric-memory number (e.g. "33.9%") for a context number ("31.4%").

# Matches integers, decimals, percentages, currency-style numbers.
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def _extract_numbers(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0) for m in _NUM_RE.finditer(text)]


def _number_in_context(num: str, context: str) -> bool:
    """A number is supported if it appears verbatim in context, OR if its
    bare-digit form does (so '31.4%' is supported by '31.4' in the text).
    Also handles thousands-separator format differences: '57.53' matches
    '57,530' or '57530' since financial docs mix billions/millions."""
    if not num or not context:
        return False
    if num in context:
        return True
    bare = num.rstrip("%")
    if bare and bare != num and bare in context:
        return True
    # Handle thousands separator: "57,530" vs "57530"
    no_comma = num.replace(',', '')
    if no_comma != num and no_comma in context:
        return True
    # Financial scale mismatch: "57.53" (billion) vs "57,530" or "57530" (million).
    # If the number has a decimal, check whether the integer-scaled form appears.
    if '.' in num:
        parts = num.split('.')
        if len(parts) == 2 and parts[1].isdigit() and parts[0].isdigit():
            # e.g. "57.53" → try "57,530", "5753", "57530"
            scaled = parts[0] + parts[1]           # "5753"
            if scaled in context or scaled.replace('', ',') in context:
                return True
            scaled_padded = parts[0] + parts[1].ljust(3, '0')  # "57530"
            if scaled_padded in context:
                return True
            # Also check with comma: "57,530"
            if len(parts[1]) <= 2:
                with_comma = parts[0] + ',' + parts[1].ljust(3, '0')
                if with_comma in context:
                    return True
    return False


def _unsupported_numbers(
    answer: str,
    docs: List[Dict],
    query: str = "",
) -> List[str]:
    """Return numeric tokens in `answer` that do NOT appear in any doc.

    Numbers excluded from the guard (they are not financial claims):
    - 4-digit years in the range 1900-2099
    - Calendar day numbers (1-31) when they are short integers
    - Any number that also appears verbatim in the query (the LLM just
      echoed a date/year the user provided — not a fabricated figure)
    """
    nums = _extract_numbers(_strip_cite_tags(answer))
    if not nums:
        return []
    context = " ".join(str(d.get("text", "") or "") for d in docs)
    query_nums = set(_extract_numbers(query)) if query else set()

    unsupported: List[str] = []
    seen: set = set()
    for n in nums:
        if n in seen:
            continue
        seen.add(n)

        # Skip if the number appeared in the user's query (date echo-back)
        if n in query_nums:
            continue

        # Skip 4-digit years (1900–2099) — not financial data
        if re.fullmatch(r'(?:19|20)\d{2}', n):
            continue

        # Skip small calendar integers (1–31) without decimals — likely dates
        bare_n = n.replace(',', '')
        if bare_n.isdigit() and 1 <= int(bare_n) <= 31 and '.' not in n:
            continue

        if not _number_in_context(n, context):
            unsupported.append(n)
    return unsupported


# PII SCRUB ANSWER BEFORE RETURNING

def _scrub_answer_pii(text: str) -> str:
    if not settings.PII_DETECTION_ENABLED:
        return text
    try:
        from app.guardrails.pii import _get_engines
        analyzer, anonymizer = _get_engines()
        if analyzer is None or anonymizer is None:
            return text

        # Protect citation tags like [file.docx p.4] before scrubbing.
        # Presidio's URL recognizer matches file extensions that are valid TLDs:
        # .do (Dominican Republic) inside .docx, .jp (Japan) inside .jpg, etc.
        # PERSON recognizer also fires on product names like "Mac", "iPad".
        # Replace all [bracket] tags with opaque tokens, scrub, then restore.
        slots: dict = {}
        protected = text
        for i, m in enumerate(re.finditer(r'\[[^\]]{1,300}\]', text)):
            token = f'__CITE_{i}__'
            slots[token] = m.group()
        for token, tag in slots.items():
            protected = protected.replace(tag, token)

        # Use a narrow entity list — URL and PERSON cause false positives on
        # financial-document answers (filename TLDs, product names, tickers).
        safe_entities = [
            "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
            "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
            "US_BANK_NUMBER", "US_PASSPORT",
        ]
        results = analyzer.analyze(text=protected, entities=safe_entities, language="en",
                                   score_threshold=0.6)
        if results:
            from presidio_anonymizer.entities import OperatorConfig
            operators = {e: OperatorConfig("replace", {"new_value": f"<{e}>"})
                         for e in safe_entities}
            protected = anonymizer.anonymize(
                text=protected, analyzer_results=results, operators=operators,
            ).text

        # Restore citation tags
        for token, tag in slots.items():
            protected = protected.replace(token, tag)
        return protected

    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pii_scrub_answer_failed", error=str(exc))
    return text


# KNOWLEDGE PREPARATION FROM RETRIEVED DOCS

def _prepare_knowledge(
    docs:     List[Dict],
    max_docs: int                            = None,
    max_chars: int                           = None,
    sources:  Optional[List[Dict[str, Any]]] = None,
) -> str:
    if not docs:
        return ""

    max_docs  = max_docs  or settings.RAG_TOP_K
    max_chars = max_chars or settings.RAG_DOC_MAX_CHARS

    # If canonical sources[] were built upstream, use their cite_keys for stable tags.
    # Map by (doc_id, chunk_id) and fall back to source+page-derived key.
    cite_by_key: Dict[Tuple[Optional[str], Optional[str]], str] = {}
    if sources:
        for s in sources:
            k = (s.get("doc_id"), s.get("chunk_id"))
            ck = s.get("cite_key")
            if ck:
                cite_by_key[k] = ck

    seen:  set        = set()
    parts: List[str]  = []

    # Presidio PII placeholders (e.g. <PERSON>, <LOCATION>) may have been
    # injected during ingestion when the entity list included those types.
    # Strip them now so the LLM does not copy them verbatim into answers.
    _PII_PLACEHOLDER_RE = re.compile(
        r"<(?:PERSON|LOCATION|ORG|NRP|GPE|DATE_TIME|AGE|ID|MEDICAL_LICENSE"
        r"|URL|IP_ADDRESS|US_SSN|CREDIT_CARD|IBAN_CODE|PHONE_NUMBER"
        r"|EMAIL_ADDRESS|US_BANK_NUMBER|US_PASSPORT|UK_NHS|AU_ABN"
        r"|AU_ACN|AU_TFN|AU_MEDICARE|NL_BSN|ES_NIF|SG_NRIC_FIN"
        r"|IN_PAN|IN_AADHAAR|CRYPTO|MEDICAL_RECORD)>",
        re.IGNORECASE,
    )

    for d in docs[:max_docs]:
        text = _normalize(d.get("text", ""))
        if not text:
            continue

        # Strip PII placeholder tags stored during ingestion.
        text = _PII_PLACEHOLDER_RE.sub("", text).strip()
        if not text:
            continue

        h = _hash(text[:200])
        if h in seen:
            continue
        seen.add(h)

        meta = d.get("metadata", {}) or {}
        doc_id   = meta.get("doc_id")
        chunk_id = meta.get("chunk_id")

        cite_key = cite_by_key.get((doc_id, chunk_id))
        if not cite_key:
            # Build a key inline as a fallback. Match build_sources() format.
            try:
                from app.core.response import _make_cite_key
                cite_key = _make_cite_key(
                    source=meta.get("source") or "unknown",
                    page=meta.get("page") if isinstance(meta.get("page"), int) else None,
                    sheet=meta.get("sheet"),
                    timestamp_start=meta.get("timestamp_start"),
                )
            except Exception:
                cite_key = f"[{meta.get('source', 'unknown')}]"

        # Also strip PII placeholders from cite keys — e.g. "call.mp3" can be
        # stored as "call.<URL>3" if old ingestion ran Presidio with URL entity.
        cite_key = _PII_PLACEHOLDER_RE.sub("", cite_key).strip()

        parts.append(f"{cite_key} {text[:max_chars]}")

    knowledge = "\n\n".join(parts)

    # KEY-FACT PREFIX: for acquisition/M&A queries, extract the most relevant
    # sentences from the top chunks and place them at the very start of the
    # knowledge block. Because the prompt is flattened to a single line by the
    # LLM normaliser, sentences buried mid-chunk are easy to miss. Putting the
    # key sentence first ensures the LLM reads it before everything else.
    return knowledge


def _prepend_key_facts_knowledge(docs: List[Dict], query: str, knowledge: str) -> str:
    """Prepend extracted M&A facts to the knowledge block when the query asks
    about an acquisition, merger, or deal."""
    _MA_Q  = frozenset(["acquisition", "merger", "acquired", "deal", "takeover", "purchased"])
    _MA_CK = frozenset(["acquired", "acquisition", "merger", "assumed", "fdic", "purchase"])
    if not query or not any(kw in query.lower() for kw in _MA_Q):
        return knowledge
    facts = []
    for doc in docs[:3]:
        text = doc.get("text", "") or ""
        for sent in text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|"):
            sl = sent.lower()
            if any(kw in sl for kw in _MA_CK) and len(sent.strip()) > 30:
                facts.append(sent.strip())
        if len(facts) >= 3:
            break
    if not facts:
        return knowledge
    prefix_body = " | ".join(facts[:3])[:300]
    prefix = "KEY FACTS (answer the query from these): " + prefix_body + " "
    return prefix + knowledge


# MEMORY PREPARATION

def _prepare_memory(memory: str) -> str:
    if not memory:
        return ""
    memory = _normalize(memory)
    return memory[:settings.MEMORY_MAX_CONTEXT_CHARS]


# STEP-LEVEL TRACE RECORD

def _record_step(
    trace_log: List[Dict],
    step_type: str,
    input_text: str,
    output_text: str,
    latency: float,
    metadata: Optional[Dict] = None,
) -> None:
    trace_log.append({
        "step":      step_type,
        "input":     input_text[:200],
        "output":    output_text[:200],
        "latency":   latency,
        "metadata":  metadata or {},
        "timestamp": time.time(),
    })


# CHAIN-OF-THOUGHT PROMPT BUILDER

def _build_cot_prompt(
    query: str,
    knowledge: str,
    memory: str,
    query_type: str = "factual",
    cite_keys: Optional[List[str]] = None,
) -> str:

    # NOTE: these instructions must NOT ask the model to show its reasoning or
    # emit "Step 1/Step 2..." text. The small GGUF model echoes such scaffolding
    # verbatim into the answer (visible as reasoning-leak preambles). Reason
    # silently; output only the final answer.
    type_instructions = {
        "factual": (
            "Answer factually using ONLY the provided knowledge. "
            "Give only the final answer — do not show your reasoning."
        ),
        "comparative": (
            "Compare the entities using ONLY the provided knowledge. "
            "Describe each, then state the key differences in a single final "
            "answer. Do not show your reasoning or number your steps."
        ),
        "temporal": (
            "Answer with temporal context using ONLY the provided knowledge. "
            "Keep events in chronological order in one final answer. "
            "Do not show your reasoning or number your steps."
        ),
        "aggregation": (
            "Aggregate information using ONLY the provided knowledge. "
            "Include every relevant item in one final answer. "
            "Do not show your reasoning or number your steps."
        ),
        "multihop": (
            "Reason silently across the provided knowledge and give one "
            "synthesized final answer. Do not show your reasoning or steps."
        ),
    }

    type_instruction = type_instructions.get(query_type, type_instructions["factual"])

    instruction = (
        "You are a strict, grounded AI assistant. Use ONLY the provided KNOWLEDGE chunks.\n"
        "\n"
        "ABSOLUTE RULES — VIOLATING ANY OF THESE IS A FAILURE:\n"
        "1. Every number, percentage, date, name, and proper noun in your Answer\n"
        "   MUST appear verbatim in at least one KNOWLEDGE chunk. Do NOT round,\n"
        "   convert, paraphrase, or substitute numbers. If KNOWLEDGE says '31.4%',\n"
        "   you MUST write '31.4%' — never '33.9%', '~31%', or 'about 31'.\n"
        "2. Do NOT use prior training knowledge. If a fact is not in KNOWLEDGE,\n"
        "   you do not know it. Say 'No relevant information was found in your\n"
        "   knowledge base to answer this question.'\n"
        "   CRITICAL: If KNOWLEDGE contains the words 'acquired', 'acquisition',\n"
        "   'merger', 'assumed liabilities', or any M&A term, those ARE acquisitions.\n"
        "   Report exactly what KNOWLEDGE says. NEVER say 'did not complete' or\n"
        "   'no acquisition' if KNOWLEDGE describes one — that would contradict the\n"
        "   document and violate this rule.\n"
        "3. You MAY combine facts from multiple chunks, but every individual fact\n"
        "   in the combination must still come from KNOWLEDGE verbatim.\n"
        "4. The MEMORY section is prior conversation context only. NEVER cite\n"
        "   MEMORY, never copy MEMORY timestamps like '[6m ago]', never list\n"
        "   MEMORY as a source.\n"
        "5. Be concise. Answer only what the QUERY asks. Do NOT add extra sentences\n"
        "   about related topics that the QUERY did not ask for. Quote numbers and\n"
        "   names exactly as they appear in KNOWLEDGE.\n"
        "6. TEMPORAL GROUNDING — if the question asks about a specific year or period\n"
        "   (e.g. 'FY2024', 'last year', 'reported'), prefer chunks that contain\n"
        "   words like 'audited', 'reported', 'fiscal year 2024', or the exact year.\n"
        "   NEVER cite a 'guidance', 'outlook', or 'FY2025' figure as a historical\n"
        "   actual. If the KNOWLEDGE chunks DO contain the requested figure (even\n"
        "   partway through a chunk), use it — do NOT refuse just because a chunk\n"
        "   also contains unrelated content.\n"
        "7. COMPLETENESS — if a KNOWLEDGE chunk contains a numbered or bulleted list\n"
        "   (e.g. competitors, risk factors, strategic priorities), include ALL items\n"
        "   from that list in your answer. Never stop after the first item.\n"
        "\n"
    )

    # CITATION RULES — make the LLM emit the exact bracket tags shown in KNOWLEDGE.
    tag_list = ""
    if cite_keys:
        tag_list = "Available source tags (use ONLY these, verbatim): " + ", ".join(cite_keys[:12]) + "\n"
    citation_rules = (
        "CITATION RULES:\n"
        "- Write a full prose answer that directly addresses the QUERY using facts from KNOWLEDGE.\n"
        "- After each factual sentence, append the matching source tag from KNOWLEDGE.\n"
        "- Use ONLY tags listed below. Never invent tags or filenames.\n"
        f"{tag_list}\n"
    )

    memory_block    = f"MEMORY:\n{memory}\n\n"    if memory    else ""
    knowledge_block = f"KNOWLEDGE:\n{knowledge}\n\n" if knowledge else ""
    query_block     = f"QUERY:\n{query}\n\n"

    # Few-shot example: STRUCTURE ONLY, no numbers or named entities that could
    # leak into the answer. Mistral-7B Q4 has been observed copying numeric
    # tokens from the example into the answer.
    example_tag = cite_keys[0] if cite_keys else "[source.txt]"
    example_tag2 = cite_keys[1] if len(cite_keys) > 1 else example_tag
    example_block = (
        "EXAMPLE OUTPUT (structure only — never copy these words):\n"
        f"Answer: The subject performed the action. {example_tag} "
        f"The second fact follows. {example_tag2}\n"
        f"Answer Tags: {example_tag}, {example_tag2}\n"
        "Confidence: 0.9\n"
        "Sources Used: 2\n\n"
        "Now answer the actual QUERY. Use only numbers and names from KNOWLEDGE verbatim.\n\n"
    )

    output_format = (
        "Output format — fill in each field, do NOT write the field description:\n"
        "Answer: [your prose answer with inline source tags]\n"
        "Answer Tags: [tags used, comma-separated]\n"
        "Confidence: [0.0–1.0]\n"
        "Sources Used: [integer]\n"
    )

    return (
        instruction
        + citation_rules
        + memory_block
        + knowledge_block
        + example_block
        + query_block
        + output_format
    )


# REACT PROMPT BUILDER — FOR TOOL-AUGMENTED REASONING

def _build_react_prompt(
    query: str,
    knowledge: str,
    memory: str,
    step_history: List[Dict],
) -> str:

    history_text = ""
    for step in step_history[-5:]:
        history_text += (
            f"Thought: {step.get('thought', '')}\n"
            f"Action: {step.get('action', '')}\n"
            f"Observation: {step.get('observation', '')}\n\n"
        )

    instruction = (
        "You are a ReAct agent. Use Thought/Action/Observation cycles.\n"
        "Available actions: search_knowledge, answer\n"
        "Rules:\n"
        "- Use ONLY provided knowledge\n"
        "- No hallucination\n\n"
    )

    memory_block    = f"MEMORY:\n{memory}\n\n"    if memory    else ""
    knowledge_block = f"KNOWLEDGE:\n{knowledge}\n\n" if knowledge else ""
    history_block   = f"HISTORY:\n{history_text}\n" if history_text else ""
    query_block     = f"QUERY:\n{query}\n\n"

    output_format = (
        "FORMAT:\n"
        "Thought: [reasoning]\n"
        "Action: answer\n"
        "Answer: [final answer]\n"
        "Confidence: [0.0-1.0]\n"
        "Sources Used: [integer]\n"
    )

    return (
        instruction
        + memory_block
        + knowledge_block
        + history_block
        + query_block
        + output_format
    )


# RESPONSE PARSER

def _parse_response(
    text: str,
    docs: List[Dict],
    valid_cite_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:

    if not text:
        return _fallback_response()

    try:
        answer:     str   = ""
        # `confidence` is `None` until the LLM actually emits a Confidence: line.
        # Downstream code uses this to distinguish "LLM said 0.5" from
        # "LLM omitted the field" so the pipeline can fall through to the
        # retrieval-derived confidence instead of a misleading hardcoded 0.5.
        confidence: Optional[float] = None
        sources:    int   = 0
        reasoning:  str   = ""
        tags_line:  str   = ""

        # NORMALIZE — strip leading bracket wrappers that Mistral emits in
        # several variants. All of these break the line-based prefix matcher
        # below if not flattened first:
        #   "[Answer]: foo"     (closed bracket)
        #   "[Answer: foo"      (open bracket only — most common, was missed)
        #   "[ Answer ]: foo"   (whitespace inside)
        # We match ANY of these and replace with the bare "<Key>:" form so the
        # downstream line-scanner can recognise the prefix correctly.
        norm_text = text
        keys_for_norm = ("Answer Tags", "Sources Used", "Confidence",
                         "Reasoning", "Answer")
        for key in keys_for_norm:
            key_re = re.escape(key)
            # Variant 1: "[Key]:" or "[ Key ]:"
            norm_text = re.sub(
                r"\[\s*" + key_re + r"\s*\]\s*:",
                key + ":", norm_text, flags=re.IGNORECASE,
            )
            # Variant 2: "[Key:" (open bracket only — no matching ])
            norm_text = re.sub(
                r"\[\s*" + key_re + r"\s*:",
                key + ":", norm_text, flags=re.IGNORECASE,
            )

        # Insert newlines before each format key when they appear inline.
        # Allow optional preceding "[" so "...years. [Answer Tags:" splits too —
        # the normalization above already removed those that started at line
        # start, but inline occurrences still keep their "[" until here.
        for key in ("Answer Tags:", "Confidence:", "Sources Used:", "Reasoning:"):
            norm_text = re.sub(
                r"\s+(?=\[?\s*" + re.escape(key) + r")",
                "\n", norm_text, flags=re.IGNORECASE,
            )

        answer_lines: list = []
        in_answer = False

        _FORMAT_KEYS = (
            "confidence:", "sources used:", "reasoning:", "answer tags:",
        )

        for line in norm_text.split("\n"):
            ll = line.lower().strip()

            # IMPORTANT: check "answer tags:" FIRST, because the broader
            # "answer " prefix below would otherwise swallow it (a line like
            # "Answer Tags: foo" starts with "answer ", which would wrongly
            # treat the tag line as a new Answer field, overwriting the real
            # answer with the tag content).
            if ll.startswith("answer tags:") or ll.startswith("answer tags "):
                in_answer = False
                tags_line = line.split(":", 1)[-1].strip()

            elif ll.startswith("answer:") or ll.startswith("answer "):
                val = line.split(":", 1)[-1].strip()
                # strip any leading placeholder the LLM echoed from the format template
                val = re.sub(r'^[\[<][^\]>]{4,140}[\]>]\s*', '', val).strip()
                answer_lines = [val] if val else []
                in_answer = True

            elif ll.startswith("reasoning:") or ll.startswith("reasoning "):
                in_answer = False
                reasoning = line.split(":", 1)[-1].strip()

            elif ll.startswith("confidence:") or ll.startswith("confidence "):
                in_answer = False
                # Parse the value; leave `confidence` as None if it's
                # unparseable so the pipeline still falls through to the
                # retrieval-derived confidence. A malformed Confidence: line
                # is no more informative than a missing one.
                try:
                    confidence = float(line.split(":", 1)[-1].strip())
                except Exception:
                    confidence = None

            elif ll.startswith("sources used:") or ll.startswith("sources used "):
                in_answer = False
                try:
                    sources = int(line.split(":", 1)[-1].strip())
                except Exception:
                    sources = 0

            elif in_answer and not any(ll.startswith(k) for k in _FORMAT_KEYS):
                answer_lines.append(line)

        answer = " ".join(answer_lines).strip()

        # Strip leading <placeholder> text Mistral sometimes copies from the
        # output-format instructions (e.g. "<your prose answer with inline...>").
        # Real source tags use square brackets, so angle-bracket content at the
        # start of the answer is always an echoed instruction fragment.
        answer = re.sub(r'^<[^>]{4,140}>\s*', '', answer).strip()

        # POST-CLEAN — strip any trailing format key fragments that survived.
        # Mistral emits these in several bracket variants:
        #   "Answer Tags:"        (clean)
        #   "[Answer Tags]:"      (closed bracket)
        #   "[Answer Tags:"       (open bracket only)  ← was missed
        #   " [Answer Tags:"      (open bracket with leading space)
        if answer:
            # Build a regex that catches any of these key markers with optional
            # surrounding brackets and optional leading whitespace+bracket.
            # "Answer Tags" must come BEFORE "Answer" in the alternation so the
            # longer match wins — otherwise the regex would split at the first
            # "Answer" and leave " Tags:" stranded.
            keys = (
                "Answer Tags", "Sources Used",
                "Confidence", "Reasoning", "Answer",
            )
            key_pattern = (
                r"\s*\[?\s*(?:"
                + "|".join(re.escape(k) for k in keys)
                + r")\s*\]?\s*:"
            )
            m = re.search(key_pattern, answer, flags=re.IGNORECASE)
            if m and m.start() > 0:
                answer = answer[:m.start()].strip()
            # Trim any dangling open bracket / trailing punctuation left over
            # from a partial format fragment that didn't match the pattern.
            answer = re.sub(r"\s*\[\s*$", "", answer).strip()
            # Strip "Source: <source number>" template leakage from some prompts
            answer = re.sub(r"^\s*Source\s*:\s*<source[^>]*>\s*\n?", "", answer, flags=re.IGNORECASE)
            answer = re.sub(r"\s*Source\s*:\s*$", "", answer, flags=re.IGNORECASE | re.MULTILINE)
            # Strip TXT section-marker artifacts like "( Item 1. Business. Overview)"
            # or "(Human capital)" that the LLM copies verbatim from chunk headers.
            answer = re.sub(r"\s*\(\s*Item\s+\d+[^()]{0,80}\)", "", answer)
            answer = re.sub(
                r"\s*\(\s*[A-Z][a-zA-Z ]{1,50}\)\s*(?=[.,;!?]|$)",
                "",
                answer,
            )
            # Strip space-before-period left when a cite tag is removed mid-sentence:
            # e.g. "legal risks . Liquidity" → "legal risks. Liquidity"
            answer = re.sub(r" +\.(?=[\s,;!?]|$)", ".", answer)
            # Strip orphaned citation separators left after tag removal:
            # e.g. "Management. ," or "Management., " → "Management."
            answer = re.sub(r"([.!?])\s*[,;]\s*$", r"\1", answer).strip()
            answer = answer.rstrip(",;: \t")
            # Repair common Mistral malformation: closing the last list item
            # with "}" or ")" instead of the natural sentence terminator. Only
            # touch the FINAL character to stay conservative.
            if answer.endswith("}") or answer.endswith(")"):
                # Replace only if there's no matching opener for the brace/paren
                # in the answer (i.e. it's a malformed closer, not balanced).
                last_char = answer[-1]
                opener = "{" if last_char == "}" else "("
                if answer.count(opener) < answer.count(last_char):
                    answer = answer[:-1].rstrip() + "."

            # Repair citation-in-data-slot artifacts. Mistral occasionally
            # emits a sentence where the citation tag has been inserted into
            # the grammatical position of a missing value — typically because
            # the numeric guard's retry stripped a fabricated number and the
            # regeneration substituted the tag where the number used to be.
            # Pattern: "<word> of [..long_tag..] <word>" where the tag is in
            # the noun-phrase slot rather than at end-of-sentence/clause.
            # The fix drops the malformed clause up to the next clause break
            # (", but", "; ", ". "), preserving the rest of the answer.
            citation_in_slot = re.compile(
                r"\b(?:of|is|are|was|were|at|reaches|reached|approximately|about|around|"
                r"holds?|has|have|had|totaling|totalling|valued|worth)\s+"
                r"\[[^\[\]\n]{8,200}\]\s+(?:in|at|on|of|for|across|globally|worldwide|"
                r"per|annually|monthly|yearly|today|currently)",
                flags=re.IGNORECASE,
            )
            m_slot = citation_in_slot.search(answer)
            if m_slot:
                # Cut the offending clause: from the start of the matched
                # phrase to the next clause boundary (or end of string).
                cut_start = m_slot.start()
                # Walk back to the start of the current clause (after the
                # nearest preceding sentence/clause boundary).
                lookbehind = answer[:cut_start]
                clause_start = max(
                    lookbehind.rfind(". "),
                    lookbehind.rfind(", but "),
                    lookbehind.rfind(", however "),
                    lookbehind.rfind("; "),
                )
                clause_start = clause_start + 2 if clause_start >= 0 else 0
                # Find the end of the broken clause.
                tail = answer[cut_start:]
                clause_end_rel = -1
                for sep in (", but ", ", however ", "; ", ". "):
                    idx = tail.find(sep)
                    if idx >= 0 and (clause_end_rel < 0 or idx < clause_end_rel):
                        clause_end_rel = idx + len(sep)
                if clause_end_rel < 0:
                    # No following clause — drop the broken phrase entirely.
                    answer = answer[:clause_start].rstrip().rstrip(",;:") or answer
                else:
                    answer = (
                        answer[:clause_start]
                        + answer[cut_start + clause_end_rel:]
                    ).strip()
                # Capitalize first letter if we trimmed away the original opener.
                if answer and answer[0].islower():
                    answer = answer[0].upper() + answer[1:]
                logger.info("reasoning_repaired_citation_in_slot")

            # POST-REPAIR INTEGRITY CHECK
            # If the answer STILL contains citation-in-data-slot patterns after
            # one repair pass — i.e. multiple broken slots chained without any
            # refusal text between them — the LLM has produced an unsalvageable
            # response (it tried to invent a value the document doesn't have,
            # and the citation tag was substituted into the value's grammatical
            # position multiple times). Replace the whole answer with an
            # explicit refusal. This is the case Q10 hit when Mistral wrote
            # "holds a market share of [tag] in [...] segment [tag]." with no
            # "but the document doesn't contain..." follow-up.
            REFUSAL_MARKERS = (
                "does not contain", "do not contain",
                "not mentioned", "not specified",
                "is not provided", "are not provided",
                "no information", "cannot find",
                "i don't know", "i do not know",
            )
            has_refusal = any(
                m in (answer or "").lower() for m in REFUSAL_MARKERS
            )
            still_broken = citation_in_slot.search(answer or "") is not None
            if still_broken and not has_refusal:
                logger.warning(
                    "reasoning_unsalvageable_citation_substitution",
                    answer_prefix=(answer or "")[:120],
                )
                answer = (
                    "No relevant information was found in your knowledge base "
                    "to answer this question."
                )

        # REJECT CITATION-ONLY ANSWERS — Mistral sometimes echoes only the
        # example tag (e.g. "[file.pdf p.4]") with no prose. Strip all bracketed
        # tags and check whether any real content remains.
        if answer:
            stripped = re.sub(r"\[[^\]]+\]", "", answer).strip()
            if len(stripped) < 10:
                answer = ""

        # FALLBACK IF ANSWER NOT CLEANLY PARSED
        if not answer or len(answer) < 10:
            # strip trailing format lines from raw text
            clean_lines = []
            for line in norm_text.split("\n"):
                ll = line.lower().strip()
                if any(ll.startswith(k) for k in (
                    "confidence:", "sources used:", "reasoning:",
                    "answer:", "answer tags:",
                )):
                    break
                clean_lines.append(line)
            answer = " ".join(clean_lines).strip() or text.strip()

        # NaN/INF GUARD ON CONFIDENCE. When the LLM omitted the Confidence:
        # line entirely, `confidence` is None — leave it that way so the
        # pipeline can substitute a retrieval-derived score downstream
        # instead of a misleading hardcoded 0.5.
        if confidence is not None:
            if math.isnan(confidence) or math.isinf(confidence):
                confidence = None
            else:
                confidence = max(0.0, min(confidence, 1.0))

        # CITATION EXTRACTION — inline tags + Answer Tags: line, validated.
        cited_tags: List[str] = []
        if valid_cite_keys:
            inline = _extract_cite_tags(answer, valid_cite_keys)
            cited_tags.extend(inline)
            if tags_line:
                # Pull tag-shaped substrings out of the tags line
                for tag in _extract_cite_tags(tags_line, valid_cite_keys):
                    if tag not in cited_tags:
                        cited_tags.append(tag)

        # AUTO SOURCES COUNT
        if sources == 0:
            if cited_tags:
                sources = len(cited_tags)
            elif docs:
                sources = min(
                    len([d for d in docs if d.get("text")]),
                    len(docs),
                )

        sources = min(sources, max(len(docs), len(cited_tags)))

        return {
            "answer":       answer,
            "reasoning":    reasoning,
            "confidence":   confidence,
            "sources_used": sources,
            "cited_tags":   cited_tags,
        }

    except Exception:
        return _fallback_response(text)


# FALLBACK RESPONSE

def _fallback_response(text: str = "") -> Dict[str, Any]:
    return {
        "answer": (
            text.strip()
            if text and len(text.strip()) >= 10
            else "I couldn't generate a reliable answer."
        ),
        "reasoning":    "",
        "confidence":   0.3,
        "sources_used": 0,
        "cited_tags":   [],
    }


# PROMPT TRUNCATION — PRESERVE HEADER AND QUERY, TRIM KNOWLEDGE

def _truncate_prompt(prompt: str, max_chars: int) -> str:
    parts = prompt.split("KNOWLEDGE:")
    if len(parts) < 2:
        return prompt[:max_chars]
    header  = parts[0]
    body    = "KNOWLEDGE:" + parts[1]
    allowed = max_chars - len(header) - 20
    return header + body[:max(allowed, 0)]


class ReasoningEngine:

    def __init__(self, llm: Any) -> None:
        self.llm              = llm
        self.max_prompt_chars = settings.MAX_PROMPT_CHARS
        self.model_name       = getattr(settings, "PRIMARY_LLM_PROVIDER", "gguf")

    # LLM CALL WITH RETRY AND TIMEOUT

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=False,
    )
    def _call_llm(
        self,
        prompt: str,
        session_id: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Optional[str]:

        if len(prompt) > self.max_prompt_chars:
            prompt = _truncate_prompt(prompt, self.max_prompt_chars)

        t_start = time.time()

        try:
            response = self.llm.generate(
                prompt,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=temperature,
                session_id=session_id,
            )
            llm_latency = round(time.time() - t_start, 2)

            _llm_call_duration.labels(model=self.model_name).observe(llm_latency)

            if llm_latency > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    "reasoning_llm_timeout",
                    llm_latency=llm_latency,
                    session_id=session_id,
                )

            return response

        except Exception as exc:
            llm_latency = round(time.time() - t_start, 2)
            _llm_call_duration.labels(model=self.model_name).observe(llm_latency)
            logger.error(
                "reasoning_llm_call_failed",
                error=str(exc),
                session_id=session_id,
            )
            return None

    # MAIN SYNC GENERATE ANSWER

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = "",
        session_id: str = "default",
        query_type: str = "factual",
        use_react: bool = False,
        step_history: Optional[List[Dict]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        if not query:
            return _fallback_response()

        start      = time.time()
        trace_log: List[Dict] = []

        # Build sources lazily here if pipeline didn't supply them (back-compat).
        if sources is None:
            try:
                from app.core.response import build_sources
                sources = build_sources(retrieved_docs)
            except Exception:
                sources = []

        cite_keys: List[str] = list(dict.fromkeys(
            s.get("cite_key") for s in (sources or []) if s.get("cite_key")
        ))

        with tracer.start_as_current_span("reasoning_generate_answer") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("docs.count", len(retrieved_docs))
            span.set_attribute("session.id", session_id)
            span.set_attribute("query.type", query_type)
            span.set_attribute("use_react", use_react)

            try:
                query    = _normalize(query)
                knowledge = _prepare_knowledge(retrieved_docs, sources=sources)
                knowledge = _prepend_key_facts_knowledge(retrieved_docs, query, knowledge)
                memory   = _prepare_memory(memory_context)

                # RECORD RETRIEVE STEP
                t_retrieve = time.time()
                _record_step(
                    trace_log,
                    STEP_RETRIEVE,
                    query,
                    knowledge[:200],
                    round(time.time() - t_retrieve, 3),
                    {"docs_count": len(retrieved_docs)},
                )

                # BUILD PROMPT — REACT OR COT
                if use_react:
                    prompt = _build_react_prompt(
                        query,
                        knowledge,
                        memory,
                        step_history or [],
                    )
                else:
                    prompt = _build_cot_prompt(
                        query,
                        knowledge,
                        memory,
                        query_type=query_type,
                        cite_keys=cite_keys,
                    )

                # PROMPT BUDGET WARNING
                budget_pct = len(prompt) / self.max_prompt_chars
                if budget_pct > 0.8:
                    logger.warning(
                        "reasoning_prompt_near_limit",
                        budget_pct=round(budget_pct, 2),
                        prompt_chars=len(prompt),
                        session_id=session_id,
                    )

                # LLM INFERENCE
                t_reason = time.time()
                response = self._call_llm(prompt, session_id)
                reason_latency = round(time.time() - t_reason, 3)

                _record_step(
                    trace_log,
                    STEP_REASON,
                    prompt[:200],
                    (response or "")[:200],
                    reason_latency,
                )

                if not response:
                    span.set_status(Status(StatusCode.OK))
                    return _fallback_response()

                # PARSE RESPONSE (validate citation tags against the closed set)
                parsed = _parse_response(response, retrieved_docs, valid_cite_keys=cite_keys)

                cited_tags = parsed.get("cited_tags") or []

                # NUMERIC FAITHFULNESS — catches the "31.4 → 33.9" substitution.
                # If unsupported numbers are detected, retry ONCE with a hardened
                # prompt that explicitly tells the model which numbers were
                # fabricated AND instructs it to refuse if the doc lacks the data.
                bad_nums = _unsupported_numbers(parsed["answer"], retrieved_docs, query=query)
                if bad_nums:
                    logger.warning(
                        "reasoning_numeric_mismatch",
                        unsupported=bad_nums[:5],
                        session_id=session_id,
                    )
                    retry_prompt = (
                        prompt
                        + "\n\nIMPORTANT CORRECTION: Your previous answer contained "
                        f"the number(s) {', '.join(bad_nums)} which do NOT appear "
                        "anywhere in KNOWLEDGE. You MUST use only numbers that "
                        "appear verbatim in KNOWLEDGE. If KNOWLEDGE does not "
                        "contain the value the QUERY asks for, respond with "
                        "exactly: 'No relevant information was found in your "
                        "knowledge base to answer this question.' Do NOT "
                        "invent a substitute number.\n"
                    )
                    retry_response = self._call_llm(
                        retry_prompt, session_id, temperature=0.0,
                    )
                    if retry_response:
                        retry_parsed = _parse_response(
                            retry_response, retrieved_docs, valid_cite_keys=cite_keys,
                        )
                        retry_bad = _unsupported_numbers(
                            retry_parsed["answer"], retrieved_docs, query=query,
                        )
                        if len(retry_bad) < len(bad_nums):
                            parsed = retry_parsed
                            cited_tags = parsed.get("cited_tags") or []
                            bad_nums = retry_bad

                # If, even after one retry, the answer still contains
                # "unsupported" numbers — BUT the LLM cited real source tags —
                # the mismatch is almost always a financial scale/format
                # difference (e.g. "57.53 billion" vs "57,530" in millions).
                # The LLM is grounded; the string-match check just can't resolve
                # the unit difference. Trust the citation and clear bad_nums.
                if bad_nums and cited_tags:
                    logger.info(
                        "reasoning_numeric_guard_bypassed_by_citations",
                        unsupported=bad_nums[:5],
                        cited_tags=cited_tags[:3],
                        session_id=session_id,
                    )
                    bad_nums = []

                # Hard-fail only for genuinely uncited answers where the LLM
                # is likely pulling numbers from training-data memory.
                if bad_nums:
                    logger.warning(
                        "reasoning_replacing_unfaithful_answer",
                        unsupported=bad_nums[:5],
                        session_id=session_id,
                    )
                    parsed["answer"] = (
                        "No relevant information was found in your knowledge base "
                        "to answer this question."
                    )
                    parsed["cited_tags"] = []
                    cited_tags = []

                # HALLUCINATION GUARD
                t_verify = time.time()
                is_hallucinated, support_score = _hallucination_guard(
                    parsed["answer"],
                    retrieved_docs,
                )
                # CITATION-BASED RELAXATION:
                # If the LLM cited >=1 VALID tag AND no numeric mismatch remains,
                # treat the answer as grounded regardless of surface overlap.
                if cited_tags and not bad_nums:
                    is_hallucinated = False
                # NUMERIC MISMATCH OVERRIDES citation relaxation — citations
                # cannot rescue an answer that contains fabricated numbers.
                if bad_nums:
                    is_hallucinated = True

                verify_latency = round(time.time() - t_verify, 3)

                _record_step(
                    trace_log,
                    STEP_VERIFY,
                    parsed["answer"][:200],
                    f"support={support_score}",
                    verify_latency,
                    {
                        "hallucinated":  is_hallucinated,
                        "support_score": support_score,
                        "cited_tags":    len(cited_tags),
                    },
                )

                if is_hallucinated:
                    _hallucination_flags.inc()
                    logger.warning(
                        "reasoning_hallucination_detected",
                        support_score=support_score,
                        unsupported_numbers=bad_nums[:5] if bad_nums else [],
                        session_id=session_id,
                    )
                    # Numeric mismatch is a hard fail — clamp confidence low.
                    # When the LLM omitted Confidence: (parsed["confidence"]
                    # is None), force a low value here since we KNOW the
                    # answer is suspect — we can't defer to retrieval scoring.
                    cap = 0.2 if bad_nums else 0.4
                    cur = parsed.get("confidence")
                    parsed["confidence"] = cap if cur is None else min(cur, cap)
                    parsed["hallucination_warning"] = True
                else:
                    parsed["hallucination_warning"] = False

                # FILTERED SOURCES — subset of input sources that the LLM actually cited.
                ans_lower = (parsed.get("answer") or "").lower()
                _REFUSAL_SENTINELS = (
                    "i don't know",
                    "i dont know",
                    "insufficient knowledge",
                    "i don't have sufficient",
                    "i do not have sufficient",
                    "couldn't generate a reliable answer",
                    "couldn't generate",
                    "no answer generated",
                    "something went wrong",
                    # New: phrases emitted by the post-repair refusal path
                    # in _parse_response, plus general document-doesn't-contain
                    # language Mistral itself may emit.
                    "no relevant information was found",
                    "not found in your knowledge base",
                    "do not contain the information",
                    "does not contain the information",
                    "do not contain this information",
                    "does not contain this information",
                    "documents do not contain",
                    "document does not contain",
                    "not provided in the document",
                    "not specified in the document",
                    "not mentioned in the document",
                )
                is_refusal = any(s in ans_lower for s in _REFUSAL_SENTINELS)

                if is_refusal:
                    # Refusals carry no real grounding — don't fake sources,
                    # and force a low-confidence + warning signal so callers
                    # and the UI can flag it. Without this, a refusal that
                    # the LLM emitted with Confidence: 1.0 would look like a
                    # confident grounded answer.
                    parsed["sources"] = []
                    parsed["confidence"] = min(parsed.get("confidence") or 0.1, 0.1)
                    parsed["hallucination_warning"] = True
                elif cited_tags and sources:
                    cited_set = set(cited_tags)
                    parsed["sources"] = [
                        s for s in sources if s.get("cite_key") in cited_set
                    ]
                else:
                    # Fallback to top-3 retrieved sources so the UI always has citations.
                    parsed["sources"] = list((sources or [])[:3])

                parsed["sources_used"] = len(parsed["sources"])

                # PII SCRUB ANSWER
                parsed["answer"] = _scrub_answer_pii(parsed["answer"])

                # FINAL SYNTHESIS STEP
                _record_step(
                    trace_log,
                    STEP_SYNTHESIZE,
                    parsed["answer"][:200],
                    f"confidence={parsed['confidence']}",
                    0.0,
                )

                latency = round(time.time() - start, 2)

                _reasoning_duration.labels(status="success").observe(latency)

                span.set_attribute("answer.confidence", parsed["confidence"])
                span.set_attribute("answer.support_score", support_score)
                span.set_attribute("answer.hallucinated", is_hallucinated)
                span.set_attribute("sources.used", parsed["sources_used"])
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "reasoning_success",
                    knowledge_chars=len(knowledge),
                    memory_chars=len(memory),
                    llm_latency=reason_latency,
                    latency=latency,
                    confidence=parsed["confidence"],
                    support_score=support_score,
                    hallucinated=is_hallucinated,
                    sources_used=parsed["sources_used"],
                    session_id=session_id,
                )

                parsed["execution_trace"] = trace_log
                return parsed

            except Exception as exc:
                latency    = round(time.time() - start, 2)
                error_type = type(exc).__name__

                _reasoning_duration.labels(status="error").observe(latency)
                _reasoning_errors.labels(error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "reasoning_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )
                return _fallback_response()

    # ASYNC WRAPPER

    async def generate_answer_async(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = "",
        session_id: str = "default",
        query_type: str = "factual",
        use_react: bool = False,
        step_history: Optional[List[Dict]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        async with _get_semaphore():
            return await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.generate_answer(
                    query,
                    retrieved_docs,
                    memory_context,
                    session_id,
                    query_type,
                    use_react,
                    step_history,
                    sources,
                ),
            )

