import asyncio
import hashlib
import math
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

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)

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

    thr = threshold or getattr(settings, "HALLUCINATION_THRESHOLD", 0.5)

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
        for s in answer.replace("!", ".").replace("?", ".").split(".")
        if len(s.strip()) > 20
    ]

    if not sentences:
        return False, 1.0

    supported = 0
    for sentence in sentences:
        words = [w.lower() for w in sentence.split() if len(w) > 4]
        if not words:
            supported += 1
            continue
        # CHECK IF MAJORITY OF SIGNIFICANT WORDS APPEAR IN DOCS
        hits = sum(1 for w in words if w in all_doc_text)
        if hits / max(len(words), 1) >= 0.4:
            supported += 1

    support_score  = supported / max(len(sentences), 1)
    is_hallucinated = support_score < thr

    return is_hallucinated, round(support_score, 3)


# PII SCRUB ANSWER BEFORE RETURNING

def _scrub_answer_pii(text: str) -> str:
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
        logger.warning("pii_scrub_answer_failed", error=str(exc))
    return text


# KNOWLEDGE PREPARATION FROM RETRIEVED DOCS

def _prepare_knowledge(
    docs: List[Dict],
    max_docs: int = None,
    max_chars: int = None,
) -> str:
    if not docs:
        return ""

    max_docs  = max_docs  or settings.RAG_TOP_K
    max_chars = max_chars or settings.RAG_DOC_MAX_CHARS
    seen:  set        = set()
    parts: List[str]  = []

    for d in docs[:max_docs]:
        text = _normalize(d.get("text", ""))
        if not text:
            continue

        h = _hash(text[:200])
        if h in seen:
            continue
        seen.add(h)

        meta         = d.get("metadata", {}) or {}
        source       = meta.get("source", "unknown")
        modality     = meta.get("modality", "text")
        subtype      = meta.get("subtype", "")
        page         = meta.get("page")
        language     = meta.get("language")
        ts_start     = meta.get("timestamp_start")
        speaker      = meta.get("speaker")
        heading      = meta.get("heading_level")

        label = f"[{modality.upper()}"
        if subtype:
            label += f"/{subtype}"
        if page:
            label += f" | p{page}"
        if ts_start is not None:
            label += f" | t={ts_start}s"
        if speaker:
            label += f" | spk={speaker}"
        if language and language != "en":
            label += f" | lang={language}"
        label += f" | {source}]"

        parts.append(f"{label} {text[:max_chars]}")

    return "\n\n".join(parts)


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
) -> str:

    type_instructions = {
        "factual": (
            "Answer factually using ONLY the provided knowledge.\n"
            "State your reasoning step by step before giving the final answer."
        ),
        "comparative": (
            "Compare the entities mentioned using ONLY the provided knowledge.\n"
            "Step 1: Describe entity A.\n"
            "Step 2: Describe entity B.\n"
            "Step 3: State key differences and similarities.\n"
            "Step 4: Give a final comparative answer."
        ),
        "temporal": (
            "Answer with temporal context using ONLY the provided knowledge.\n"
            "Step 1: Identify relevant time periods.\n"
            "Step 2: Order events chronologically.\n"
            "Step 3: Give a final temporal answer."
        ),
        "aggregation": (
            "Aggregate information using ONLY the provided knowledge.\n"
            "Step 1: List all relevant items found.\n"
            "Step 2: Count or summarize as needed.\n"
            "Step 3: Give a final aggregated answer."
        ),
        "multihop": (
            "Reason across multiple hops using ONLY the provided knowledge.\n"
            "Step 1: Answer sub-question 1.\n"
            "Step 2: Use that answer for sub-question 2.\n"
            "Step 3: Synthesize into a final answer."
        ),
    }

    type_instruction = type_instructions.get(query_type, type_instructions["factual"])

    instruction = (
        "You are a grounded AI assistant.\n"
        "RULES:\n"
        "- Use ONLY the provided knowledge\n"
        "- If knowledge is insufficient → say 'I don't know based on available data'\n"
        "- No hallucination\n"
        "- Be concise and factual\n"
        f"- {type_instruction}\n\n"
    )

    memory_block    = f"MEMORY:\n{memory}\n\n"    if memory    else ""
    knowledge_block = f"KNOWLEDGE:\n{knowledge}\n\n" if knowledge else ""
    query_block     = f"QUERY:\n{query}\n\n"

    output_format = (
        "FORMAT:\n"
        "Reasoning: <step by step reasoning>\n"
        "Answer: <final answer>\n"
        "Confidence: <0.0-1.0>\n"
        "Sources Used: <integer>\n"
    )

    return instruction + memory_block + knowledge_block + query_block + output_format


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
        "Thought: <reasoning>\n"
        "Action: answer\n"
        "Answer: <final answer>\n"
        "Confidence: <0.0-1.0>\n"
        "Sources Used: <integer>\n"
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
) -> Dict[str, Any]:

    if not text:
        return _fallback_response()

    try:
        answer:     str   = ""
        confidence: float = 0.5
        sources:    int   = 0
        reasoning:  str   = ""

        for line in text.split("\n"):
            ll = line.lower().strip()

            if ll.startswith("answer:") or ll.startswith("answer "):
                answer = line.split(":", 1)[-1].strip()

            elif ll.startswith("reasoning:") or ll.startswith("reasoning "):
                reasoning = line.split(":", 1)[-1].strip()

            elif ll.startswith("confidence:") or ll.startswith("confidence "):
                try:
                    confidence = float(line.split(":", 1)[-1].strip())
                except Exception:
                    confidence = 0.5

            elif ll.startswith("sources used:") or ll.startswith("sources used "):
                try:
                    sources = int(line.split(":", 1)[-1].strip())
                except Exception:
                    sources = 0

        # FALLBACK IF ANSWER NOT CLEANLY PARSED
        if not answer or len(answer) < 10:
            answer = text.strip()

        # NaN/INF GUARD ON CONFIDENCE
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.5

        confidence = max(0.0, min(confidence, 1.0))

        # AUTO SOURCES COUNT
        if sources == 0 and docs:
            sources = min(
                len([d for d in docs if d.get("text")]),
                len(docs),
            )

        sources = min(sources, len(docs))

        return {
            "answer":       answer,
            "reasoning":    reasoning,
            "confidence":   confidence,
            "sources_used": sources,
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
        temperature: float = 0.1,
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
    ) -> Dict[str, Any]:

        if not query:
            return _fallback_response()

        start      = time.time()
        trace_log: List[Dict] = []

        with tracer.start_as_current_span("reasoning_generate_answer") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("docs.count", len(retrieved_docs))
            span.set_attribute("session.id", session_id)
            span.set_attribute("query.type", query_type)
            span.set_attribute("use_react", use_react)

            try:
                query    = _normalize(query)
                knowledge = _prepare_knowledge(retrieved_docs)
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

                # PARSE RESPONSE
                parsed = _parse_response(response, retrieved_docs)

                # HALLUCINATION GUARD
                t_verify = time.time()
                is_hallucinated, support_score = _hallucination_guard(
                    parsed["answer"],
                    retrieved_docs,
                )
                verify_latency = round(time.time() - t_verify, 3)

                _record_step(
                    trace_log,
                    STEP_VERIFY,
                    parsed["answer"][:200],
                    f"support={support_score}",
                    verify_latency,
                    {"hallucinated": is_hallucinated, "support_score": support_score},
                )

                if is_hallucinated:
                    _hallucination_flags.inc()
                    logger.warning(
                        "reasoning_hallucination_detected",
                        support_score=support_score,
                        session_id=session_id,
                    )
                    # DOWNGRADE CONFIDENCE WHEN HALLUCINATION DETECTED
                    parsed["confidence"] = min(parsed["confidence"], 0.4)
                    parsed["hallucination_warning"] = True
                else:
                    parsed["hallucination_warning"] = False

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
    ) -> Dict[str, Any]:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.generate_answer(
                    query,
                    retrieved_docs,
                    memory_context,
                    session_id,
                    query_type,
                    use_react,
                    step_history,
                ),
            )

