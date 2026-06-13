import asyncio
import json
import re
import threading
import time
import unicodedata
from typing import Dict, Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.agents.agent_schema import AgentDecision
from app.core.config import settings
from app.core.model_loader import model_loader

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_router_duration = Histogram(
    "agent_router_duration_seconds",
    "Agent router duration",
    ["route", "status"],
)
_router_errors = Counter(
    "agent_router_errors_total",
    "Agent router errors by type",
    ["error_type"],
)
_router_decisions = Counter(
    "agent_router_decisions_total",
    "Agent router decision counts by action",
    ["action", "method"],
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

# HARD RULE KEYWORD SETS
_RECENT_WORDS     = {"latest", "today", "news", "recent", "update", "current", "now", "live"}
# Web/market-data signals — queries mentioning these almost certainly need live external
# data that won't be inside an ingested document (stock prices, analyst ratings, etc.).
# NOTE: "stock" alone is intentionally excluded — "stock price reaction mentioned in the
# CNBC video" asks about KB content, not live data. Only include terms that ALWAYS imply
# live external data when not paired with a media-reference phrase.
_WEB_WORDS        = {"stocks", "analyst", "analysts", "consensus", "sentiment",
                     "market cap", "ticker", "earnings reaction", "post-earnings"}

# If any of these phrases appears, the query is asking about content inside an ingested
# media file — force RAG regardless of other signals (e.g. "stock" keyword).
_MEDIA_REFERENCE_PHRASES = frozenset({
    "mentioned in the", "mentioned in", "in the video", "in the audio",
    "in the clip", "from the video", "from the audio", "shown in the",
    "in the cnbc", "in the earnings video", "in the highlight",
    "according to the video", "according to the audio",
})
# Explicit web-request phrases — user directly asked to fetch from the internet.
# These force a pure "search" route (not hybrid) so file citations never mix in.
_EXPLICIT_WEB_PHRASES = frozenset({
    "from web", "from the web", "search web", "search the web",
    "get from web", "get it from web", "web search", "find online",
    "search online", "look online", "from internet", "from the internet",
    "find on the internet", "look it up",
    "search the internet", "search internet", "search for current",
    "search for historical", "search for recent", "search for latest",
})
_MEMORY_WORDS     = {
    "earlier", "previous", "last time", "we discussed", "you said", "before",
    "you mentioned", "earlier you", "last conversation", "what did we",
    "you told me", "earlier you said", "what did we talk", "we talked about",
    "our conversation", "you mentioned earlier", "recall", "remember when",
    "previously you", "in our last",
}
_COMPLEX_KEYWORDS = {"compare", "difference", "process", "steps", "vs", "versus", "explain"}
_REASONING_WORDS  = {"why", "how", "explain", "reason", "cause", "because"}
_MULTIMODAL_WORDS = {"image", "video", "diagram", "chart", "audio", "photo", "picture", "figure"}
_CODE_WORDS       = {"code", "function", "implement", "script", "syntax", "class", "debug"}
_GREETING_WORDS   = {"hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", "good morning", "good evening", "how are you", "how r you"}
_MATH_WORDS       = {"calculate", "compute", "solve", "equation", "formula", "integral", "derivative"}
# Arithmetic pattern: digits with operators e.g. "2+2", "10 * 5", "100/4"
_ARITHMETIC_RE    = re.compile(r"^\s*\d[\d\s]*[+\-*/^%]\s*\d[\d\s]*[=?]?\s*$")
_SECURITY_WORDS   = {"password", "credential", "secret", "token", "api key", "hack", "exploit"}

# Finance domain routing keyword sets (Phase 7)
_MARKET_DATA: frozenset = frozenset({
    "stock price", "market cap", "p/e", "pe ratio", "analyst target",
    "price target", "consensus", "52-week", "trading at", "share price",
})
_EARNINGS_CALL: frozenset = frozenset({
    "earnings call", "conference call", "transcript", "said about",
    "cfo said", "ceo said", "management said", "what did the cfo",
    "what did the ceo", "prepared remarks",
})
_REGULATORY: frozenset = frozenset({
    "10-k", "10-q", "8-k", "sec filing", "annual report", "proxy",
    "form 10", "10k", "10q", "regulatory filing", "disclosure",
})
_FINANCIAL_MODEL: frozenset = frozenset({
    "dcf", "lbo", "valuation", "projection", "sensitivity",
    "wacc", "financial model", "assumptions", "model output",
})


# NORMALIZE QUERY

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    return " ".join(query.strip().split())


# INJECTION PATTERN DETECTION — delegates to unified guardrail (Issue A fix)

def _detect_injection(query: str) -> bool:
    from app.guardrails.input_guard import _check_injection, _load_policy
    _load_policy()
    return _check_injection(query) is not None


class AgentRouter:

    # NORMALIZE

    def _normalize(self, query: str) -> str:
        return _normalize(query)

    # MAIN ROUTE

    def route(self, query: str, session_id: str) -> AgentDecision:

        if not query:
            return self._decision("direct", "empty_query", 0.0, session_id)

        start = time.time()
        query = self._normalize(query)

        with tracer.start_as_current_span("agent_router") as span:
            span.set_attribute("query.length", len(query))
            span.set_attribute("session.id", session_id)

            try:
                # INJECTION GUARD
                if _detect_injection(query):
                    logger.warning(
                        "router_injection_detected",
                        query_prefix=query[:80],
                        session_id=session_id,
                    )
                    _router_decisions.labels(
                        action="reject",
                        method="injection_guard",
                    ).inc()
                    return self._decision("direct", "injection_detected", 0.0, session_id)

                signals = self._analyze(query)

                # HARD RULE: EXPLICIT WEB REQUEST — "get from web", "search online" etc.
                # Force pure search so no file citations contaminate the result.
                if any(phrase in query.lower() for phrase in _EXPLICIT_WEB_PHRASES):
                    d = self._decision("search", "explicit_web_request", 0.98, session_id)
                    _router_decisions.labels(action="search", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # HARD RULE: GREETING / CHITCHAT
                if signals["is_greeting"]:
                    d = self._decision("direct", "greeting_detected", 0.95, session_id)
                    _router_decisions.labels(action="direct", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # HARD RULE: CODE QUERY
                if signals["is_code"]:
                    d = self._decision("direct", "code_query", 0.9, session_id)
                    _router_decisions.labels(action="direct", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # HARD RULE: MATH QUERY
                if signals["is_math"]:
                    d = self._decision("direct", "math_query", 0.9, session_id)
                    _router_decisions.labels(action="direct", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # HARD RULE: MEDIA REFERENCE — query asks about content inside an ingested
                # video/audio/image file (e.g. "mentioned in the CNBC video").
                # Force pure RAG so web results never contaminate KB-grounded answers.
                if any(phrase in query.lower() for phrase in _MEDIA_REFERENCE_PHRASES):
                    d = self._decision("rag", "media_reference_kb", 0.97, session_id)
                    _router_decisions.labels(action="rag", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # HARD RULE: RECENT QUERY — RAG + web fusion
                if signals["is_recent"]:
                    d = self._decision("hybrid", "recent_hybrid", 0.95, session_id)
                    _router_decisions.labels(action="hybrid", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # HARD RULE: MEMORY REFERENCE
                if signals["is_memory"]:
                    d = self._decision("memory", "memory_reference", 0.9, session_id)
                    _router_decisions.labels(action="memory", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # HARD RULE: WEB/MARKET DATA — analyst ratings, consensus ratings
                # are live external data not found in any document.
                if signals["is_web"]:
                    d = self._decision("hybrid", "web_market_signal", 0.88, session_id)
                    _router_decisions.labels(action="hybrid", method="hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # FINANCE DOMAIN HARD RULES (Phase 7)
                if signals.get("is_market_data_query"):
                    d = self._finance_decision(
                        "search", "market_data_live", 0.93, session_id, signals,
                    )
                    _router_decisions.labels(action="search", method="finance_hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                if signals.get("is_earnings_call_query"):
                    d = self._finance_decision(
                        "rag", "earnings_call_kb", 0.92, session_id, signals,
                        modality_hint="mp3",
                    )
                    _router_decisions.labels(action="rag", method="finance_hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                if signals.get("is_regulatory_query"):
                    d = self._finance_decision(
                        "rag", "regulatory_filing_kb", 0.91, session_id, signals,
                        source_type_filter=["pdf", "docx"],
                    )
                    _router_decisions.labels(action="rag", method="finance_hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                if signals.get("is_financial_model_query"):
                    d = self._finance_decision(
                        "rag", "financial_model_kb", 0.90, session_id, signals,
                        subtype_filter=["table", "assumptions"],
                    )
                    _router_decisions.labels(action="rag", method="finance_hard_rule").inc()
                    self._log_decision(d, signals, start, session_id)
                    return d

                # LLM ROUTING FOR AMBIGUOUS QUERIES
                try:
                    decision = self._llm_route(query, signals, session_id)
                except Exception as exc:
                    logger.warning(
                        "router_llm_route_failed",
                        error=str(exc),
                        session_id=session_id,
                    )
                    decision = self._decision("rag", "llm_route_exception", 0.5, session_id)
                validated = self._validate(decision, signals, session_id)
                validated.set_latency(start)

                _router_decisions.labels(
                    action=validated.action,
                    method="llm",
                ).inc()
                _router_duration.labels(
                    route=validated.action,
                    status="success",
                ).observe(validated.latency_ms / 1000.0 if validated.latency_ms else 0)

                span.set_attribute("decision.action", validated.action)
                span.set_attribute("decision.confidence", validated.confidence)
                span.set_status(Status(StatusCode.OK))

                self._log_decision(validated, signals, start, session_id)

                return validated

            except Exception as exc:
                latency    = round(time.time() - start, 3)
                error_type = type(exc).__name__

                _router_errors.labels(error_type=error_type).inc()
                _router_duration.labels(route="fallback", status="error").observe(latency)

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "router_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )

                return self._decision("rag", "router_exception", 0.5, session_id)

    # LOG DECISION

    def _log_decision(
        self,
        decision: AgentDecision,
        signals: Dict,
        start: float,
        session_id: str,
    ) -> None:
        logger.info(
            "router_final",
            action=decision.action,
            confidence=decision.confidence,
            reason=decision.reason,
            signal_summary={k: v for k, v in signals.items() if v},
            latency_ms=round((time.time() - start) * 1000, 1),
            session_id=session_id,
        )

    # SIGNAL ANALYSIS

    def _analyze(self, query: str) -> Dict:
        q      = query.lower()
        tokens = set(q.split())

        is_market_data      = any(phrase in q for phrase in _MARKET_DATA)
        is_earnings_call    = any(phrase in q for phrase in _EARNINGS_CALL)
        is_regulatory       = any(phrase in q for phrase in _REGULATORY)
        is_financial_model  = any(phrase in q for phrase in _FINANCIAL_MODEL)

        return {
            "is_recent":              bool(tokens & _RECENT_WORDS),
            "is_web":                 bool(tokens & _WEB_WORDS),
            "is_memory":              any(
                re.search(r"\b" + re.escape(w) + r"\b", q, re.IGNORECASE)
                for w in _MEMORY_WORDS
            ),
            "is_complex":             (
                len(tokens) > settings.DECOMPOSITION_MIN_WORDS or
                bool(tokens & _COMPLEX_KEYWORDS)
            ),
            "is_reasoning":           bool(tokens & _REASONING_WORDS),
            "has_multimodal_hint":    bool(tokens & _MULTIMODAL_WORDS),
            "is_code":                bool(tokens & _CODE_WORDS),
            "is_greeting":            (bool(tokens & _GREETING_WORDS) and len(tokens) <= 6)
                                      or bool(re.search(r"\bhow are you\b", q)),
            "is_math":                bool(tokens & _MATH_WORDS) or bool(_ARITHMETIC_RE.match(q)),
            "is_security":            bool(tokens & _SECURITY_WORDS),
            "token_count":            len(tokens),
            "has_question_mark":      "?" in query,
            "multi_question":         query.count("?") > 1,
            # Finance domain signals
            "is_market_data_query":      is_market_data,
            "is_earnings_call_query":    is_earnings_call,
            "is_regulatory_query":       is_regulatory,
            "is_financial_model_query":  is_financial_model,
        }

    # PROMPT BUILDER

    def _build_prompt(self, query: str, signals: dict) -> str:
        instruction = (
            "Route the query to exactly ONE of these options:\n"
            "rag | search | direct | memory | hybrid\n\n"
            "ROUTING RULES:\n"
            "- rag: ANY question that asks for a specific fact, name, number, date,\n"
            "  specification, person, project, product, organization, or event.\n"
            "  This is the DEFAULT for any question containing 'what', 'who',\n"
            "  'when', 'where', 'which', or 'how much/many'. Always prefer rag\n"
            "  when the answer could plausibly live in an ingested document.\n"
            "- search: recent events, news, today's information, live data only\n"
            "- direct: ONLY pure chitchat, greetings, math arithmetic, or code\n"
            "  syntax help. NEVER use direct for factual questions about\n"
            "  companies, people, products, or technical specifications.\n"
            "- memory: explicit references to previous conversation turns\n"
            "- hybrid: complex multi-part questions needing both docs and search\n\n"
            "When uncertain, choose rag. Return JSON only. No explanation.\n\n"
        )

        signal_str   = str({k: v for k, v in signals.items() if v})
        body         = f"Signals: {signal_str}\nQuery: {query}\n"
        format_block = '{"action":"<route>", "reason":"<brief reason>"}'

        max_chars  = settings.MAX_PROMPT_CHARS
        available  = max_chars - len(instruction) - len(format_block) - 50

        return instruction + body[:max(available, 0)] + "\n" + format_block

    # LLM ROUTING WITH RETRY

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _llm_route(
        self,
        query: str,
        signals: dict,
        session_id: str,
    ) -> AgentDecision:

        try:
            llm = model_loader.get_llm()
        except Exception:
            return self._decision("rag", "llm_unavailable", 0.5, session_id)

        prompt = self._build_prompt(query, signals)

        try:
            t_start  = time.time()
            response = llm.generate(
                prompt,
                max_tokens=80,
                temperature=0.0,
                top_p=1.0,
                session_id=session_id,
            )
            elapsed = time.time() - t_start

            if elapsed > settings.AGENT_STEP_TIMEOUT_SEC:
                logger.warning(
                    "router_llm_timeout",
                    elapsed=round(elapsed, 2),
                    session_id=session_id,
                )
                return self._decision("rag", "llm_timeout", 0.5, session_id)

            return self._parse(response, signals, session_id)

        except Exception as exc:
            logger.warning(
                "router_llm_failed",
                error=str(exc),
                session_id=session_id,
            )
            return self._decision("rag", "llm_failure", 0.5, session_id)

    # JSON EXTRACTION

    def _extract_json(self, text: str) -> str:
        text = text.strip()

        if "```" in text:
            parts = text.split("```")
            text  = next((p for p in parts if "{" in p), text)

        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("NO_JSON_FOUND")

    # PARSE LLM RESPONSE

    def _parse(
        self,
        text: str,
        signals: dict,
        session_id: str,
    ) -> AgentDecision:
        try:
            cleaned = self._extract_json(text)
            data    = json.loads(cleaned)

            action = str(data.get("action", "")).lower().strip()
            reason = str(data.get("reason", ""))

            confidence = self._score_confidence(action, signals)

            return AgentDecision(
                action=action,
                reason=reason,
                confidence=confidence,
                signals=signals,
                session_id=session_id,
            )

        except Exception:
            return self._decision("rag", "parse_failure", 0.6, session_id)

    # CONFIDENCE SCORING

    def _score_confidence(self, action: str, signals: dict) -> float:
        score = 0.6

        if signals.get("is_recent") and action in {"search", "hybrid"}:
            score += 0.30

        if signals.get("is_memory") and action == "memory":
            score += 0.25

        if signals.get("is_complex") and action in {"rag", "hybrid"}:
            score += 0.20

        if signals.get("has_multimodal_hint") and action in {"rag", "hybrid"}:
            score += 0.10

        if signals.get("is_reasoning") and action in {"rag", "direct"}:
            score += 0.05

        if signals.get("multi_question") and action in {"hybrid", "rag"}:
            score += 0.05

        return min(round(score, 3), 0.95)

    # VALIDATE DECISION

    _INTERROGATIVES = {"what", "who", "when", "where", "which", "whose", "whom"}

    def _validate(
        self,
        decision: AgentDecision,
        signals: dict,
        session_id: str,
    ) -> AgentDecision:

        if decision.action not in {"rag", "search", "direct", "memory", "hybrid"}:
            return self._decision("rag", "invalid_action_fallback", 0.5, session_id)

        # OVERRIDE: RECENT SIGNAL ROUTES TO HYBRID — RAG + web fusion
        if signals.get("is_recent"):
            return self._decision("hybrid", "override_recent_hybrid", 0.95, session_id)

        # OVERRIDE: SECURITY KEYWORDS FORCE DIRECT — AVOID RAG ON CREDENTIALS
        if signals.get("is_security") and decision.action == "rag":
            return self._decision("direct", "override_security_signal", 0.8, session_id)

        # OVERRIDE: MULTI-QUESTION BENEFITS FROM HYBRID
        if signals.get("multi_question") and decision.action == "direct":
            return self._decision("hybrid", "override_multi_question", 0.75, session_id)

        # OVERRIDE: LLM picked `direct` but the query looks like a factual
        # lookup (interrogative + non-trivial length + not greeting/code/math).
        # Force rag so we never hallucinate a doc answer from parametric memory.
        if (
            decision.action == "direct"
            and not signals.get("is_greeting")
            and not signals.get("is_code")
            and not signals.get("is_math")
            and signals.get("token_count", 0) >= 5
        ):
            return self._decision(
                "rag", "override_factual_lookup", 0.8, session_id,
            )

        return decision

    # DECISION FACTORY

    def _decision(
        self,
        action: str,
        reason: str,
        confidence: float,
        session_id: str = "default",
    ) -> AgentDecision:
        return AgentDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            signals={},
            session_id=session_id,
        )

    # FINANCE DECISION FACTORY — carries retrieval filter hints

    def _finance_decision(
        self,
        action: str,
        reason: str,
        confidence: float,
        session_id: str,
        signals: dict,
        modality_hint: Optional[str] = None,
        source_type_filter: Optional[list] = None,
        subtype_filter: Optional[list] = None,
        call_section_filter: Optional[str] = None,
    ) -> AgentDecision:
        from app.agents.agent_schema import AgentSignals
        return AgentDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            signals=signals,
            session_id=session_id,
            modality_hint=modality_hint,
            source_type_filter=source_type_filter or [],
            subtype_filter=subtype_filter or [],
            call_section_filter=call_section_filter,
        )

    # ASYNC ROUTE WRAPPER

    async def route_async(
        self,
        query: str,
        session_id: str,
    ) -> AgentDecision:

        async with _get_semaphore():
            return await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.route(query, session_id),
            )

