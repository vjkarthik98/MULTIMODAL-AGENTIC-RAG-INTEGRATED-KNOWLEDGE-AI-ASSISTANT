import asyncio
import hashlib
import time
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.model_loader import model_loader

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_search_duration = Histogram(
    "web_search_duration_seconds",
    "Web search total duration",
    ["status"],
)
_search_errors = Counter(
    "web_search_errors_total",
    "Web search errors by type",
    ["error_type"],
)
_search_results_count = Histogram(
    "web_search_results_count",
    "Number of web search results used",
)
_api_duration = Histogram(
    "web_search_api_duration_seconds",
    "Tavily API call duration",
)

# SEMAPHORE
_semaphore = asyncio.Semaphore(5)

# LOW QUALITY DOMAIN BLOCKLIST
_BLOCKED_DOMAINS: Set[str] = {
    "pinterest.com",
    "quora.com",
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    "tumblr.com",
    "youtube.com",
    "youtu.be",
    "linkedin.com",
}

# SSRF PREVENTION — BLOCK PRIVATE IP RANGES
_PRIVATE_IP_PATTERNS = [
    "127.",
    "192.168.",
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "localhost",
    "0.0.0.0",
    "::1",
    "169.254.",
]


# NORMALIZE QUERY

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    return " ".join(query.strip().split())[:settings.MAX_PROMPT_CHARS]


# SHA-256 HASH FOR DEDUP

def _hash(text: str) -> str:
    return hashlib.sha256(text[:300].encode("utf-8")).hexdigest()


# SSRF GUARD — BLOCK PRIVATE IP RANGES AND LOCALHOST

def _is_ssrf_risk(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in _PRIVATE_IP_PATTERNS)


# DOMAIN BLOCK CHECK

def _is_blocked(url: Optional[str]) -> bool:
    if not url:
        return False
    if _is_ssrf_risk(url):
        logger.warning("web_search_ssrf_blocked", url=url[:100])
        return True
    return any(domain in url for domain in _BLOCKED_DOMAINS)


# INJECTION SANITIZATION FOR SEARCH QUERIES

_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all instructions",
    "disregard",
    "forget everything",
    "you are now",
    "act as",
    "jailbreak",
    "system prompt",
]


def _sanitize_query(query: str) -> str:
    lower = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            idx   = query.lower().find(pattern)
            query = query[:idx].strip()
            logger.warning("web_search_injection_sanitized", pattern=pattern)
            break
    return query


# RESULT QUALITY SCORING

def _quality_score(result: Dict) -> float:
    score   = float(result.get("score", 0.5))
    content = str(result.get("content", "") or "")
    title   = str(result.get("title", "") or "")

    # LENGTH BONUS
    if len(content) > 500:
        score = min(score + 0.1, 1.0)

    # TITLE PRESENCE BONUS
    if title and len(title) > 5:
        score = min(score + 0.05, 1.0)

    # BLOCKED DOMAIN PENALTY
    url = result.get("url", "")
    if _is_blocked(url):
        score = 0.0

    return round(score, 3)


class WebSearchTool:

    def __init__(self) -> None:
        if not settings.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY_MISSING")

        from tavily import TavilyClient
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        self.max_results       = settings.WEB_MAX_RESULTS
        self.max_docs          = settings.WEB_MAX_DOCS
        self.max_doc_chars     = settings.WEB_DOC_MAX_CHARS
        self.max_context_chars = settings.WEB_CONTEXT_MAX_CHARS

        logger.info("web_search_tool_initialized")

    # MAIN EXECUTE

    def execute(
        self,
        query: str,
        context: Any = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:

        start = time.time()

        with tracer.start_as_current_span("web_search_execute") as span:
            span.set_attribute("session.id", session_id)
            span.set_attribute("query.length", len(query))

            if not query:
                return self._empty()

            try:
                query = _normalize(query)
                query = _sanitize_query(query)

                if not query.strip():
                    return self._empty()

                # TAVILY API CALL
                t_api      = time.time()
                raw        = self._search(query, session_id)
                api_latency = round(time.time() - t_api, 3)

                _api_duration.observe(api_latency)

                if api_latency > settings.RETRIEVAL_TIMEOUT:
                    logger.warning(
                        "web_search_api_slow",
                        api_latency=api_latency,
                        threshold=settings.RETRIEVAL_TIMEOUT,
                        session_id=session_id,
                    )

                processed = self._process(raw)

                if not processed["documents"]:
                    logger.warning("web_search_no_documents", session_id=session_id)
                    return self._empty()

                # LLM SUMMARIZATION
                t_llm       = time.time()
                answer      = self._summarize(query, processed["documents"], session_id)
                llm_latency = round(time.time() - t_llm, 3)

                confidence  = self._confidence(processed)
                latency     = round(time.time() - start, 3)

                _search_duration.labels(status="success").observe(latency)
                _search_results_count.observe(len(processed["documents"]))

                span.set_attribute("results.count", len(processed["documents"]))
                span.set_attribute("confidence", confidence)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "web_search_success",
                    results_used=len(processed["documents"]),
                    api_latency=api_latency,
                    llm_latency=llm_latency,
                    latency=latency,
                    session_id=session_id,
                )

                return {
                    "answer":    answer,
                    "sources":   processed["sources"],
                    "documents": processed["documents"],
                    "confidence": confidence,
                    "metadata": {
                        "results_used":  len(processed["documents"]),
                        "api_latency":   api_latency,
                        "llm_latency":   llm_latency,
                        "total_latency": latency,
                    },
                }

            except Exception as exc:
                latency    = round(time.time() - start, 3)
                error_type = type(exc).__name__

                _search_duration.labels(status="error").observe(latency)
                _search_errors.labels(error_type=error_type).inc()

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)

                logger.error(
                    "web_search_failed",
                    error=str(exc),
                    error_type=error_type,
                    session_id=session_id,
                )
                return self._error(str(exc))

    # TAVILY API CALL WITH RETRY

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _search(self, query: str, session_id: str = "") -> Dict:
        try:
            return self.client.search(
                query=query,
                search_depth=settings.WEB_SEARCH_DEPTH,
                max_results=self.max_results,
            )
        except Exception as exc:
            logger.error(
                "tavily_api_failed",
                error=str(exc),
                session_id=session_id,
            )
            return {}

    # PROCESS AND FILTER RESULTS

    def _process(self, response: Dict) -> Dict[str, List]:
        documents: List[str]  = []
        sources:   List[str]  = []
        seen:      set        = set()

        results = response.get("results", [])

        # SORT BY QUALITY SCORE
        results = sorted(
            results,
            key=lambda r: _quality_score(r),
            reverse=True,
        )

        for r in results:
            url     = r.get("url", "")
            content = str(r.get("content", "") or "").strip()
            title   = str(r.get("title", "") or "").strip()

            # SSRF + BLOCKED DOMAIN GUARD
            if _is_blocked(url):
                continue

            if len(content) < 40:
                continue

            # PREPEND TITLE FOR BETTER CONTEXT
            full_text = f"{title}: {content}" if title else content
            text      = full_text[:self.max_doc_chars]

            h = _hash(text)
            if h in seen:
                continue
            seen.add(h)

            documents.append(text)

            if url:
                sources.append(url)

        return {
            "documents": documents[:self.max_docs],
            "sources":   list(dict.fromkeys(sources))[:self.max_docs],
        }

    # LLM SUMMARIZATION OF SEARCH RESULTS

    def _summarize(
        self,
        query: str,
        docs: List[str],
        session_id: str = "default",
    ) -> str:

        context = "\n\n".join(docs)[:self.max_context_chars]

        instruction = (
            "Answer ONLY using the provided web search results.\n"
            "Rules:\n"
            "- Use ONLY the provided results\n"
            "- If insufficient → say 'I don't know based on available data'\n"
            "- Be concise and factual\n"
            "- No hallucination\n\n"
        )

        body    = f"WEB RESULTS:\n{context}\n\nQUERY:\n{query}\n\nAnswer:"
        allowed = settings.MAX_PROMPT_CHARS - len(instruction) - 50
        prompt  = instruction + body[:max(allowed, 0)]

        try:
            llm      = model_loader.get_llm()
            t_start  = time.time()

            response = llm.generate(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.1,
                session_id=session_id,
            )

            elapsed = time.time() - t_start

            if elapsed > settings.MODEL_TIMEOUT_SEC:
                logger.warning(
                    "web_search_llm_timeout",
                    elapsed=round(elapsed, 2),
                    session_id=session_id,
                )
                return "Summary took too long to generate."

            return (response or "").strip() or "No answer generated."

        except Exception as exc:
            logger.warning(
                "web_search_llm_failed",
                error=str(exc),
                session_id=session_id,
            )
            return "Summary unavailable."

    # CONFIDENCE SCORING

    def _confidence(self, processed: Dict) -> float:
        n = len(processed.get("documents", []))
        if n >= 5:
            return 0.85
        if n >= 3:
            return 0.70
        if n >= 1:
            return 0.50
        return 0.30

    # EMPTY RESPONSE

    def _empty(self) -> Dict[str, Any]:
        return {
            "answer":    "No relevant results found.",
            "sources":   [],
            "documents": [],
            "confidence": 0.3,
            "metadata":  {},
        }

    # ERROR RESPONSE

    def _error(self, msg: str) -> Dict[str, Any]:
        return {
            "answer":    "Search failed.",
            "sources":   [],
            "documents": [],
            "confidence": 0.2,
            "metadata":  {"error": msg},
        }

    # ASYNC EXECUTE WRAPPER

    async def execute_async(
        self,
        query: str,
        context: Any = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:

        async with _semaphore:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.execute(query, context, session_id),
            )

