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

# FINANCE DOMAIN SCORE BOOST (Phase 7)
_FINANCE_BOOST_DOMAINS: Dict[str, float] = {
    "sec.gov":       1.30,
    "ft.com":        1.20,
    "reuters.com":   1.20,
    "bloomberg.com": 1.20,
    "wsj.com":       1.15,
    "cnbc.com":      1.10,
    "marketwatch.com": 1.10,
}
# Finance query detection for topic= param and freshness filter
_FINANCE_QUERY_RE = _re.compile(
    r'\b(?:revenue|earnings|eps|stock|dividend|margin|quarterly|annual report'
    r'|10-k|10-q|8-k|sec|investor|analyst|fiscal|guidance|EBITDA|P/E|valuation)\b',
    _re.IGNORECASE,
)
_YEAR_RE = _re.compile(r'\b20\d{2}\b')
_FRESHNESS_CUTOFF_DAYS = 90


def _is_finance_query(query: str) -> bool:
    return bool(_FINANCE_QUERY_RE.search(query))


def _parse_date(date_str: str) -> Optional[datetime.date]:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(date_str[:19], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


# SSRF PREVENTION — delegated to consolidated ssrf.py guard (Phase 26)


# NORMALIZE QUERY

def _normalize(query: str) -> str:
    query = unicodedata.normalize("NFC", str(query or ""))
    return " ".join(query.strip().split())[:settings.MAX_PROMPT_CHARS]


# WEB ANSWER CLEANER — the small GGUF model sometimes narrates raw search
# results ("and here's what we found. | 1:30 AM PST | Dec 28, 2025 | by Sarah
# Perez | TechCrunch In this article, it is stated that ..."). This deterministic
# pass removes narration intros, pipe-separated article metadata, and self-
# referential "the article mentions" phrasing, leaving a clean factual answer.
import datetime
import re as _re

_WEB_NARRATION_RE = _re.compile(
    r"^\s*(?:and\s+)?here'?s what (?:we|i) (?:found|have)\b[.:]?\s*"
    r"|^\s*the question\s*:?\s*[^?]*\?\s*"                # echoed "The question: ...?"
    r"|\bin this article,?\s*it is stated that\s*"
    r"|\bthe article (?:also )?(?:mentions|states|notes) that\s*"
    r"|\bit (?:also )?mentions that\s*"
    r"|\baccording to the (?:article|results?),?\s*"
    r"|\b(?:however,?\s*)?it'?s important to note that\s*"  # hedge filler
    r"|\bbased on the (?:available data|web search results?),?\s*",
    _re.IGNORECASE,
)
# "The answer, according to X, is" → "according to X, it is"  (keeps attribution).
_WEB_ANSWER_REPHRASE_RE = _re.compile(
    r'\bthe answer,?\s*according to\s+([^,]+?),?\s*is\b', _re.IGNORECASE,
)
_WEB_ANSWER_PLAIN_RE = _re.compile(r'\bthe answer\s+is\b', _re.IGNORECASE)
# Runs of short pipe-separated metadata cells: "| 1:30 AM PST | Dec 28, 2025 |
# by Sarah Perez |". Requires a trailing pipe so the run stops at the last
# delimiter and never consumes the prose sentence that follows it.
_WEB_PIPE_META_RE = _re.compile(r'(?:\s*\|\s*[^|\n]{1,32}){2,}\s*\|')
# A lone publication/byline token left dangling right before the real sentence
# (e.g. "TechCrunch In this article ...") once the pipe run is removed.
_WEB_ORPHAN_LEAD_RE = _re.compile(
    r'^\s*(?:by\s+)?[A-Z][A-Za-z0-9.\']{1,20}(?:\s+[A-Z][A-Za-z0-9.\']{1,20}){0,2}\s+'
    r'(?=(?:in this article|it is stated|tim |the |a |an |according))',
    _re.IGNORECASE,
)


def _clean_web_answer(text: str) -> str:
    if not text:
        return text
    out = text.strip()
    out = _WEB_PIPE_META_RE.sub(" ", out)             # strip article metadata runs
    out = _WEB_ANSWER_REPHRASE_RE.sub(r'according to \1, it is', out)
    out = _WEB_ANSWER_PLAIN_RE.sub('it is', out)
    # Remove narration intros / self-references anywhere they appear (may chain).
    for _ in range(5):
        new = _WEB_NARRATION_RE.sub("", out)
        new = _WEB_ORPHAN_LEAD_RE.sub("", new)
        if new == out:
            break
        out = new
    out = _re.sub(r'^\s*answers?\s*:\s*', '', out, flags=_re.IGNORECASE)  # "Answer:" prefix
    out = _re.sub(r'^\s*[:\-—|]\s*', '', out)          # leading stray punctuation
    out = _re.sub(r'\s{2,}', ' ', out).strip()
    # Re-capitalise the first letter of each sentence (clause removals can leave
    # a lowercase word exposed after a period).
    out = _re.sub(r'(^|[.!?]\s+)([a-z])',
                  lambda m: m.group(1) + m.group(2).upper(), out)
    return out


# SHA-256 HASH FOR DEDUP

def _hash(text: str) -> str:
    return hashlib.sha256(text[:300].encode("utf-8")).hexdigest()


# SSRF GUARD — delegates to consolidated ssrf.py (Phase 26)

def _is_ssrf_risk(url: str) -> bool:
    from app.guardrails.ssrf import is_ssrf_risk
    return is_ssrf_risk(url)


# DOMAIN BLOCK CHECK

def _is_blocked(url: Optional[str]) -> bool:
    if not url:
        return False
    if _is_ssrf_risk(url):
        logger.warning("web_search_ssrf_blocked", url=url[:100])
        return True
    return any(domain in url for domain in _BLOCKED_DOMAINS)


# INJECTION SANITIZATION FOR SEARCH QUERIES

def _sanitize_query(query: str) -> str:
    from app.guardrails.input_guard import sanitize as _guard_sanitize
    return _guard_sanitize(query, surface="web_search")


# RESULT QUALITY SCORING

def _quality_score(result: Dict, finance_query: bool = False) -> float:
    score   = float(result.get("score", 0.5))
    content = str(result.get("content", "") or "")
    title   = str(result.get("title", "") or "")
    url     = str(result.get("url", "") or "")

    # LENGTH BONUS
    if len(content) > 500:
        score = min(score + 0.1, 1.0)

    # TITLE PRESENCE BONUS
    if title and len(title) > 5:
        score = min(score + 0.05, 1.0)

    # FINANCE DOMAIN CREDIBILITY BOOST
    if finance_query:
        for domain, mult in _FINANCE_BOOST_DOMAINS.items():
            if domain in url:
                score = min(score * mult, 1.0)
                break

    # BLOCKED DOMAIN PENALTY
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

                # DETECT FINANCE QUERY FOR TOPIC PARAMETER + FRESHNESS FILTER
                is_finance      = _is_finance_query(query)
                has_year        = bool(_YEAR_RE.search(query))

                # TAVILY API CALL
                t_api      = time.time()
                raw        = self._search(query, session_id, is_finance=is_finance)
                api_latency = round(time.time() - t_api, 3)

                _api_duration.observe(api_latency)

                if api_latency > settings.RETRIEVAL_TIMEOUT:
                    logger.warning(
                        "web_search_api_slow",
                        api_latency=api_latency,
                        threshold=settings.RETRIEVAL_TIMEOUT,
                        session_id=session_id,
                    )

                processed = self._process(raw, is_finance=is_finance, has_year=has_year)

                if not processed["documents"]:
                    logger.warning("web_search_no_documents", session_id=session_id)
                    return self._empty()

                # RERANK WEB RESULTS BY QUERY RELEVANCE (H-04)
                reranked_docs, reranked_sources = self._rerank_web_docs(
                    query,
                    processed["_raw_docs"],
                    processed["_raw_sources"],
                )
                processed["documents"] = reranked_docs[:self.max_docs]
                processed["sources"]   = reranked_sources[:self.max_docs]

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
    def _search(self, query: str, session_id: str = "", is_finance: bool = False) -> Dict:
        try:
            kwargs: Dict[str, Any] = dict(
                query=query,
                search_depth=settings.WEB_SEARCH_DEPTH,
                max_results=self.max_results,
            )
            if is_finance:
                kwargs["topic"] = "finance"
            return self.client.search(**kwargs)
        except Exception as exc:
            logger.error(
                "tavily_api_failed",
                error=str(exc),
                session_id=session_id,
            )
            return {}

    # PROCESS AND FILTER RESULTS

    def _process(
        self,
        response: Dict,
        is_finance: bool = False,
        has_year: bool = False,
    ) -> Dict[str, List]:
        documents: List[str]  = []
        sources:   List[str]  = []
        seen:      set        = set()

        results = response.get("results", [])
        cutoff  = (datetime.date.today() - datetime.timedelta(days=_FRESHNESS_CUTOFF_DAYS))

        # SORT BY QUALITY SCORE (with finance domain boost when applicable)
        results = sorted(
            results,
            key=lambda r: _quality_score(r, finance_query=is_finance),
            reverse=True,
        )

        for r in results:
            url        = r.get("url", "")
            content    = str(r.get("content", "") or "").strip()
            title      = str(r.get("title", "") or "").strip()
            pub_date   = r.get("published_date") or r.get("published_at") or ""

            # SSRF + BLOCKED DOMAIN GUARD
            if _is_blocked(url):
                continue

            if len(content) < 40:
                continue

            # FRESHNESS FILTER — drop stale market/finance results unless query pins a year
            if is_finance and not has_year and pub_date:
                parsed = _parse_date(pub_date)
                if parsed and parsed < cutoff:
                    logger.debug("web_search_stale_result_dropped", url=url[:80], pub_date=pub_date)
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
            "_raw_docs": documents,  # kept for reranking
            "_raw_sources": list(dict.fromkeys(sources)),
        }

    def _rerank_web_docs(
        self,
        query: str,
        documents: List[str],
        sources: List[str],
    ) -> tuple[List[str], List[str]]:
        """Rerank web documents by relevance to the query using CrossEncoder."""
        if len(documents) <= 1:
            return documents, sources
        try:
            from app.core.model_loader import model_loader
            reranker = model_loader.get_reranker()
            if reranker is None:
                return documents, sources
            pairs = [(query, doc[:512]) for doc in documents]
            scores = reranker.predict(pairs)
            ranked = sorted(
                zip(scores, documents, sources + [""] * len(documents)),
                key=lambda x: x[0],
                reverse=True,
            )
            docs_out    = [r[1] for r in ranked][:self.max_docs]
            sources_out = [r[2] for r in ranked if r[2]][:self.max_docs]
            return docs_out, sources_out
        except Exception as exc:
            logger.warning("web_rerank_failed", error=str(exc))
            return documents, sources

    # LLM SUMMARIZATION OF SEARCH RESULTS

    def _summarize(
        self,
        query: str,
        docs: List[str],
        session_id: str = "default",
    ) -> str:

        context = "\n\n".join(docs)[:self.max_context_chars]

        instruction = (
            "You are a web research assistant. Using ONLY the web search results "
            "below, write a direct, factual answer to the QUERY in 1-3 clean "
            "sentences.\n"
            "Rules:\n"
            "- State the facts directly. Do NOT narrate (no 'here's what we "
            "found', 'in this article it is stated', 'the article also "
            "mentions').\n"
            "- Do NOT include article timestamps, dates, author bylines, or "
            "publication names as inline or pipe-separated metadata.\n"
            "- Preserve ALL dollar amounts, percentages, and dates verbatim. "
            "Do NOT paraphrase or round any numbers.\n"
            "- Do NOT restate these rules or add labels.\n"
            "- If the results do not answer the query, say 'I don't know based "
            "on available data'.\n\n"
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

            return _clean_web_answer(response or "") or "No answer generated."

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

