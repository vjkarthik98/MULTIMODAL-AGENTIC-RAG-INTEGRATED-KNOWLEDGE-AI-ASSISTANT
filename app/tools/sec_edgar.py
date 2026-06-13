"""
SEC EDGAR full-text search tool.

Uses the free EDGAR EFTS API (efts.us.sec.gov/hits.json) — no API key needed.
Returns filing metadata: accession_number, filing_date, company_name, cik,
form_type, document_url, snippet.

Called by the agent when is_regulatory_query=True, before web_search.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
import structlog

logger = structlog.get_logger(__name__)

_EDGAR_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index?q={q}&dateRange=custom&startdt={start}&enddt={end}&forms={forms}&hits.hits.total.value=true&hits.hits._source.period_of_report=true&hits.hits._source.entity_name=true&hits.hits._source.file_date=true&hits.hits._source.form_type=true&hits.hits._source.accession_no=true"
_EDGAR_SEARCH_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_DOC_BASE    = "https://www.sec.gov/Archives/edgar/data"
_TIMEOUT_SEC       = 10.0
_MAX_RESULTS       = 5


@dataclass
class EdgarResult:
    accession_number: str
    filing_date:      str
    company_name:     str
    cik:              str
    form_type:        str
    document_url:     str
    snippet:          str


def _edgar_url(query: str, form_type: str, ticker: Optional[str], date_range: Optional[tuple]) -> str:
    params: Dict[str, Any] = {
        "q":    f'"{query}"' if " " in query else query,
        "forms": form_type,
        "hits.hits._source.accession_no": "true",
        "hits.hits._source.entity_name":  "true",
        "hits.hits._source.file_date":    "true",
        "hits.hits._source.form_type":    "true",
        "hits.hits._source.period_of_report": "true",
    }
    if ticker:
        params["entity"] = ticker
    if date_range and len(date_range) == 2:
        params["dateRange"] = "custom"
        params["startdt"]   = date_range[0]
        params["enddt"]     = date_range[1]
    return f"https://efts.sec.gov/LATEST/search-index?{urlencode(params)}"


def _parse_hits(hits: List[Dict]) -> List[EdgarResult]:
    results: List[EdgarResult] = []
    for h in hits[:_MAX_RESULTS]:
        src = h.get("_source", {})
        acc = src.get("accession_no", "").replace("-", "")
        cik_raw = src.get("entity_id") or src.get("cik") or ""
        cik     = str(cik_raw).lstrip("0") or "0"

        if acc and cik:
            doc_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={src.get('form_type','')}&dateb=&owner=include&count=40"
        else:
            doc_url = "https://www.sec.gov/cgi-bin/browse-edgar"

        results.append(EdgarResult(
            accession_number = src.get("accession_no", ""),
            filing_date      = src.get("file_date", ""),
            company_name     = src.get("entity_name", ""),
            cik              = cik,
            form_type        = src.get("form_type", ""),
            document_url     = doc_url,
            snippet          = (h.get("highlight", {}).get("file_date", [""])[0] or
                                src.get("period_of_report", "")),
        ))
    return results


async def search_edgar(
    query: str,
    form_type: str = "10-K",
    ticker: Optional[str] = None,
    date_range: Optional[tuple] = None,
) -> List[EdgarResult]:
    """
    Free EDGAR EFTS full-text search.

    Args:
        query:      plain-text query (e.g. "revenue recognition policy")
        form_type:  SEC form (10-K, 10-Q, 8-K, etc.)
        ticker:     optional company ticker to narrow results
        date_range: optional (start_date, end_date) ISO strings e.g. ("2023-01-01","2024-12-31")

    Returns:
        Up to 5 EdgarResult objects.
    """
    url = _edgar_url(query, form_type, ticker, date_range)

    try:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=_TIMEOUT_SEC) as client:
            resp = await client.get(url, headers={"User-Agent": "multimodal-rag-assistant research@example.com"})
            resp.raise_for_status()
        data  = resp.json()
        hits  = (data.get("hits", {}).get("hits") or [])
        elapsed = round(time.time() - t0, 2)
        results = _parse_hits(hits)
        logger.info(
            "edgar_search_success",
            query=query[:80],
            form_type=form_type,
            ticker=ticker,
            hits=len(hits),
            returned=len(results),
            latency=elapsed,
        )
        return results
    except httpx.HTTPStatusError as exc:
        logger.warning("edgar_search_http_error", status=exc.response.status_code, query=query[:80])
        return []
    except Exception as exc:
        logger.warning("edgar_search_failed", error=str(exc), query=query[:80])
        return []


def search_edgar_sync(
    query: str,
    form_type: str = "10-K",
    ticker: Optional[str] = None,
    date_range: Optional[tuple] = None,
) -> List[EdgarResult]:
    """Synchronous wrapper for non-async callers."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    search_edgar(query, form_type, ticker, date_range),
                )
                return future.result(timeout=_TIMEOUT_SEC + 2)
        return loop.run_until_complete(
            search_edgar(query, form_type, ticker, date_range)
        )
    except Exception as exc:
        logger.warning("edgar_search_sync_failed", error=str(exc))
        return []
