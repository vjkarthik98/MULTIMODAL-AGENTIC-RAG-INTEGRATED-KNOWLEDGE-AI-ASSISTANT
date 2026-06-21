"""
Finance tool runner — wires financial_calculator and sec_edgar into the query pipeline.

Called from rag_pipeline.py and query_pipeline.py after reranking, before LLM reasoning.
Results are injected as synthetic docs so the LLM sees verified numbers and EDGAR filing
metadata in its context window.  EDGAR results also become clickable source chips in the UI.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

# ── FINANCIAL METRIC DETECTION ────────────────────────────────────────────────

# Priority order — more specific / less common metrics first so a query that
# mentions "EV/EBITDA margin" matches ev_ebitda rather than margin.
_METRIC_PRIORITY = [
    "ev_ebitda", "pe_ratio", "current_ratio", "debt_to_equity",
    "eps_basic", "eps_diluted", "cagr", "margin", "yoy_change",
]

_METRIC_SIGNALS: Dict[str, frozenset] = {
    "yoy_change": frozenset({
        "yoy", "year over year", "year-over-year",
        "grew by", "growth rate", "revenue growth", "revenue change",
        "increased by", "decreased by", "declined by",
        "compared to last year", "compared to prior year",
        "vs last year", "vs prior year", "prior year",
        "from last year", "from prior year", "annual change",
        "change from", "change in revenue",
    }),
    "margin": frozenset({
        "gross margin", "operating margin", "net margin", "profit margin",
        "margin %", "margin percentage", "what is the margin",
        "gross profit margin", "ebitda margin", "net profit margin",
    }),
    "cagr": frozenset({
        "cagr", "compound annual", "compounded annual",
        "5-year growth", "3-year growth", "annual growth rate",
        "over the past", "over the last", "historical growth",
    }),
    "eps_diluted": frozenset({
        "earnings per share", "eps", "diluted eps",
        "per share earnings", "diluted earnings per share",
    }),
    "eps_basic": frozenset({
        "basic eps", "basic earnings per share",
    }),
    "debt_to_equity": frozenset({
        "debt to equity", "d/e ratio", "debt-to-equity",
        "debt equity ratio", "leverage ratio",
    }),
    "current_ratio": frozenset({
        "current ratio", "liquidity ratio",
        "current assets to liabilities",
    }),
    "pe_ratio": frozenset({
        "p/e ratio", "pe ratio", "price to earnings",
        "price-to-earnings", "p/e multiple",
    }),
    "ev_ebitda": frozenset({
        "ev/ebitda", "ev ebitda", "enterprise multiple",
        "enterprise value to ebitda", "ev to ebitda",
    }),
}


def detect_finance_metric(query: str) -> Optional[str]:
    """Return the first matching metric name, or None if no arithmetic signal found."""
    q = query.lower()
    for metric in _METRIC_PRIORITY:
        if any(sig in q for sig in _METRIC_SIGNALS[metric]):
            return metric
    return None


# ── NUMBER EXTRACTION ─────────────────────────────────────────────────────────

_AMOUNT_RE = re.compile(
    r"\$?\s*(\d[\d,]*\.?\d*)\s*(trillion|billion|million|thousand|B|M|K|T|bn|mn)?\b",
    re.IGNORECASE,
)

_MULTIPLIERS: Dict[str, float] = {
    "trillion": 1e12, "t": 1e12,
    "billion":  1e9,  "b": 1e9,  "bn": 1e9,
    "million":  1e6,  "m": 1e6,  "mn": 1e6,
    "thousand": 1e3,  "k": 1e3,
}

_REVENUE_RE  = re.compile(r"(?:net\s+)?(?:sales|revenue|net\s+sales|total\s+revenue)", re.I)
_PROFIT_RE   = re.compile(r"(?:gross\s+profit|operating\s+income|net\s+income|ebitda|operating\s+profit)", re.I)
_DEBT_RE     = re.compile(r"(?:total\s+debt|long.?term\s+debt)\b", re.I)
_EQUITY_RE   = re.compile(r"(?:total\s+equity|shareholders.?\s+equity|stockholders.?\s+equity)", re.I)
_ASSETS_RE   = re.compile(r"(?:current\s+assets|total\s+current\s+assets)", re.I)
_LIAB_RE     = re.compile(r"(?:current\s+liabilities|total\s+current\s+liabilities)", re.I)
_SHARES_RE   = re.compile(r"(?:diluted\s+shares|shares\s+outstanding|weighted\s+average\s+shares)", re.I)
_EPS_RE      = re.compile(r"(?:earnings\s+per\s+share|diluted\s+eps|basic\s+eps|\beps\b)", re.I)
_YEARS_RE    = re.compile(r"\b(20\d{2})\b")


def _parse_amount(val_str: str, unit_str: str) -> float:
    val = float(val_str.replace(",", ""))
    return val * _MULTIPLIERS.get((unit_str or "").lower(), 1.0)


def _all_amounts(text: str) -> List[float]:
    out = []
    for m in _AMOUNT_RE.finditer(text):
        try:
            raw_val = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").lower()
            # Skip bare 4-digit years (1900-2099) when no unit suffix is present
            if not unit and 1900 <= raw_val <= 2099:
                continue
            val = raw_val * _MULTIPLIERS.get(unit, 1.0)
            if val > 0:
                out.append(val)
        except (ValueError, TypeError):
            pass
    return out


def _labeled_amount(text: str, label_re: re.Pattern, window: int = 140) -> Optional[float]:
    """Return the first amount within `window` chars after a label match."""
    for lm in label_re.finditer(text):
        snippet = text[lm.end(): lm.end() + window]
        for am in _AMOUNT_RE.finditer(snippet):
            try:
                val = _parse_amount(am.group(1), am.group(2) or "")
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return None


def extract_numbers_for_metric(text: str, metric: str) -> Optional[Dict[str, float]]:
    """Build the input dict the calculator expects for `metric` from raw doc text."""
    amounts = _all_amounts(text)
    if not amounts:
        return None

    if metric == "yoy_change":
        if len(amounts) >= 2:
            return {"current": amounts[0], "prior": amounts[1]}

    elif metric == "margin":
        numerator = _labeled_amount(text, _PROFIT_RE)
        revenue   = _labeled_amount(text, _REVENUE_RE)
        if numerator and revenue and revenue > numerator:
            return {"numerator": numerator, "revenue": revenue}
        if len(amounts) >= 2:
            s = sorted(amounts[:4], reverse=True)
            return {"numerator": s[1], "revenue": s[0]}

    elif metric == "cagr":
        if len(amounts) >= 2:
            years_found = sorted({int(y) for y in _YEARS_RE.findall(text)})
            n_years = float(years_found[-1] - years_found[0]) if len(years_found) >= 2 else 5.0
            if n_years <= 0:
                n_years = 5.0
            return {"beginning": amounts[-1], "ending": amounts[0], "years": n_years}

    elif metric in ("eps_diluted", "eps_basic"):
        net_income = _labeled_amount(text, _PROFIT_RE)
        shares     = _labeled_amount(text, _SHARES_RE)
        if net_income and shares:
            return {"net_income": net_income, "shares": shares}
        if len(amounts) >= 2:
            return {"net_income": amounts[0], "shares": amounts[1]}

    elif metric == "debt_to_equity":
        debt   = _labeled_amount(text, _DEBT_RE)
        equity = _labeled_amount(text, _EQUITY_RE)
        if debt and equity:
            return {"total_debt": debt, "total_equity": equity}
        if len(amounts) >= 2:
            return {"total_debt": amounts[0], "total_equity": amounts[1]}

    elif metric == "current_ratio":
        assets = _labeled_amount(text, _ASSETS_RE)
        liabs  = _labeled_amount(text, _LIAB_RE)
        if assets and liabs:
            return {"current_assets": assets, "current_liabilities": liabs}
        if len(amounts) >= 2:
            return {"current_assets": amounts[0], "current_liabilities": amounts[1]}

    elif metric == "pe_ratio":
        eps_val = _labeled_amount(text, _EPS_RE)
        if eps_val:
            candidates = [a for a in amounts if eps_val < a < 5000]
            if candidates:
                return {"share_price": candidates[0], "eps": eps_val}
        if len(amounts) >= 2:
            s = sorted(amounts[:4])
            return {"share_price": s[-1], "eps": s[0]}

    elif metric == "ev_ebitda":
        if len(amounts) >= 2:
            return {"enterprise_value": amounts[0], "ebitda": amounts[1]}

    return None


# ── FINANCIAL CALCULATOR RUNNER ───────────────────────────────────────────────

def run_financial_calculator(
    query: str,
    docs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Detect the finance metric in `query`, extract inputs from `docs`, call
    calculate(), and return a synthetic doc to prepend to the context window.
    Returns None when no arithmetic signal is found or extraction fails.

    The synthetic doc has modality="tool" so _build_p248_sources skips it —
    the calculator result appears in the answer text, not as a source chip.
    """
    metric = detect_finance_metric(query)
    if not metric:
        return None

    combined_text = "\n".join(d.get("text", "") for d in docs[:3])
    ctx = extract_numbers_for_metric(combined_text, metric)
    if not ctx:
        logger.info("finance_tool_runner_no_numbers", metric=metric)
        return None

    try:
        from app.tools.financial_calculator import calculate
        result = calculate(metric, ctx)
        if result.error or result.formatted in ("N/M", ""):
            logger.info("finance_tool_runner_calc_skipped",
                        metric=metric, error=result.error)
            return None

        verified_text = (
            f"[VERIFIED ARITHMETIC — financial_calculator]\n"
            f"Metric  : {metric}\n"
            f"Inputs  : {', '.join(f'{k}={v:.4g}' for k, v in ctx.items())}\n"
            f"Formula : {result.formula}\n"
            f"Result  : {result.formatted}\n"
        )
        logger.info("finance_tool_runner_calc_ok",
                    metric=metric, result=result.formatted)
        return {
            "text": verified_text,
            "metadata": {
                "source":   "financial_calculator",
                "modality": "tool",
                "doc_id":   f"calc_{metric}",
            },
            "score": 0.99,
        }
    except Exception as exc:
        logger.warning("finance_tool_runner_calc_error",
                       metric=metric, error=str(exc))
        return None


# ── SEC EDGAR RUNNER ──────────────────────────────────────────────────────────

_REGULATORY_SIGNALS = frozenset({
    "10-k", "10k", "10-q", "10q", "8-k", "8k",
    "sec filing", "sec report", "sec edgar", "edgar",
    "annual report", "proxy statement", "def 14a", "form 10",
    "regulatory filing", "disclosure", "s-1", "prospectus",
    "securities filing", "10 k", "10 q", "ipo filing",
})

# Single-letter and common English / finance acronyms that are NOT tickers
_TICKER_STOPWORDS = frozenset({
    "A", "I", "IN", "OF", "THE", "AND", "OR", "FOR", "TO", "AT",
    "BY", "AS", "IS", "IT", "ON", "IF", "DO", "NO", "SO", "UP",
    "US", "UK", "EU", "GDP", "AI", "SEC", "CEO", "CFO", "COO",
    "IPO", "ETF", "Q1", "Q2", "Q3", "Q4", "FY", "YOY", "EPS",
    "DCF", "LBO", "RAG", "PII", "KPI", "ROI", "ROE", "CAGR",
    "EV", "PE", "DE",
})

_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")


def is_regulatory_query(query: str) -> bool:
    """True when the query is about an SEC regulatory filing."""
    q = query.lower()
    return any(sig in q for sig in _REGULATORY_SIGNALS)


def extract_ticker(query: str) -> Optional[str]:
    """Return the first plausible stock ticker found in the query."""
    for m in _TICKER_RE.finditer(query):
        tok = m.group(1)
        if tok not in _TICKER_STOPWORDS:
            return tok
    return None


def run_sec_edgar_lookup(
    query: str,
    session_id: str = "default",
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Search EDGAR for the query and return:
      - a synthetic doc (appended to final_docs so LLM sees filing metadata)
      - source chips (added to p248_sources so UI renders clickable EDGAR links)

    Both are empty/None when EDGAR returns no results or errors.
    """
    try:
        from app.tools.sec_edgar import search_edgar_sync
        ticker  = extract_ticker(query)
        results = search_edgar_sync(query, ticker=ticker)
        if not results:
            logger.info("finance_tool_runner_edgar_empty",
                        query=query[:60], session_id=session_id)
            return None, []

        # LLM context text
        lines = ["[SEC EDGAR — supplemental regulatory filings]"]
        for r in results[:3]:
            line = f"• {r.company_name} | {r.form_type} | Filed: {r.filing_date}"
            if r.snippet:
                line += f" | {r.snippet}"
            lines.append(line)

        edgar_doc: Dict[str, Any] = {
            "text": "\n".join(lines),
            "metadata": {
                "source":   "sec_edgar",
                "modality": "tool",
                "doc_id":   "edgar_results",
            },
            "score": 0.80,
        }

        # UI source chips — same shape as web sources so MessageBubble renders
        # them as clickable links with a globe icon
        chips: List[Dict[str, Any]] = []
        for r in results[:3]:
            if r.document_url:
                chips.append({
                    "text":        f"{r.form_type} — {r.company_name} ({r.filing_date})",
                    "score":       0.75,
                    "source":      r.document_url,
                    "page_number": None,
                    "start_time":  None,
                    "end_time":    None,
                    "modality":    "web",
                    "doc_id":      None,
                })

        logger.info("finance_tool_runner_edgar_ok",
                    results=len(results), chips=len(chips),
                    ticker=ticker, session_id=session_id)
        return edgar_doc, chips

    except Exception as exc:
        logger.warning("finance_tool_runner_edgar_error",
                       error=str(exc), session_id=session_id)
        return None, []
