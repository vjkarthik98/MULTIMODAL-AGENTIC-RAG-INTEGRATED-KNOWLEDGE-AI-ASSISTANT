"""
Financial arithmetic tool — prevents LLM numeric errors.

Handles: yoy_change, cagr, margin, eps_basic, eps_diluted,
         debt_to_equity, current_ratio, pe_ratio, ev_ebitda.

Called by the agent when the query contains arithmetic signals
("grew by", "change", "YoY", "margin", "per share").
Result is fed into LLM context as a verified, source-safe number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CalculatorResult:
    metric:    str
    result:    float
    formatted: str
    formula:   str
    error:     Optional[str] = None


def _fmt(value: float, pct: bool = False, decimals: int = 2) -> str:
    if pct:
        sign = "+" if value > 0 else ""
        return f"{sign}{round(value, decimals)}%"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.2f}K"
    return f"{round(value, decimals)}"


def yoy_change(current: float, prior: float) -> CalculatorResult:
    if prior == 0:
        return CalculatorResult(
            metric="yoy_change", result=float("nan"),
            formatted="N/M", formula=f"({current} - {prior}) / |{prior}| × 100",
            error="prior_is_zero",
        )
    result = (current - prior) / abs(prior) * 100
    return CalculatorResult(
        metric="yoy_change",
        result=result,
        formatted=_fmt(result, pct=True),
        formula=f"({_fmt(current)} - {_fmt(prior)}) / |{_fmt(prior)}| × 100",
    )


def cagr(beginning: float, ending: float, years: float) -> CalculatorResult:
    if beginning <= 0 or years <= 0:
        return CalculatorResult(
            metric="cagr", result=float("nan"),
            formatted="N/M", formula="(ending/beginning)^(1/years) - 1",
            error="invalid_input",
        )
    result = ((ending / beginning) ** (1 / years) - 1) * 100
    return CalculatorResult(
        metric="cagr",
        result=result,
        formatted=_fmt(result, pct=True),
        formula=f"({_fmt(ending)}/{_fmt(beginning)})^(1/{years}) - 1",
    )


def margin(numerator: float, revenue: float) -> CalculatorResult:
    if revenue == 0:
        return CalculatorResult(
            metric="margin", result=float("nan"),
            formatted="N/M", formula=f"{numerator} / {revenue} × 100",
            error="revenue_is_zero",
        )
    result = numerator / revenue * 100
    return CalculatorResult(
        metric="margin",
        result=result,
        formatted=_fmt(result, pct=True),
        formula=f"{_fmt(numerator)} / {_fmt(revenue)} × 100",
    )


def eps(net_income: float, shares: float, diluted: bool = True) -> CalculatorResult:
    label = "eps_diluted" if diluted else "eps_basic"
    if shares == 0:
        return CalculatorResult(
            metric=label, result=float("nan"),
            formatted="N/M", formula=f"{net_income} / {shares}",
            error="shares_is_zero",
        )
    result = net_income / shares
    return CalculatorResult(
        metric=label,
        result=result,
        formatted=f"${result:.2f}",
        formula=f"{_fmt(net_income)} / {_fmt(shares, decimals=0)}",
    )


def debt_to_equity(total_debt: float, total_equity: float) -> CalculatorResult:
    if total_equity == 0:
        return CalculatorResult(
            metric="debt_to_equity", result=float("nan"),
            formatted="N/M", formula=f"{total_debt} / {total_equity}",
            error="equity_is_zero",
        )
    result = total_debt / total_equity
    return CalculatorResult(
        metric="debt_to_equity",
        result=result,
        formatted=f"{result:.2f}x",
        formula=f"{_fmt(total_debt)} / {_fmt(total_equity)}",
    )


def current_ratio(current_assets: float, current_liabilities: float) -> CalculatorResult:
    if current_liabilities == 0:
        return CalculatorResult(
            metric="current_ratio", result=float("nan"),
            formatted="N/M", formula=f"{current_assets} / {current_liabilities}",
            error="liabilities_is_zero",
        )
    result = current_assets / current_liabilities
    return CalculatorResult(
        metric="current_ratio",
        result=result,
        formatted=f"{result:.2f}x",
        formula=f"{_fmt(current_assets)} / {_fmt(current_liabilities)}",
    )


def pe_ratio(share_price: float, eps_value: float) -> CalculatorResult:
    if eps_value == 0:
        return CalculatorResult(
            metric="pe_ratio", result=float("nan"),
            formatted="N/M", formula=f"{share_price} / {eps_value}",
            error="eps_is_zero",
        )
    result = share_price / eps_value
    return CalculatorResult(
        metric="pe_ratio",
        result=result,
        formatted=f"{result:.1f}x",
        formula=f"${share_price:.2f} / ${eps_value:.2f}",
    )


def ev_ebitda(enterprise_value: float, ebitda: float) -> CalculatorResult:
    if ebitda == 0:
        return CalculatorResult(
            metric="ev_ebitda", result=float("nan"),
            formatted="N/M", formula=f"{enterprise_value} / {ebitda}",
            error="ebitda_is_zero",
        )
    result = enterprise_value / ebitda
    return CalculatorResult(
        metric="ev_ebitda",
        result=result,
        formatted=f"{result:.1f}x",
        formula=f"{_fmt(enterprise_value)} / {_fmt(ebitda)}",
    )


# DISPATCH TABLE

_DISPATCH = {
    "yoy_change":     lambda c: yoy_change(c["current"], c["prior"]),
    "cagr":           lambda c: cagr(c["beginning"], c["ending"], c["years"]),
    "margin":         lambda c: margin(c["numerator"], c["revenue"]),
    "eps_diluted":    lambda c: eps(c["net_income"], c["shares"], diluted=True),
    "eps_basic":      lambda c: eps(c["net_income"], c["shares"], diluted=False),
    "debt_to_equity": lambda c: debt_to_equity(c["total_debt"], c["total_equity"]),
    "current_ratio":  lambda c: current_ratio(c["current_assets"], c["current_liabilities"]),
    "pe_ratio":       lambda c: pe_ratio(c["share_price"], c["eps"]),
    "ev_ebitda":      lambda c: ev_ebitda(c["enterprise_value"], c["ebitda"]),
}


def calculate(expression: str, context: Dict[str, Any]) -> CalculatorResult:
    """
    Agent-facing entry point.

    Args:
        expression: metric name (e.g. "yoy_change", "margin", "cagr")
        context:    dict of named float inputs required by the metric

    Returns:
        CalculatorResult with result, formatted string, and formula used.
    """
    metric = expression.strip().lower()
    fn = _DISPATCH.get(metric)
    if fn is None:
        logger.warning("financial_calculator_unknown_metric", metric=metric)
        return CalculatorResult(
            metric=metric, result=float("nan"),
            formatted="N/M", formula="",
            error=f"unknown_metric: {metric}",
        )
    try:
        result = fn(context)
        logger.info(
            "financial_calculator_result",
            metric=metric,
            result=result.result,
            formatted=result.formatted,
        )
        return result
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        logger.warning("financial_calculator_error", metric=metric, error=str(exc))
        return CalculatorResult(
            metric=metric, result=float("nan"),
            formatted="N/M", formula="",
            error=str(exc),
        )
