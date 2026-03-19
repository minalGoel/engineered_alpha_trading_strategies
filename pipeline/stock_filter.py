"""Stock filtering (Phase 5).

A stock passes for a strategy if ALL hold:
- >= 30 trades
- Profit factor > 1.2
- Sharpe > 0.5
- Max drawdown < 50% of total PnL (only when PnL > 0)

Do NOT filter on win rate.
"""
from __future__ import annotations
import logging

from pipeline.config import (
    FILTER_MIN_TRADES, FILTER_MIN_PROFIT_FACTOR,
    FILTER_MIN_SHARPE, FILTER_MAX_DD_TO_PNL,
    MIN_PASSING_STOCKS,
)

log = logging.getLogger(__name__)


def filter_stock(per_stock_metrics: dict) -> tuple[bool, list[str]]:
    """Apply stock filter to a single stock's metrics.

    Returns (passed, list_of_failure_reasons).
    """
    failures = []
    m = per_stock_metrics

    total_trades = m.get("total_trades", 0)
    profit_factor = m.get("profit_factor", 0.0)
    sharpe = m.get("sharpe", 0.0)
    total_pnl = m.get("total_pnl", 0.0)
    max_dd = m.get("max_drawdown", 0.0)

    if total_trades < FILTER_MIN_TRADES:
        failures.append(f"trades={total_trades} < {FILTER_MIN_TRADES}")

    if profit_factor < FILTER_MIN_PROFIT_FACTOR:
        failures.append(f"profit_factor={profit_factor:.2f} < {FILTER_MIN_PROFIT_FACTOR}")

    if sharpe < FILTER_MIN_SHARPE:
        failures.append(f"sharpe={sharpe:.2f} < {FILTER_MIN_SHARPE}")

    if total_pnl > 0 and max_dd > 0:
        dd_ratio = max_dd / total_pnl
        if dd_ratio > FILTER_MAX_DD_TO_PNL:
            failures.append(f"dd/pnl={dd_ratio:.2f} > {FILTER_MAX_DD_TO_PNL}")

    passed = len(failures) == 0
    return passed, failures


def filter_all_stocks(
    per_stock_results: dict[str, dict],
) -> tuple[dict, list[str], list[str]]:
    """Filter all stocks and return filter results.

    Returns:
        stock_filter_json: dict for stock_filter.json
        passing_symbols: list of passing stock symbols
        failing_symbols: list of failing stock symbols
    """
    passing = []
    failing = []
    stock_details = {}

    for symbol, metrics in per_stock_results.items():
        passed, failures = filter_stock(metrics)
        metrics["passed_filter"] = passed
        metrics["filter_failures"] = failures
        stock_details[symbol] = {
            "passed": passed,
            "failures": failures,
            "trades": metrics.get("total_trades", 0),
            "sharpe": metrics.get("sharpe", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "total_pnl": metrics.get("total_pnl", 0.0),
        }
        if passed:
            passing.append(symbol)
        else:
            failing.append(symbol)

    stock_filter_json = {
        "total_stocks_tested": len(per_stock_results),
        "passing_stocks": len(passing),
        "failing_stocks": len(failing),
        "passing_symbols": sorted(passing),
        "failing_symbols": sorted(failing),
        "min_passing_required": MIN_PASSING_STOCKS,
        "passed_phase": len(passing) >= MIN_PASSING_STOCKS,
        "stocks": stock_details,
    }

    return stock_filter_json, passing, failing
