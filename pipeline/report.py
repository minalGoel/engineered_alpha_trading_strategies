"""Report generation for option trading strategies.

Generates human-readable Markdown reports with option-specific fields
(strike, expiry, premium, costs per trade).
"""
from __future__ import annotations
import polars as pl
from typing import Optional

from pipeline.cost_model import min_points_to_breakeven


def generate_report(
    strategy_name: str,
    raw_json: dict,
    optimization: Optional[dict],
    train_metrics: Optional[dict],
    test_metrics: Optional[dict],
    verdict: str,
    verdict_reason: str = "",
    cv_day_results: Optional[list] = None,
    test_trades_df: Optional[pl.DataFrame] = None,
    assumptions: Optional[list[str]] = None,
    lot_size: int = 75,
) -> str:
    """Generate the full Markdown report for an option strategy."""
    lines = []

    # ── Header ──
    underlying = raw_json.get("universe", raw_json.get("underlying", "BOTH"))
    lines.append(f"# {strategy_name}")
    lines.append(f"**Underlying:** {underlying} | **Verdict:** {verdict}")
    if verdict_reason:
        lines.append(f"\n**Verdict Reason:** {verdict_reason}")
    lines.append("")

    # ── Thesis & Mechanism ──
    lines.append("## Thesis")
    lines.append(raw_json.get("thesis", "No thesis provided."))
    lines.append("")

    mechanism = raw_json.get("mechanism", "")
    if mechanism:
        lines.append("## Microstructure Mechanism")
        lines.append(mechanism)
        lines.append("")

    # ── Mechanics ──
    lines.append("## Mechanics")
    lines.append(f"**Timeframe:** {raw_json.get('timeframe', '5s')}")
    lines.append(f"**Session:** {raw_json.get('session', '09:20-15:25 IST')}")
    lines.append(f"**Max Hold:** {raw_json.get('max_hold_bars', 24)} bars ({raw_json.get('max_hold_bars', 24) * 5}s)")
    lines.append(f"**Trade Type:** Buy CE (bullish) / Buy PE (bearish) only")
    lines.append("")

    # Cost viability
    breakeven = min_points_to_breakeven(lot_size)
    lines.append(f"**Min breakeven move:** {breakeven:.1f} premium points (lot={lot_size})")
    lines.append("")

    # ── Indicators ──
    lines.append("**Indicators:**")
    for ind in raw_json.get("indicators", []):
        if isinstance(ind, dict):
            lines.append(f"- `{ind.get('name', '')}` = `{ind.get('formula', '')}` "
                         f"(lookback: {ind.get('lookback', 'N/A')})")
        else:
            lines.append(f"- {ind}")
    lines.append("")

    # ── Parameters ──
    if optimization:
        lines.append("## Parameters")
        lines.append("| Parameter | Default | Optimized | Bound |")
        lines.append("|-----------|---------|-----------|-------|")
        default_p = optimization.get("default_params", {})
        optimized_p = optimization.get("optimized_params", {})
        bounds = optimization.get("param_bounds", {})
        for k in default_p:
            d = default_p.get(k, "N/A")
            o = optimized_p.get(k, "N/A")
            b = bounds.get(k, ["N/A", "N/A"])
            lines.append(f"| {k} | {d} | {o} | [{b[0]}, {b[1]}] |")
        lines.append("")

        ds = optimization.get("default_sharpe", 0)
        os_ = optimization.get("optimized_sharpe", 0)
        improvement = ((os_ - ds) / abs(ds) * 100) if ds != 0 else 0
        lines.append(f"Sharpe: {ds} -> {os_} ({improvement:+.1f}%)")
        lines.append(f"Overfit flag: {'yes' if optimization.get('overfit_flag') else 'no'}")
        lines.append("")

    # ── Leave-one-day-out CV ──
    if cv_day_results:
        profitable_count = sum(1 for d in cv_day_results if d.get("profitable"))
        lines.append(f"## Leave-One-Day-Out CV ({profitable_count}/{len(cv_day_results)} days profitable)")
        lines.append("")
        lines.append("| Day | Date | PnL | Trades | Profitable |")
        lines.append("|-----|------|-----|--------|------------|")
        for d in cv_day_results:
            lines.append(
                f"| {d['day']} | {d.get('date', '')} | INR {d['pnl']:,.0f} | "
                f"{d['trades']} | {'Yes' if d['profitable'] else 'No'} |"
            )
        lines.append("")

    # ── Training Performance ──
    if train_metrics and train_metrics.get("total_trades", 0) > 0:
        lines.append("## Training Performance")
        _add_metrics_table(lines, train_metrics)
        lines.append("")
    else:
        lines.append("## Training Performance")
        lines.append("N/A - insufficient trades.")
        lines.append("")

    # ── Test Performance ──
    if test_metrics and test_metrics.get("total_trades", 0) > 0:
        lines.append("## Test Performance")
        _add_metrics_table(lines, test_metrics)
        lines.append("")
    else:
        lines.append("## Test Performance")
        lines.append(f"N/A - strategy did not qualify ({verdict})")
        lines.append("")

    # ── Recent Trades ──
    if test_trades_df is not None and not test_trades_df.is_empty():
        lines.append("## Last 10 Trades (test period)")
        lines.append("| Entry Time | Side | Entry Prem | Exit Prem | PnL | Exit Reason |")
        lines.append("|------------|------|------------|-----------|-----|-------------|")
        recent = test_trades_df.sort("exit_time", descending=True).head(10)
        for row in recent.iter_rows(named=True):
            entry_ts = str(row.get("entry_time", ""))[:19]
            side = "CE" if row.get("side") == 1 else "PE"
            ep = row.get("entry_premium", 0)
            xp = row.get("exit_premium", 0)
            pnl = row.get("pnl", 0)
            reason = row.get("exit_reason", "")
            sign = "+" if pnl > 0 else ""
            lines.append(f"| {entry_ts} | {side} | {ep:.1f} | {xp:.1f} | {sign}{pnl:.0f} | {reason} |")
        lines.append("")

    # ── Known Weaknesses ──
    lines.append("## Known Weaknesses")
    for w in raw_json.get("weaknesses", []):
        lines.append(f"- {w}")
    lines.append("- **12 days of data is very thin.** Results must be validated with more data.")
    lines.append("")

    # ── Assumptions ──
    if assumptions:
        lines.append("## Assumptions Made")
        for a in assumptions:
            lines.append(f"- {a}")
        lines.append("")

    lines.append("## Standard Assumptions")
    lines.append("- All option entries fill at bar close premium")
    lines.append("- Cost model ON: bid-ask spread, STT, brokerage, exchange charges")
    lines.append("- EOD flatten at 15:25 IST, expiry day flatten at 15:20 IST")
    lines.append("- One position at a time (no simultaneous CE + PE)")
    lines.append("- ATM strike = nearest strike to spot close at entry")
    lines.append("- Leave-one-day-out CV with 12 trading days")
    lines.append("- Strategy must be profitable on >= 9 of 12 test days")
    lines.append("")

    return "\n".join(lines)


def _add_metrics_table(lines: list[str], m: dict):
    """Add a metrics table to the report lines."""
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total trades | {m.get('total_trades', 0)} |")
    lines.append(f"| Win rate | {m.get('win_rate', 0):.1%} |")
    lines.append(f"| Profit factor | {m.get('profit_factor', 0):.2f} |")
    lines.append(f"| Sharpe (annualized) | {m.get('sharpe_annualized', 0):.4f} |")
    lines.append(f"| Total PnL | INR {m.get('total_pnl', 0):,.0f} |")
    lines.append(f"| Max drawdown | INR {m.get('max_drawdown', 0):,.0f} |")
    lines.append(f"| Avg trade PnL | INR {m.get('avg_trade_pnl', 0):.1f} |")
    lines.append(f"| Avg points/trade | {m.get('avg_points_per_trade', 0):.2f} pts |")
    lines.append(f"| Avg holding | {m.get('avg_holding_bars', 0):.1f} bars ({m.get('avg_holding_seconds', 0):.0f}s) |")
    lines.append(f"| Best trade | INR {m.get('best_trade_pnl', 0):,.0f} |")
    lines.append(f"| Worst trade | INR {m.get('worst_trade_pnl', 0):,.0f} |")
