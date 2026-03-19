"""Report generation (Phase 7).

Generates human-readable Markdown reports for each strategy.
"""
from __future__ import annotations
import polars as pl
from typing import Optional


def generate_report(
    strategy_name: str,
    raw_json: dict,
    parsed_strategy,
    optimization: Optional[dict],
    stock_filter: Optional[dict],
    train_metrics: Optional[dict],
    test_metrics: Optional[dict],
    verdict: str,
    verdict_reason: str = "",
    cv_fold_results: Optional[list] = None,
    per_stock_results: Optional[dict] = None,
    test_trades_df: Optional[pl.DataFrame] = None,
    assumptions: Optional[list[str]] = None,
) -> str:
    """Generate the full Markdown report for a strategy."""
    lines = []

    # ── Header ──────────────────────────────────────────────────────────
    family = raw_json.get("_dedup_metadata", {}).get("family", "unknown")
    author = raw_json.get("author", "unknown")
    lines.append(f"# {strategy_name}")
    lines.append(f"**Family:** {family} | **Author:** {author} | **Verdict:** {verdict}")
    if verdict_reason:
        lines.append(f"\n**Verdict Reason:** {verdict_reason}")
    lines.append("")

    # ── Thesis ──────────────────────────────────────────────────────────
    lines.append("## Thesis")
    lines.append(raw_json.get("thesis", "No thesis provided."))
    lines.append("")

    # ── Mechanics ───────────────────────────────────────────────────────
    lines.append("## Mechanics")
    lines.append(f"**Timeframe:** {raw_json.get('timeframe', 'unknown')}")
    lines.append(f"**Session:** {raw_json.get('session', 'unknown')}")
    lines.append(f"**Max Hold:** {raw_json.get('max_hold_bars', 'N/A')} bars")
    lines.append("")

    # Entry conditions
    entry = raw_json.get("entry", {})
    long_conds = entry.get("long", {}).get("conditions", []) if isinstance(entry.get("long"), dict) else []
    lines.append("**Entry Long:**")
    for c in long_conds:
        lines.append(f"- {c}")
    if not long_conds:
        lines.append("- No conditions")
    lines.append("")

    short_conds = entry.get("short", {}).get("conditions", []) if isinstance(entry.get("short"), dict) else []
    if short_conds and str(short_conds[0]).lower() not in ("n/a", "not_applicable", "none", ""):
        lines.append("**Entry Short:**")
        for c in short_conds:
            lines.append(f"- {c}")
    else:
        lines.append("**Entry Short:** Long-only strategy")
    lines.append("")

    # Exit rules
    exit_data = raw_json.get("exit", {})
    lines.append("**Exit:**")
    for key in ("target", "stop_loss", "trailing_stop", "time_stop", "signal_exit"):
        val = exit_data.get(key, "none")
        lines.append(f"- {key}: {val}")
    lines.append("")

    # Filters
    filters = raw_json.get("filters", {})
    lines.append("**Filters:**")
    for key in ("vix_filter", "time_filter", "volume_filter", "trend_filter"):
        val = filters.get(key, "none")
        lines.append(f"- {key}: {val}")
    lines.append("")

    # Indicators
    lines.append("**Indicators:**")
    for ind in raw_json.get("indicators", []):
        lines.append(f"- `{ind.get('name', '')}` = `{ind.get('formula', '')}` (lookback: {ind.get('lookback', 'N/A')})")
    lines.append("")

    # ── Parameters ──────────────────────────────────────────────────────
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
        lines.append(f"Sharpe: {ds} → {os_} ({improvement:+.1f}%)")
        lines.append(f"Overfit flag: {'yes' if optimization.get('overfit_flag') else 'no'}")

        if cv_fold_results:
            profitable_count = sum(1 for f in cv_fold_results if f.get("profitable"))
            lines.append(f"CV folds profitable: {profitable_count}/5")
            lines.append("")
            lines.append("| Fold | Period | PnL | Trades | Profitable |")
            lines.append("|------|--------|-----|--------|------------|")
            for f in cv_fold_results:
                lines.append(
                    f"| {f['fold']} | {f['period']} | ₹{f['pnl']:,.0f} | "
                    f"{f['trades']} | {'Yes' if f['profitable'] else 'No'} |"
                )
        lines.append("")

    # ── Training Performance ────────────────────────────────────────────
    if train_metrics and train_metrics.get("total_trades", 0) > 0:
        lines.append("## Training Performance (2022–2024, passing stocks only)")
        _add_metrics_table(lines, train_metrics)
        lines.append("")

        # Best/worst months
        monthly = train_metrics.get("monthly_pnl", {})
        if monthly:
            best_month = max(monthly.items(), key=lambda x: x[1]) if monthly else ("N/A", 0)
            worst_month = min(monthly.items(), key=lambda x: x[1]) if monthly else ("N/A", 0)
            lines.append(f"**Best month:** {best_month[0]} ₹{best_month[1]:,.0f}")
            lines.append(f"**Worst month:** {worst_month[0]} ₹{worst_month[1]:,.0f}")
            lines.append("")
    else:
        lines.append("## Training Performance")
        lines.append("N/A — strategy did not produce sufficient trades.")
        lines.append("")

    # ── Stock Coverage ──────────────────────────────────────────────────
    if stock_filter:
        passing = stock_filter.get("passing_stocks", 0)
        total = stock_filter.get("total_stocks_tested", 0)
        lines.append("## Stock Coverage")
        lines.append(f"- Tested: {total} | Passing: {passing} | Failing: {total - passing}")

        # Top 5 passing stocks by Sharpe
        if per_stock_results:
            passing_stocks = [(sym, m.get("sharpe", 0))
                              for sym, m in per_stock_results.items()
                              if m.get("passed_filter", False)]
            passing_stocks.sort(key=lambda x: x[1], reverse=True)
            top5 = passing_stocks[:5]
            if top5:
                lines.append(f"- **Top 5 stocks:** {', '.join(f'{s} (Sharpe {v:.2f})' for s, v in top5)}")

            # Sample failures
            failing_stocks = [(sym, m.get("filter_failures", []))
                              for sym, m in per_stock_results.items()
                              if not m.get("passed_filter", True)]
            samples = failing_stocks[:3]
            if samples:
                lines.append("- **Sample failures:**")
                for sym, reasons in samples:
                    lines.append(f"  - {sym}: {'; '.join(reasons)}")
        lines.append("")

    # ── Test Performance ────────────────────────────────────────────────
    if test_metrics and test_metrics.get("total_trades", 0) > 0:
        lines.append("## Test Performance (2025, passing stocks only)")
        _add_metrics_table(lines, test_metrics)
        lines.append("")
    else:
        lines.append("## Test Performance (2025)")
        lines.append(f"N/A — strategy did not qualify ({verdict})")
        lines.append("")

    # ── Train vs Test Comparison ────────────────────────────────────────
    if (train_metrics and test_metrics and
            train_metrics.get("total_trades", 0) > 0 and
            test_metrics.get("total_trades", 0) > 0):
        lines.append("## Train vs Test Comparison")
        lines.append("| Metric | Train | Test | Change |")
        lines.append("|--------|-------|------|--------|")
        for key, label in [
            ("sharpe_annualized", "Sharpe"),
            ("profit_factor", "Profit factor"),
            ("win_rate", "Win rate"),
            ("avg_trade_pnl", "Avg trade PnL"),
        ]:
            tr = train_metrics.get(key, 0)
            te = test_metrics.get(key, 0)
            if tr != 0:
                change = (te - tr) / abs(tr) * 100
                change_str = f"{change:+.1f}%"
            else:
                change_str = "N/A"
            if key == "win_rate":
                lines.append(f"| {label} | {tr:.1%} | {te:.1%} | {change_str} |")
            elif key == "avg_trade_pnl":
                lines.append(f"| {label} | ₹{tr:.1f} | ₹{te:.1f} | {change_str} |")
            else:
                lines.append(f"| {label} | {tr:.2f} | {te:.2f} | {change_str} |")
        lines.append("")

    # ── Last 20 Signals ─────────────────────────────────────────────────
    if test_trades_df is not None and not test_trades_df.is_empty():
        lines.append("## Last 10 Trades (test period)")
        lines.append("| Timestamp | Symbol | Action | Price | Trade PnL |")
        lines.append("|-----------|--------|--------|-------|-----------|")
        recent = test_trades_df.sort("exit_time", descending=True).head(10)
        for row in recent.iter_rows(named=True):
            entry_ts = str(row["entry_time"])[:16] if row.get("entry_time") else ""
            exit_ts = str(row["exit_time"])[:16] if row.get("exit_time") else ""
            sym = row.get("symbol", "")
            side = row.get("side", "")
            ep = row.get("entry_price", 0)
            xp = row.get("exit_price", 0)
            pnl = row.get("pnl", 0)
            entry_action = "BUY" if side == "LONG" else "SHORT"
            exit_action = "SELL" if side == "LONG" else "COVER"
            lines.append(f"| {entry_ts} | {sym} | {entry_action} | {ep:.2f} | — |")
            lines.append(f"| {exit_ts} | {sym} | {exit_action} | {xp:.2f} | {'+'if pnl > 0 else ''}₹{pnl:.2f} |")
        lines.append("")

    # ── Known Weaknesses ────────────────────────────────────────────────
    lines.append("## Known Weaknesses")
    for w in raw_json.get("weaknesses", []):
        lines.append(f"- {w}")
    # Discovered weaknesses
    if stock_filter and stock_filter.get("passing_stocks", 0) < 20:
        total_stocks = stock_filter.get('total_stocks_tested', stock_filter.get('passing_stocks', 0))
        lines.append(f"- Only works on {stock_filter.get('passing_stocks', 0)}/{total_stocks} stocks")
    if test_metrics and train_metrics:
        train_sharpe = train_metrics.get("sharpe_annualized", 0)
        test_sharpe = test_metrics.get("sharpe_annualized", 0)
        if train_sharpe > 0 and test_sharpe > 0:
            decay = (1 - test_sharpe / train_sharpe) * 100
            if decay > 50:
                lines.append(f"- Sharpe collapsed {decay:.0f}% OOS")
    lines.append("")

    # ── Assumptions ─────────────────────────────────────────────────────
    if assumptions:
        lines.append("## Assumptions Made")
        for a in assumptions:
            lines.append(f"- {a}")
        lines.append("")

    # Standard assumptions that always apply
    lines.append("## Standard Assumptions")
    lines.append("- All entries fill at bar close price")
    lines.append("- No transaction costs (signals trigger option trades via separate OMS)")
    lines.append("- EOD flatten always at 15:20 IST, no overnight positions")
    lines.append("- One position at a time per stock per strategy")
    lines.append("- Stop/target checked against bar high/low; if both hit same bar, stop assumed first")
    lines.append("- Expiry days approximated as every Thursday")
    lines.append("- **Sharpe ratio note:** Absolute Sharpe values are computed using per-trade capital "
                 "(₹100K) as the denominator, NOT total deployed capital (~50 stocks × ₹100K = ₹50L). "
                 "This inflates absolute values. Relative comparisons between strategies remain valid.")
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
    lines.append(f"| Total PnL | ₹{m.get('total_pnl', 0):,.0f} |")
    lines.append(f"| Max drawdown | ₹{m.get('max_drawdown', 0):,.0f} |")
    lines.append(f"| Avg trade PnL | ₹{m.get('avg_trade_pnl', 0):.1f} |")
    lines.append(f"| Avg winner | ₹{m.get('avg_winner', 0):.1f} |")
    lines.append(f"| Avg loser | ₹{m.get('avg_loser', 0):.1f} |")
    tf = m.get("avg_holding_bars", 0)
    lines.append(f"| Avg holding time | {tf:.1f} bars |")
    lines.append(f"| Best trade PnL | ₹{m.get('best_trade_pnl', 0):,.0f} |")
    lines.append(f"| Worst trade PnL | ₹{m.get('worst_trade_pnl', 0):,.0f} |")
