"""Performance metrics computation.

All metrics follow Ernest Chan's methodology:
- Sharpe ratio is primary (annualized, daily returns, zero-fill days without trades)
- Minimum trade count enforced
- No transaction costs
"""
from __future__ import annotations
import numpy as np
import polars as pl
import math
from typing import Optional

from pipeline.config import DEFAULT_CAPITAL_PER_TRADE


def compute_metrics(
    trades_df: pl.DataFrame,
    capital_per_trade: float = DEFAULT_CAPITAL_PER_TRADE,
    total_trading_days: Optional[int] = None,
) -> dict:
    """Compute aggregate performance metrics from a trades DataFrame.

    Args:
        trades_df: DataFrame with columns: pnl, pnl_pct, entry_time, exit_time,
                   holding_bars, exit_reason, side
        capital_per_trade: Capital per trade for return calculations.
        total_trading_days: Total calendar trading days in the period (for Sharpe).
                           If None, inferred from data.

    Returns:
        Metrics dict matching the schema in the spec.
    """
    if trades_df is None or trades_df.is_empty():
        return _empty_metrics()

    n_trades = len(trades_df)
    pnls = trades_df["pnl"].to_numpy().astype(np.float64)
    pnl_pcts = trades_df["pnl_pct"].to_numpy().astype(np.float64)

    # ── Basic PnL metrics ───────────────────────────────────────────────
    total_pnl = float(np.nansum(pnls))
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0.0

    avg_winner = float(np.mean(winners)) if len(winners) > 0 else 0.0
    avg_loser = float(np.mean(losers)) if len(losers) > 0 else 0.0
    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    avg_trade_pnl = float(np.mean(pnls))
    best_trade = float(np.max(pnls)) if n_trades > 0 else 0.0
    worst_trade = float(np.min(pnls)) if n_trades > 0 else 0.0

    # ── Holding time ────────────────────────────────────────────────────
    holding_bars = trades_df["holding_bars"].to_numpy().astype(np.float64)
    avg_holding = float(np.mean(holding_bars)) if n_trades > 0 else 0.0

    # ── Drawdown ────────────────────────────────────────────────────────
    # Sort trades by exit time to compute drawdown on a proper time-ordered
    # equity curve, not in arbitrary iteration order.
    if "exit_time" in trades_df.columns:
        sorted_pnls = trades_df.sort("exit_time")["pnl"].to_numpy().astype(np.float64)
    else:
        sorted_pnls = pnls
    cumulative_pnl = np.cumsum(sorted_pnls)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # ── Daily returns for Sharpe ratio ──────────────────────────────────
    # Sum PnL by day, fill missing days with zero
    if "exit_time" in trades_df.columns:
        trades_with_date = trades_df.with_columns(
            pl.col("exit_time").cast(pl.Date).alias("trade_date")
        )
        daily_pnl = trades_with_date.group_by("trade_date").agg(
            pl.col("pnl").sum().alias("daily_pnl")
        ).sort("trade_date")

        days_with_trades = len(daily_pnl)

        if total_trading_days is None:
            # Infer from date range
            if "entry_time" in trades_df.columns:
                min_date = trades_df["entry_time"].min()
                max_date = trades_df["exit_time"].max()
                if min_date is not None and max_date is not None:
                    date_range = (max_date - min_date).days
                    total_trading_days = max(int(date_range * 252 / 365), days_with_trades)
                else:
                    total_trading_days = days_with_trades
            else:
                total_trading_days = days_with_trades

        # Compute daily return as daily_pnl / capital_per_trade
        daily_returns_arr = daily_pnl["daily_pnl"].to_numpy() / capital_per_trade

        # Zero-fill non-trading days
        total_days = max(total_trading_days, days_with_trades)
        zero_fill_count = total_days - days_with_trades
        if zero_fill_count > 0:
            daily_returns_full = np.concatenate([
                daily_returns_arr,
                np.zeros(zero_fill_count)
            ])
        else:
            daily_returns_full = daily_returns_arr

    else:
        daily_returns_full = np.array([total_pnl / capital_per_trade])
        total_trading_days = 1
        days_with_trades = 1

    # ── Sharpe ratio ────────────────────────────────────────────────────
    mean_daily = float(np.mean(daily_returns_full))
    std_daily = float(np.std(daily_returns_full, ddof=1)) if len(daily_returns_full) > 1 else 0.0
    if std_daily > 0:
        sharpe = mean_daily / std_daily * math.sqrt(252)
    else:
        sharpe = 0.0

    # ── Streaks ─────────────────────────────────────────────────────────
    longest_win = _longest_streak(pnls > 0)
    longest_loss = _longest_streak(pnls < 0)

    # ── Monthly PnL ─────────────────────────────────────────────────────
    monthly_pnl = _compute_monthly_pnl(trades_df)

    return {
        "total_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(min(profit_factor, 999.0), 2),
        "sharpe_annualized": round(sharpe, 4),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_drawdown, 2),
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "avg_holding_bars": round(avg_holding, 1),
        "total_trading_days": total_trading_days,
        "days_with_trades": days_with_trades,
        "best_trade_pnl": round(best_trade, 2),
        "worst_trade_pnl": round(worst_trade, 2),
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "monthly_pnl": monthly_pnl,
    }


def compute_per_stock_metrics(
    trades_df: pl.DataFrame,
    symbol: str,
    capital_per_trade: float = DEFAULT_CAPITAL_PER_TRADE,
    total_trading_days: Optional[int] = None,
) -> dict:
    """Compute per-stock metrics matching the spec schema."""
    m = compute_metrics(trades_df, capital_per_trade, total_trading_days)
    return {
        "symbol": symbol,
        "total_trades": m["total_trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "sharpe": m["sharpe_annualized"],
        "total_pnl": m["total_pnl"],
        "max_drawdown": m["max_drawdown"],
        "avg_trade_pnl": m["avg_trade_pnl"],
        "avg_winner": m["avg_winner"],
        "avg_loser": m["avg_loser"],
        "avg_holding_bars": m["avg_holding_bars"],
        "best_trade_pnl": m["best_trade_pnl"],
        "worst_trade_pnl": m["worst_trade_pnl"],
        "passed_filter": False,   # Set later by stock_filter
        "filter_failures": [],
    }


def compute_sharpe_from_trades(
    trades_df: pl.DataFrame,
    capital_per_trade: float,
    total_trading_days: int,
) -> float:
    """Quick Sharpe computation for Optuna objective."""
    if trades_df is None or trades_df.is_empty() or len(trades_df) < 5:
        return -999.0

    trades_with_date = trades_df.with_columns(
        pl.col("exit_time").cast(pl.Date).alias("trade_date")
    )
    daily_pnl = trades_with_date.group_by("trade_date").agg(
        pl.col("pnl").sum().alias("daily_pnl")
    )

    daily_returns = daily_pnl["daily_pnl"].to_numpy() / capital_per_trade
    days_with_trades = len(daily_returns)

    # Zero fill
    zero_fill = max(0, total_trading_days - days_with_trades)
    if zero_fill > 0:
        daily_returns = np.concatenate([daily_returns, np.zeros(zero_fill)])

    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    if std_r > 0:
        return float(mean_r / std_r * math.sqrt(252))
    return 0.0


def _longest_streak(mask: np.ndarray) -> int:
    """Compute longest consecutive True streak in a boolean array."""
    if len(mask) == 0:
        return 0
    max_streak = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _compute_monthly_pnl(trades_df: pl.DataFrame) -> dict[str, float]:
    """Compute monthly PnL from trades."""
    if trades_df is None or trades_df.is_empty():
        return {}
    if "exit_time" not in trades_df.columns:
        return {}

    try:
        monthly = trades_df.with_columns(
            pl.col("exit_time").dt.strftime("%Y-%m").alias("month")
        ).group_by("month").agg(
            pl.col("pnl").sum().alias("monthly_pnl")
        ).sort("month")

        return {row["month"]: round(row["monthly_pnl"], 2)
                for row in monthly.iter_rows(named=True)}
    except Exception:
        return {}


def _empty_metrics() -> dict:
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe_annualized": 0.0,
        "total_pnl": 0.0,
        "max_drawdown": 0.0,
        "avg_trade_pnl": 0.0,
        "avg_winner": 0.0,
        "avg_loser": 0.0,
        "avg_holding_bars": 0.0,
        "total_trading_days": 0,
        "days_with_trades": 0,
        "best_trade_pnl": 0.0,
        "worst_trade_pnl": 0.0,
        "longest_win_streak": 0,
        "longest_loss_streak": 0,
        "monthly_pnl": {},
    }
