"""Performance metrics computation for option trading.

All metrics follow Ernest Chan's methodology:
- Sharpe ratio is primary (annualized, daily returns, zero-fill days without trades)
- Minimum trade count enforced
- Transaction costs included (cost model ON)

PnL is in INR: (exit_premium - entry_premium) × lot_size - costs.
"""
from __future__ import annotations
import numpy as np
import polars as pl
import math
from typing import Optional

from pipeline.config import TOTAL_TRADING_DAYS
from pipeline.cost_model import net_pnl_quick, SPREAD_POINTS


def compute_metrics(
    trades_df: pl.DataFrame,
    lot_size: int = 75,
    total_trading_days: Optional[int] = None,
) -> dict:
    """Compute aggregate performance metrics from a trades DataFrame.

    Args:
        trades_df: DataFrame with columns: pnl, entry_premium, exit_premium,
                   entry_time, exit_time, holding_bars, exit_reason, side
        lot_size: Contract lot size (75 for NIFTY, 15 for BANKNIFTY).
        total_trading_days: Total calendar trading days (for Sharpe).

    Returns:
        Metrics dict.
    """
    if trades_df is None or trades_df.is_empty():
        return _empty_metrics()

    if total_trading_days is None:
        total_trading_days = TOTAL_TRADING_DAYS

    n_trades = len(trades_df)
    pnls = trades_df["pnl"].to_numpy().astype(np.float64)

    # ── Apply costs if not already applied ──
    if "net_pnl" in trades_df.columns:
        net_pnls = trades_df["net_pnl"].to_numpy().astype(np.float64)
    else:
        # Compute net PnL with costs
        if "entry_premium" in trades_df.columns and "exit_premium" in trades_df.columns:
            entry_prems = trades_df["entry_premium"].to_numpy()
            exit_prems = trades_df["exit_premium"].to_numpy()
            net_pnls = np.array([
                net_pnl_quick(ep, xp, lot_size) for ep, xp in zip(entry_prems, exit_prems)
            ])
        else:
            net_pnls = pnls  # fallback

    # ── Basic PnL metrics ──
    total_pnl = float(np.nansum(net_pnls))
    winners = net_pnls[net_pnls > 0]
    losers = net_pnls[net_pnls < 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0.0

    avg_winner = float(np.mean(winners)) if len(winners) > 0 else 0.0
    avg_loser = float(np.mean(losers)) if len(losers) > 0 else 0.0
    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    avg_trade_pnl = float(np.mean(net_pnls))
    best_trade = float(np.max(net_pnls)) if n_trades > 0 else 0.0
    worst_trade = float(np.min(net_pnls)) if n_trades > 0 else 0.0

    # ── Holding time ──
    if "holding_bars" in trades_df.columns:
        holding_bars = trades_df["holding_bars"].to_numpy().astype(np.float64)
        avg_holding = float(np.mean(holding_bars))
        avg_holding_seconds = avg_holding * 5  # 1 bar = 5 seconds
    else:
        avg_holding = 0.0
        avg_holding_seconds = 0.0

    # ── Drawdown (use net PnL, not gross) ──
    if "exit_time" in trades_df.columns:
        # Sort trades by exit time, then use NET PnL for drawdown
        sorted_df = trades_df.sort("exit_time")
        if "entry_premium" in sorted_df.columns and "exit_premium" in sorted_df.columns:
            sorted_entry = sorted_df["entry_premium"].to_numpy()
            sorted_exit = sorted_df["exit_premium"].to_numpy()
            sorted_net = np.array([
                net_pnl_quick(ep, xp, lot_size) for ep, xp in zip(sorted_entry, sorted_exit)
            ])
        else:
            sorted_net = sorted_df["pnl"].to_numpy().astype(np.float64)
    else:
        sorted_net = net_pnls
    cumulative_pnl = np.cumsum(sorted_net)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # ── Daily returns for Sharpe ratio (use NET PnL, not gross) ──
    if "exit_time" in trades_df.columns:
        # Add net_pnl column for daily aggregation
        if "entry_premium" in trades_df.columns and "exit_premium" in trades_df.columns:
            entry_prems_daily = trades_df["entry_premium"].to_numpy()
            exit_prems_daily = trades_df["exit_premium"].to_numpy()
            net_pnl_arr = np.array([
                net_pnl_quick(ep, xp, lot_size)
                for ep, xp in zip(entry_prems_daily, exit_prems_daily)
            ])
            trades_with_net = trades_df.with_columns(
                pl.Series("_net_pnl", net_pnl_arr)
            )
        else:
            trades_with_net = trades_df.with_columns(
                pl.col("pnl").alias("_net_pnl")
            )
        trades_with_date = trades_with_net.with_columns(
            pl.col("exit_time").cast(pl.Date).alias("trade_date")
        )
        daily_pnl = trades_with_date.group_by("trade_date").agg(
            pl.col("_net_pnl").sum().alias("daily_pnl")
        ).sort("trade_date")

        days_with_trades = len(daily_pnl)
        # Use lot_size as a proxy for capital deployed
        capital_proxy = lot_size * 200  # approx premium × lot_size
        daily_returns_arr = daily_pnl["daily_pnl"].to_numpy() / max(capital_proxy, 1)

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
        capital_proxy = lot_size * 200
        daily_returns_full = np.array([total_pnl / max(capital_proxy, 1)])
        days_with_trades = 1

    # ── Sharpe ratio ──
    mean_daily = float(np.mean(daily_returns_full))
    std_daily = float(np.std(daily_returns_full, ddof=1)) if len(daily_returns_full) > 1 else 0.0
    if std_daily > 0:
        sharpe = mean_daily / std_daily * math.sqrt(252)
    else:
        sharpe = 0.0

    # ── Streaks ──
    longest_win = _longest_streak(net_pnls > 0)
    longest_loss = _longest_streak(net_pnls < 0)

    # ── Avg premium points per trade ──
    if "entry_premium" in trades_df.columns and "exit_premium" in trades_df.columns:
        avg_points = float(np.mean(
            trades_df["exit_premium"].to_numpy() - trades_df["entry_premium"].to_numpy()
        ))
    else:
        avg_points = avg_trade_pnl / max(lot_size, 1)

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
        "avg_holding_seconds": round(avg_holding_seconds, 1),
        "avg_points_per_trade": round(avg_points, 2),
        "total_trading_days": total_trading_days,
        "days_with_trades": days_with_trades,
        "best_trade_pnl": round(best_trade, 2),
        "worst_trade_pnl": round(worst_trade, 2),
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
    }


def compute_sharpe_from_trades(
    trades_df: pl.DataFrame,
    lot_size: int,
    total_trading_days: int,
) -> float:
    """Quick Sharpe computation for Optuna objective. Uses NET PnL (after costs)."""
    if trades_df is None or trades_df.is_empty() or len(trades_df) < 5:
        return -999.0

    # Compute net PnL per trade
    if "entry_premium" in trades_df.columns and "exit_premium" in trades_df.columns:
        entry_p = trades_df["entry_premium"].to_numpy()
        exit_p = trades_df["exit_premium"].to_numpy()
        net_pnl_arr = np.array([
            net_pnl_quick(ep, xp, lot_size) for ep, xp in zip(entry_p, exit_p)
        ])
        trades_with_net = trades_df.with_columns(pl.Series("_net_pnl", net_pnl_arr))
    else:
        trades_with_net = trades_df.with_columns(pl.col("pnl").alias("_net_pnl"))

    trades_with_date = trades_with_net.with_columns(
        pl.col("exit_time").cast(pl.Date).alias("trade_date")
    )
    daily_pnl = trades_with_date.group_by("trade_date").agg(
        pl.col("_net_pnl").sum().alias("daily_pnl")
    )

    capital_proxy = lot_size * 200
    daily_returns = daily_pnl["daily_pnl"].to_numpy() / max(capital_proxy, 1)
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
        "avg_holding_seconds": 0.0,
        "avg_points_per_trade": 0.0,
        "total_trading_days": 0,
        "days_with_trades": 0,
        "best_trade_pnl": 0.0,
        "worst_trade_pnl": 0.0,
        "longest_win_streak": 0,
        "longest_loss_streak": 0,
    }
