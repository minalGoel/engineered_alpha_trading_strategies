"""Afternoon Profit Fade — NIFTY/BANKNIFTY ATM options.

Mechanism:
  On NIFTY/BANKNIFTY, when the index has moved more than 0.5% from session open
  by 14:45 IST, leveraged intraday retail and prop-desk participants simultaneously
  unwind directional positions to avoid NSE EOD MTM margin calls and overnight risk.
  This creates concentrated one-sided order flow that pushes the index 15-25 spot
  points back toward session VWAP in the 14:45-15:15 window. We buy CE on down-days
  (expect covering rally) and PE on up-days (expect profit-booking selloff).

Converted from: trading_strategies/unique_strategies_all/Strategy_348.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam

# Afternoon entry window (IST minutes from midnight)
_AFTERNOON_START = 885  # 14:45 IST
_AFTERNOON_END = 920    # 15:20 IST


def _compute_session_vwap(spot_df: pl.DataFrame) -> np.ndarray:
    """Cumulative session VWAP, reset each day_id."""
    result = (
        spot_df
        .with_columns([
            (
                (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
                * pl.col("volume").cast(pl.Float64)
            ).alias("_tp_vol"),
        ])
        .with_columns([
            pl.col("_tp_vol").cum_sum().over("day_id").alias("_cum_tp_vol"),
            pl.col("volume").cast(pl.Float64).cum_sum().over("day_id").alias("_cum_vol"),
        ])
        .with_columns([
            (pl.col("_cum_tp_vol") / pl.col("_cum_vol")).alias("_vwap"),
        ])
    )["_vwap"].fill_null(strategy="forward").to_numpy()

    # Replace any remaining NaN (e.g., first bar with zero volume) with close
    close = spot_df["close"].to_numpy()
    result = np.where(np.isnan(result) | (result == 0.0), close, result)
    return result


def _compute_session_open(spot_df: pl.DataFrame) -> np.ndarray:
    """First bar's close per day_id — proxy for session open price."""
    result = (
        spot_df
        .with_columns([
            pl.col("close").first().over("day_id").alias("_sess_open"),
        ])
    )["_sess_open"].fill_null(strategy="forward").to_numpy()

    close = spot_df["close"].to_numpy()
    result = np.where(np.isnan(result) | (result <= 0.0), close, result)
    return result


class Strategy(BaseStrategy):
    name = "afternoon_profit_fade"
    underlying = "BOTH"
    # Full session start so VWAP has all-day history when compute() is called
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST — flatten before close
    max_trades_per_day = 3        # one directional fade per session per underlying
    max_lookback = 240            # 20 min warmup (for VWAP to stabilise)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("day_return_threshold", 0.005, 0.002, 0.012),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)
        close = spot_df["close"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        threshold = params.get("day_return_threshold", 0.005)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # Session VWAP (cumulative, per day) — the mean-reversion magnet
        vwap = _compute_session_vwap(spot_df)

        # Session open price (first bar's close per day) — basis for day_return
        session_open = _compute_session_open(spot_df)

        # Intraday return from session open
        day_ret = np.where(
            session_open > 0,
            (close - session_open) / session_open,
            0.0,
        )

        # Afternoon trading window: 14:45-15:20 IST only
        in_window = (time_min >= _AFTERNOON_START) & (time_min < _AFTERNOON_END)

        # buy CE: down-day fade (NIFTY down >threshold AND still below VWAP)
        # Expect EOD short-covering to push index back toward VWAP
        buy_ce = in_window & (day_ret < -threshold) & (close < vwap)

        # buy PE: up-day fade (NIFTY up >threshold AND still above VWAP)
        # Expect EOD profit-booking to push index back toward VWAP
        buy_pe = in_window & (day_ret > threshold) & (close > vwap)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=120,           # 10 min max hold — bulk of EOD fade is done by then
            max_trades_per_day=self.max_trades_per_day,
        )
