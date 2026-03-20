"""VWAP + RSI Confluence Mean Reversion — NIFTY 5-second options strategy.

Mechanism: On NIFTY, when the index is >0.3% below session VWAP while the
2-minute RSI drops below 30, VWAP-benchmarked institutional algorithms and
delta-hedging market makers both generate buy flow simultaneously, creating
10-20 spot point reversions within 30-90 seconds. RSI slope turning positive
within the oversold extreme (detected at 5s resolution) allows entry 45-55
seconds before the 1-minute bar confirmation, capturing the steepest part of
the reversion.

Original: equity strategy on NIFTY200 stocks, 1-min bars, 10-40 min hold.
Converted: NIFTY index, 5s bars, 30-90 second hold, ATM options.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothed RSI. Returns 50.0 for bars before warmup completes."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average over first period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


class Strategy(BaseStrategy):
    name = "vwap_rsi_confluence_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — RSI(24) needs warmup, skip opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 288            # 24 min warmup (safe for RSI(24) + VWAP from open)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_threshold", 0.30, 0.15, 0.60),
            TunableParam("rsi_oversold", 30.0, 22.0, 38.0),
            TunableParam("rsi_overbought", 70.0, 62.0, 78.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot columns (forward-fill before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # --- Parameters ---
        vwap_dev_threshold = params.get("vwap_dev_threshold", 0.30)
        rsi_oversold = params.get("rsi_oversold", 30.0)
        rsi_overbought = params.get("rsi_overbought", 70.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # --- VIX: join_asof to align with spot bars ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- VWAP: cumulative typical_price*volume, reset per day ---
        # typical_price = (high + low + close) / 3
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            tp = (high[i] + low[i] + close[i]) / 3.0
            v = max(volume[i], 0.0)
            cum_tp_vol += tp * v
            cum_vol += v
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        # VWAP deviation as percentage
        vwap_dev_pct = (close - vwap) / np.where(vwap > 0, vwap, 1.0) * 100.0

        # --- RSI(24) on 5s bars = 2-minute RSI ---
        # 24 bars chosen to match the 30-90s trade horizon — detects exhaustion
        # within the last 2 minutes without being overwhelmed by 5s bar noise.
        rsi_24 = _compute_rsi(close, 24)

        # --- RSI slope over 6 bars (30 seconds) ---
        # Detects RSI turning up/down within the extreme zone.
        # 6 bars (30s) compressed from original 3-min slope to match 5s trade entry.
        rsi_slope = np.zeros(n)
        rsi_slope[6:] = rsi_24[6:] - rsi_24[:-6]

        # --- Session and VIX filters ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max

        # --- Entry signals ---
        # buy_ce: NIFTY oversold below VWAP — VWAP algo buyers + MM unwind converge
        buy_ce = (
            in_session
            & vix_ok
            & (vwap_dev_pct < -vwap_dev_threshold)
            & (rsi_24 < rsi_oversold)
            & (rsi_slope > 0.0)        # RSI turning up within oversold
            & (close > open_)          # bullish candle confirms buying absorption
        )

        # buy_pe: NIFTY overbought above VWAP — mirror logic for shorts
        buy_pe = (
            in_session
            & vix_ok
            & (vwap_dev_pct > vwap_dev_threshold)
            & (rsi_24 > rsi_overbought)
            & (rsi_slope < 0.0)        # RSI turning down within overbought
            & (close < open_)          # bearish candle confirms selling pressure
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,               # 90 seconds — reversion happens in <90s or not at all
            max_trades_per_day=self.max_trades_per_day,
        )
