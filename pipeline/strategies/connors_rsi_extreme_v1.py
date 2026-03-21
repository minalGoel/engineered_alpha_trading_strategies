"""connors_rsi_extreme_v1 — Connors RSI Extreme Reversion on NIFTY

Mechanism: On NIFTY, when the Connors RSI composite (RSI(6) of 30s price, RSI(2) of
consecutive-bar streak, and 100-bar percentile rank of 5s returns) drops below 10,
three independent exhaustion signals converge simultaneously. VWAP-benchmarked
institutional programs treat sub-VWAP prices at this CRSI depth as below-benchmark
accumulation opportunities. We enter on the bar where CRSI first crosses back above
the oversold threshold, confirming exhaustion is complete.

Converted from: trading_strategies/unique_strategies_all/Strategy_79.json
Original: Connors RSI < 10 on NIFTY 100 stocks, 1-min bars, hold 10-45 min.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing RSI. Returns 50.0 for bars before warmup."""
    n = len(prices)
    rsi = np.full(n, 50.0)
    if period < 1 or n <= period:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed with simple average over first `period` bars
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    if avg_loss[period] == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain[period] / avg_loss[period]
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for remaining bars
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_streak(prices: np.ndarray) -> np.ndarray:
    """Consecutive up/down streak: positive = up streak, negative = down streak."""
    n = len(prices)
    streak = np.zeros(n)
    for i in range(1, n):
        if prices[i] > prices[i - 1]:
            streak[i] = max(streak[i - 1], 0.0) + 1.0
        elif prices[i] < prices[i - 1]:
            streak[i] = min(streak[i - 1], 0.0) - 1.0
        else:
            streak[i] = 0.0
    return streak


def _compute_pct_rank(values: np.ndarray, window: int) -> np.ndarray:
    """Percentile rank of current value within the last `window` bars (0-100)."""
    n = len(values)
    pct_rank = np.full(n, 50.0)
    for i in range(window, n):
        hist = values[i - window:i]
        pct_rank[i] = float(np.sum(hist < values[i])) / window * 100.0
    return pct_rank


class Strategy(BaseStrategy):
    name = "connors_rsi_extreme_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min pre-open noise
    session_end_minutes = 920     # 15:20 IST — flatten before EOD illiquidity
    max_trades_per_day = 8
    max_lookback = 120            # 100 bars for pct_rank + 20 bar buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("crsi_oversold", 10.0, 5.0, 15.0),
            TunableParam("crsi_overbought", 90.0, 85.0, 95.0),
            TunableParam("vwap_band_pct", 0.003, 0.001, 0.006),
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        crsi_oversold = params.get("crsi_oversold", 10.0)
        crsi_overbought = params.get("crsi_overbought", 90.0)
        vwap_band_pct = params.get("vwap_band_pct", 0.003)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # Forward-fill then extract numpy arrays
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- VWAP (cumulative per day, reset on day change) ---
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tp_vol = 0.0
                cum_vol = 0.0
            tp = (high[i] + low[i] + close[i]) / 3.0
            cum_tp_vol += tp * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        # --- Connors RSI components ---
        # RSI(6) on close: 30-second price momentum (original RSI(3) on 1min = 3min; we compress to 30s)
        rsi6 = _compute_rsi(close, 6)

        # Streak of consecutive up/down closes
        streak = _compute_streak(close)

        # RSI(2) of streak: turning-point detector for the streak (same 2-bar lookback as original)
        streak_rsi2 = _compute_rsi(streak, 2)

        # Percentile rank of 1-bar return over 100 bars (8.3 min of intraday history)
        ret_1bar = np.zeros(n)
        ret_1bar[1:] = close[1:] - close[:-1]
        pct_rank = _compute_pct_rank(ret_1bar, 100)

        # Connors RSI composite
        crsi = (rsi6 + streak_rsi2 + pct_rank) / 3.0

        # --- VIX filter ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Previous bar CRSI for cross detection ---
        crsi_prev = np.empty(n)
        crsi_prev[0] = 50.0
        crsi_prev[1:] = crsi[:-1]

        # --- Session filter ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # VIX must be below 22: above this NIFTY moves become gappy and 90s hold breaks down
        vix_ok = vix_close < 22.0

        # VWAP proximity: price within band (not an extreme runaway move)
        near_vwap_long = (close < vwap) & (close > vwap * (1.0 - vwap_band_pct))
        near_vwap_short = (close > vwap) & (close < vwap * (1.0 + vwap_band_pct))

        # CRSI cross above oversold: prev bar was extreme low, current bar recovers
        crsi_cross_up = (crsi_prev < crsi_oversold) & (crsi >= crsi_oversold)

        # CRSI cross below overbought: prev bar was extreme high, current bar fades
        crsi_cross_down = (crsi_prev > crsi_overbought) & (crsi <= crsi_overbought)

        # --- Entry signals ---
        # Buy CE (bullish): CRSI exhaustion confirmed, price sub-VWAP, low vol regime
        buy_ce = in_session & vix_ok & near_vwap_long & crsi_cross_up

        # Buy PE (bearish): CRSI exhaustion at top confirmed, price above VWAP, low vol regime
        buy_pe = in_session & vix_ok & near_vwap_short & crsi_cross_down

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,   # 90 seconds: CRSI reversals play out within 60-90s or not at all
            max_trades_per_day=self.max_trades_per_day,
        )
