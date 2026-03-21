"""Opening Drive Reversal — NIFTY 5-second index options.

Fades the post-auction opening drive on NIFTY when price deviates ≥0.3% from
session VWAP with extreme 30-second RSI(6), a visible rejection wick (≥35% of
bar range), and an elevated-volume exhaustion bar. Enters on the first reversal
5s bar; holds 30-90 seconds toward VWAP mean-reversion.

Only active in the 09:20-10:00 IST opening window.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "opening_drive_reversal_001"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip auction noise
    session_end_minutes = 600     # 10:00 IST — opening drive window only
    max_trades_per_day = 5
    max_lookback = 72             # 6-min warmup for rolling relative volume

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_threshold", 0.003, 0.002, 0.006),
            TunableParam("rsi_os_threshold", 15.0, 5.0, 25.0),
            TunableParam("rsi_ob_threshold", 85.0, 75.0, 95.0),
            TunableParam("wick_threshold", 0.35, 0.20, 0.55),
            TunableParam("vol_threshold", 1.5, 1.2, 2.5),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN in Polars before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        vwap_dev_thr = params.get("vwap_dev_threshold", 0.003)
        rsi_os = params.get("rsi_os_threshold", 15.0)
        rsi_ob = params.get("rsi_ob_threshold", 85.0)
        wick_thr = params.get("wick_threshold", 0.35)
        vol_thr = params.get("vol_threshold", 1.5)

        # --- Session filter: 09:20-10:00 only ---
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # --- Session VWAP (cumulative, resets each day) ---
        vwap = _compute_session_vwap(close, high, low, volume, day_id, n)

        # --- VWAP deviation ---
        vwap_dev = np.where(vwap > 0.0, (close - vwap) / vwap, 0.0)

        # --- RSI(6) — 30-second RSI for opening exhaustion ---
        rsi_6 = _compute_rsi(close, 6, n)

        # --- Wick ratios ---
        bar_range = high - low
        body_low = np.minimum(open_, close)
        body_high = np.maximum(open_, close)
        denom = np.where(bar_range > 0.001, bar_range, 0.001)
        wick_down = (body_low - low) / denom   # lower wick ratio (bullish exhaustion)
        wick_up = (high - body_high) / denom   # upper wick ratio (bearish exhaustion)

        # --- Relative volume vs 5-min rolling mean (60 bars) ---
        rel_vol = _compute_rel_volume(volume, 60, n)

        # --- VWAP slope over 60 bars (5 min) as fraction of close ---
        vwap_slope = np.zeros(n)
        for i in range(60, n):
            vwap_slope[i] = vwap[i] - vwap[i - 60]
        vwap_slope_rel = np.where(close > 0.0, np.abs(vwap_slope) / close, 0.0)

        # --- Confirmation: bullish or bearish bar ---
        bull_bar = close > open_
        bear_bar = close < open_

        # --- Signals ---
        # Buy CE: down-drive exhaustion → buy NIFTY CE
        buy_ce = (
            in_session
            & (vwap_dev <= -vwap_dev_thr)
            & (rsi_6 <= rsi_os)
            & (wick_down >= wick_thr)
            & (rel_vol >= vol_thr)
            & (vwap_slope_rel <= 0.002)
            & bull_bar
        )

        # Buy PE: up-drive exhaustion → buy NIFTY PE
        buy_pe = (
            in_session
            & (vwap_dev >= vwap_dev_thr)
            & (rsi_6 >= rsi_ob)
            & (wick_up >= wick_thr)
            & (rel_vol >= vol_thr)
            & (vwap_slope_rel <= 0.002)
            & bear_bar
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4),    # 4 pts = ~8 NIFTY spot pts; drive resumption threshold
            target_points=np.full(n, 8),  # 7 pts = ~14 NIFTY spot pts; 60% of typical reversal leg
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_session_vwap(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
    n: int,
) -> np.ndarray:
    """Cumulative session VWAP, reset at each new day."""
    vwap = np.zeros(n)
    typical = (high + low + close) / 3.0
    cum_tpv = 0.0
    cum_vol = 0.0
    cur_day = -1
    for i in range(n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            cum_tpv = 0.0
            cum_vol = 0.0
        cum_tpv += typical[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0.0 else close[i]
    return vwap


def _compute_rsi(close: np.ndarray, period: int, n: int) -> np.ndarray:
    """Wilder smoothed RSI."""
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            gains[i] = diff
        else:
            losses[i] = -diff

    # Seed: simple average of first `period` moves
    ag = float(np.mean(gains[1 : period + 1]))
    al = float(np.mean(losses[1 : period + 1]))
    if al == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + ag / al)

    # Wilder smoothing
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        if al == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + ag / al)

    return rsi


def _compute_rel_volume(volume: np.ndarray, period: int, n: int) -> np.ndarray:
    """Rolling relative volume: current bar / mean(last `period` bars)."""
    rel_vol = np.ones(n)
    for i in range(period, n):
        avg = float(np.mean(volume[i - period : i]))
        rel_vol[i] = volume[i] / avg if avg > 0.0 else 1.0
    return rel_vol
