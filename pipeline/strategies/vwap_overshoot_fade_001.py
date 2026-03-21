"""vwap_overshoot_fade_001 — VWAP Overshoot Fade with RSI(24) Exhaustion

Converted from: trading_strategies/unique_strategies_all/Strategy_135.json
Original: VWAP mean-reversion on top-120 FnO equities, 1-min bars, RSI(2)+ADX(14).

Conversion logic:
- RSI(2) on 1-min  → RSI(24) on 5s: same 2-minute window, not 12x scaling
- ADX(14) on 1-min → ADX(36) on 5s: compressed to 3-min for 15-90s hold time
- VWAP: cumulative from session open (no scaling — always cumulative)
- Deviation threshold: 0.5% → 0.3% (NIFTY moves less % per bar than individual stocks)
- Session: 09:20-11:15 IST (early-session VWAP inventory correction window)
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI using Wilder's smoothing. Returns 50.0 for warmup bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed with SMA over first `period` bars
    avg_gain[period] = np.mean(gains[1:period + 1])
    avg_loss[period] = np.mean(losses[1:period + 1])

    # Wilder smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ADX using Wilder's smoothing. Returns 25.0 (neutral) for warmup bars."""
    n = len(close)
    adx = np.full(n, 25.0)
    if n < period * 2 + 2:
        return adx

    # True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # +DM and -DM
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0.0:
            dm_plus[i] = up
        if down > up and down > 0.0:
            dm_minus[i] = down

    # Wilder-smoothed TR, +DM, -DM
    atr_w = np.zeros(n)
    dmp_w = np.zeros(n)
    dmm_w = np.zeros(n)

    atr_w[period] = np.sum(tr[1:period + 1])
    dmp_w[period] = np.sum(dm_plus[1:period + 1])
    dmm_w[period] = np.sum(dm_minus[1:period + 1])

    for i in range(period + 1, n):
        atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / period + tr[i]
        dmp_w[i] = dmp_w[i - 1] - dmp_w[i - 1] / period + dm_plus[i]
        dmm_w[i] = dmm_w[i - 1] - dmm_w[i - 1] / period + dm_minus[i]

    # DI+, DI-, DX
    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = np.where(atr_w > 0, 100.0 * dmp_w / atr_w, 0.0)
        di_minus = np.where(atr_w > 0, 100.0 * dmm_w / atr_w, 0.0)
        denom = di_plus + di_minus
        dx = np.where(denom > 0, 100.0 * np.abs(di_plus - di_minus) / denom, 0.0)

    # Smooth DX into ADX
    adx_val = np.full(n, 25.0)
    start = period * 2
    if start < n:
        adx_val[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx_val[i] = (adx_val[i - 1] * (period - 1) + dx[i]) / period

    return adx_val


def _compute_vwap_deviation(
    close: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Session-cumulative VWAP deviation: (close - vwap) / vwap.

    Resets each new day_id. Returns 0.0 for bars where cumulative volume == 0.
    """
    n = len(close)
    dev = np.zeros(n)

    cum_pv = 0.0
    cum_v = 0.0
    current_day = day_id[0]

    for i in range(n):
        if day_id[i] != current_day:
            cum_pv = 0.0
            cum_v = 0.0
            current_day = day_id[i]
        v = volume[i] if volume[i] > 0 else 1.0
        cum_pv += close[i] * v
        cum_v += v
        vwap_i = cum_pv / cum_v
        dev[i] = (close[i] - vwap_i) / vwap_i

    return dev


class Strategy(BaseStrategy):
    name = "vwap_overshoot_fade_001"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 675     # 11:15 IST — early session only
    max_trades_per_day = 5
    max_lookback = 480            # 40 min warmup: ADX(36) needs 72+ bars to stabilize

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_threshold", 0.003, 0.002, 0.006),
            TunableParam("rsi_low", 15.0, 8.0, 25.0),
            TunableParam("rsi_high", 85.0, 75.0, 92.0),
            TunableParam("adx_max", 22.0, 15.0, 30.0),
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy (per CONVENTIONS §7)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        vwap_dev_threshold = params.get("vwap_dev_threshold", 0.003)
        rsi_low = params.get("rsi_low", 15.0)
        rsi_high = params.get("rsi_high", 85.0)
        adx_max = params.get("adx_max", 22.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # Session VWAP deviation — cumulative, resets each day
        vwap_dev = _compute_vwap_deviation(close, volume, day_id)

        # RSI(24) = 2-minute RSI at 5s resolution
        # Preserves the exact 2-minute window of RSI(2) on 1-min bars from the original.
        rsi_24 = _compute_rsi(close, 24)

        # ADX(36) = 3-minute trend strength at 5s resolution
        # Compressed from original ADX(14) on 1-min (14 min) to 3 min,
        # matching our 15-90s hold horizon.
        adx_36 = _compute_adx(high, low, close, 36)

        # Early session filter — opening inventory correction window
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # Range-bound filter: no active directional institutional program
        low_adx = adx_36 < adx_max

        # Buy CE: NIFTY >= 0.3% below VWAP + RSI(24) oversold + range-bound tape
        buy_ce = (
            in_session
            & low_adx
            & (vwap_dev < -vwap_dev_threshold)
            & (rsi_24 < rsi_low)
        )

        # Buy PE: NIFTY >= 0.3% above VWAP + RSI(24) overbought + range-bound tape
        buy_pe = (
            in_session
            & low_adx
            & (vwap_dev > vwap_dev_threshold)
            & (rsi_24 > rsi_high)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90s max hold
            max_trades_per_day=self.max_trades_per_day,
        )
