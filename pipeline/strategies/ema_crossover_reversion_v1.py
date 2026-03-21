"""EMA Crossover Reversion — NIFTY 5-second index options strategy.

Thesis: EMA(12)/(36) crossovers on NIFTY fire 1-2 minutes AFTER the directional impulse,
attracting late retail algo entries at exhaustion. In range-bound sessions (low ATR regime),
we fade these crossovers after a 2-bar (10s) confirmation that price has already recrossed
the slow EMA — capturing 30-90 second reversions as the over-extension unwinds.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "ema_crossover_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of pre-open noise
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 432            # 36 min warmup (EMA36 + 30-min ATR baseline)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_ratio_threshold", 1.3, 0.8, 2.0),
            TunableParam("vwap_buffer", 0.001, 0.0003, 0.003),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        atr_ratio = params.get("atr_ratio_threshold", 1.3)
        vwap_buf  = params.get("vwap_buffer", 0.001)
        stop_pts  = params.get("stop_pts", 4.0)
        tgt_pts   = params.get("target_pts", 6.0)

        # ── Indicators ──
        ema12 = _ema(close, 12)
        ema36 = _ema(close, 36)

        # Replace leading NaN with the first valid close so booleans don't NaN-propagate
        _fill_leading_nan(ema12, close)
        _fill_leading_nan(ema36, close)

        # ATR(60) — 5-minute current volatility
        atr60 = _atr(high, low, close, 60)
        _fill_leading_nan(atr60, np.full(n, 0.0))

        # Rolling 30-min mean of ATR(60) as baseline (360 bars)
        atr_base = _rolling_mean_o1(atr60, 360)
        _fill_leading_nan(atr_base, np.full(n, 1.0))

        # Session VWAP (cumulative per day)
        vwap = _session_vwap(close, volume, day_id, n)

        # ── Crossover detection (vectorized) ──
        # cross_up[i]: EMA12 crossed above EMA36 at bar i (bullish crossover → fade = buy PE)
        cross_up = np.zeros(n, dtype=bool)
        cross_dn = np.zeros(n, dtype=bool)
        cross_up[1:] = (ema12[1:] > ema36[1:]) & (ema12[:-1] <= ema36[:-1])
        cross_dn[1:] = (ema12[1:] < ema36[1:]) & (ema12[:-1] >= ema36[:-1])

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Range-bound filter ──
        safe_base = np.where(atr_base > 0.0, atr_base, 1.0)
        range_bound = (atr60 / safe_base) < atr_ratio

        # ── 2-bar confirmation entries ──
        # At bar j: crossover fired at j-2; confirm price has recrossed slow EMA
        buy_pe = np.zeros(n, dtype=bool)  # fade bullish cross → bearish → buy PE
        buy_ce = np.zeros(n, dtype=bool)  # fade bearish cross → bullish → buy CE

        for j in range(2, n):
            if not in_session[j] or not range_bound[j]:
                continue
            k = j - 2  # crossover bar

            # Fade bullish cross: EMA12 crossed above EMA36 at bar k,
            # price at crossover was above VWAP (over-extension long side),
            # price now has fallen back below EMA36 (reversion confirmed)
            if (cross_up[k]
                    and close[j] < ema36[j]
                    and close[k] > vwap[k] * (1.0 + vwap_buf)):
                buy_pe[j] = True

            # Fade bearish cross: EMA12 crossed below EMA36 at bar k,
            # price at crossover was below VWAP (over-extension short side),
            # price now has risen back above EMA36 (reversion confirmed)
            if (cross_dn[k]
                    and close[j] > ema36[j]
                    and close[k] < vwap[k] * (1.0 - vwap_buf)):
                buy_ce[j] = True

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, tgt_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )


# ── Helper functions ──────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with Wilder-style initialisation."""
    n = len(arr)
    out = np.full(n, np.nan)
    k = 2.0 / (period + 1)
    # Find first finite value
    start = 0
    while start < n and not np.isfinite(arr[start]):
        start += 1
    if start >= n:
        return out
    out[start] = arr[start]
    for i in range(start + 1, n):
        val = arr[i] if np.isfinite(arr[i]) else out[i - 1]
        out[i] = val * k + out[i - 1] * (1.0 - k)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _rolling_mean_o1(arr: np.ndarray, window: int) -> np.ndarray:
    """O(n) rolling mean using cumulative sum; NaN treated as 0 (ATR is always non-negative)."""
    n = len(arr)
    out = np.full(n, np.nan)
    arr_f = np.nan_to_num(arr, nan=0.0)
    valid = (~np.isnan(arr)).astype(np.float64)
    cs = np.cumsum(arr_f)
    cv = np.cumsum(valid)
    # From bar (window-1) onward
    end_cs = cs[window - 1:]
    end_cv = cv[window - 1:]
    beg_cs = np.concatenate([[0.0], cs[:n - window]])
    beg_cv = np.concatenate([[0.0], cv[:n - window]])
    s = end_cs - beg_cs
    c = end_cv - beg_cv
    mask = c > 0
    result = np.full(len(s), np.nan)
    result[mask] = s[mask] / c[mask]
    out[window - 1:] = result
    return out


def _session_vwap(close: np.ndarray, volume: np.ndarray,
                  day_id: np.ndarray, n: int) -> np.ndarray:
    """Cumulative VWAP per trading day."""
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v  = 0.0
    prev_day = -999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v  = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v  += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _fill_leading_nan(arr: np.ndarray, fallback: np.ndarray) -> None:
    """Replace leading NaN values in-place with corresponding fallback values."""
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = fallback[i]
        else:
            break
