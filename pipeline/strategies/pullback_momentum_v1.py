"""pullback_momentum_v1 — VWAP Pullback Momentum (NIFTY, 5s bars)

Thesis: On NIFTY, pullbacks to session VWAP during an established 5-minute
uptrend are absorbed by VWAP-benchmarked institutional algorithms receiving
fills at/below their benchmark. Entry on first positive tick after VWAP touch,
targeting 15-25 spot point reversion within 60-90 seconds.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_session_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Cumulative session VWAP, reset at each new day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            cum_pv = close[i] * volume[i]
            cum_v = volume[i]
        else:
            cum_pv += close[i] * volume[i]
            cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Rolling ATR over `period` bars."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(n)
    # Initial seed: first full window
    if period <= n:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        # Back-fill warmup with first valid value
        fill = atr[period - 1]
        for i in range(period - 1):
            atr[i] = fill
    return atr


class Strategy(BaseStrategy):
    name = "pullback_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_lookback = 120             # 10-min warmup (120 × 5s)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_zscore_threshold", 0.5,  0.3,  1.2),
            TunableParam("momentum_threshold",     0.001, 0.0005, 0.003),
            TunableParam("vix_max",                19.0, 14.0, 25.0),
            TunableParam("stop_pts",               4.0,  2.0,  8.0),
            TunableParam("target_pts",             6.0,  3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        # ── Parameters ──
        zscore_thr = params.get("vwap_zscore_threshold", 0.5)
        mom_thr    = params.get("momentum_threshold", 0.001)
        vix_max    = params.get("vix_max", 19.0)
        stop_pts   = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Indicators ──

        # 1. Session VWAP (cumulative, resets each day)
        vwap = _compute_session_vwap(close, volume, day_id)

        # 2. ATR(24) — 2-minute rolling ATR to normalize VWAP deviation
        atr24 = _compute_atr(high, low, close, 24)
        atr24 = np.where(atr24 < 0.01, 0.01, atr24)   # guard division by zero

        # 3. VWAP deviation in ATR units
        vwap_dev = (close - vwap) / atr24

        # 4. 5-minute momentum (60 bars × 5s = 5 min) — trend context filter
        mom_60 = np.zeros(n)
        denom = np.where(close > 0.0, close, 1.0)
        for i in range(60, n):
            mom_60[i] = (close[i] - close[i - 60]) / denom[i - 60]

        # 5. 1-bar return (5-second) — entry direction confirmation
        ret_1 = np.zeros(n)
        ret_1[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] > 0.0, close[:-1], 1.0)

        # ── VIX filter (join_asof against spot timestamps) ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok     = vix_close < vix_max

        # ── Signals ──
        # Buy CE: price dipped below VWAP, trend is bullish, first uptick confirms bounce
        buy_ce = (
            in_session
            & vix_ok
            & (vwap_dev < -zscore_thr)
            & (mom_60 > mom_thr)
            & (ret_1 > 0.0)
        )

        # Buy PE: price pushed above VWAP, trend is bearish, first downtick confirms rejection
        buy_pe = (
            in_session
            & vix_ok
            & (vwap_dev > zscore_thr)
            & (mom_60 < -mom_thr)
            & (ret_1 < 0.0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds — VWAP bounce expected quickly
            max_trades_per_day=self.max_trades_per_day,
        )
