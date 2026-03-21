"""ATR Channel Breakout — NIFTY 5-second index options.

When NIFTY closes above its 2-minute ATR-adaptive Keltner Channel with above-average
volume, passive sell-side liquidity has been absorbed by institutional buyers.
Momentum algorithms and delta-hedgers amplify the move for 30-90 seconds.

Original: Keltner Channel breakout on 1-min FnO stocks, hold 15-45 min.
Converted: 2-min ATR channel on 5-second NIFTY bars, hold 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Returns NaN until first valid value."""
    k = 2.0 / (period + 1)
    out = np.full(len(arr), np.nan)
    started = False
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        if not started:
            out[i] = arr[i]
            started = True
        else:
            out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range via EMA smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _ema(tr, period)


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average. Returns NaN for first (period-1) bars."""
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.mean(arr[i - period + 1 : i + 1])
    return out


class Strategy(BaseStrategy):
    name = "atr_channel_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 144            # 12 min warmup (144 × 5s = 720s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("kc_multiplier", 2.0, 1.5, 3.0),
            TunableParam("vol_ratio_threshold", 1.5, 1.0, 3.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        kc_mult = params.get("kc_multiplier", 2.0)
        vol_thresh = params.get("vol_ratio_threshold", 1.5)
        vix_max = params.get("vix_max", 22.0)

        # --- Keltner Channel (2-minute at 5s = 24 bars) ---
        ema_24 = _ema(close, 24)
        atr_24 = _atr(high, low, close, 24)
        kc_upper = ema_24 + kc_mult * atr_24
        kc_lower = ema_24 - kc_mult * atr_24

        # --- Relative volume (1-min rolling baseline = 12 bars) ---
        vol_sma_12 = _sma(volume, 12)
        safe_vol_sma = np.where(vol_sma_12 > 0, vol_sma_12, 1.0)
        rel_vol = volume / safe_vol_sma

        # --- VIX: join asof to align timestamps ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Replace NaN with neutral values before boolean masks ---
        kc_upper = np.where(np.isnan(kc_upper), close + 1e6, kc_upper)  # never fires if warmup
        kc_lower = np.where(np.isnan(kc_lower), close - 1e6, kc_lower)
        rel_vol = np.where(np.isnan(rel_vol), 0.0, rel_vol)

        # --- Session and regime filters ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max

        # --- Entry signals ---
        # Bullish: close breaks above upper Keltner Channel with above-average volume
        buy_ce = (
            in_session
            & vix_ok
            & (close > kc_upper)
            & (rel_vol > vol_thresh)
        )

        # Bearish: close breaks below lower Keltner Channel with above-average volume
        buy_pe = (
            in_session
            & vix_ok
            & (close < kc_lower)
            & (rel_vol > vol_thresh)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # 4 pts stop: ~8 spot pts at delta 0.5 — genuine breakout should not re-enter channel
            stop_points=np.full(n, 4),
            # 7 pts target: ~14 spot pts — captures ~60% of typical 15-25 pt continuation impulse
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
