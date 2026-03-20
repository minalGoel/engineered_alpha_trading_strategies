"""
Keltner Channel Reversion v1
============================
NIFTY index 5-second options strategy.

Mechanism: When NIFTY touches the outer Keltner Channel (EMA(24) ± 2.0×ATR(24)) and the bar
itself closes back inside with a directional body, delta-hedging dealers and EMA-anchored
mean-reversion desks fade the ATR-normalised extreme. The ATR-based bands don't inflate on
single-spike outliers (unlike Bollinger), making each touch a cleaner signal of genuine
micro-volatility-normalised exhaustion. Entry on the reversal bar; target is KC_mid (2-min EMA)
within 30-90 seconds.

Hold time: 30-90 seconds (time_stop_bars=18)
Stop: 3 option pts (~6 spot pts — channel-walk confirmation threshold)
Target: 5 option pts (~10 spot pts — 50-60% reversion to KC_mid)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (EMA) via recursive formula."""
    alpha = 2.0 / (period + 1)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR via EMA of TrueRange."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _ema(tr, period)


class Strategy(BaseStrategy):
    name = "keltner_channel_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # ATR band multiplier: 2.0 ATRs (NIFTY index is smoother than individual stocks)
            TunableParam("kc_multiplier", 2.0, 1.5, 3.0),
            # KC_position threshold: close must be this close to the touched band
            TunableParam("kc_position_threshold", 0.15, 0.05, 0.25),
            # Minimum fractional bar body strength (close_to_low or high_to_close fraction)
            TunableParam("body_strength_threshold", 0.55, 0.40, 0.75),
            # Max VIX: above this, ATR bands widen and channel-walk risk exceeds reversion edge
            TunableParam("vix_max", 25.0, 18.0, 35.0),
            # EMA slope threshold (normalized): above this = trending, skip mean reversion
            TunableParam("slope_threshold", 0.0003, 0.0001, 0.0010),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract OHLCV (forward-fill NaN before numpy) ──────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ─────────────────────────────────────────────────────────
        kc_mult    = float(params.get("kc_multiplier", 2.0))
        pos_thresh = float(params.get("kc_position_threshold", 0.15))
        body_min   = float(params.get("body_strength_threshold", 0.55))
        vix_max    = float(params.get("vix_max", 25.0))
        slope_thr  = float(params.get("slope_threshold", 0.0003))

        # ── Keltner Channel (period = 24 bars = 2 minutes) ─────────────────────
        period = 24
        kc_mid   = _ema(close, period)
        kc_atr   = _atr(high, low, close, period)
        kc_upper = kc_mid + kc_mult * kc_atr
        kc_lower = kc_mid - kc_mult * kc_atr

        # KC position: 0 = at lower band, 1 = at upper band
        kc_width = kc_upper - kc_lower
        kc_width = np.where(kc_width > 0.0, kc_width, 1.0)   # guard div-by-zero
        kc_pos   = (close - kc_lower) / kc_width

        # ── EMA slope filter (30-second window = 6 bars) ───────────────────────
        # Detect ranging vs trending: skip when EMA is steeply directional
        kc_slope = np.zeros(n, dtype=np.float64)
        for i in range(6, n):
            kc_slope[i] = (kc_mid[i] - kc_mid[i - 6]) / kc_mid[i - 6]
        kc_ranging = np.abs(kc_slope) < slope_thr   # True = EMA is flat (good for reversion)

        # ── Bar body strength ──────────────────────────────────────────────────
        bar_range = np.where(high > low, high - low, 1e-6)
        # Bullish strength: how much of the bar is close-above-low
        close_to_low_frac = (close - low) / bar_range
        # Bearish strength: how much of the bar is high-above-close
        high_to_close_frac = (high - close) / bar_range

        # ── VIX filter ─────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0, dtype=np.float64)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(np.float64)
        vix_ok = vix_close < vix_max

        # ── Session and warmup filters ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed     = np.arange(n) >= period

        # ── Buy CE: lower band touch + bullish reversal ────────────────────────
        low_touched      = low <= kc_lower                           # wick to/through lower band
        closed_inside_lo = close > kc_lower                         # close back inside
        near_lower       = kc_pos < pos_thresh                       # still near lower band
        bullish_body     = (close > open_) & (close_to_low_frac > body_min)  # strong bounce bar

        buy_ce = (
            in_session & warmed & vix_ok & kc_ranging
            & low_touched & closed_inside_lo & near_lower & bullish_body
        )

        # ── Buy PE: upper band touch + bearish reversal ────────────────────────
        high_touched     = high >= kc_upper                          # wick to/through upper band
        closed_inside_hi = close < kc_upper                         # close back inside
        near_upper       = kc_pos > (1.0 - pos_thresh)              # still near upper band
        bearish_body     = (close < open_) & (high_to_close_frac > body_min)  # strong rejection bar

        buy_pe = (
            in_session & warmed & vix_ok & kc_ranging
            & high_touched & closed_inside_hi & near_upper & bearish_body
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop 3 pts: channel-walk confirmation threshold (~6 spot pts at delta 0.5)
            stop_points=np.full(n, 3.0),
            # Target 5 pts: ~50-60% reversion to KC_mid (~10 spot pts) within 30-90s
            target_points=np.full(n, 5.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,                   # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
