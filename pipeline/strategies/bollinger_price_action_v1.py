"""bollinger_price_action_v1 — Bollinger Band pinbar rejection on NIFTY/BANKNIFTY.

Mechanism: When NIFTY/BANKNIFTY probes a 5-minute Bollinger Band extreme (±2σ)
and a 5-second bar forms a rejection pinbar (large wick relative to body, bar
closes away from the band), institutional limit orders and market-maker delta
hedging have absorbed directional flow. In low-VIX regimes (VIX < 18), the
index mean-reverts toward the 5-minute midband (10-20 spot pts) within 30-90
seconds.

Converted from: trading_strategies/unique_strategies_all/Strategy_15.json
Original: BB(20) + pinbar on 1-min FnO equity stocks, hold 5-20 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_mean_std(arr: np.ndarray, window: int):
    """Rolling mean and std (population ddof=1). Pre-warmup values are NaN."""
    n = len(arr)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(window - 1, n):
        segment = arr[i - window + 1 : i + 1]
        mean[i] = segment.mean()
        s = segment.std(ddof=1)
        std[i] = s if s > 0.0 else 0.0
    return mean, std


class Strategy(BaseStrategy):
    name = "bollinger_price_action_v1"
    underlying = "BOTH"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 60-bar BB + 60-bar safety buffer (10 min total)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # How close the bar's extreme must approach the band (fraction of band level).
            # 0.001 = within 0.1% of bb_lower e.g. ~22 pts if lower band = 22000 NIFTY.
            TunableParam("band_touch_pct", 0.001, 0.0, 0.004),
            # Lower wick must be >= wick_ratio × body to count as bullish pinbar.
            TunableParam("wick_ratio", 2.0, 1.2, 4.0),
            # VIX ceiling — mean reversion only in low-volatility, orderly regimes.
            TunableParam("vix_max", 18.0, 14.0, 24.0),
            # Stop and target in option premium points.
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract OHLC (forward-fill NaN in Polars before to_numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        band_touch_pct = params.get("band_touch_pct", 0.001)
        wick_ratio = params.get("wick_ratio", 2.0)
        vix_max = params.get("vix_max", 18.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Bollinger Bands: 60-bar (5-minute) window ──
        # 60 bars chosen because:
        #   - At 5s bars, 60 bars = 5 minutes of context → enough to identify
        #     statistically valid band extremes without being stale
        #   - Holding 30-90s, the 5-min band is the right regime context
        #   - Shorter (20-30 bars) produces erratic bands; longer (240 bars) too slow
        bb_window = 60
        bb_mean_raw, bb_std_raw = _rolling_mean_std(close, bb_window)

        # Replace pre-warmup NaN with neutral: bands == close (never near band)
        bb_mean = np.where(np.isnan(bb_mean_raw), close, bb_mean_raw)
        bb_std = np.where(np.isnan(bb_std_raw), 0.0, bb_std_raw)
        bb_upper = bb_mean + 2.0 * bb_std
        bb_lower = bb_mean - 2.0 * bb_std

        # ── Pinbar components ──
        body = np.abs(close - open_)
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low

        # Floor the body so near-doji bars don't trivially satisfy wick_ratio.
        # 0.5 spot points is a minimal meaningful body for NIFTY/BANKNIFTY.
        body_safe = np.maximum(body, 0.5)

        # Bullish pinbar at lower band:
        #   - Lower wick dominates (buyers aggressively absorbed selling pressure)
        #   - Bar closes bullish (close >= open)
        bullish_pinbar = (lower_wick >= wick_ratio * body_safe) & (close >= open_)

        # Bearish pinbar at upper band:
        #   - Upper wick dominates (sellers absorbed buying pressure)
        #   - Bar closes bearish (close <= open)
        bearish_pinbar = (upper_wick >= wick_ratio * body_safe) & (close <= open_)

        # ── Band touch: the bar's extreme reached the band (±buffer) ──
        # buffer allows bars that nearly touch the band (within band_touch_pct of level)
        touched_lower = low <= bb_lower * (1.0 + band_touch_pct)
        touched_upper = high >= bb_upper * (1.0 - band_touch_pct)

        # ── VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = vix_close < vix_max

        # ── Session and warmup filters ──
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        warmup_done = np.arange(n) >= bb_window

        # ── Entry signals ──
        buy_ce = in_session & warmup_done & vix_ok & touched_lower & bullish_pinbar
        buy_pe = in_session & warmup_done & vix_ok & touched_upper & bearish_pinbar

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
