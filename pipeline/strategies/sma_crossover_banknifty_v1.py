"""SMA Crossover BANKNIFTY v1 — Triple SMA alignment pullback entry.

Original: Triple SMA(10/29/100) on BankNifty 1-min bars. Hold 5-20 min.
Converted: Triple SMA(36/180/720) on BankNifty 5s bars. Hold 15-60s.

Mechanism: On BANKNIFTY, large institutional TWAP/VWAP algorithms working
directional orders create persistent micro-trends visible as SMA alignment
across 3-min, 15-min, and 60-min windows. When all three align bullishly,
continuous demand absorption sustains the trend. We enter on micro-pullbacks
(price touching the 3-min SMA) and ride the next 20-30 spot point push.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "sma_crossover_banknifty_v1"
    underlying = "BANKNIFTY"
    session_start_minutes = 620   # 10:20 IST — after 60-min warmup for sma_720
    session_end_minutes = 915     # 15:15 IST — avoid expiry gamma effects
    max_trades_per_day = 8
    max_lookback = 720            # 60-min warmup for sma_720

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pullback_pct", 0.03, 0.01, 0.08),   # % within which close touches sma_36
            TunableParam("vix_max", 25.0, 18.0, 35.0),        # max VIX for trend trading
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        pullback_pct = params.get("pullback_pct", 0.03)
        vix_max = params.get("vix_max", 25.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Compute SMAs ─────────────────────────────────────────────────────
        # SMA(36) = 3-min micro-trend (trade trigger — pullback entry)
        # SMA(180) = 15-min session trend (context filter)
        # SMA(720) = 60-min session anchor (regime filter)

        sma_36 = _rolling_sma(close, 36, day_id)
        sma_180 = _rolling_sma(close, 180, day_id)
        sma_720 = _rolling_sma(close, 720, day_id)

        # ── VIX filter ───────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward"
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(np.float64)

        # ── Session and warmup masks ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Replace NaN in SMAs (before warmup) with 0 so comparisons work correctly
        sma_36 = np.where(np.isnan(sma_36), 0.0, sma_36)
        sma_180 = np.where(np.isnan(sma_180), 0.0, sma_180)
        sma_720 = np.where(np.isnan(sma_720), 0.0, sma_720)

        # ── Pullback-touch conditions ────────────────────────────────────────
        # Price is within pullback_pct% of sma_36 (touching the micro-trend SMA)
        sma_36_safe = np.where(sma_36 > 0.0, sma_36, 1.0)
        pct_dist = np.abs(close - sma_36) / sma_36_safe * 100.0  # % distance from sma_36

        # ── Bullish: triple alignment AND pullback to sma_36 ─────────────────
        bullish_alignment = (sma_36 > 0) & (sma_180 > 0) & (sma_720 > 0) & \
                            (close > sma_36) & (sma_36 > sma_180) & (sma_180 > sma_720)
        bullish_pullback = pct_dist <= pullback_pct  # price touching sma_36

        buy_ce = in_session & warmed & bullish_alignment & bullish_pullback & (vix_close < vix_max)

        # ── Bearish: inverted alignment AND pullback to sma_36 from below ────
        bearish_alignment = (sma_36 > 0) & (sma_180 > 0) & (sma_720 > 0) & \
                            (close < sma_36) & (sma_36 < sma_180) & (sma_180 < sma_720)
        # For bearish, pullback is close touching sma_36 from below (close within pct of sma_36)
        buy_pe = in_session & warmed & bearish_alignment & bullish_pullback & (vix_close < vix_max)

        # Ensure no simultaneous buy_ce and buy_pe
        both = buy_ce & buy_pe
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,          # 60s time stop
            max_trades_per_day=self.max_trades_per_day,
        )


def _rolling_sma(arr: np.ndarray, period: int, day_id: np.ndarray) -> np.ndarray:
    """Compute rolling SMA, resetting at day boundaries.

    Returns NaN for bars where there are fewer than `period` bars
    in the current day (no cross-day contamination).
    """
    n = len(arr)
    result = np.full(n, np.nan)
    current_day = -1
    day_start = 0

    for i in range(n):
        if day_id[i] != current_day:
            current_day = day_id[i]
            day_start = i
        bars_in_day = i - day_start + 1
        if bars_in_day >= period:
            result[i] = np.mean(arr[i - period + 1: i + 1])

    return result
