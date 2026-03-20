"""VIX-Filtered Index Trend Strategy (vix_filtered_index_trend_v124)

When India VIX is elevated (>21, <=35), NIFTY's intraday order flow shifts to
sustained institutional directional flow. EMA(12) crossing EMA(60) with VWAP
confirmation signals the onset of these institutional waves. Enter at the first
5-second crossover bar; hold 30-90 seconds.

Converted from: trading_strategies/unique_strategies_all/Strategy_117.json
Original: VIX-filtered EMA(9/21) crossover on top-50 FnO stocks, 1-min bars.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA with standard smoothing factor k = 2/(period+1)."""
    result = np.empty(len(arr))
    k = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


class Strategy(BaseStrategy):
    name = "vix_filtered_index_trend_v124"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening noise/price discovery
    session_end_minutes = 920     # 15:20 IST — EOD flatten
    max_trades_per_day = 8
    max_lookback = 120            # 120 bars × 5s = 600s = 10 min warmup for EMA(60)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_threshold", 21.0, 15.0, 30.0),
            TunableParam("vix_max", 35.0, 28.0, 45.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        vix_threshold = params.get("vix_threshold", 21.0)
        vix_max = params.get("vix_max", 35.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Spot data (forward-fill before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        vol_arr = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── VIX: join_asof backward onto spot bars ──
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

        # ── Session VWAP (cumulative per day from 09:15) ──
        typical = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tpv = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tpv = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            cum_tpv += typical[i] * vol_arr[i]
            cum_vol += vol_arr[i]
            vwap[i] = cum_tpv / cum_vol if cum_vol > 0.0 else close[i]

        # ── EMA indicators ──
        ema_fast = _ema(close, 12)   # 60-second fast EMA — trade trigger
        ema_slow = _ema(close, 60)   # 5-minute slow EMA — context filter

        # ── Crossover detection ──
        fast_above = ema_fast > ema_slow
        prev_fast_above = np.empty(n, dtype=bool)
        prev_fast_above[0] = fast_above[0]
        prev_fast_above[1:] = fast_above[:-1]

        cross_up = fast_above & ~prev_fast_above    # EMA fast crosses above slow (bullish)
        cross_dn = ~fast_above & prev_fast_above    # EMA fast crosses below slow (bearish)

        # ── Filters ──
        vix_regime = (vix_close >= vix_threshold) & (vix_close <= vix_max)
        close_above_vwap = close > vwap
        close_below_vwap = close < vwap
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.arange(n) >= 60   # require at least 60 bars for EMA(60) to stabilize

        # ── Signals ──
        buy_ce = warmed_up & in_session & vix_regime & cross_up & close_above_vwap
        buy_pe = warmed_up & in_session & vix_regime & cross_dn & close_below_vwap

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
