"""
VWAP Pinch Breakout — NIFTY 5-second index options strategy.

When NIFTY's 2-minute EMA and session VWAP converge to within 0.05% of spot
(the "pinch"), price is in micro-equilibrium. A breakout above/below both anchors
with a volume surge signals institutional order flow resumption, producing a
10-20 spot point continuation in 15-90 seconds.

Converted from: trading_strategies/unique_strategies_all/Strategy_343.json
Original: EMA(20)+VWAP pinch breakout on nifty100 stocks, 1-min bars, 20-60 min hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_pinch_breakout"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 60             # 5-min warmup for EMA to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pinch_threshold", 0.0005, 0.0002, 0.0015),
            TunableParam("vol_multiplier",  1.2,    1.0,    2.5),
            TunableParam("stop_pts",        4.0,    2.0,    8.0),
            TunableParam("target_pts",      7.0,    4.0,    14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays ──────────────────────────────────────────────────────
        close   = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume  = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id  = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────
        pinch_threshold = float(params.get("pinch_threshold", 0.0005))
        vol_multiplier  = float(params.get("vol_multiplier",  1.2))
        stop_pts        = float(params.get("stop_pts",        4.0))
        target_pts      = float(params.get("target_pts",      7.0))

        # ── EMA(24) — 2-minute exponential moving average ───────────────────
        # Compressed from original EMA(20) on 1-min bars (= 20 min) to 24 bars
        # (= 2 min) because we're targeting a 15-90s micro-continuation, not
        # the 20-60 min trend the original traded.
        ema_period = 24
        alpha = 2.0 / (ema_period + 1.0)
        ema = np.empty(n, dtype=np.float64)
        ema[0] = close[0]
        for i in range(1, n):
            ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]

        # ── Session VWAP — cumulative from daily open ────────────────────────
        vwap = np.empty(n, dtype=np.float64)
        cum_pv = 0.0
        cum_v  = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv = 0.0
                cum_v  = 0.0
                prev_day = day_id[i]
            cum_pv += close[i] * volume[i]
            cum_v  += volume[i]
            vwap[i] = (cum_pv / cum_v) if cum_v > 0.0 else close[i]

        # ── Rolling 12-bar (1-min) average volume ───────────────────────────
        avg_vol = np.empty(n, dtype=np.float64)
        avg_vol[:12] = volume[:12]
        for i in range(12, n):
            avg_vol[i] = np.mean(volume[i - 12 : i])

        # ── Pinch condition ──────────────────────────────────────────────────
        safe_close = np.where(close > 0.0, close, 1.0)
        pinch = np.abs(ema - vwap) / safe_close < pinch_threshold

        # ── Directional filters ──────────────────────────────────────────────
        above = (close > ema) & (close > vwap)   # bullish break
        below = (close < ema) & (close < vwap)   # bearish break

        # ── Volume surge ─────────────────────────────────────────────────────
        safe_avg_vol = np.where(avg_vol > 0.0, avg_vol, 1.0)
        vol_surge = volume > (vol_multiplier * safe_avg_vol)

        # ── Session and warmup masks ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed     = np.arange(n) >= ema_period

        # ── Signals ──────────────────────────────────────────────────────────
        buy_ce = in_session & warmed & pinch & above & vol_surge
        buy_pe = in_session & warmed & pinch & below & vol_surge

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts,   dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
