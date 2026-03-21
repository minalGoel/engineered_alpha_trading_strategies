"""VWAP Z-Score Mean Reversion — 5-second NIFTY index options.

Original: equity stock VWAP reversion on nifty200, 1-min bars, 15-45 min hold.
Converted: NIFTY index VWAP deviation snap-back, 5-second bars, 30-90s hold.

Mechanism: On NIFTY, institutional VWAP-benchmarked algorithms accelerate their
counter-flow when the index deviates 2.5+ sigma from session VWAP in a non-trending
regime, producing a snap-back of 10-20 spot points within 30-90 seconds.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_zscore_mean_reversion"
    underlying = "NIFTY"
    session_start_minutes = 600   # 10:00 IST — avoid opening auction noise
    session_end_minutes = 870     # 14:30 IST — avoid late-session trend dominance
    max_lookback = 120            # 10 min warmup for rolling std
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.5, 1.5, 4.0),
            TunableParam("de_threshold", 0.30, 0.15, 0.45),
            TunableParam("vix_max", 20.0, 12.0, 30.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays ──────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────
        zscore_threshold = params.get("zscore_threshold", 2.5)
        de_threshold = params.get("de_threshold", 0.30)
        vix_max = params.get("vix_max", 20.0)

        # ── VIX (aligned to spot bars) ───────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session VWAP (cumulative per day, typical price) ─────────────────
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tp_vol = typical_price[i] * volume[i]
                cum_vol = volume[i]
            else:
                cum_tp_vol += typical_price[i] * volume[i]
                cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else typical_price[i]

        # ── Rolling 120-bar (10-min) std of close for z-score ────────────────
        # 10-min local vol is more responsive to the current session regime
        # than original's 20-min window; we're trading the fast snap-back.
        std_120 = np.zeros(n)
        for i in range(120, n):
            s = np.std(close[i - 120:i])
            std_120[i] = s if s > 0.0 else 1.0
        std_120[:120] = 1.0  # neutral — prevents division by zero during warmup

        vwap_z = np.zeros(n)
        for i in range(120, n):
            vwap_z[i] = (close[i] - vwap[i]) / std_120[i]

        # ── Directional efficiency over 60 bars (5 min) — replaces ADX ───────
        # de = |net move| / sum(|bar-to-bar moves|). Low (<0.30) = choppy = good
        # for mean reversion. 60 bars chosen so filter reacts within 5 minutes
        # to a trending shift — faster than Wilder ADX(14 min) at 5s resolution.
        de_60 = np.full(n, 0.5)  # neutral (assume trending during warmup)
        for i in range(60, n):
            bars = close[i - 60:i + 1]
            bar_moves = np.abs(np.diff(bars))
            total_path = np.sum(bar_moves)
            net_move = abs(bars[-1] - bars[0])
            de_60[i] = net_move / total_path if total_path > 0.0 else 0.0

        # ── Session filter ───────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & \
                     (time_min < self.session_end_minutes)

        # ── Reversal bar confirmation ────────────────────────────────────────
        bullish_bar = close > open_   # price recovering from oversold extreme
        bearish_bar = close < open_   # price reversing from overbought extreme

        # ── Signals ─────────────────────────────────────────────────────────
        # buy_ce: NIFTY 2.5+ sigma BELOW VWAP + choppy + calm VIX + bullish bar
        buy_ce = (
            in_session
            & (vwap_z < -zscore_threshold)
            & (de_60 < de_threshold)
            & (vix_close < vix_max)
            & bullish_bar
        )

        # buy_pe: NIFTY 2.5+ sigma ABOVE VWAP + choppy + calm VIX + bearish bar
        buy_pe = (
            in_session
            & (vwap_z > zscore_threshold)
            & (de_60 < de_threshold)
            & (vix_close < vix_max)
            & bearish_bar
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop: 4 pts (~8 NIFTY spot pts). Deeper deviation invalidates thesis.
            stop_points=np.full(n, 4.0),
            # target: 7 pts (~14 NIFTY spot pts). Captures 60-70% of snap-back.
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
