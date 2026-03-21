"""vwap_mean_reversion_v12 — Volume-Exhaustion VWAP Reversion on NIFTY

Mechanism: When NIFTY flushes below session VWAP with a volume spike (rel_vol > 1.3x
1-minute average), the surge represents capitulation by VWAP-benchmarked algorithms
forced to liquidate. Once the high-volume bar clears the book, resting limit buy orders
absorb remaining supply and price snaps back within 30-75 seconds. Symmetric for above-
VWAP high-volume run-ups.

Differentiator vs v18/v19: Uses volume exhaustion (intrabar volume spike vs 1-min
average) as primary entry filter, not deviation momentum (v18) or range contraction (v19).
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v12"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST (flatten before EOD)
    max_trades_per_day = 6
    max_lookback = 120            # 10 min warmup for volume SMA and stddev to stabilize

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("vol_ratio_threshold", 1.3, 1.1, 2.0),
            TunableParam("vix_max", 28.0, 20.0, 35.0),
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        zscore_threshold = params.get("zscore_threshold", 1.5)
        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.3)
        vix_max = params.get("vix_max", 28.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── VIX filter ───────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session VWAP (cumulative, reset per day) ─────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_v = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv = 0.0
                cum_v = 0.0
                prev_day = day_id[i]
            vol_i = volume[i] if volume[i] > 0 else 1.0
            cum_pv += close[i] * vol_i
            cum_v += vol_i
            vwap[i] = cum_pv / cum_v

        # ── VWAP deviation ───────────────────────────────────────────────────
        dev = close - vwap  # raw deviation in index points

        # ── Rolling 3-min (36-bar) stddev of close for z-score ──────────────
        # 36 bars chosen: proportional to 30-90s hold; detects current-regime spike severity
        window_std = 36
        rolling_std = np.ones(n) * 5.0  # neutral default ~5 NIFTY pts
        for i in range(window_std, n):
            s = np.std(close[i - window_std:i])
            rolling_std[i] = s if s > 0.5 else 0.5

        vwap_zscore = dev / rolling_std

        # ── Rolling 1-min (12-bar) volume SMA for relative volume ─────────────
        # 12 bars (1 min) vs original's 20-bar 1-min SMA (20 min):
        # detects intrabar capitulation spikes vs RECENT pace, not historical
        window_vol = 12
        vol_sma = np.ones(n)
        for i in range(window_vol, n):
            avg = np.mean(volume[i - window_vol:i])
            vol_sma[i] = avg if avg > 0 else 1.0

        rel_vol = np.zeros(n)
        for i in range(n):
            rel_vol[i] = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 0.0

        # ── Single-bar 5s return (confirms flush still active) ───────────────
        bar_ret = np.zeros(n)
        bar_ret[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] > 0, close[:-1], 1.0)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── VIX filter ────────────────────────────────────────────────────────
        vix_ok = vix_close < vix_max

        # ── Warmup mask (need at least max_lookback bars) ─────────────────────
        warmup = np.zeros(n, dtype=bool)
        warmup[self.max_lookback:] = True

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: VWAP flush with volume capitulation — bullish reversion
        buy_ce = (
            in_session
            & vix_ok
            & warmup
            & (vwap_zscore < -zscore_threshold)    # below VWAP by threshold z-scores
            & (rel_vol > vol_ratio_threshold)       # volume surge = capitulation spike
            & (bar_ret < 0)                          # flush still active on this bar
        )

        # buy_pe: VWAP run-up with volume capitulation — bearish reversion
        buy_pe = (
            in_session
            & vix_ok
            & warmup
            & (vwap_zscore > zscore_threshold)     # above VWAP by threshold z-scores
            & (rel_vol > vol_ratio_threshold)       # volume surge = exhaustion spike
            & (bar_ret > 0)                          # run-up still active on this bar
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds — reversion should occur within 90s
            max_trades_per_day=self.max_trades_per_day,
        )
