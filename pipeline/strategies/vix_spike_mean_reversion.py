"""VIX Spike Mean Reversion — 5-second NIFTY index options.

Thesis: A rapid 5%+ surge in India VIX within 10 minutes forces NIFTY options market
makers to delta-hedge aggressively (selling NIFTY futures), pushing NIFTY below session
VWAP. Once VIX stabilizes (stops rising), delta-hedge selling dissipates and resting buy
orders absorbed below VWAP drive a 30-90 second mean reversion toward VWAP.

Converted from: trading_strategies/unique_strategies_all/Strategy_355.json
Original timeframe: 1-min bars, 15-45 min hold.
Key changes: RSI compressed 14-min → 3-min (36 bars) for fast reversion; added VIX
stabilization check (not in original) possible only at 5s resolution; universe shifted
from NIFTY50 stocks → NIFTY index (strengthens the mechanism: VIX IS NIFTY's IV).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vix_spike_mean_reversion"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — ensures 10-min VIX lookback is populated
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 4
    max_lookback = 120            # 10-min warmup for VIX spike window (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_spike_threshold", 0.05, 0.03, 0.10),
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        vix_spike_threshold = params.get("vix_spike_threshold", 0.05)
        rsi_oversold = params.get("rsi_oversold", 35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── Spot arrays ──────────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── VIX: 10-min spike detection + 30s stabilization ─────────────────────
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

        # VIX spike: current vs 120 bars ago (10 minutes)
        vix_spike = np.zeros(n)
        for i in range(120, n):
            past = vix_close[i - 120]
            if past > 0.0:
                vix_spike[i] = (vix_close[i] - past) / past

        # VIX stabilizing: not printing new highs in last 6 bars (30 seconds)
        # True when VIX is ≤ its value 30 seconds ago — selling pressure abating
        vix_stabilizing = np.zeros(n, dtype=bool)
        for i in range(6, n):
            vix_stabilizing[i] = vix_close[i] <= vix_close[i - 6]

        # ── Session VWAP on NIFTY (cumulative, resets each day) ─────────────────
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            v = volume[i]
            cum_tp_vol += typical_price[i] * v
            cum_vol += v
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else typical_price[i]

        # ── RSI(36) on NIFTY spot — 3-minute oversold/overbought ────────────────
        # 36 bars = 3 min (compressed from original 14-min; we target the fast bounce)
        rsi_period = 36
        rsi = np.full(n, 50.0)
        if n > rsi_period:
            gains = np.zeros(n)
            losses = np.zeros(n)
            for i in range(1, n):
                diff = close[i] - close[i - 1]
                if diff > 0.0:
                    gains[i] = diff
                else:
                    losses[i] = -diff

            avg_gain = np.zeros(n)
            avg_loss = np.zeros(n)
            avg_gain[rsi_period] = np.mean(gains[1 : rsi_period + 1])
            avg_loss[rsi_period] = np.mean(losses[1 : rsi_period + 1])
            for i in range(rsi_period + 1, n):
                avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gains[i]) / rsi_period
                avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + losses[i]) / rsi_period

            for i in range(rsi_period, n):
                if avg_loss[i] == 0.0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain[i] / avg_loss[i]
                    rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        # ── Signals ──────────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        has_spike = vix_spike >= vix_spike_threshold

        # Buy CE: VIX spiked → now stabilizing → NIFTY oversold below VWAP → bounce up
        buy_ce = (
            in_session
            & has_spike
            & vix_stabilizing
            & (rsi < rsi_oversold)
            & (close < vwap)
        )

        # Buy PE: VIX spiked → now stabilizing → NIFTY overbought above VWAP → fade down
        # (rarer — occurs when VIX spike is macro fear but NIFTY has lagged reversal)
        buy_pe = (
            in_session
            & has_spike
            & vix_stabilizing
            & (rsi > rsi_overbought)
            & (close > vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
