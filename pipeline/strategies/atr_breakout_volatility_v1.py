"""ATR Breakout Volatility Strategy (atr_breakout_volatility_v1)

Thesis: When NIFTY's 5-min ATR expands above its 10-min baseline AND price
breaches a 2.0*ATR channel anchored to session VWAP, an institutional
directional order (TWAP or stop-cascade) is overwhelming passive resting
liquidity. We enter within the first 1-2 bars of the breach — before most
1-min momentum algos confirm on their bar close.

Converted from: trading_strategies/unique_strategies_all/Strategy_211.json
(ATR channel breakout on 1-min equity bars, hold 15-60 min)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "atr_breakout_volatility_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — allow 10 min warmup after open
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 180            # 15 min warmup (ATR 60 + ATR baseline 120)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_multiplier",      2.0, 1.5, 3.0),
            TunableParam("atr_ratio_threshold", 1.2, 1.05, 1.6),
            TunableParam("vol_ratio_threshold", 1.3, 1.1, 2.0),
            TunableParam("stop_pts",            4.0, 2.0, 8.0),
            TunableParam("target_pts",          7.0, 4.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Pull arrays (forward-fill NaN before numpy) ──────────────────────
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        atr_mult        = params.get("atr_multiplier",      2.0)
        atr_ratio_thresh = params.get("atr_ratio_threshold", 1.2)
        vol_ratio_thresh = params.get("vol_ratio_threshold", 1.3)
        stop_pts        = params.get("stop_pts",            4.0)
        target_pts      = params.get("target_pts",          7.0)

        # ── True Range ───────────────────────────────────────────────────────
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i]  - close[i - 1]),
            )

        # ── ATR(60) — Wilder smoothing ────────────────────────────────────────
        period = 60
        atr = np.zeros(n)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            # back-fill early bars with first valid value
            first_val = atr[period - 1]
            for i in range(period - 1):
                atr[i] = first_val
        else:
            atr[:] = np.mean(tr) if n > 0 else 1.0

        # ── ATR expansion ratio: atr / SMA(atr, 120) ─────────────────────────
        baseline = 120
        atr_sma = np.empty(n)
        for i in range(baseline, n):
            atr_sma[i] = np.mean(atr[i - baseline:i])
        # fill early bars with first available value (or atr itself)
        fill_val = atr_sma[baseline] if n > baseline else (np.mean(atr) if n > 0 else 1.0)
        for i in range(min(baseline, n)):
            atr_sma[i] = fill_val

        atr_ratio = atr / np.maximum(atr_sma, 1e-6)

        # ── Cumulative VWAP (reset per day) ──────────────────────────────────
        vwap = np.empty(n)
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
            vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

        # ── ATM channels ─────────────────────────────────────────────────────
        upper_channel = vwap + atr_mult * atr
        lower_channel = vwap - atr_mult * atr

        # ── SMA(60) trend filter (replaces ADX — too noisy at 5s) ────────────
        sma_60 = np.empty(n)
        for i in range(60, n):
            sma_60[i] = np.mean(close[i - 60:i])
        # back-fill early bars
        fill_sma = sma_60[60] if n > 60 else close[0]
        for i in range(min(60, n)):
            sma_60[i] = fill_sma

        # ── Volume ratio: volume / SMA(volume, 60) ───────────────────────────
        vol_sma = np.empty(n)
        for i in range(60, n):
            vol_sma[i] = np.mean(volume[i - 60:i])
        fill_vol = vol_sma[60] if n > 60 else max(volume[0], 1.0)
        for i in range(min(60, n)):
            vol_sma[i] = fill_vol

        vol_ratio = volume / np.maximum(vol_sma, 1.0)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Signals ───────────────────────────────────────────────────────────
        atr_expanding  = atr_ratio  > atr_ratio_thresh
        vol_confirming = vol_ratio  > vol_ratio_thresh

        buy_ce = (
            in_session
            & (close > upper_channel)
            & atr_expanding
            & (close > sma_60)
            & vol_confirming
        )

        buy_pe = (
            in_session
            & (close < lower_channel)
            & atr_expanding
            & (close < sma_60)
            & vol_confirming
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,            # 24 bars × 5s = 120s max hold
            max_trades_per_day=self.max_trades_per_day,
        )
