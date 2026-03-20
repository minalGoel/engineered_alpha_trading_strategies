"""VWAP Mean Reversion Basic v1 — volume-surge + moderate RSI + two-bar confirmation.

On NIFTY, moderate 0.10-0.20% VWAP deviations coinciding with volume spikes mark the
climax of retail/algo flow. When RSI(24) shows moderate exhaustion (35/65) and two
consecutive 5-second bars confirm the reversal has begun, VWAP-benchmarked institutional
buyers/sellers have absorbed the flow and the mean-reversion snapback follows.

Original: equity VWAP mean reversion on NIFTY200 stocks (1-min, hold 15-60 min).
Conversion: NIFTY index options at 5s, hold 30-90s.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_basic_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST
    session_end_minutes = 870     # 14:30 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20-min warmup for volume SMA baseline

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_threshold", 0.12, 0.05, 0.25),  # % deviation
            TunableParam("rsi_low", 35.0, 22.0, 45.0),
            TunableParam("rsi_high", 65.0, 55.0, 78.0),
            TunableParam("volume_ratio_min", 1.5, 1.0, 3.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN before to_numpy) ──────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()
        day_id = spot_df["day_id"].fill_null(0).to_numpy()

        # ── Parameters ─────────────────────────────────────────────────────────
        dev_thr = params.get("vwap_dev_threshold", 0.12)
        rsi_low = params.get("rsi_low", 35.0)
        rsi_high = params.get("rsi_high", 65.0)
        vol_min = params.get("volume_ratio_min", 1.5)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── VIX filter ──────────────────────────────────────────────────────────
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

        # ── Session VWAP (cumulative, resets per day) ──────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_vol = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_pv = close[i] * volume[i]
                cum_vol = volume[i]
            else:
                cum_pv += close[i] * volume[i]
                cum_vol += volume[i]
            vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close[i]

        # VWAP deviation %
        vwap_safe = np.where(vwap > 0.0, vwap, close)
        vwap_dev = (close - vwap_safe) / vwap_safe * 100.0

        # ── RSI(24) — Wilder smoothing, 2-minute window ────────────────────────
        RSI_PERIOD = 24
        rsi = np.full(n, 50.0)
        if n > RSI_PERIOD + 1:
            delta = np.zeros(n)
            delta[1:] = close[1:] - close[:-1]
            up = np.where(delta > 0.0, delta, 0.0)
            dn = np.where(delta < 0.0, -delta, 0.0)

            avg_up = np.zeros(n)
            avg_dn = np.zeros(n)
            avg_up[RSI_PERIOD] = np.mean(up[1 : RSI_PERIOD + 1])
            avg_dn[RSI_PERIOD] = np.mean(dn[1 : RSI_PERIOD + 1])
            for i in range(RSI_PERIOD + 1, n):
                avg_up[i] = (avg_up[i - 1] * (RSI_PERIOD - 1) + up[i]) / RSI_PERIOD
                avg_dn[i] = (avg_dn[i - 1] * (RSI_PERIOD - 1) + dn[i]) / RSI_PERIOD

            for i in range(RSI_PERIOD, n):
                if avg_dn[i] < 1e-10:
                    rsi[i] = 100.0
                else:
                    rs = avg_up[i] / avg_dn[i]
                    rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        # ── Volume ratio: current / 20-min (240 bar) rolling mean ──────────────
        VOL_PERIOD = 240
        cumvol = np.cumsum(volume)
        # rolling sum ending at index i (inclusive) over VOL_PERIOD bars
        # sum[i] = cumvol[i] - cumvol[i - VOL_PERIOD]  for i >= VOL_PERIOD
        roll_sum = np.empty(n)
        roll_sum[:VOL_PERIOD] = cumvol[:VOL_PERIOD]   # partial window — mean underestimates
        roll_sum[VOL_PERIOD:] = cumvol[VOL_PERIOD:] - cumvol[:n - VOL_PERIOD]
        roll_count = np.minimum(np.arange(1, n + 1), VOL_PERIOD).astype(float)
        roll_mean = roll_sum / roll_count
        roll_mean = np.where(roll_mean > 0.0, roll_mean, 1.0)

        vol_ratio = volume / roll_mean
        vol_ratio[:VOL_PERIOD] = 1.0  # insufficient history — neutral

        # ── Two-bar momentum confirmation ──────────────────────────────────────
        two_bar_bull = np.zeros(n, dtype=bool)
        two_bar_bear = np.zeros(n, dtype=bool)
        if n > 2:
            two_bar_bull[2:] = (close[2:] > close[1:-1]) & (close[1:-1] > close[:-2])
            two_bar_bear[2:] = (close[2:] < close[1:-1]) & (close[1:-1] < close[:-2])

        # ── Composite filters ──────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        low_vix = vix_close < vix_max
        vol_surge = vol_ratio > vol_min

        # BUY CE: NIFTY below VWAP by dev_thr%, RSI oversold, volume surge, two-bar bounce
        buy_ce = (
            in_session
            & low_vix
            & (vwap_dev < -dev_thr)
            & (rsi < rsi_low)
            & vol_surge
            & two_bar_bull
        )

        # BUY PE: NIFTY above VWAP by dev_thr%, RSI overbought, volume surge, two-bar fade
        buy_pe = (
            in_session
            & low_vix
            & (vwap_dev > dev_thr)
            & (rsi > rsi_high)
            & vol_surge
            & two_bar_bear
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
