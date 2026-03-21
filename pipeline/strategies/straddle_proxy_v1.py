"""straddle_proxy_v1 — Vol-Compression Pivot Breakout on NIFTY

Thesis:
    After the first 45 minutes of the session, NIFTY frequently consolidates
    near the previous day's close (PDC) — a reference level monitored by every
    institutional VWAP desk and passive fund. When the 5-min ATR compresses
    below 75% of its 20-min average AND Bollinger Band width falls to its
    lowest 20th percentile of the past 20 min, the order book is in equilibrium.
    Once price crosses PDC ± 0.5×ATR(5min), stop-loss cascades and gamma
    hedging amplify the move. Buy CE on upside breaks, buy PE on downside breaks.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "straddle_proxy_v1"
    underlying = "NIFTY"
    session_start_minutes = 600   # 10:00 IST — 45 min after open for pivot to establish
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 300            # 300 × 5s = 25 min (ATR60 + SMA240 both ready)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_compression_ratio", 0.75, 0.55, 0.90),
            TunableParam("bb_pctl_threshold", 20.0, 10.0, 35.0),
            TunableParam("vix_low", 12.0, 8.0, 16.0),
            TunableParam("vix_high", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        day_ids = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        atr_compression_ratio = params.get("atr_compression_ratio", 0.75)
        bb_pctl_threshold = params.get("bb_pctl_threshold", 20.0)
        vix_low_thr = params.get("vix_low", 12.0)
        vix_high_thr = params.get("vix_high", 22.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Previous Day Close (PDC) ──────────────────────────────────────────
        # Build a map: day_id → last close of that day, then shift by one day
        day_last_close: dict[int, float] = {}
        for i in range(n):
            day_last_close[int(day_ids[i])] = close[i]

        unique_days = sorted(day_last_close.keys())
        prev_close_map: dict[int, float] = {}
        for j, d in enumerate(unique_days):
            if j > 0:
                prev_close_map[d] = day_last_close[unique_days[j - 1]]
            else:
                prev_close_map[d] = np.nan  # no prior day for first day

        pdc = np.array([prev_close_map.get(int(d), np.nan) for d in day_ids])
        # Fill first-day NaN with the session's first close (neutral fallback)
        first_valid = close[0] if n > 0 else 0.0
        pdc = np.where(np.isnan(pdc), first_valid, pdc)

        # ── True Range & ATR(60) — 5-minute ATR ──────────────────────────────
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr60 = np.zeros(n)
        for i in range(60, n):
            atr60[i] = np.mean(tr[i - 60:i])

        # ── SMA(ATR60, 240) — 20-min rolling average of 5-min ATR ────────────
        atr60_sma240 = np.zeros(n)
        for i in range(300, n):          # need 60+240 bars before this is stable
            atr60_sma240[i] = np.mean(atr60[i - 240:i])

        # ATR compression: current 5-min ATR < ratio × 20-min baseline
        atr_compressed = (
            (atr60 > 0)
            & (atr60_sma240 > 0)
            & (atr60 < atr_compression_ratio * atr60_sma240)
        )

        # Recent compression: any compression in the past 6 bars (30 seconds)
        recent_compressed = np.zeros(n, dtype=bool)
        for i in range(6, n):
            if np.any(atr_compressed[i - 6:i + 1]):
                recent_compressed[i] = True

        # ── Bollinger Band width & percentile rank ────────────────────────────
        bb_window = 60
        bb_width = np.zeros(n)
        for i in range(bb_window, n):
            w = close[i - bb_window:i]
            bb_width[i] = 4.0 * np.std(w)  # = upper - lower (2σ bands)

        pctl_window = 240
        bb_pctl = np.full(n, 50.0)   # neutral default
        for i in range(bb_window + pctl_window, n):
            window = bb_width[i - pctl_window:i]
            total = len(window)
            bb_pctl[i] = np.sum(window < bb_width[i]) / total * 100.0

        bb_compressed = bb_pctl < bb_pctl_threshold

        # ── Trigger levels ────────────────────────────────────────────────────
        half_atr = 0.5 * atr60
        upper_trigger = pdc + half_atr
        lower_trigger = pdc - half_atr

        # Price crossing trigger: previous bar on opposite side, current bar crosses
        cross_up = np.zeros(n, dtype=bool)
        cross_dn = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if atr60[i] <= 0:
                continue
            if close[i - 1] < upper_trigger[i - 1] and close[i] >= upper_trigger[i]:
                cross_up[i] = True
            if close[i - 1] > lower_trigger[i - 1] and close[i] <= lower_trigger[i]:
                cross_dn[i] = True

        # ── VIX filter ────────────────────────────────────────────────────────
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

        vix_ok = (vix_close >= vix_low_thr) & (vix_close <= vix_high_thr)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Combine ───────────────────────────────────────────────────────────
        valid = in_session & recent_compressed & bb_compressed & vix_ok & (atr60 > 0)

        buy_ce = valid & cross_up
        buy_pe = valid & cross_dn

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
