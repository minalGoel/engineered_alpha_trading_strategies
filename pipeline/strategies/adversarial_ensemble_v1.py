"""
adversarial_ensemble_v1 — Multi-signal adversarial ensemble on NIFTY 5-second bars.

Runs 5 sub-strategies simultaneously (VWAP z-score, ORB direction, RSI extreme,
volume spike, EMA trend/pullback) and only enters when at least 3 of 5 agree on
direction AND none of the remaining actively contradict. Targets 30-120 second
holds on NIFTY ATM options.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder-style EMA (exponential moving average). Input must be NaN-free."""
    n = len(arr)
    result = np.empty(n)
    result[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns 50.0 for warm-up bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = float(np.mean(gain[:period]))
    avg_loss = float(np.mean(loss[:period]))
    if avg_loss > 0.0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0.0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "adversarial_ensemble_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30-min warmup for VWAP stddev window

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ensemble_threshold", 3.0, 2.0, 5.0),
            TunableParam("vix_max", 22.0, 18.0, 28.0),
            TunableParam("vwap_zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("volume_spike_ratio", 1.8, 1.3, 3.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-filled before numpy conversion) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        vix_max = float(params.get("vix_max", 22.0))
        vwap_z_thresh = float(params.get("vwap_zscore_threshold", 1.5))
        vol_ratio = float(params.get("volume_spike_ratio", 1.8))
        ens_thresh = int(round(float(params.get("ensemble_threshold", 3.0))))

        # ── VIX (aligned to spot bars via asof join) ──
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

        # ────────────────────────────────────────────────────────────
        # Sub-strategy 1: VWAP z-score
        # Session VWAP (cumulative from open, reset daily) with rolling
        # 360-bar stddev of deviation for normalization.
        # Signal: +1 if z < -threshold (price below VWAP, expect reversion up)
        #         -1 if z > +threshold (price above VWAP, expect reversion down)
        # ────────────────────────────────────────────────────────────
        tp = (high + low + close) / 3.0
        cum_tp_vol = np.zeros(n)
        cum_vol = np.zeros(n)
        for i in range(n):
            v = volume[i]
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tp_vol[i] = tp[i] * v
                cum_vol[i] = v
            else:
                cum_tp_vol[i] = cum_tp_vol[i - 1] + tp[i] * v
                cum_vol[i] = cum_vol[i - 1] + v
        vwap = cum_tp_vol / np.maximum(cum_vol, 1.0)
        dev = close - vwap

        dev_std = np.ones(n)
        for i in range(360, n):
            s = float(np.std(dev[i - 360:i]))
            dev_std[i] = s if s > 0.1 else 0.1

        vwap_z = dev / dev_std
        vwap_sig = np.where(
            vwap_z < -vwap_z_thresh, 1,
            np.where(vwap_z > vwap_z_thresh, -1, 0)
        ).astype(np.int8)

        # ────────────────────────────────────────────────────────────
        # Sub-strategy 2: Opening Range Breakout (20 min = 240 bars)
        # Fixed time window — same 20-min calendar period regardless of bar size.
        # Signal: +1 if close > ORB high, -1 if close < ORB low
        # ────────────────────────────────────────────────────────────
        orb_high = np.zeros(n)
        orb_low = np.zeros(n)
        day_starts: dict[int, int] = {}
        day_orb_h: dict[int, float] = {}
        day_orb_l: dict[int, float] = {}

        for i in range(n):
            d = int(day_id[i])
            if d not in day_starts:
                day_starts[d] = i
                day_orb_h[d] = high[i]
                day_orb_l[d] = low[i]
            bars_in = i - day_starts[d]
            if bars_in < 240:
                if high[i] > day_orb_h[d]:
                    day_orb_h[d] = high[i]
                if low[i] < day_orb_l[d]:
                    day_orb_l[d] = low[i]
            orb_high[i] = day_orb_h[d]
            orb_low[i] = day_orb_l[d]

        orb_sig = np.where(
            close > orb_high, 1,
            np.where(close < orb_low, -1, 0)
        ).astype(np.int8)

        # ────────────────────────────────────────────────────────────
        # Sub-strategy 3: RSI extreme (36 bars = 3 minutes)
        # Compressed from original 40-bar/40-min to 36-bar/3-min.
        # We target the fast exhaustion signal, not the slow divergence.
        # Signal: +1 if RSI < 35 (oversold, expect bounce)
        #         -1 if RSI > 65 (overbought, expect fade)
        # ────────────────────────────────────────────────────────────
        rsi = _rsi(close, 36)
        rsi_sig = np.where(
            rsi < 35.0, 1,
            np.where(rsi > 65.0, -1, 0)
        ).astype(np.int8)

        # ────────────────────────────────────────────────────────────
        # Sub-strategy 4: Volume spike on directional bar
        # 60-bar (5-min) rolling volume baseline.
        # Signal: +1 if vol spike AND close > open (green bar)
        #         -1 if vol spike AND close < open (red bar)
        # ────────────────────────────────────────────────────────────
        vol_ma = np.zeros(n)
        for i in range(60, n):
            vol_ma[i] = float(np.mean(volume[i - 60:i]))

        vol_spike = volume > (vol_ratio * np.maximum(vol_ma, 1.0))
        bar_green = close > open_
        bar_red = close < open_
        vol_sig = np.where(
            vol_spike & bar_green, 1,
            np.where(vol_spike & bar_red, -1, 0)
        ).astype(np.int8)

        # ────────────────────────────────────────────────────────────
        # Sub-strategy 5: EMA trend + micro-pullback
        # EMA(120) = 10-min trend context (same order of magnitude as
        # original EMA(20) on 1-min = 20-min). EMA(60) = 5-min trigger.
        # Signal: +1 if price crosses above EMA(60) while above EMA(120)
        #         -1 if price crosses below EMA(60) while below EMA(120)
        # ────────────────────────────────────────────────────────────
        ema120 = _ema(close, 120)
        ema60 = _ema(close, 60)

        uptrend = close > ema120
        downtrend = close < ema120

        ema_cross_up = np.zeros(n, dtype=bool)
        ema_cross_dn = np.zeros(n, dtype=bool)
        for i in range(1, n):
            ema_cross_up[i] = (
                close[i] > ema60[i] and close[i - 1] <= ema60[i - 1]
            )
            ema_cross_dn[i] = (
                close[i] < ema60[i] and close[i - 1] >= ema60[i - 1]
            )

        ema_sig = np.where(
            uptrend & ema_cross_up, 1,
            np.where(downtrend & ema_cross_dn, -1, 0)
        ).astype(np.int8)

        # ────────────────────────────────────────────────────────────
        # Ensemble logic + adversarial veto
        # ────────────────────────────────────────────────────────────
        ensemble = (
            vwap_sig.astype(np.int32)
            + orb_sig.astype(np.int32)
            + rsi_sig.astype(np.int32)
            + vol_sig.astype(np.int32)
            + ema_sig.astype(np.int32)
        )

        # Adversarial veto: any sub-strategy actively opposing the direction
        has_bearish_sub = (
            (vwap_sig == -1)
            | (orb_sig == -1)
            | (rsi_sig == -1)
            | (vol_sig == -1)
            | (ema_sig == -1)
        )
        has_bullish_sub = (
            (vwap_sig == 1)
            | (orb_sig == 1)
            | (rsi_sig == 1)
            | (vol_sig == 1)
            | (ema_sig == 1)
        )

        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = vix_close < vix_max

        # Raw signal (before 2-bar persistence)
        bull_raw = (ensemble >= ens_thresh) & (~has_bearish_sub)
        bear_raw = (ensemble <= -ens_thresh) & (~has_bullish_sub)

        # 2-bar persistence: both current AND previous bar must satisfy signal
        # (10-second confirmation — proportionally same as original's 2-bar on 1-min)
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if in_session[i] and vix_ok[i]:
                buy_ce[i] = bull_raw[i] and bull_raw[i - 1]
                buy_pe[i] = bear_raw[i] and bear_raw[i - 1]

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4),
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
