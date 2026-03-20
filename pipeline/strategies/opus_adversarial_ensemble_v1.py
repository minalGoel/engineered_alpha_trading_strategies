"""Adversarial Ensemble — Opus_50

Thesis: Combine 5 independent sub-signals into an ensemble score.
Only enter when >= 3 signals agree with no contradictions (no opposing
signal is active). Sub-signals: VWAP zscore, ORB breakout, RSI
divergence, volume spike + green bar, pullback to EMA(20) in trend.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(arr, period):
    """EMA with standard multiplier."""
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    k = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


class Strategy(BaseStrategy):
    name = "opus_adversarial_ensemble_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("vol_spike_ratio", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("trailing_stop_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 1.5)
        vol_spike = params.get("vol_spike_ratio", 2.0)
        vix_max = params.get("vix_max", 22.0)
        tgt_pct = params.get("target_pct", 0.004)
        stp_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_stop_pct", 0.002)
        trail_act = params.get("trailing_activate_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Z-score ──
        dev = close - vwap
        std_30 = df["close"].rolling_std(30).to_numpy().astype(np.float64)
        std_30 = np.nan_to_num(std_30, nan=1e10)
        std_30 = np.clip(std_30, 1e-10, None)
        zscore = dev / std_30
        zscore = np.nan_to_num(zscore, nan=0.0)

        rsi9 = _compute_rsi(close, 9)
        ema20 = _compute_ema(close, 20)
        atr = _compute_atr(high, low, close, 20)

        # ── ORB per day (555-570) ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        prev_day = -1
        d_hi = 0.0
        d_lo = 1e18

        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                d_hi = high[i]
                d_lo = low[i]
            if time_mins[i] <= 570:
                d_hi = max(d_hi, high[i])
                d_lo = min(d_lo, low[i])
            orb_high[i] = d_hi
            orb_low[i] = d_lo

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        green_bar = close > open_
        red_bar = close < open_

        # ── RSI divergence (20-bar lookback) ──
        lookback = 20
        bullish_div = np.zeros(n, dtype=np.bool_)
        bearish_div = np.zeros(n, dtype=np.bool_)
        for i in range(lookback, n):
            ws = i - lookback
            pmin_idx = ws + np.argmin(low[ws:i])
            if low[i] < low[pmin_idx] and rsi9[i] > rsi9[pmin_idx]:
                bullish_div[i] = True
            pmax_idx = ws + np.argmax(high[ws:i])
            if high[i] > high[pmax_idx] and rsi9[i] < rsi9[pmax_idx]:
                bearish_div[i] = True

        # ── 5 sub-signals: each +1 for long, -1 for short, 0 neutral ──
        sig1 = np.zeros(n, dtype=np.int32)  # VWAP zscore
        sig2 = np.zeros(n, dtype=np.int32)  # ORB breakout
        sig3 = np.zeros(n, dtype=np.int32)  # RSI divergence
        sig4 = np.zeros(n, dtype=np.int32)  # vol spike + green/red
        sig5 = np.zeros(n, dtype=np.int32)  # pullback to EMA(20) in trend

        after_orb = time_mins > 570

        # Signal 1: VWAP zscore
        sig1[zscore < -zs_thresh] = 1
        sig1[zscore > zs_thresh] = -1

        # Signal 2: ORB breakout
        sig2[(close > orb_high) & after_orb] = 1
        sig2[(close < orb_low) & after_orb] = -1

        # Signal 3: RSI divergence
        sig3[bullish_div] = 1
        sig3[bearish_div] = -1

        # Signal 4: Volume spike + green/red bar
        vol_high = volume > vol_spike * avg_vol
        sig4[vol_high & green_bar] = 1
        sig4[vol_high & red_bar] = -1

        # Signal 5: Pullback to EMA(20) in trend
        # Uptrend: close > ema20 overall, current low touches ema20
        safe_ema = np.where(ema20 > 0, ema20, 1.0)
        near_ema = np.abs(low - ema20) / safe_ema < 0.002
        near_ema_high = np.abs(high - ema20) / safe_ema < 0.002
        uptrend = close > ema20
        downtrend = close < ema20
        sig5[near_ema & uptrend & green_bar] = 1
        sig5[near_ema_high & downtrend & red_bar] = -1

        # ── Ensemble score ──
        score = sig1 + sig2 + sig3 + sig4 + sig5

        # Count positives and negatives
        pos_count = ((sig1 > 0).astype(np.int32) + (sig2 > 0).astype(np.int32) +
                     (sig3 > 0).astype(np.int32) + (sig4 > 0).astype(np.int32) +
                     (sig5 > 0).astype(np.int32))
        neg_count = ((sig1 < 0).astype(np.int32) + (sig2 < 0).astype(np.int32) +
                     (sig3 < 0).astype(np.int32) + (sig4 < 0).astype(np.int32) +
                     (sig5 < 0).astype(np.int32))

        vix_ok = vix < vix_max
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: score >= 3 AND no signal is -1 AND VIX < max
        long_entry = ((score >= 3) & (neg_count == 0) & vix_ok & time_ok)

        # Short: score <= -3 AND no signal is +1
        short_entry = ((score <= -3) & (pos_count == 0) & vix_ok & time_ok)

        # Signal exit: score drops below +2 (longs) or above -2 (shorts)
        exit_long = (score < 2) & time_ok
        exit_short = (score > -2) & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=90,
        )
