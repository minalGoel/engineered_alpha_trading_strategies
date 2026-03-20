"""Momentum Score Composite v1 — cursor_opus46max_026

Thesis: A composite momentum score aggregating 5 normalized sub-signals
(RSI(14), MACD histogram, ROC(10), ADX direction, volume momentum) into
a single -100 to +100 score. When composite > 60, all sub-signals agree
on strong directional momentum. Requires 3+ consecutive bars of elevated
score and at least 4 of 5 sub-scores agreeing on direction.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


def _compute_ema(data, period):
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = data[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema


def _compute_adx(high, low, close, period):
    n = len(close)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return plus_di, minus_di, adx
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    # Wilder smoothing
    sm_tr = np.zeros(n, dtype=np.float64)
    sm_pdm = np.zeros(n, dtype=np.float64)
    sm_mdm = np.zeros(n, dtype=np.float64)
    sm_tr[period] = np.sum(tr[1:period+1])
    sm_pdm[period] = np.sum(plus_dm[1:period+1])
    sm_mdm[period] = np.sum(minus_dm[1:period+1])
    for i in range(period+1, n):
        sm_tr[i] = sm_tr[i-1] - sm_tr[i-1]/period + tr[i]
        sm_pdm[i] = sm_pdm[i-1] - sm_pdm[i-1]/period + plus_dm[i]
        sm_mdm[i] = sm_mdm[i-1] - sm_mdm[i-1]/period + minus_dm[i]
    for i in range(period, n):
        if sm_tr[i] > 0:
            plus_di[i] = 100.0 * sm_pdm[i] / sm_tr[i]
            minus_di[i] = 100.0 * sm_mdm[i] / sm_tr[i]
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s
    if n >= 2 * period:
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return plus_di, minus_di, adx


class Strategy(BaseStrategy):
    name = "cursor_opus46max_026"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("composite_thresh", default=60.0, low=40.0, high=80.0),
            TunableParam("sustain_thresh", default=50.0, low=30.0, high=70.0),
            TunableParam("sustain_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.005),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        comp_thresh = params.get("composite_thresh", 60.0)
        sustain_thresh = params.get("sustain_thresh", 50.0)
        sustain_bars = int(params.get("sustain_bars", 3.0))
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        tgt_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Sub-score 1: RSI(14) score ──
        rsi = _compute_rsi(close, 14)
        rsi_score = np.clip((rsi - 50.0) / 2.5, -20.0, 20.0)

        # ── Sub-score 2: MACD histogram score ──
        ema12 = _compute_ema(close, 12)
        ema26 = _compute_ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = _compute_ema(macd_line, 9)
        macd_hist = macd_line - macd_signal
        safe_atr = np.where(atr > 0, atr, 1e-10)
        macd_score = np.clip(macd_hist / safe_atr * 200.0, -20.0, 20.0)

        # ── Sub-score 3: ROC(10) score ──
        roc10 = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if close[i-10] > 0:
                roc10[i] = (close[i] - close[i-10]) / close[i-10] * 100.0
        roc_score = np.clip(roc10 * 50.0, -20.0, 20.0)

        # ── Sub-score 4: ADX direction score ──
        plus_di, minus_di, adx = _compute_adx(high, low, close, 14)
        adx_dir_score = np.sign(plus_di - minus_di) * np.minimum(adx, 40.0) / 2.0

        # ── Sub-score 5: Volume momentum score ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        price_dir = np.sign(close - opn)
        vol_mom_score = np.clip(
            (volume - avg_vol_20) / avg_vol_20 * 20.0 * price_dir,
            -20.0, 20.0
        )

        # ── Composite score ──
        composite = rsi_score + macd_score + roc_score + adx_dir_score + vol_mom_score

        # ── Count positive/negative sub-scores ──
        pos_count = ((rsi_score > 0).astype(int) + (macd_score > 0).astype(int) +
                     (roc_score > 0).astype(int) + (adx_dir_score > 0).astype(int) +
                     (vol_mom_score > 0).astype(int))
        neg_count = ((rsi_score < 0).astype(int) + (macd_score < 0).astype(int) +
                     (roc_score < 0).astype(int) + (adx_dir_score < 0).astype(int) +
                     (vol_mom_score < 0).astype(int))

        # ── Sustained composite ──
        sustained_bull = np.zeros(n, dtype=np.bool_)
        sustained_bear = np.zeros(n, dtype=np.bool_)
        bull_count = 0
        bear_count = 0
        for i in range(n):
            if composite[i] > sustain_thresh:
                bull_count += 1
            else:
                bull_count = 0
            if composite[i] < -sustain_thresh:
                bear_count += 1
            else:
                bear_count = 0
            sustained_bull[i] = bull_count >= sustain_bars
            sustained_bear[i] = bear_count >= sustain_bars

        # ── Volume filter ──
        vol_ok = volume > avg_vol_20

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 585) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (composite > comp_thresh) & sustained_bull & (pos_count >= 4) &
            (close > vwap) & vol_ok & vix_ok & time_ok
        )
        short_entry = (
            (composite < -comp_thresh) & sustained_bear & (neg_count >= 4) &
            (close < vwap) & vol_ok & vix_ok & time_ok
        )

        # ── Signal exits: composite crosses zero ──
        signal_exit_long = composite < 0
        signal_exit_short = composite > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=45,
        )
