"""VWAP Reclaim Trend — cursor_gpt54_vwap_reclaim_trend_001

Thesis: Exploits trend continuation after a temporary VWAP dislocation is
absorbed and price reclaims the benchmark.  Many intraday participants use
VWAP as a risk-transfer anchor and re-enter only after confirmation.

Entry long: below_vwap_count(8) >= 6, close > VWAP, EMA(9) > EMA(21),
    ADX(14) >= 20, index VWAP dev >= 0, close > prev high.
Entry short: above_vwap_count(8) >= 6, close < VWAP, EMA(9) < EMA(21),
    ADX(14) >= 20, index VWAP dev <= 0, close < prev low.
Target: 0.80% (approx 2R).  Stop: 0.8*ATR.  Time stop: 24 bars.
Signal exit: close crosses back through VWAP.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    alpha = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema


def _compute_adx(high, low, close, period):
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period * 2:
        return adx
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr_s = np.zeros(n, dtype=np.float64)
    pdm_s = np.zeros(n, dtype=np.float64)
    mdm_s = np.zeros(n, dtype=np.float64)
    atr_s[period] = np.sum(tr[1:period + 1])
    pdm_s[period] = np.sum(plus_dm[1:period + 1])
    mdm_s[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if atr_s[i] > 0:
            plus_di[i] = 100.0 * pdm_s[i] / atr_s[i]
            minus_di[i] = 100.0 * mdm_s[i] / atr_s[i]
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


class Strategy(BaseStrategy):
    name = "vwap_reclaim_trend_001"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 810     # 13:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_count_min", default=6.0, low=4.0, high=8.0),
            TunableParam("adx_min", default=20.0, low=15.0, high=25.0),
            TunableParam("rel_vol_min", default=1.2, low=1.0, high=2.0),
            TunableParam("stop_atr_mult", default=0.8, low=0.5, high=1.2),
            TunableParam("target_pct", default=0.008, low=0.005, high=0.012),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_cnt_min = int(params.get("vwap_count_min", 6.0))
        adx_min = params.get("adx_min", 20.0)
        rel_vol_min = params.get("rel_vol_min", 1.2)
        stop_atr = params.get("stop_atr_mult", 0.8)
        target_pct = params.get("target_pct", 0.008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        index_close = df["index_close"].to_numpy().astype(np.float64)

        atr = _compute_atr(high, low, close, 14)
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)
        adx = _compute_adx(high, low, close, 14)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Index VWAP dev (simple SMA proxy) ──
        unique_days = np.unique(day_ids)
        idx_vwap = np.zeros(n, dtype=np.float64)
        for d in unique_days:
            mask = day_ids == d
            idx_d = np.where(mask)[0]
            if len(idx_d) == 0:
                continue
            cum = np.cumsum(index_close[idx_d])
            cnt = np.arange(1, len(idx_d) + 1, dtype=np.float64)
            idx_vwap[idx_d] = cum / cnt
        idx_vwap = np.where(idx_vwap > 0, idx_vwap, 1.0)
        index_vwap_dev = (index_close - idx_vwap) / idx_vwap

        # ── below_vwap_count / above_vwap_count over last 8 bars ──
        below_vwap_count = np.zeros(n, dtype=np.float64)
        above_vwap_count = np.zeros(n, dtype=np.float64)
        lookback = 8
        for i in range(lookback, n):
            below_cnt = 0
            above_cnt = 0
            for j in range(i - lookback, i):
                if close[j] < vwap[j]:
                    below_cnt += 1
                if close[j] > vwap[j]:
                    above_cnt += 1
            below_vwap_count[i] = below_cnt
            above_vwap_count[i] = above_cnt

        # ── Relative volume ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── Prev bar high/low ──
        prev_high = np.zeros(n, dtype=np.float64)
        prev_low = np.zeros(n, dtype=np.float64)
        prev_high[1:] = high[:-1]
        prev_low[1:] = low[:-1]

        # ── Time filter: 09:35-13:30 (575-810) ──
        time_ok = (time_mins >= 575) & (time_mins <= 810)

        long_entry = (
            (below_vwap_count >= vwap_cnt_min) &
            (close > vwap) &
            (ema9 > ema21) &
            (adx >= adx_min) &
            (index_vwap_dev >= 0) &
            (close > prev_high) &
            (rel_vol >= rel_vol_min) &
            time_ok
        )

        short_entry = (
            (above_vwap_count >= vwap_cnt_min) &
            (close < vwap) &
            (ema9 < ema21) &
            (adx >= adx_min) &
            (index_vwap_dev <= 0) &
            (close < prev_low) &
            (rel_vol >= rel_vol_min) &
            time_ok
        )

        # ── Signal exit: close crosses back through VWAP ──
        sig_exit_long = (close < vwap) & (vwap > 0)
        sig_exit_short = (close > vwap) & (vwap > 0)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_pct=target_pct,
            time_stop_bars=24,
        )
