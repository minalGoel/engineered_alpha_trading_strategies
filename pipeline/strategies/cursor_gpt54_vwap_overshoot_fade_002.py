"""VWAP Overshoot Fade 002 — cursor_gpt54_vwap_overshoot_fade_002

Thesis: Same family as 001 but uses RSI(3) with wider thresholds (15/85) and
slightly deeper VWAP deviation (0.6%/0.6%).  VIX < 24.

Entry long: close <= VWAP*0.994, RSI(3) <= 15, rel_vol >= 1.2, ADX(14) <= 22,
    close > BB lower, abs(index_ret_15) <= 0.004, bullish bar.
Entry short: mirror with close >= VWAP*1.006, RSI(3) >= 85.
Target: VWAP (use_target_indicator).  Stop: 0.45%.  BE after +0.25%.
Time stop: 12 bars.
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


def _compute_rsi_wilder(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


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
    name = "vwap_overshoot_fade_002"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 675     # 11:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_long", default=0.006, low=0.003, high=0.010),
            TunableParam("vwap_dev_short", default=0.006, low=0.003, high=0.010),
            TunableParam("rsi_long_thresh", default=15.0, low=8.0, high=22.0),
            TunableParam("rsi_short_thresh", default=85.0, low=78.0, high=92.0),
            TunableParam("adx_max", default=22.0, low=16.0, high=28.0),
            TunableParam("rel_vol_min", default=1.2, low=1.0, high=2.0),
            TunableParam("index_ret_max", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0045, low=0.003, high=0.007),
            TunableParam("breakeven_pct", default=0.0025, low=0.002, high=0.004),
            TunableParam("vix_max", default=24.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_dev_l = params.get("vwap_dev_long", 0.006)
        vwap_dev_s = params.get("vwap_dev_short", 0.006)
        rsi_l = params.get("rsi_long_thresh", 15.0)
        rsi_s = params.get("rsi_short_thresh", 85.0)
        adx_max = params.get("adx_max", 22.0)
        rel_vol_min = params.get("rel_vol_min", 1.2)
        idx_ret_max = params.get("index_ret_max", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0045)
        be_pct = params.get("breakeven_pct", 0.0025)
        vix_max = params.get("vix_max", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        atr = _compute_atr(high, low, close, 14)
        rsi_fast = _compute_rsi_wilder(close, 3)  # RSI(3) for this variant
        adx = _compute_adx(high, low, close, 14)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── BB (SMA20, 2*std) ──
        close_sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        close_sma20 = np.nan_to_num(close_sma20, nan=0.0)
        close_std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        close_std20 = np.nan_to_num(close_std20, nan=0.0)
        bb_upper = close_sma20 + 2.0 * close_std20
        bb_lower = close_sma20 - 2.0 * close_std20

        # ── Relative volume ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── Index return 15 bars ──
        idx_ret_15 = np.zeros(n, dtype=np.float64)
        for i in range(15, n):
            if index_close[i - 15] > 0:
                idx_ret_15[i] = index_close[i] / index_close[i - 15] - 1.0

        # ── OR range filter ──
        # AUDIT FIX: The original threshold of 1.8 * ATR(14) was calibrated for a
        # 1-min ATR, but the opening range covers ~15 bars.  A 15-bar range is
        # naturally ~sqrt(15) ≈ 3.9x the 1-min ATR under random-walk assumptions.
        # Using 1.8 excluded every single day.  Corrected to 6.0 * ATR(14) which
        # only filters genuinely extreme gap/spike open days (>6σ equivalent).
        unique_days = np.unique(day_ids)
        or_range_ok = np.ones(n, dtype=np.bool_)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            or_bars = idx[time_mins[idx] < 570]
            if len(or_bars) == 0:
                continue
            or_rng = np.max(high[or_bars]) - np.min(low[or_bars])
            atr_val = atr[or_bars[-1]] if atr[or_bars[-1]] > 0 else atr[idx[-1]]
            if atr_val > 0 and or_rng > 6.0 * atr_val:
                or_range_ok[idx] = False

        # ── Time filter: 09:25-11:15 (565-675) ──
        time_ok = (time_mins >= 565) & (time_mins <= 675)
        vix_ok = vix < vix_max

        bullish = close > open_
        bearish = close < open_

        long_entry = (
            (close <= vwap * (1 - vwap_dev_l)) &
            (rsi_fast <= rsi_l) &
            (rel_vol >= rel_vol_min) &
            (adx <= adx_max) &
            (close > bb_lower) &
            (np.abs(idx_ret_15) <= idx_ret_max) &
            bullish &
            time_ok &
            vix_ok &
            or_range_ok
        )

        short_entry = (
            (close >= vwap * (1 + vwap_dev_s)) &
            (rsi_fast >= rsi_s) &
            (rel_vol >= rel_vol_min) &
            (adx <= adx_max) &
            (close < bb_upper) &
            (np.abs(idx_ret_15) <= idx_ret_max) &
            bearish &
            time_ok &
            vix_ok &
            or_range_ok
        )

        # ── Signal exit: close crosses VWAP OR RSI crosses 55/45 against ──
        sig_exit_long = ((close >= vwap) & (vwap > 0)) | (rsi_fast >= 55)
        sig_exit_short = ((close <= vwap) & (vwap > 0)) | (rsi_fast <= 45)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=12,
        )
