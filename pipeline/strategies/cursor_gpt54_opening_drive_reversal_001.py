"""Opening Drive Reversal — cursor_gpt54_opening_drive_reversal_001

Thesis: Exploits opening-drive exhaustion after the first burst of post-auction
directional flow in liquid NSE names.  Market orders, stop runs, and emotional
retail participation exaggerate the initial impulse.

Entry long: close <= VWAP*0.995, RSI(2) <= 6, wick_down >= 0.35, flat VWAP slope,
    rel_vol >= 1.5, index_return_5 > -0.003, bullish bar.
Entry short: close >= VWAP*1.005, RSI(2) >= 94, wick_up >= 0.35, flat VWAP slope,
    rel_vol >= 1.5, index_return_5 < 0.003, bearish bar.
Target: VWAP (use_target_indicator).  Stop: 0.45%.  Time stop: 12 bars.
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
    """RSI using Wilder's smoothing."""
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


class Strategy(BaseStrategy):
    name = "opening_drive_reversal_001"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 615     # 10:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_long", default=0.005, low=0.003, high=0.010),
            TunableParam("vwap_dev_short", default=0.005, low=0.003, high=0.010),
            TunableParam("rsi_long_thresh", default=6.0, low=3.0, high=12.0),
            TunableParam("rsi_short_thresh", default=94.0, low=88.0, high=97.0),
            TunableParam("wick_min", default=0.35, low=0.20, high=0.50),
            TunableParam("rel_vol_min", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_loss_pct", default=0.0045, low=0.003, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_dev_l = params.get("vwap_dev_long", 0.005)
        vwap_dev_s = params.get("vwap_dev_short", 0.005)
        rsi_l = params.get("rsi_long_thresh", 6.0)
        rsi_s = params.get("rsi_short_thresh", 94.0)
        wick_min = params.get("wick_min", 0.35)
        rel_vol_min = params.get("rel_vol_min", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.0045)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        index_close = df["index_close"].to_numpy().astype(np.float64)

        atr = _compute_atr(high, low, close, 14)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── RSI(2) Wilder ──
        rsi_fast = _compute_rsi_wilder(close, 2)

        # ── Wick ratios ──
        body_high = np.maximum(open_, close)
        body_low = np.minimum(open_, close)
        bar_range = high - low
        bar_range_safe = np.where(bar_range < 0.0001, 0.0001, bar_range)
        wick_down_ratio = (body_low - low) / bar_range_safe
        wick_up_ratio = (high - body_high) / bar_range_safe

        # ── VWAP slope over 5 bars ──
        vwap_slope_5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            vwap_slope_5[i] = vwap[i] - vwap[i - 5]

        # ── Relative volume ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── Index return 5 bars ──
        idx_ret_5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if index_close[i - 5] > 0:
                idx_ret_5[i] = index_close[i] / index_close[i - 5] - 1.0

        # ── OR range filter ──
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
            if atr_val > 0 and or_rng > 1.8 * atr_val:
                or_range_ok[idx] = False

        # ── Time filter: 09:20-10:15 (560-615) ──
        time_ok = (time_mins >= 560) & (time_mins <= 615)

        flat_vwap = np.abs(vwap_slope_5) <= 0.08 * atr
        bullish = close > open_
        bearish = close < open_

        long_entry = (
            (close <= vwap * (1 - vwap_dev_l)) &
            (rsi_fast <= rsi_l) &
            (wick_down_ratio >= wick_min) &
            flat_vwap &
            (rel_vol >= rel_vol_min) &
            (idx_ret_5 > -0.003) &
            bullish &
            time_ok &
            or_range_ok
        )

        short_entry = (
            (close >= vwap * (1 + vwap_dev_s)) &
            (rsi_fast >= rsi_s) &
            (wick_up_ratio >= wick_min) &
            flat_vwap &
            (rel_vol >= rel_vol_min) &
            (idx_ret_5 < 0.003) &
            bearish &
            time_ok &
            or_range_ok
        )

        # ── Signal exit: close touches VWAP OR impulse resumes through signal bar extreme ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vwap[i] > 0 and close[i] >= vwap[i]:
                sig_exit_long[i] = True
            if vwap[i] > 0 and close[i] <= vwap[i]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=12,
        )
