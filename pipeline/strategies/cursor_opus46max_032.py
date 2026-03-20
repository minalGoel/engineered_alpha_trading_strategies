"""Intraday Trend Channel v1 — cursor_opus46max_032

Thesis: A linear regression channel fitted on last 60 bars identifies
intraday trend. Buy at lower channel boundary in uptrend (pullback to
support), sell at upper boundary in downtrend. R-squared filter ensures
trend channel is a good fit.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_032"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("lr_period", default=60.0, low=40.0, high=80.0),
            TunableParam("channel_pos_long", default=0.15, low=0.05, high=0.25),
            TunableParam("channel_pos_short", default=0.85, low=0.75, high=0.95),
            TunableParam("r_squared_min", default=0.7, low=0.5, high=0.85),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        lr_period = int(params.get("lr_period", 60.0))
        cp_long = params.get("channel_pos_long", 0.15)
        cp_short = params.get("channel_pos_short", 0.85)
        r2_min = params.get("r_squared_min", 0.7)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Rolling linear regression channel ──
        linreg_val = np.zeros(n, dtype=np.float64)
        linreg_slope = np.zeros(n, dtype=np.float64)
        linreg_std = np.zeros(n, dtype=np.float64)
        r_squared = np.zeros(n, dtype=np.float64)

        x = np.arange(lr_period, dtype=np.float64)
        x_mean = np.mean(x)
        x_var = np.sum((x - x_mean) ** 2)

        for i in range(lr_period - 1, n):
            y = close[i - lr_period + 1:i + 1]
            y_mean = np.mean(y)
            if x_var < 1e-10:
                continue
            cov_xy = np.sum((x - x_mean) * (y - y_mean))
            slope = cov_xy / x_var
            intercept = y_mean - slope * x_mean
            linreg_slope[i] = slope
            linreg_val[i] = intercept + slope * (lr_period - 1)  # current fitted value
            residuals = y - (intercept + slope * x)
            linreg_std[i] = np.std(residuals)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            if ss_tot > 1e-10:
                r_squared[i] = 1.0 - ss_res / ss_tot

        # ── Channel bounds ──
        linreg_upper = linreg_val + 2.0 * linreg_std
        linreg_lower = linreg_val - 2.0 * linreg_std
        channel_width = linreg_upper - linreg_lower
        channel_width = np.where(channel_width > 0, channel_width, 1e-10)
        channel_pos = (close - linreg_lower) / channel_width

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── Slope consistency: same sign for 30+ bars ──
        slope_consistent = np.zeros(n, dtype=np.bool_)
        pos_count = 0
        neg_count = 0
        for i in range(n):
            if linreg_slope[i] > 0:
                pos_count += 1
                neg_count = 0
            elif linreg_slope[i] < 0:
                neg_count += 1
                pos_count = 0
            else:
                pos_count = 0
                neg_count = 0
            slope_consistent[i] = pos_count >= 30 or neg_count >= 30

        # ── Time filter ──
        time_ok = (time_mins >= 615) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (linreg_slope > 0) & (channel_pos < cp_long) &
            (close > linreg_lower) & (close > opn) &
            (r_squared > r2_min) & slope_consistent &
            vol_ok & time_ok
        )
        short_entry = (
            (linreg_slope < 0) & (channel_pos > cp_short) &
            (close < linreg_upper) & (close < opn) &
            (r_squared > r2_min) & slope_consistent &
            vol_ok & time_ok
        )

        # ── Signal exit: slope flips sign ──
        signal_exit_long = linreg_slope < 0
        signal_exit_short = linreg_slope > 0

        # ── Target indicator: linreg midline ──
        target_ind = linreg_val.copy()

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=target_ind,
            use_target_indicator=True,
            stop_loss_pct=stop_pct,
            time_stop_bars=45,
        )
