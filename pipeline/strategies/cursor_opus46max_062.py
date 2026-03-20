"""Gap Fill Partial v1 — cursor_opus46max_062

Thesis: Target 50% gap fill for ~78% win rate. EMA 9/21 cross confirmation.
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


def _ema(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    out[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_062"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.004, low=0.002, high=0.008),
            TunableParam("gap_max", default=0.012, low=0.008, high=0.02),
            TunableParam("vix_min", default=11.0, low=8.0, high=14.0),
            TunableParam("vix_max", default=18.0, low=14.0, high=24.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.004)
        gap_max = params.get("gap_max", 0.012)
        vix_min = params.get("vix_min", 11.0)
        vix_max = params.get("vix_max", 18.0)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        atr10 = _compute_atr(high_, low_, close, 10)

        # Gap and half-gap target per day
        gap_pct = np.zeros(n, dtype=np.float64)
        half_gap_target = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close = np.nan
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            day_open = open_[idx[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap = (day_open - prev_close) / prev_close
                gap_pct[idx] = gap
                half_gap_target[idx] = (day_open + prev_close) / 2.0
            prev_close = close[idx[-1]]

        # Volume ratio
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        vix_ok = (vix >= vix_min) & (vix <= vix_max)
        time_ok = (time_mins >= 560) & (time_mins <= 600)  # 09:20-10:00

        # EMA 9/21 cross
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            gp = gap_pct[i]
            # Gap down: EMA9 crosses above EMA21
            if (gp < -gap_min and gp > -gap_max
                    and close[i] > ema9[i] and vol_ratio[i] > 1.0
                    and vix_ok[i] and time_ok[i]
                    and ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i]):
                long_entry[i] = True
            # Gap up: EMA9 crosses below EMA21
            if (gp > gap_min and gp < gap_max
                    and close[i] < ema9[i] and vol_ratio[i] > 1.0
                    and vix_ok[i] and time_ok[i]
                    and ema9[i-1] >= ema21[i-1] and ema9[i] < ema21[i]):
                short_entry[i] = True

        # Signal exit: price reaches half-gap target
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if gap_pct[i] < 0 and half_gap_target[i] > 0 and close[i] >= half_gap_target[i]:
                signal_exit_long[i] = True
            if gap_pct[i] > 0 and half_gap_target[i] > 0 and close[i] <= half_gap_target[i]:
                signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr10,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=0.4,
            time_stop_bars=90,
        )
