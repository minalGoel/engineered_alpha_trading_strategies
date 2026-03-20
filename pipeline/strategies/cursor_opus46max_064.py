"""Gap Continuation v1 — cursor_opus46max_064

Thesis: Gap-and-go pattern. Stock gaps on heavy volume, breaks opening range,
continues in gap direction. Enter on ORB with volume confirmation.
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
    name = "cursor_opus46max_064"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 720     # 12:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.005, low=0.003, high=0.01),
            TunableParam("vol_surge_min", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_min", default=13.0, low=10.0, high=16.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("target_pct", default=0.007, low=0.003, high=0.012),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.005)
        vol_surge_min = params.get("vol_surge_min", 2.0)
        vix_min = params.get("vix_min", 13.0)
        vix_max = params.get("vix_max", 25.0)
        target_pct = params.get("target_pct", 0.007)
        stop_pct = params.get("stop_loss_pct", 0.005)

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

        atr10 = _compute_atr(high_, low_, close, 10)
        ema5 = _ema(close, 5)
        ema13 = _ema(close, 13)

        # Gap, opening range per day
        gap_pct = np.zeros(n, dtype=np.float64)
        or_high = np.full(n, -np.inf, dtype=np.float64)
        or_low = np.full(n, np.inf, dtype=np.float64)
        vol_surge = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close = np.nan

        # Avg first-5-min volume (approximate with 20-bar avg)
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)

        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            day_open = open_[idx[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap_pct[idx] = (day_open - prev_close) / prev_close
            if len(idx) >= 5:
                orh = np.max(high_[idx[:5]])
                orl = np.min(low_[idx[:5]])
                or_high[idx] = orh
                or_low[idx] = orl
                total_vol_5 = np.sum(volume[idx[:5]])
                avg_5 = np.mean(avg_vol[idx[:5]]) * 5
                if avg_5 > 0:
                    vol_surge[idx] = total_vol_5 / avg_5
            prev_close = close[idx[-1]]

        vix_ok = (vix >= vix_min) & (vix <= vix_max)
        time_ok = (time_mins >= 560) & (time_mins <= 615)  # 09:20-10:15

        # Bar body ratio
        bar_range = high_ - low_
        bar_body = np.abs(close - open_)
        body_ratio = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if bar_range[i] > 1e-8:
                body_ratio[i] = bar_body[i] / bar_range[i]

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            gp = gap_pct[i]
            # Gap up continuation: breaks above OR high
            if (gp > gap_min and close[i] > or_high[i]
                    and vol_surge[i] > vol_surge_min
                    and close[i] > ema5[i] and ema5[i] > ema13[i]
                    and body_ratio[i] > 0.6
                    and vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            # Gap down continuation: breaks below OR low
            if (gp < -gap_min and close[i] < or_low[i]
                    and vol_surge[i] > vol_surge_min
                    and close[i] < ema5[i] and ema5[i] < ema13[i]
                    and body_ratio[i] > 0.6
                    and vix_ok[i] and time_ok[i]):
                short_entry[i] = True

        # Signal exit: EMA5 crosses EMA13 against position
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema5[i-1] >= ema13[i-1] and ema5[i] < ema13[i]:
                signal_exit_long[i] = True
            if ema5[i-1] <= ema13[i-1] and ema5[i] > ema13[i]:
                signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr10,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.003,
            time_stop_bars=60,
        )
