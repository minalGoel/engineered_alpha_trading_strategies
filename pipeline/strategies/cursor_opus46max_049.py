"""Multi Timeframe Breakout v1 — cursor_opus46max_049

Thesis: A breakout confirmed on both 1-min AND 5-min timeframes has
significantly higher reliability than either alone. The 5-min chart filters
noise from 1-min false breakouts; the 1-min provides precise entry timing.
We use 60-bar Donchian on 1-min and simulate 5-min via 150-bar Donchian on
1-min data (equivalent to 30 bars on 5-min chart).
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
    name = "cursor_opus46max_049"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 840     # 14:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dc_1m_period", default=60.0, low=40.0, high=80.0),
            TunableParam("dc_5m_period", default=150.0, low=100.0, high=200.0),
            TunableParam("vol_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.006, low=0.004, high=0.010),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dc_1m = int(params.get("dc_1m_period", 60.0))
        dc_5m = int(params.get("dc_5m_period", 150.0))
        vol_mult = params.get("vol_mult", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.003)
        tgt_pct = params.get("target_pct", 0.006)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── 1-min Donchian Channel (dc_1m bars) ──
        dc1_upper = np.zeros(n, dtype=np.float64)
        dc1_lower = np.zeros(n, dtype=np.float64)
        for i in range(dc_1m, n):
            dc1_upper[i] = np.max(high[i - dc_1m:i])
            dc1_lower[i] = np.min(low[i - dc_1m:i])

        # ── 5-min equivalent Donchian Channel (dc_5m bars on 1-min) ──
        dc5_upper = np.zeros(n, dtype=np.float64)
        dc5_lower = np.zeros(n, dtype=np.float64)
        for i in range(dc_5m, n):
            dc5_upper[i] = np.max(high[i - dc_5m:i])
            dc5_lower[i] = np.min(low[i - dc_5m:i])

        # ── Breakout flags ──
        break_1m_long = np.zeros(n, dtype=np.bool_)
        break_1m_short = np.zeros(n, dtype=np.bool_)
        break_5m_long = np.zeros(n, dtype=np.bool_)
        break_5m_short = np.zeros(n, dtype=np.bool_)

        for i in range(dc_5m, n):
            if dc1_upper[i] > 0 and close[i] > dc1_upper[i]:
                break_1m_long[i] = True
            if dc1_lower[i] > 0 and close[i] < dc1_lower[i]:
                break_1m_short[i] = True
            if dc5_upper[i] > 0 and close[i] > dc5_upper[i]:
                break_5m_long[i] = True
            if dc5_lower[i] > 0 and close[i] < dc5_lower[i]:
                break_5m_short[i] = True

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 840)

        # ── Entries: dual-timeframe confirmation ──
        long_entry = (
            break_1m_long & break_5m_long &
            (close > vwap) & vol_ok & time_ok
        )
        short_entry = (
            break_1m_short & break_5m_short &
            (close < vwap) & vol_ok & time_ok
        )

        # ── Signal exit: 5-min Donchian reversal (close back inside 5m channel for 2 bars) ──
        inside_5m_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if (dc5_upper[i] > 0 and dc5_lower[i] > 0 and
                    close[i] >= dc5_lower[i] and close[i] <= dc5_upper[i]):
                inside_5m_count[i] = inside_5m_count[i-1] + 1
            else:
                inside_5m_count[i] = 0

        signal_exit_long = inside_5m_count >= 2
        signal_exit_short = inside_5m_count >= 2

        # ── Donchian midpoint for target indicator ──
        dc_mid = np.where(
            (dc1_upper > 0) & (dc1_lower > 0),
            (dc1_upper + dc1_lower) / 2.0,
            0.0
        )

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
            trailing_activate_pct=0.0035,
            time_stop_bars=60,
        )
