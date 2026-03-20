"""Williams %R Reversion v1 — cursor_opus46max_020

Thesis: Williams %R(20) extremes (<-90 or >-10) combined with volume spike
(>2x) identify capitulation events. Enter when %R turns back from extreme.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_020"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("wr_long_thresh", default=-90.0, low=-98.0, high=-80.0),
            TunableParam("wr_short_thresh", default=-10.0, low=-20.0, high=-2.0),
            TunableParam("vol_spike_thresh", default=2.0, low=1.2, high=3.0),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("target_pct", default=0.0025, low=0.001, high=0.005),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        wr_lt = params.get("wr_long_thresh", -90.0)
        wr_st = params.get("wr_short_thresh", -10.0)
        vol_spike = params.get("vol_spike_thresh", 2.0)
        vix_max = params.get("vix_max", 25.0)
        tgt_pct = params.get("target_pct", 0.0025)
        stop_pct = params.get("stop_loss_pct", 0.002)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Williams %R(20) ──
        wr = np.full(n, -50.0, dtype=np.float64)
        for i in range(19, n):
            hh = np.max(high[i - 19:i + 1])
            ll = np.min(low[i - 19:i + 1])
            denom = hh - ll
            if denom > 1e-10:
                wr[i] = ((hh - close[i]) / denom) * -100.0
            else:
                wr[i] = -50.0

        # ── %R slope (3-bar) ──
        wr_slope = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            wr_slope[i] = wr[i] - wr[i - 3]

        # ── Volume spike ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Bars in extreme ──
        bars_extreme = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if wr[i] < wr_lt or wr[i] > wr_st:
                bars_extreme[i] = bars_extreme[i - 1] + 1

        # ── Bullish / bearish bar ──
        bullish = close > opn
        bearish = close < opn

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (wr < wr_lt)
            & (vol_ratio > vol_spike)
            & (close > vwap * 0.99)
            & (wr_slope > 0) & bullish
            & (bars_extreme <= 5)
            & vix_ok & time_ok
        )
        short_entry = (
            (wr > wr_st)
            & (vol_ratio > vol_spike)
            & (close < vwap * 1.01)
            & (wr_slope < 0) & bearish
            & (bars_extreme <= 5)
            & vix_ok & time_ok
        )

        # ── Signal exit: %R crosses -50 or re-enters extreme ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if wr[i] >= -50 and wr[i - 1] < -50:
                sig_exit_long[i] = True
            if wr[i] < -95:
                sig_exit_long[i] = True
            if wr[i] <= -50 and wr[i - 1] > -50:
                sig_exit_short[i] = True
            if wr[i] > -5:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=20,
        )
